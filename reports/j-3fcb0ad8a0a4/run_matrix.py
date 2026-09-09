#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time


PYTHON = "/opt/venv/bin/python"
CHILD = Path(__file__).with_name("sampler_child.py")
REPO_ROOT = Path(__file__).resolve().parents[2]


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def parse_prefixed_lines(path, prefix):
    records = []
    if not path.exists():
        return records
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith(prefix + " "):
            try:
                records.append(json.loads(line[len(prefix) + 1 :]))
            except json.JSONDecodeError:
                continue
    return records


def command_output(command, cwd=None):
    process = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return f"$ {' '.join(command)}\n{process.stdout}"


def wait_for_files(paths, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if all(path.exists() for path in paths):
            return True
        time.sleep(0.05)
    return False


def kill_process_group(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def run_scenario(
    name,
    ranks,
    cache_dir,
    output_dir,
    trials,
    timeout,
    hash_seed_base,
    loader_trace=False,
):
    output_dir.mkdir(parents=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    ready_dir = output_dir / "ready"
    ready_dir.mkdir()
    go_file = output_dir / "go"
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{REPO_ROOT}:{existing_pythonpath}" if existing_pythonpath else str(REPO_ROOT)
    )
    environment.update(
        {
            "AITER_ROOT_DIR": str(cache_dir),
            "AITER_JIT_DIR": str(cache_dir / "jit"),
            "AITER_LOG_MORE": "2",
            "AITER_LOG_LEVEL": "WARNING",
            "AITER_DEBUG": "0",
            "GPU_ARCHS": "gfx942",
            "PYTHONFAULTHANDLER": "1",
            "TORCH_SHOW_CPP_STACKTRACES": "1",
            "LIBC_FATAL_STDERR_": "1",
            "HSA_ENABLE_COREDUMP": "1",
            "AMDGPU_ENABLE_COREDUMP": "1",
        }
    )
    if loader_trace:
        environment["LD_DEBUG"] = "libs"

    processes = []
    for rank in ranks:
        stdout_path = output_dir / f"rank-{rank}.stdout.log"
        stderr_path = output_dir / f"rank-{rank}.stderr.log"
        ready_file = ready_dir / str(rank)
        child_environment = environment.copy()
        child_environment["HIP_VISIBLE_DEVICES"] = str(rank)
        child_environment["PYTHONHASHSEED"] = str(hash_seed_base + rank)
        command = [
            PYTHON,
            str(CHILD),
            "--rank",
            str(rank),
            "--cache-dir",
            str(cache_dir),
            "--ready-file",
            str(ready_file),
            "--go-file",
            str(go_file),
            "--trials",
            str(trials),
        ]
        with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
            process = subprocess.Popen(
                command,
                stdout=stdout,
                stderr=stderr,
                env=child_environment,
                start_new_session=True,
            )
        processes.append(
            {
                "rank": rank,
                "pid": process.pid,
                "process": process,
                "stdout_path": stdout_path,
                "stderr_path": stderr_path,
                "ready_file": ready_file,
                "command": command,
                "pythonhashseed": child_environment["PYTHONHASHSEED"],
                "hip_visible_devices": child_environment["HIP_VISIBLE_DEVICES"],
            }
        )

    ready = wait_for_files(
        [record["ready_file"] for record in processes], min(timeout, 180.0)
    )
    scenario_started = time.time_ns()
    if ready:
        go_file.write_text(str(scenario_started))
    else:
        for record in processes:
            kill_process_group(record["process"])

    deadline = time.monotonic() + timeout
    timed_out = False
    while any(record["process"].poll() is None for record in processes):
        if time.monotonic() >= deadline:
            timed_out = True
            for record in processes:
                kill_process_group(record["process"])
            break
        time.sleep(0.05)

    process_records = []
    events = []
    results = []
    errors = []
    for record in processes:
        process = record["process"]
        return_code = process.returncode
        signal_number = -return_code if return_code is not None and return_code < 0 else None
        result_records = parse_prefixed_lines(record["stdout_path"], "RESULT_JSON")
        error_records = parse_prefixed_lines(record["stdout_path"], "ERROR_JSON")
        event_records = parse_prefixed_lines(record["stdout_path"], "EVENT_JSON")
        events.extend(event_records)
        results.extend(result_records)
        errors.extend(error_records)
        process_records.append(
            {
                "rank": record["rank"],
                "pid": record["pid"],
                "command": record["command"],
                "stdout_path": str(record["stdout_path"]),
                "stderr_path": str(record["stderr_path"]),
                "pythonhashseed": record["pythonhashseed"],
                "hip_visible_devices": record["hip_visible_devices"],
                "ready": record["ready_file"].exists(),
                "return_code": return_code,
                "signal": signal_number,
                "timed_out": timed_out,
                "result": result_records[-1] if result_records else None,
                "error": error_records[-1] if error_records else None,
            }
        )

    events.sort(key=lambda event: event["time_ns"])
    with (output_dir / "events.jsonl").open("w") as events_file:
        for event in events:
            events_file.write(json.dumps(event, sort_keys=True) + "\n")
    write_json(output_dir / "processes.json", process_records)

    library_paths = sorted(
        {
            event["library_path"]
            for event in events
            if "library_path" in event and Path(event["library_path"]).exists()
        }
    )
    artifact_records = []
    for library_path in library_paths:
        path = Path(library_path)
        artifact = {
            "path": str(path),
            "size": path.stat().st_size,
            "sha256": subprocess.run(
                ["sha256sum", str(path)], text=True, capture_output=True, check=True
            ).stdout.split()[0],
        }
        artifact["file"] = command_output(["file", str(path)])
        artifact["ldd"] = command_output(["ldd", str(path)])
        artifact["readelf_header"] = command_output(["readelf", "-h", str(path)])
        artifact["readelf_dynamic"] = command_output(["readelf", "-d", str(path)])
        artifact["undefined_symbols"] = command_output(
            ["nm", "-D", "--undefined-only", str(path)]
        )
        artifact_records.append(artifact)
    write_json(output_dir / "artifacts.json", artifact_records)

    compile_starts = [event for event in events if event["event"] == "compile_start"]
    compile_ends = [event for event in events if event["event"] == "compile_end"]
    load_starts = [event for event in events if event["event"] == "load_start"]
    load_ends = [event for event in events if event["event"] == "load_end"]
    all_return_zero = all(record["return_code"] == 0 for record in process_records)
    all_contract_pass = bool(results) and all(
        result.get("contract_pass") for result in results
    )
    summary = {
        "name": name,
        "ranks": list(ranks),
        "cache_dir": str(cache_dir),
        "trials": trials,
        "timeout": timeout,
        "ready": ready,
        "timed_out": timed_out,
        "started_ns": scenario_started,
        "process_count": len(process_records),
        "all_return_zero": all_return_zero,
        "all_contract_pass": all_contract_pass,
        "compile_starts": len(compile_starts),
        "compile_ends": len(compile_ends),
        "load_starts": len(load_starts),
        "load_ends": len(load_ends),
        "library_paths": library_paths,
        "result_count": len(results),
        "error_count": len(errors),
        "passed": ready and all_return_zero and all_contract_pass,
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cold-cycles", type=int, default=6)
    parser.add_argument("--trials", type=int, default=4096)
    parser.add_argument("--child-timeout", type=float, default=900.0)
    args = parser.parse_args()

    if args.cache_root.exists() and any(args.cache_root.iterdir()):
        raise SystemExit(f"cache root is not empty: {args.cache_root}")
    args.cache_root.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)

    (args.output_root / "rocm-smi-before.txt").write_text(
        command_output(
            [
                "rocm-smi",
                "--showproductname",
                "--showserial",
                "--showbus",
                "--showid",
            ]
        )
    )
    (args.output_root / "hipcc-version.txt").write_text(
        command_output(["hipcc", "--version"])
    )

    scenarios = [
        ("single-cold", [0], args.cache_root / "single-cold", args.trials, 100000),
    ]
    for cycle in range(1, args.cold_cycles + 1):
        scenarios.append(
            (
                f"eight-cold-{cycle}",
                list(range(8)),
                args.cache_root / f"eight-cold-{cycle}",
                args.trials,
                200000 + cycle,
            )
        )
    scenarios.extend(
        [
            ("prebuild", [0], args.cache_root / "prebuild", 1, 300000),
            (
                "warm-eight",
                list(range(8)),
                args.cache_root / "prebuild",
                args.trials,
                400000,
            ),
            (
                "loader-trace",
                [0],
                args.cache_root / "prebuild",
                1,
                500000,
            ),
        ]
    )

    summaries = []
    for name, ranks, cache_dir, trials, hash_seed_base in scenarios:
        loader_trace = name == "loader-trace"
        summary = run_scenario(
            name,
            ranks,
            cache_dir,
            args.output_root / name,
            trials,
            args.child_timeout,
            hash_seed_base,
            loader_trace=loader_trace,
        )
        summaries.append(summary)
        print(json.dumps(summary, sort_keys=True), flush=True)

    (args.output_root / "rocm-smi-after.txt").write_text(
        command_output(
            [
                "rocm-smi",
                "--showproductname",
                "--showserial",
                "--showbus",
                "--showid",
            ]
        )
    )
    write_json(args.output_root / "matrix-summary.json", summaries)



if __name__ == "__main__":
    main()
