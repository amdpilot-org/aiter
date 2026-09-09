#!/usr/bin/env python3
"""Bounded fresh-process reproducer for ``top_k_renorm_probs``.

Each case runs in a child process because the native operator keeps its
workspace in function-local static storage. A fault in one case therefore
cannot mask later cases, and a repeated call cannot be treated as an
independent trial.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class TestCase:
    case_id: str
    batch_size: int
    vocab_size: int
    dtype: str
    layout: str
    k_mode: str
    k_values: tuple[int, ...]


def case_matrix() -> list[TestCase]:
    return [
        TestCase("scalar-v111-k1", 3, 111, "float32", "contiguous", "scalar", (1,)),
        TestCase("scalar-v111-k50", 3, 111, "float32", "contiguous", "scalar", (50,)),
        TestCase("scalar-v111-k110", 3, 111, "float32", "contiguous", "scalar", (110,)),
        TestCase("scalar-v111-k111", 3, 111, "float32", "contiguous", "scalar", (111,)),
        TestCase("scalar-v111-k112", 3, 111, "float32", "contiguous", "scalar", (112,)),
        TestCase("scalar-v500-k100", 3, 500, "float32", "contiguous", "scalar", (100,)),
        TestCase("scalar-v32000-k50", 3, 32000, "float32", "contiguous", "scalar", (50,)),
        TestCase("scalar-v32000-k100", 3, 32000, "float32", "contiguous", "scalar", (100,)),
        TestCase("scalar-v32000-k31999", 3, 32000, "float32", "contiguous", "scalar", (31999,)),
        TestCase("scalar-v32000-k32000", 3, 32000, "float32", "contiguous", "scalar", (32000,)),
        TestCase("scalar-v128256-k100", 3, 128256, "float32", "contiguous", "scalar", (100,)),
        TestCase("scalar-v128256-k128255", 3, 128256, "float32", "contiguous", "scalar", (128255,)),
        TestCase("scalar-v128256-k128256", 3, 128256, "float32", "contiguous", "scalar", (128256,)),
        TestCase("scalar-v128256-k128257", 3, 128256, "float32", "contiguous", "scalar", (128257,)),
        TestCase("per-row-v500-mixed", 3, 500, "float32", "contiguous", "per-row", (10, 100, 500)),
        TestCase("per-row-v32000-mixed", 3, 32000, "float32", "contiguous", "per-row", (10, 100, 32000)),
        TestCase("per-row-v32000-k50-disabled", 2, 32000, "float32", "contiguous", "per-row", (50, 32000)),
        TestCase("per-row-v128256-k50-disabled", 2, 128256, "float32", "contiguous", "per-row", (50, 128256)),
        TestCase("per-row-v1048576-k50-disabled", 2, 1048576, "float32", "contiguous", "per-row", (50, 1048576)),
        TestCase("per-row-v1048576-mixed", 3, 1048576, "float32", "contiguous", "per-row", (10, 100, 1048576)),
        TestCase("per-row-v1048576-boundary", 2, 1048576, "float32", "contiguous", "per-row", (1048575, 50)),
        TestCase("per-row-v32000-float16", 3, 32000, "float16", "contiguous", "per-row", (10, 100, 32000)),
        TestCase("per-row-v32000-bfloat16", 3, 32000, "bfloat16", "contiguous", "per-row", (10, 100, 32000)),
        TestCase("scalar-v128256-float16-full", 3, 128256, "float16", "contiguous", "scalar", (128256,)),
        TestCase("scalar-v128256-bfloat16-near", 3, 128256, "bfloat16", "contiguous", "scalar", (128255,)),
        TestCase("per-row-v32000-offset", 3, 32000, "float32", "storage-offset", "per-row", (10, 100, 32000)),
        TestCase("per-row-v1048576-offset-disabled", 2, 1048576, "float32", "storage-offset", "per-row", (50, 1048576)),
    ]


def truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def run_child(case: TestCase, timeout: int, repo_root: Path) -> dict[str, object]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--repo-root",
        str(repo_root),
        "--case-id",
        case.case_id,
        "--batch-size",
        str(case.batch_size),
        "--vocab-size",
        str(case.vocab_size),
        "--dtype",
        case.dtype,
        "--layout",
        case.layout,
        "--k-mode",
        case.k_mode,
        "--k-values",
        ",".join(str(value) for value in case.k_values),
    ]
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=repo_root,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        elapsed = time.monotonic() - started
        child_result: dict[str, object] | None = None
        for line in reversed(completed.stdout.splitlines()):
            if line.startswith("CHILD_RESULT "):
                child_result = json.loads(line[len("CHILD_RESULT ") :])
                break
        result = {
            "returncode": completed.returncode,
            "timed_out": False,
            "elapsed_seconds": round(elapsed, 3),
            "child_result": child_result,
            "stdout_tail": truncate(completed.stdout, 8000),
            "stderr_tail": truncate(completed.stderr, 12000),
        }
    except subprocess.TimeoutExpired as error:
        elapsed = time.monotonic() - started
        stdout = error.stdout.decode() if isinstance(error.stdout, bytes) else error.stdout or ""
        stderr = error.stderr.decode() if isinstance(error.stderr, bytes) else error.stderr or ""
        result = {
            "returncode": None,
            "timed_out": True,
            "elapsed_seconds": round(elapsed, 3),
            "child_result": None,
            "stdout_tail": truncate(stdout, 8000),
            "stderr_tail": truncate(stderr, 12000),
        }
    return result


def collect_environment(image_id: str, repo_root: Path) -> dict[str, object]:
    rocm_smI = subprocess.run(
        ["/opt/rocm/bin/rocm-smi", "--showproductname", "--showid"],
        capture_output=True,
        text=True,
        check=False,
    )
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    return {
        "image_id": image_id,
        "git_commit": git_commit,
        "python_executable": sys.executable,
        "repo_root": str(repo_root),
        "aiter_root_dir": os.environ.get("AITER_ROOT_DIR"),
        "rocm_smi_returncode": rocm_smI.returncode,
        "rocm_smi_stdout": rocm_smI.stdout,
        "rocm_smi_stderr": rocm_smI.stderr,
    }


def run_parent(args: argparse.Namespace) -> None:
    repo_root = Path(args.repo_root).resolve()
    cases = [case for case in case_matrix() if args.filter in case.case_id]
    if not cases:
        raise SystemExit(f"No case ID contains {args.filter!r}")

    output = {
        "schema_version": 1,
        "label": args.label,
        "environment": collect_environment(args.image_id, repo_root),
        "cases": [],
    }
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for case in cases:
        for repeat in range(1, args.repeat + 1):
            print(f"RUN {case.case_id} repeat={repeat}", flush=True)
            result = run_child(case, args.timeout, repo_root)
            result["case"] = asdict(case)
            result["repeat"] = repeat
            result["passed"] = result["returncode"] == 0
            output["cases"].append(result)
            output_path.write_text(json.dumps(output, indent=2) + "\n")
            status = "PASS" if result["passed"] else "FAIL"
            print(
                f"{status} {case.case_id} repeat={repeat} "
                f"returncode={result['returncode']} elapsed={result['elapsed_seconds']}s",
                flush=True,
            )

    output["summary"] = {
        "total_runs": len(output["cases"]),
        "passed": sum(result["passed"] for result in output["cases"]),
        "failed": sum(not result["passed"] for result in output["cases"]),
    }
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output["summary"]))
    if output["summary"]["failed"]:
        raise SystemExit(1)


def make_probabilities(case: TestCase):
    import torch

    torch.manual_seed(0x5A17E)
    if case.layout == "contiguous":
        probabilities = torch.rand(
            (case.batch_size, case.vocab_size), device="cuda", dtype=torch.float32
        )
    elif case.layout == "storage-offset":
        flat = torch.rand(
            (case.batch_size + 1) * case.vocab_size, device="cuda", dtype=torch.float32
        )
        probabilities = flat[case.vocab_size :].view(case.batch_size, case.vocab_size)
    else:
        raise ValueError(f"Unsupported layout: {case.layout}")

    probabilities = probabilities.clamp_min(0.05)
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
    dtype = getattr(torch, case.dtype)
    probabilities = probabilities.to(dtype)
    if not torch.all(probabilities > 0):
        raise AssertionError("Synthetic probabilities must be nonzero")
    return probabilities


def reference_renormalization(probabilities, k_values: list[int]):
    import torch

    float_probabilities = probabilities.float()
    reference = float_probabilities.clone()
    for row_index, k_value in enumerate(k_values):
        if k_value < probabilities.size(-1):
            ordered = torch.sort(float_probabilities[row_index], descending=True).values
            pivot = ordered[k_value - 1]
            reference[row_index][float_probabilities[row_index] < pivot] = 0
    return reference / reference.sum(dim=-1, keepdim=True)


def run_child_case(args: argparse.Namespace) -> None:
    import torch

    repo_root = Path(args.repo_root).resolve()
    sys.path.insert(0, str(repo_root))
    from aiter.ops import sampling  # noqa: F401
    from csrc.cpp_itfs import utils as jit_utils
    from csrc.cpp_itfs.sampling import top_k_renorm_probs as interface

    case = TestCase(
        args.case_id,
        args.batch_size,
        args.vocab_size,
        args.dtype,
        args.layout,
        args.k_mode,
        tuple(int(value) for value in args.k_values.split(",")),
    )
    probabilities = make_probabilities(case)
    if case.k_mode == "scalar":
        maybe_top_k_arr = None
        top_k_val = case.k_values[0]
    else:
        maybe_top_k_arr = torch.tensor(case.k_values, dtype=torch.int32, device="cuda")
        top_k_val = 0

    actual = torch.ops.aiter.top_k_renorm_probs(
        probabilities, maybe_top_k_arr, top_k_val
    )
    torch.cuda.synchronize()
    reference_k_values = (
        list(case.k_values)
        if case.k_mode == "per-row"
        else [case.k_values[0]] * case.batch_size
    )
    reference = reference_renormalization(probabilities, reference_k_values)
    difference = (actual.float() - reference).abs()
    max_abs_difference = difference.max().item()
    numerical_match = torch.allclose(actual, reference, rtol=1e-3, atol=1e-3)

    native_libraries = sorted(
        str(path)
        for path in Path(jit_utils.BUILD_DIR).glob("top_k_renorm_probs_*/lib.so")
        if path.is_file()
    )
    result = {
        "case_id": case.case_id,
        "input_dtype": str(probabilities.dtype),
        "input_layout": case.layout,
        "input_is_contiguous": probabilities.is_contiguous(),
        "output_dtype": str(actual.dtype),
        "output_shape": list(actual.shape),
        "max_abs_difference": max_abs_difference,
        "numerical_match": numerical_match,
        "actual_nonzero_counts": (actual != 0).sum(dim=-1).tolist(),
        "reference_nonzero_counts": (reference != 0).sum(dim=-1).tolist(),
        "python_wrapper_path": str(repo_root / "aiter/ops/sampling.py"),
        "python_interface_path": str(interface.__file__),
        "native_source_path": str(repo_root / "csrc/cpp_itfs/sampling/sampling.cuh"),
        "native_library_paths": native_libraries,
    }
    print("CHILD_RESULT " + json.dumps(result))
    if not numerical_match:
        raise AssertionError(
            f"Independent reference mismatch: max_abs_difference={max_abs_difference}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--label", default="main")
    parser.add_argument("--output", default="results.json")
    parser.add_argument("--filter", default="")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--image-id", default="unknown")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--case-id")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--vocab-size", type=int)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--layout", choices=["contiguous", "storage-offset"])
    parser.add_argument("--k-mode", choices=["scalar", "per-row"])
    parser.add_argument("--k-values")
    args = parser.parse_args()
    if args.child:
        required = [
            args.case_id,
            args.batch_size,
            args.vocab_size,
            args.dtype,
            args.layout,
            args.k_mode,
            args.k_values,
        ]
        if any(value is None for value in required):
            parser.error("child mode requires all case arguments")
    return args


def main() -> None:
    args = parse_args()
    if args.child:
        run_child_case(args)
    else:
        if args.repeat < 1:
            raise SystemExit("--repeat must be positive")
        run_parent(args)


if __name__ == "__main__":
    main()
