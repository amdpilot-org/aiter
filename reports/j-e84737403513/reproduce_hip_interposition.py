#!/usr/bin/env python3
import argparse
import ctypes
import ctypes.util
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


CASES = {
    "clean": {"stub_mode": "none", "stub_order": "before"},
    "stub_global_before": {"stub_mode": "global", "stub_order": "before"},
    "stub_local_before": {"stub_mode": "local", "stub_order": "before"},
    "aiter_then_stub_global": {"stub_mode": "global", "stub_order": "after"},
}

BINDING_SYMBOLS = (
    "hipGetDevicePropertiesR0600",
    "hipModuleLaunchKernel",
    "hipGetDevice",
    "hipGetDeviceCount",
)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DlInfo(ctypes.Structure):
    _fields_ = [
        ("dli_fname", ctypes.c_char_p),
        ("dli_fbase", ctypes.c_void_p),
        ("dli_sname", ctypes.c_char_p),
        ("dli_saddr", ctypes.c_void_p),
    ]


def dladdr(function):
    libdl = ctypes.CDLL("libdl.so.2")
    libdl.dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(DlInfo)]
    address = ctypes.cast(function, ctypes.c_void_p).value
    info = DlInfo()
    if libdl.dladdr(address, ctypes.byref(info)) == 0:
        return None
    return {
        "address": hex(address),
        "file": os.path.realpath(info.dli_fname.decode()),
        "symbol": info.dli_sname.decode(),
    }


def mapped_paths(patterns):
    paths = set()
    with open("/proc/self/maps") as handle:
        for line in handle:
            path = line.rstrip().split(maxsplit=5)[-1]
            if any(pattern in path for pattern in patterns):
                paths.add(os.path.realpath(path))
    return sorted(paths)


def global_symbol_owner(symbol):
    try:
        return dladdr(getattr(ctypes.CDLL(None), symbol))
    except AttributeError:
        return None


def load_stub(path, mode):
    if mode == "global":
        return ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
    if mode == "local":
        return ctypes.CDLL(path, mode=ctypes.RTLD_LOCAL)
    return None


def run_child(args):
    import torch
    import aiter
    from aiter.jit import core
    from aiter.ops.mha import fmha_v3_varlen_fwd

    torch.manual_seed(123)
    stub_handle = None
    if args.stub_order == "before":
        stub_handle = load_stub(args.stub_path, args.stub_mode)

    operation = fmha_v3_varlen_fwd
    core.get_module_custom_op("module_fmha_v3_varlen_fwd")
    extension_module = core.__mds["module_fmha_v3_varlen_fwd"]
    extension_path = extension_module.__file__

    if args.stub_order == "after":
        stub_handle = load_stub(args.stub_path, args.stub_mode)

    q = torch.randn((128, 8, 128), device="cuda", dtype=torch.bfloat16)
    cu_seqlens = torch.tensor([0, 128], device="cuda", dtype=torch.int32)
    scale = 1.0 / math.sqrt(128)
    q_float = q.float().transpose(0, 1).unsqueeze(0)
    reference = torch.softmax(q_float @ q_float.transpose(-2, -1) * scale, dim=-1)
    reference = reference @ q_float
    reference = reference.squeeze(0).transpose(0, 1)
    torch.cuda.synchronize()

    aiter_error = None
    max_abs_error = None
    try:
        output, _, _, _ = operation(
            q,
            q,
            q,
            cu_seqlens,
            cu_seqlens,
            128,
            128,
            128,
            0.0,
            scale,
            0.0,
            False,
            False,
            -1,
            -1,
            False,
            False,
            1,
        )
        torch.cuda.synchronize()
        max_abs_error = float((output.float() - reference).abs().max())
    except Exception as error:
        aiter_error = f"{type(error).__name__}: {error}"

    real_hip_name = ctypes.util.find_library("amdhip64")
    real_hip = ctypes.CDLL(real_hip_name)
    real_symbol_owner = dladdr(
        getattr(real_hip, "hipGetDevicePropertiesR0600")
    )
    result = {
        "case": args.case,
        "deepbind": args.deepbind,
        "stub_mode": args.stub_mode,
        "stub_order": args.stub_order,
        "torch_version": torch.__version__,
        "torch_hip_version": torch.version.hip,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
        "aiter_source": os.path.realpath(aiter.__file__),
        "extension_path": os.path.realpath(extension_path),
        "extension_sha256": sha256(extension_path),
        "stub_path": os.path.realpath(args.stub_path),
        "stub_sha256": sha256(args.stub_path),
        "real_hip_path": real_symbol_owner["file"],
        "global_symbol_owner": global_symbol_owner(
            "hipGetDevicePropertiesR0600"
        ),
        "real_symbol_owner": real_symbol_owner,
        "stub_symbol_owner": (
            dladdr(getattr(stub_handle, "hipGetDevicePropertiesR0600"))
            if stub_handle is not None
            else None
        ),
        "reference_gpu_pass": True,
        "aiter_gpu_pass": aiter_error is None and max_abs_error < 0.05,
        "max_abs_error": max_abs_error,
        "aiter_error": aiter_error,
        "mapped_paths": mapped_paths(
            [
                "libamdhip64.so",
                "libhip_stub.so",
                "module_fmha_v3_varlen_fwd.so",
                "libtorch_hip.so",
            ]
        ),
    }
    with open(args.result_file, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")


def parse_bindings(debug_log, extension_path):
    bindings = {}
    pattern = re.compile(
        r"binding file (\S+) \[0\] to (\S+) \[0\]: normal symbol `([^']+)'"
    )
    for line in debug_log.splitlines():
        match = pattern.search(line)
        if not match or match.group(1) != extension_path:
            continue
        requester, owner, symbol = match.groups()
        if symbol in BINDING_SYMBOLS:
            bindings[symbol] = os.path.realpath(owner)
    return bindings


def run_matrix(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for case_name, case_args in CASES.items():
        with tempfile.TemporaryDirectory(prefix=f"aiter-{case_name}-") as temp_dir:
            result_file = Path(temp_dir) / "result.json"
            child_env = os.environ.copy()
            child_env.update(
                {
                    "PYTHONPATH": str(Path(args.aiter_root).resolve()),
                    "AITER_JIT_DIR": str(Path(args.jit_dir).resolve()),
                    "TORCH_EXTENSIONS_DIR": str(
                        Path(args.torch_extensions_dir).resolve()
                    ),
                    "LD_DEBUG": "bindings",
                }
            )
            if args.deepbind == "off":
                child_env["AITER_DISABLE_DEEPBIND"] = "1"
            else:
                child_env.pop("AITER_DISABLE_DEEPBIND", None)
            command = [
                sys.executable,
                os.path.abspath(__file__),
                "--child",
                "--case",
                case_name,
                "--stub-path",
                str(Path(args.stub_path).resolve()),
                "--stub-mode",
                case_args["stub_mode"],
                "--stub-order",
                case_args["stub_order"],
                "--deepbind",
                args.deepbind,
                "--result-file",
                str(result_file),
            ]
            try:
                completed = subprocess.run(
                    command,
                    env=child_env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=args.timeout,
                    check=False,
                )
                returncode = completed.returncode
                stdout = completed.stdout
                stderr = completed.stderr
            except subprocess.TimeoutExpired as error:
                returncode = 124
                stdout = error.stdout or ""
                stderr = error.stderr or ""

            debug_path = output_dir / f"lddebug-{case_name}-{args.deepbind}.log"
            debug_path.write_text(stderr)
            child_result = None
            if result_file.exists():
                child_result = json.loads(result_file.read_text())
            bindings = {}
            if child_result:
                bindings = parse_bindings(stderr, child_result["extension_path"])
            result = {
                "case": case_name,
                "deepbind": args.deepbind,
                "returncode": returncode,
                "bindings": {
                    symbol: {
                        "owner": owner,
                        "expected_owner": "libamdhip64.so",
                        "correct": "libamdhip64.so" in os.path.basename(owner),
                    }
                    for symbol, owner in bindings.items()
                },
                "child": child_result,
            }
            results.append(result)

    report_path = output_dir / f"results-{args.deepbind}.json"
    with open(report_path, "w") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0 if all(result["returncode"] == 0 for result in results) else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--case")
    parser.add_argument("--stub-path")
    parser.add_argument("--stub-mode", choices=("none", "global", "local"))
    parser.add_argument("--stub-order", choices=("before", "after"))
    parser.add_argument("--deepbind", choices=("on", "off"))
    parser.add_argument("--result-file")
    parser.add_argument("--aiter-root")
    parser.add_argument("--jit-dir")
    parser.add_argument("--torch-extensions-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    if args.child:
        run_child(args)
        return 0

    required = ("aiter_root", "stub_path", "jit_dir", "torch_extensions_dir", "output_dir")
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error(f"missing arguments: {', '.join(missing)}")
    return run_matrix(args)


if __name__ == "__main__":
    raise SystemExit(main())
