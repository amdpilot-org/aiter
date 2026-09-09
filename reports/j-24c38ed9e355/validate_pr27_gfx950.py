import ctypes
import ctypes.util
import json
import os
import subprocess
import sys
from pathlib import Path


os.environ.setdefault("TRITON_CACHE_DIR", "/job/.triton-cache")
os.environ.setdefault("TRITON_HOME", "/job/.triton-home")
Path(os.environ["TRITON_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["TRITON_HOME"]).mkdir(parents=True, exist_ok=True)

import torch
import triton

from aiter.ops.shuffle import shuffle_weight
from aiter.ops.triton._gluon_kernels.gfx950.gemm.basic.gemm_a8w8 import (
    _gemm_a8w8_kernel,
    _gemm_a8w8_preshuffled_kernel,
)
from aiter.ops.triton.gemm.basic.gemm_a8w8 import (
    gemm_a8w8,
    gemm_a8w8_preshuffle,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.device_info import (
    _get_num_xcds,
    get_num_xcds,
)
from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config
from aiter.ops.triton.utils.types import (
    get_scaled_dot_format_string,
    torch_to_triton_dtype,
)
from op_tests.triton_tests.gemm.basic.test_gemm_a8w8 import (
    generate_gemm_a8w8_inputs,
)


REPO = Path("/job/aiter-validation")
OUTPUT = Path("/job/pr27_gfx950_validation.json")
IN_DTYPE = torch.float8_e4m3fn
OUT_DTYPE = torch.bfloat16
FP8_FORMAT = get_scaled_dot_format_string(torch_to_triton_dtype[IN_DTYPE])


def command_output(command):
    return subprocess.check_output(command, cwd=REPO, text=True).strip()


def module_record(module):
    return {
        "module": module.__name__,
        "source": str(Path(module.__file__).resolve()),
    }


def reference(x, weight, x_scale, w_scale, bias):
    product = torch.mm(x.float(), weight.float().t())
    result = product * torch.mm(x_scale, w_scale)
    if bias is not None:
        result = result + bias
    return result.to(OUT_DTYPE)


def timed_pair(call, reference_call, iterations=5):
    for _ in range(3):
        call()
        reference_call()
    torch.cuda.synchronize()
    consumer_start = torch.cuda.Event(enable_timing=True)
    consumer_end = torch.cuda.Event(enable_timing=True)
    reference_start = torch.cuda.Event(enable_timing=True)
    reference_end = torch.cuda.Event(enable_timing=True)
    consumer_start.record()
    for _ in range(iterations):
        call()
    consumer_end.record()
    reference_start.record()
    for _ in range(iterations):
        reference_call()
    reference_end.record()
    torch.cuda.synchronize()
    return {
        "iterations": iterations,
        "consumer_us": consumer_start.elapsed_time(consumer_end) * 1000 / iterations,
        "reference_us": reference_start.elapsed_time(reference_end) * 1000 / iterations,
    }


def graph_replay(call, expected):
    for _ in range(2):
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            call()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(call(), expected, atol=0.02, rtol=1e-2)
        return {"supported": True, "replayed": True}
    except Exception as error:
        torch.cuda.synchronize()
        return {
            "supported": False,
            "replayed": False,
            "error": f"{type(error).__name__}: {error}",
        }


def validate_case(device, name, shape, shuffle, graph_result):
    torch.cuda.set_device(device)
    torch.manual_seed(27)
    m, n, k = shape
    x, weight, weight_shuffled, x_scale, w_scale, bias, _ = generate_gemm_a8w8_inputs(
        M=m,
        N=n,
        K=k,
        in_dtype=IN_DTYPE,
        out_dtype=OUT_DTYPE,
        layout="TN",
        output=False,
        shuffle=shuffle,
    )
    expected = reference(x, weight, x_scale, w_scale, bias)
    config, tuned = get_gemm_config("GEMM-A8W8", m, n, k, backend="gluon")
    y = torch.empty_like(expected)
    if shuffle:
        call = lambda: gemm_a8w8_preshuffle(
            x, weight_shuffled, x_scale, w_scale, bias, OUT_DTYPE, y, config
        )
    else:
        call = lambda: gemm_a8w8(
            x,
            weight,
            x_scale,
            w_scale,
            bias,
            OUT_DTYPE,
            y,
            config,
            backend="gluon",
        )
    call()
    torch.cuda.synchronize()
    torch.testing.assert_close(y, expected, atol=0.02, rtol=1e-2)
    timing = timed_pair(
        call,
        lambda: reference(x, weight, x_scale, w_scale, bias),
    )
    return {
        "consumer": name,
        "device": device,
        "shape": {"M": m, "N": n, "K": k},
        "input_dtype": str(IN_DTYPE),
        "output_dtype": str(OUT_DTYPE),
        "fp8_format": FP8_FORMAT,
        "architecture": arch_info.get_arch(),
        "config": config,
        "config_is_tuned": tuned,
        "max_abs_diff": float((y.float() - expected.float()).abs().max()),
        "finite": bool(torch.isfinite(y).all()),
        "reference_finite": bool(torch.isfinite(expected).all()),
        "graph_replay": graph_result,
        "synchronized_pair_timing": timing,
    }


def graph_probe_main():
    probe = json.loads(os.environ["PR27_GRAPH_PROBE"])
    device = probe["device"]
    shuffle = probe["shuffle"]
    torch.cuda.set_device(device)
    torch.manual_seed(27)
    x, weight, weight_shuffled, x_scale, w_scale, bias, _ = generate_gemm_a8w8_inputs(
        M=1024,
        N=1024,
        K=1024,
        in_dtype=IN_DTYPE,
        out_dtype=OUT_DTYPE,
        layout="TN",
        output=False,
        shuffle=shuffle,
    )
    expected = reference(x, weight, x_scale, w_scale, bias)
    config, _ = get_gemm_config("GEMM-A8W8", 1024, 1024, 1024, backend="gluon")
    y = torch.empty_like(expected)
    if shuffle:
        call = lambda: gemm_a8w8_preshuffle(
            x, weight_shuffled, x_scale, w_scale, bias, OUT_DTYPE, y, config
        )
    else:
        call = lambda: gemm_a8w8(
            x,
            weight,
            x_scale,
            w_scale,
            bias,
            OUT_DTYPE,
            y,
            config,
            backend="gluon",
        )
    result = graph_replay(call, expected)
    print(json.dumps(result), flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def run_graph_probe(device, shuffle):
    environment = os.environ.copy()
    environment["PR27_GRAPH_PROBE"] = json.dumps(
        {"device": device, "shuffle": shuffle}
    )
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve())],
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
    )
    result = {
        "device": device,
        "consumer": "gemm_a8w8_preshuffle" if shuffle else "gemm_a8w8(backend=gluon)",
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }
    try:
        parsed = json.loads(completed.stdout.strip().splitlines()[-1])
        result.update(parsed)
    except (IndexError, json.JSONDecodeError):
        result["supported"] = False
        result["replayed"] = False
    return result


def main():
    if torch.cuda.device_count() != 2:
        raise RuntimeError(f"Expected exactly 2 GPUs, got {torch.cuda.device_count()}")
    if arch_info.get_arch() != "gfx950":
        raise RuntimeError(f"Expected gfx950, got {arch_info.get_arch()}")

    libhip = ctypes.CDLL("libamdhip64.so")
    hip_values = {}
    for device in range(2):
        value = ctypes.c_int(0)
        status = libhip.hipDeviceGetAttribute(
            ctypes.byref(value), 10018, device
        )
        hip_values[device] = {
            "status": int(status),
            "attribute": value.value,
        }
        if status != 0 or value.value <= 0:
            raise RuntimeError(f"HIP XCD query failed on device {device}")

    _get_num_xcds.cache_clear()
    getter_values = []
    for device in (0, 1, 0, 1):
        torch.cuda.set_device(device)
        current = get_num_xcds()
        explicit = get_num_xcds(device)
        direct = hip_values[device]["attribute"]
        if current != direct or explicit != direct:
            raise RuntimeError(
                f"Getter mismatch on device {device}: current={current}, "
                f"explicit={explicit}, HIP={direct}"
            )
        getter_values.append(
            {
                "current_device": device,
                "current_form": current,
                "explicit_form": explicit,
                "hip_attribute": direct,
            }
        )

    cases = []
    for device in range(2):
        graph_results = {
            False: run_graph_probe(device, False),
            True: run_graph_probe(device, True),
        }
        cases.append(validate_case(device, "gemm_a8w8(backend=gluon)", (1024, 1024, 1024), False, graph_results[False]))
        cases.append(validate_case(device, "gemm_a8w8_preshuffle", (1024, 1024, 1024), True, graph_results[True]))
        boundary_graph = {
            "supported": False,
            "replayed": False,
            "reason": "graph replay checked once per consumer/device on representative shape only",
        }
        cases.append(validate_case(device, "gemm_a8w8(backend=gluon) boundary", (1, 16, 256), False, boundary_graph))
        cases.append(validate_case(device, "gemm_a8w8_preshuffle boundary", (16, 16, 256), True, boundary_graph))

    try:
        import aiter.jit.module_aiter_core as module_aiter_core

        native_module = module_record(module_aiter_core)
    except Exception as error:
        native_module = {"error": f"{type(error).__name__}: {error}"}

    result = {
        "validation_head": command_output(["git", "rev-parse", "HEAD"]),
        "validation_branch": command_output(["git", "branch", "--show-current"]) or "(detached)",
        "validation_status": command_output(["git", "status", "--porcelain"]) or "clean",
        "python": sys.version,
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "torch_hip_version": torch.version.hip,
        "triton_version": triton.__version__,
        "compiler_versions": {
            "hipcc": command_output(["hipcc", "--version"]),
            "clang": command_output(
                ["/opt/rocm-7.2.0/lib/llvm/bin/clang", "--version"]
            ),
            "rocm_smi": command_output(
                ["rocm-smi", "--showproductname", "--showserial", "--showuniqueid", "--showbus", "--json"]
            ),
        },
        "gpu_count": torch.cuda.device_count(),
        "gpus": [
            {
                "ordinal": device,
                "name": torch.cuda.get_device_name(device),
                "uuid": str(torch.cuda.get_device_properties(device).uuid),
                "major": torch.cuda.get_device_properties(device).major,
                "minor": torch.cuda.get_device_properties(device).minor,
                "compute_units": torch.cuda.get_device_properties(device).multi_processor_count,
                "total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
            }
            for device in range(2)
        ],
        "hip_xcd_attribute": hip_values,
        "getter_current_explicit_and_hip": getter_values,
        "getter_cache_info_after_switching": dict(
            hits=_get_num_xcds.cache_info().hits,
            misses=_get_num_xcds.cache_info().misses,
            maxsize=_get_num_xcds.cache_info().maxsize,
            currsize=_get_num_xcds.cache_info().currsize,
        ),
        "imported_modules": [
            module_record(torch),
            module_record(triton),
            module_record(arch_info),
            module_record(sys.modules["aiter.ops.triton.utils.device_info"]),
            module_record(sys.modules["aiter.ops.triton.gemm.basic.gemm_a8w8"]),
            module_record(sys.modules["aiter.ops.triton._gluon_kernels.gfx950.gemm.basic.gemm_a8w8"]),
            module_record(sys.modules["aiter.ops.shuffle"]),
        ],
        "native_module": native_module,
        "hip_library": {
            "requested": "libamdhip64.so",
            "resolved": ctypes.util.find_library("amdhip64"),
        },
        "cases": cases,
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    if "PR27_GRAPH_PROBE" in os.environ:
        graph_probe_main()
    else:
        main()
