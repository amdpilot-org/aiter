import argparse
import json

import torch

from aiter import dtypes
from aiter.jit.core import AITER_CONFIGS
from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime
from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale_ck, get_CKGEMM_config


SHAPES = [
    (2048, 4608, 7168),
    (2048, 7168, 2304),
    (2048, 7168, 256),
    (32768, 4608, 7168),
    (32768, 7168, 2304),
    (32768, 7168, 256),
]
SEEDS = (1234, 5678)
WARMUP_ITERS = 5
TIMING_ITERS = 20
RTOL = 1e-2
ATOL = 1e-2


def dequantized_reference(x, weight, x_scale, w_scale):
    m, k = x.shape
    n = weight.shape[0]
    x_dequantized = (
        x.to(torch.float32).view(m, k // 128, 128) * x_scale.unsqueeze(-1)
    ).reshape(m, k)
    w_block_scale = (
        w_scale.repeat_interleave(128, dim=0)
        .unsqueeze(2)
        .repeat_interleave(128, dim=2)
    )
    w_dequantized = (
        weight.to(torch.float32).view(n, k // 128, 128) * w_block_scale
    ).reshape(n, k)
    return torch.matmul(x_dequantized, w_dequantized.t()).to(torch.bfloat16)


def numeric_summary(reference, output):
    close = torch.isclose(reference, output, rtol=RTOL, atol=ATOL)
    delta = (reference.float() - output.float()).abs()
    return {
        "finite": bool(torch.isfinite(output).all().item()),
        "mismatch": int((~close).sum().item()),
        "total": int(close.numel()),
        "max_abs": float(delta.max().item()),
        "mean_abs": float(delta.mean().item()),
    }


def synchronized_timing(kernel, output):
    for _ in range(WARMUP_ITERS):
        kernel(output)
    torch.cuda.synchronize()
    samples = []
    for _ in range(2):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(TIMING_ITERS):
            kernel(output)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / TIMING_ITERS)
    return samples


def validate_shape(shape, config_file):
    m, n, k = shape
    config = get_CKGEMM_config(m, n, k, config_file)
    if config is None:
        raise RuntimeError(f"{shape}: no tuned config found in {config_file}")

    result = {
        "selected_config": {
            "libtype": config["libtype"],
            "kernelId": int(config["kernelId"]),
            "splitK": int(config["splitK"]),
            "kernelName": config["kernelName"],
            "csv_us": float(config["us"]),
        },
        "seeds": {},
    }
    for seed in SEEDS:
        torch.manual_seed(seed)
        x = (torch.rand((m, k), device="cuda") / 10).to(dtypes.fp8)
        weight = (torch.rand((n, k), device="cuda") / 10).to(dtypes.fp8)
        x_scale = torch.rand((m, k // 128), device="cuda") + 0.5
        w_scale = torch.rand((n // 128, k // 128), device="cuda") + 0.5
        reference = dequantized_reference(x, weight, x_scale, w_scale)
        selected_output = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        default_output = torch.empty_like(selected_output)
        selected_kernel = lambda output: gemm_a8w8_blockscale_ck(
            x,
            weight,
            x_scale,
            w_scale,
            output,
            splitK=int(config["splitK"]),
            kernelName=str(config["kernelName"]),
        )
        default_kernel = lambda output: gemm_a8w8_blockscale_ck(
            x, weight, x_scale, w_scale, output, splitK=0, kernelName=""
        )
        selected_kernel(selected_output)
        default_kernel(default_output)
        result["seeds"][str(seed)] = {
            "selected": {
                "timing_us": synchronized_timing(selected_kernel, selected_output),
                "numeric": numeric_summary(reference, selected_output),
            },
            "default": {
                "timing_us": synchronized_timing(default_kernel, default_output),
                "numeric": numeric_summary(reference, default_output),
            },
        }
        del reference, selected_output, default_output, x, weight, x_scale, w_scale
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="validation.json")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    config_file = AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_FILE
    result = {
        "hardware": {
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "device_name": torch.cuda.get_device_properties(0).name,
            "gfx": get_gfx_runtime(),
            "cu_num": get_cu_num(),
            "visible_device_count": torch.cuda.device_count(),
        },
        "config_file": config_file,
        "input_dtype": str(dtypes.fp8),
        "tolerance": {"rtol": RTOL, "atol": ATOL},
        "timing": {
            "warmup": WARMUP_ITERS,
            "iters": TIMING_ITERS,
            "method": "CUDA events around preallocated-output calls",
        },
        "shapes": {},
    }
    for shape in SHAPES:
        key = "x".join(map(str, shape))
        result["shapes"][key] = validate_shape(shape, config_file)
        print(key, "validated", flush=True)

    with open(args.output, "w", encoding="utf-8") as output:
        json.dump(result, output, indent=2, sort_keys=True)
        output.write("\n")


if __name__ == "__main__":
    main()
