# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Build a bounded, opt-in MoE tolerance calibration dataset.

This script intentionally does not change the verdicts used by the existing
MoE tests.  It generates seeded inputs, captures real reference and AITER
outputs, applies labeled corruptions, and reports both the stock scalar
criterion and a per-row, output-dtype-aware experimental criterion.  The
native outputs are not guaranteed to be bit-identical across runs.
"""

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from aiter import ActivationType, dtypes, get_gfx, pertoken_quant
from aiter.fused_moe import fused_topk
from aiter.fused_moe_bf16_asm import (
    asm_moe,
    asm_moe_tkw1,
    torch_moe,
    torch_moe_tkw1,
)
from aiter.int4_utils import convert_int8_to_uint32_int4, rearrange_4bit_elements
from aiter.ops.shuffle import shuffle_weight


TOKEN_COUNT = 32
MODEL_DIM = 2048
INTER_DIM = 1792
EXPERT_COUNT = 32
STOCK_RTOL = 0.01
STOCK_ATOL = 100.0
BF16_EPS = torch.finfo(torch.bfloat16).eps


@dataclass(frozen=True)
class CalibrationCase:
    case_id: str
    quant: str
    use_g1u1: bool
    topk: int
    topk_one: bool
    safety_factor: int
    seed: int
    kernel_hint: str


CASES = (
    CalibrationCase(
        "test_moe.py:g1u1_int8quant:a8w8",
        "int8quant",
        True,
        5,
        False,
        32,
        53652,
        "fmoe_bf16_pertokenInt8_g1u1_vs_silu_1tg_ps_32x256E",
    ),
    CalibrationCase(
        "test_moe.py:g1u1_fp8quant:a8w8",
        "fp8quant",
        True,
        5,
        False,
        16,
        53653,
        "fmoe_bf16_pertokenFp8_g1u1_vs_silu_1tg_ps_32x256E",
    ),
    CalibrationCase(
        "test_moe.py:g1u1_int8smoothquant:a8w8",
        "int8smoothquant",
        True,
        5,
        False,
        32,
        53654,
        "fmoe_bf16_pertokenInt8_g1u1_vs_multix_silu_1tg_ps_32x256E",
    ),
    CalibrationCase(
        "test_moe.py:g1u1_int8smoothquant:a16w8",
        "int8smoothquant",
        True,
        5,
        False,
        32,
        53654,
        "fmoe_bf16_pertokenInt8_g1u1_vs_smf_silu_1tg_1tg_32x256E",
    ),
    CalibrationCase(
        "test_moe.py:g1u1_fp8smoothquant:a8w8",
        "fp8smoothquant",
        True,
        5,
        False,
        16,
        53655,
        "fmoe_bf16_pertokenFp8_g1u1_vs_multix_silu_1tg_ps_32x256E",
    ),
    CalibrationCase(
        "test_moe.py:g1u1_wint4afp8smoothquant:a8w4",
        "wint4afp8smoothquant",
        True,
        5,
        False,
        32,
        53656,
        "fmoe_int4fp8_g1u1_subGU_256_gelu",
    ),
    CalibrationCase(
        "test_moe.py:g1u0_int8smoothquant:a8w8",
        "int8smoothquant",
        False,
        5,
        False,
        32,
        53657,
        "fmoe_int8_g1u0_subGU_448",
    ),
    CalibrationCase(
        "test_moe_tkw1.py:g1u1_fp8quant:a8w8",
        "fp8quant",
        True,
        1,
        True,
        16,
        53658,
        "CK 2-stage per-token fp8/fp8, mulWeightStage1",
    ),
)

CORRUPTIONS = (
    ("real_kernel_output", "none"),
    ("add_1_all", "absolute_offset"),
    ("add_10_all", "absolute_offset"),
    ("add_50_all", "absolute_offset"),
    ("add_99_all", "absolute_offset"),
    ("add_100_all", "absolute_offset"),
    ("add_101_all", "absolute_offset"),
    ("add_500_all", "absolute_offset"),
    ("scale_1pct_all", "relative_scale"),
    ("zero_1pct", "masked_zero"),
    ("zero_10pct", "masked_zero"),
    ("zero_50pct", "masked_zero"),
    ("sign_flip_1pct", "masked_sign_flip"),
)


def git_revision(repository: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()


def quantized_case_inputs(case: CalibrationCase):
    torch.manual_seed(case.seed)
    torch.cuda.manual_seed_all(case.seed)

    hidden_states = torch.randn(
        (TOKEN_COUNT, MODEL_DIM), dtype=dtypes.bf16, device="cuda"
    )
    if case.use_g1u1:
        weight_1 = (
            torch.randn(
                (EXPERT_COUNT, INTER_DIM * 2, MODEL_DIM),
                dtype=dtypes.bf16,
                device="cuda",
            )
            / 10.0
        )
    else:
        weight_1 = torch.randn(
            (EXPERT_COUNT, INTER_DIM, MODEL_DIM), dtype=dtypes.bf16, device="cuda"
        )
    weight_2 = torch.randn(
        (EXPERT_COUNT, MODEL_DIM, INTER_DIM), dtype=dtypes.bf16, device="cuda"
    )
    scores = torch.randn(
        (TOKEN_COUNT, EXPERT_COUNT), dtype=dtypes.bf16, device="cuda"
    )
    topk_weights, topk_ids = fused_topk(
        hidden_states, scores, case.topk, renormalize=True
    )

    use_int4 = "wint4" in case.quant
    quant_dtype = (
        dtypes.i8
        if use_int4 or case.quant.startswith("int8")
        else dtypes.fp8
    )
    dtype_max = 7 if use_int4 else None
    weight_1, fc1_scale = pertoken_quant(
        weight_1, quant_dtype=quant_dtype, dtypeMax=dtype_max
    )
    weight_2, fc2_scale = pertoken_quant(
        weight_2, quant_dtype=quant_dtype, dtypeMax=dtype_max
    )

    use_smooth = "smooth" in case.quant
    if not use_smooth:
        fc1_smooth_scale = None
        fc2_smooth_scale = None
    elif use_int4:
        fc1_smooth_scale = torch.ones(
            (EXPERT_COUNT, 1, MODEL_DIM), dtype=dtypes.fp32, device="cuda"
        )
        fc2_smooth_scale = torch.ones(
            (EXPERT_COUNT, 1, INTER_DIM), dtype=dtypes.fp32, device="cuda"
        )
    else:
        fc1_smooth_scale = torch.randn(
            (EXPERT_COUNT, 1, MODEL_DIM), dtype=dtypes.fp32, device="cuda"
        )
        fc2_smooth_scale = torch.randn(
            (EXPERT_COUNT, 1, INTER_DIM), dtype=dtypes.fp32, device="cuda"
        )

    if case.topk_one:
        reference = torch_moe_tkw1(
            hidden_states,
            weight_1,
            weight_2,
            topk_weights,
            topk_ids,
            fc1_scale,
            fc2_scale,
            fc1_smooth_scale,
            fc2_smooth_scale,
            None,
            ActivationType.Silu,
        )
    else:
        reference = torch_moe(
            hidden_states,
            weight_1,
            weight_2,
            topk_weights,
            topk_ids,
            fc1_scale,
            fc2_scale,
            fc1_smooth_scale,
            fc2_smooth_scale,
            None,
            ActivationType.Silu,
        )

    if use_int4:
        weight_1 = rearrange_4bit_elements(
            convert_int8_to_uint32_int4(weight_1)
        )
        weight_2 = rearrange_4bit_elements(
            convert_int8_to_uint32_int4(weight_2)
        )
    shuffled_weight_1 = shuffle_weight(weight_1)
    shuffled_weight_2 = shuffle_weight(weight_2)

    a16 = case.case_id.endswith(":a16w8")
    if case.topk_one:
        candidate = asm_moe_tkw1(
            hidden_states,
            shuffled_weight_1,
            shuffled_weight_2,
            topk_weights,
            topk_ids,
            fc1_scale,
            fc2_scale,
            fc1_smooth_scale,
            fc2_smooth_scale,
            a16,
            None,
            None,
            ActivationType.Silu,
        )
    else:
        candidate = asm_moe(
            hidden_states,
            shuffled_weight_1,
            shuffled_weight_2,
            topk_weights,
            topk_ids,
            fc1_scale,
            fc2_scale,
            fc1_smooth_scale,
            fc2_smooth_scale,
            a16,
            None,
            None,
            None,
            ActivationType.Silu,
        )

    return reference, candidate


def row_scale(reference: torch.Tensor) -> torch.Tensor:
    return reference.detach().abs().amax(dim=-1, keepdim=True).to(torch.float32)


def row_bf16_ulp(scale: torch.Tensor) -> torch.Tensor:
    exponent = torch.floor(torch.log2(scale.clamp_min(1.0e-30)))
    return torch.exp2(exponent) * BF16_EPS


def stock_close(reference: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    return torch.isclose(
        reference, candidate, rtol=STOCK_RTOL, atol=STOCK_ATOL
    )


def experimental_close(
    reference: torch.Tensor, candidate: torch.Tensor, safety_factor: int
) -> torch.Tensor:
    scale = row_scale(reference)
    ulp = row_bf16_ulp(scale)
    atol_row = torch.maximum(
        torch.ones_like(ulp), ulp * float(safety_factor)
    )
    difference = (reference.detach().float() - candidate.detach().float()).abs()
    threshold = atol_row + STOCK_RTOL * candidate.detach().float().abs()
    return difference <= threshold


def values_summary(values: torch.Tensor) -> dict:
    host_values = values.detach().float().cpu().flatten()
    quantiles = torch.quantile(
        host_values, torch.tensor([0.50, 0.99, 0.999])
    )
    return {
        "min": float(host_values.min().item()),
        "max": float(host_values.max().item()),
        "mean": float(host_values.mean().item()),
        "p50": float(quantiles[0].item()),
        "p99": float(quantiles[1].item()),
        "p999": float(quantiles[2].item()),
    }


def criterion_summary(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    close: torch.Tensor,
    safety_factor: int,
) -> dict:
    difference = (reference.detach().float() - candidate.detach().float()).abs()
    scale = row_scale(reference)
    ulp = row_bf16_ulp(scale)
    required_safety = difference / ulp.clamp_min(1.0e-30)
    return {
        "accepted": bool(close.all().item()),
        "mismatch_count": int((~close).sum().item()),
        "mismatch_ratio": float((~close).float().mean().item()),
        "error": values_summary(difference),
        "reference_abs": values_summary(reference.detach().float().abs()),
        "row_scale": values_summary(scale),
        "required_ulp_safety": values_summary(required_safety),
        "experimental_safety_factor": safety_factor,
    }


def corrupt(
    reference: torch.Tensor, label: str, corruption_seed: int
) -> torch.Tensor:
    generator = torch.Generator(device=reference.device)
    generator.manual_seed(corruption_seed)
    candidate = reference.detach().clone()
    if label == "real_kernel_output":
        return candidate
    if label.startswith("add_"):
        offset = float(label.removeprefix("add_").removesuffix("_all"))
        return candidate + torch.full_like(candidate, offset)
    if label == "scale_1pct_all":
        return (candidate.float() * 1.01).to(candidate.dtype)
    if label.startswith("zero_"):
        fraction = float(label.removeprefix("zero_").removesuffix("pct")) / 100.0
        mask = torch.rand(
            candidate.shape, device=candidate.device, generator=generator
        ) < fraction
        candidate[mask] = 0
        return candidate
    if label == "sign_flip_1pct":
        mask = torch.rand(
            candidate.shape, device=candidate.device, generator=generator
        ) < 0.01
        candidate[mask] = -candidate[mask]
        return candidate
    raise ValueError(f"unknown corruption: {label}")


def calibrate_case(case: CalibrationCase) -> dict:
    reference, candidate = quantized_case_inputs(case)
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        raise RuntimeError(
            f"reference/candidate mismatch: {reference.shape} {candidate.shape}"
        )

    results = []
    for corruption_index, (label, corruption_type) in enumerate(CORRUPTIONS):
        if label == "real_kernel_output":
            corrupted = candidate
        else:
            corrupted = corrupt(
                reference, label, case.seed * 1000 + corruption_index
            )
        stock_result = criterion_summary(
            reference, corrupted, stock_close(reference, corrupted), 0
        )
        experimental_result = criterion_summary(
            reference,
            corrupted,
            experimental_close(reference, corrupted, case.safety_factor),
            case.safety_factor,
        )
        results.append(
            {
                "label": label,
                "type": corruption_type,
                "stock": stock_result,
                "experimental": experimental_result,
            }
        )

    return {
        "case_id": case.case_id,
        "quant": case.quant,
        "use_g1u1": case.use_g1u1,
        "topk": case.topk,
        "topk_one": case.topk_one,
        "seed": case.seed,
        "shape": list(reference.shape),
        "dtype": str(reference.dtype),
        "kernel_hint": case.kernel_hint,
        "corruptions": results,
        "reference": reference.detach().to(torch.float32).cpu().numpy(),
        "candidate": candidate.detach().to(torch.float32).cpu().numpy(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("moe_tolerance_calibration.json"),
        help="JSON summary output path",
    )
    parser.add_argument(
        "--tensor-output",
        type=Path,
        default=None,
        help="Optional compressed .npz path for reference and AITER tensors",
    )
    parser.add_argument(
        "--image-id",
        default=(
            "amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5 "
            "sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1"
        ),
        help="Qualified image identity recorded in the report",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("MoE calibration requires an available HIP device")
    if get_gfx() != "gfx942":
        raise RuntimeError(
            f"MoE calibration selected gfx942 kernels; refusing to run on {get_gfx()}"
        )

    case_results = []
    tensor_payload = {}
    for case in CASES:
        print(f"calibrating {case.case_id}", flush=True)
        result = calibrate_case(case)
        tensor_payload[f"{case.case_id}_reference"] = result.pop("reference")
        tensor_payload[f"{case.case_id}_candidate"] = result.pop("candidate")
        case_results.append(result)

    report = {
        "schema_version": 1,
        "source_revision": git_revision(Path(__file__).resolve().parents[1]),
        "image_id": args.image_id,
        "torch_version": torch.__version__,
        "hip_version": torch.version.hip,
        "device_name": torch.cuda.get_device_name(0),
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "gfx": get_gfx(),
        "shape": {
            "token": TOKEN_COUNT,
            "model_dim": MODEL_DIM,
            "inter_dim": INTER_DIM,
            "expert": EXPERT_COUNT,
        },
        "stock_criterion": {"rtol": STOCK_RTOL, "atol": STOCK_ATOL},
        "experimental_criterion": {
            "rtol": STOCK_RTOL,
            "atol": "max(1, safety_factor * bf16_ulp(max_abs(reference_row)))",
            "safety_factors": {
                "int8": 32,
                "fp8": 16,
                "int4": 32,
            },
        },
        "cases": case_results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        json.dump(report, output_file, indent=2, sort_keys=True)
        output_file.write("\n")

    if args.tensor_output is not None:
        args.tensor_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.tensor_output, **tensor_payload)

    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
