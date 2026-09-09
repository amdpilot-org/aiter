#!/usr/bin/env python3
"""Reproduce the DeepSeek-V4 profile_run mhc_pre launch on gfx942."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch
import aiter


PRODUCTION_TOKENS = 8192
PRODUCTION_HC_MULT = 4
PRODUCTION_HIDDEN_SIZE = 4096
SUPPORTED_HIDDEN_SIZES = {7168, 4096, 2560, 1280}


@dataclass(frozen=True)
class Case:
    label: str
    num_tokens: int
    hc_mult: int
    hidden_size: int


def production_inputs(
    num_tokens: int, hc_mult: int, hidden_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hc_mult3 = 2 * hc_mult + hc_mult * hc_mult
    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260909)

    def randn(*shape: int, **kwargs):
        return torch.randn(
            *shape, generator=generator, device="cuda", **kwargs
        )

    residual = randn(num_tokens, hc_mult, hidden_size, dtype=torch.bfloat16)
    fn = randn(hc_mult3, hc_mult * hidden_size, dtype=torch.float32)
    hc_scale = randn(3, dtype=torch.float32) * 0.1
    hc_base = randn(hc_mult3, dtype=torch.float32) * 0.1
    return residual, fn, hc_scale, hc_base


def mhc_pre_reference(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hc_mult = residual.shape[1]
    hidden_size = residual.shape[2]
    residual_flat = residual.flatten(-2, -1).float()
    sqrsum = residual_flat.square().sum(-1)
    mixes = residual_flat @ fn.T
    mixes = mixes * (sqrsum.unsqueeze(-1) / fn.shape[-1] + rms_eps).rsqrt()

    expanded_scale = torch.cat(
        [
            hc_scale[0].expand(hc_mult),
            hc_scale[1].expand(hc_mult),
            hc_scale[2].expand(hc_mult * hc_mult),
        ]
    )
    mixes = mixes * expanded_scale + hc_base
    pre_mix = mixes[:, :hc_mult].sigmoid().unsqueeze(-1) + hc_pre_eps
    post_mix = (
        mixes[:, hc_mult : 2 * hc_mult].sigmoid() * hc_post_mult_value
    ).unsqueeze(-1)
    comb_mix = mixes[:, 2 * hc_mult :].view(-1, hc_mult, hc_mult)
    comb_mix = comb_mix.softmax(-1) + hc_sinkhorn_eps
    comb_mix = comb_mix / (comb_mix.sum(-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        comb_mix = comb_mix / (comb_mix.sum(-1, keepdim=True) + hc_sinkhorn_eps)
        comb_mix = comb_mix / (comb_mix.sum(-2, keepdim=True) + hc_sinkhorn_eps)

    layer_input = (residual * pre_mix).sum(-2).bfloat16()
    return post_mix, comb_mix, layer_input


def compare(
    actual: torch.Tensor, expected: torch.Tensor, rtol: float, atol: float
) -> dict[str, object]:
    difference = (actual.float() - expected.float()).abs()
    close = torch.isclose(actual, expected, rtol=rtol, atol=atol)
    return {
        "shape": list(actual.shape),
        "dtype": str(actual.dtype),
        "rtol": rtol,
        "atol": atol,
        "max_abs_error": difference.max().item(),
        "mismatch_count": int((~close).sum().item()),
        "element_count": actual.numel(),
        "passed": bool(close.all().item()),
    }


def run_case(case: Case) -> dict[str, object]:
    residual, fn, hc_scale, hc_base = production_inputs(
        case.num_tokens, case.hc_mult, case.hidden_size
    )
    result: dict[str, object] = {
        "label": case.label,
        "shape": [case.num_tokens, case.hc_mult, case.hidden_size],
        "status": "unsupported",
        "error": None,
    }
    if case.hc_mult != PRODUCTION_HC_MULT:
        result["error"] = "mhc_pre_big_fuse only supports hc_mult=4"
        return result
    if case.hidden_size not in SUPPORTED_HIDDEN_SIZES:
        result["error"] = (
            "mhc_pre_big_fuse only supports hidden_size in "
            f"{sorted(SUPPORTED_HIDDEN_SIZES)}"
        )
        return result
    try:
        outputs = aiter.mhc_pre(
            residual,
            fn,
            hc_scale,
            hc_base,
            1e-6,
            1e-6,
            1e-6,
            1.0,
            20,
        )
        torch.cuda.synchronize()
        result["status"] = "launched"
        result["output_shapes"] = [list(output.shape) for output in outputs]
        result["output_dtypes"] = [str(output.dtype) for output in outputs]

        if case.label == "production":
            reference = mhc_pre_reference(
                residual, fn, hc_scale, hc_base, 1e-6, 1e-6, 1e-6, 1.0, 20
            )
            result["reference"] = [
                compare(
                    output,
                    expected,
                    rtol=1e-3 if output.dtype == torch.float32 else 1e-2,
                    atol=1e-3 if output.dtype == torch.float32 else 1e-2,
                )
                for output, expected in zip(outputs, reference)
            ]
            result["status"] = (
                "validated"
                if all(item["passed"] for item in result["reference"])
                else "accuracy_failed"
            )
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
        try:
            torch.cuda.synchronize()
        except Exception as sync_error:
            result["sync_error"] = f"{type(sync_error).__name__}: {sync_error}"
    return result


def cases() -> list[Case]:
    production = Case("production", 8192, 4, 4096)
    return [
        production,
        Case("tokens_off_by_one_minus", 8191, 4, 4096),
        Case("tokens_off_by_one_plus", 8193, 4, 4096),
        Case("tokens_tile_boundary_minus", 8128, 4, 4096),
        Case("tokens_tile_boundary_plus", 8256, 4, 4096),
        Case("big_fuse_nt_boundary_minus", 2431, 4, 4096),
        Case("big_fuse_nt_boundary", 2432, 4, 4096),
        Case("big_fuse_nt_boundary_plus", 2433, 4, 4096),
        Case("big_fuse_dispatch_minus", 3647, 4, 4096),
        Case("big_fuse_dispatch", 3648, 4, 4096),
        Case("big_fuse_dispatch_plus", 3649, 4, 4096),
        Case("hc_mult_minus", 8192, 3, 4096),
        Case("hc_mult_plus", 8192, 5, 4096),
        Case("hidden_off_by_one_minus", 8192, 4, 4095),
        Case("hidden_off_by_one_plus", 8192, 4, 4097),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    if torch.cuda.device_count() != 1:
        raise SystemExit(
            f"expected one GPU, found {torch.cuda.device_count()}"
        )
    if torch.cuda.get_device_capability(0) != (9, 4):
        raise SystemExit(
            "expected gfx942, found "
            f"{torch.cuda.get_device_capability(0)}"
        )

    report = {
        "image": "amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5",
        "image_id": (
            "sha256:39fe745feda79ecf4c17f4d806d8ef12150"
            "bef720f2f07f5c63a20b3ccfd63f1"
        ),
        "python": "/opt/venv/bin/python",
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "production_shape": [
            PRODUCTION_TOKENS,
            PRODUCTION_HC_MULT,
            PRODUCTION_HIDDEN_SIZE,
        ],
        "cases": [],
    }

    for case in cases():
        print(f"running {case.label}: {case.num_tokens}x{case.hc_mult}x"
              f"{case.hidden_size}", flush=True)
        result = run_case(case)
        report["cases"].append(result)
        print(json.dumps(result, indent=2), flush=True)
        del result
        torch.cuda.empty_cache()

    if args.json_output:
        args.json_output.write_text(json.dumps(report, indent=2) + "\n")

    failed = [
        case
        for case in report["cases"]
        if case["status"] not in {"validated", "launched", "unsupported"}
    ]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
