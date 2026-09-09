#!/usr/bin/env python3
"""Synthetic gfx942 validation for packed KV-cache stride handling."""

from __future__ import annotations

import json
import inspect
import platform
import subprocess
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(REPO_ROOT))

import aiter

OPERATOR_SOURCE = inspect.getsource(
    aiter.ops.fused_qk_norm_rope_cache_quant.fused_qk_norm_rope_cache_pts_quant_shuffle
)
SUPPORTS_V_NORM = "v_norm: bool" in OPERATOR_SOURCE


RESULT_PATH = Path(__file__).with_name("results.json")


def rms_norm(value: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    value_f32 = value.float()
    inverse_rms = torch.rsqrt(value_f32.square().mean(dim=-1, keepdim=True) + eps)
    return (value_f32 * inverse_rms * weight.float()).to(value.dtype)


def apply_rope(
    value: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool,
) -> torch.Tensor:
    value_f32 = value.float()
    cos = cos.unsqueeze(-2).float()
    sin = sin.unsqueeze(-2).float()
    if is_neox_style:
        first, second = torch.chunk(value_f32, 2, dim=-1)
    else:
        first = value_f32[..., ::2]
        second = value_f32[..., 1::2]
    rotated_first = first * cos - second * sin
    rotated_second = second * cos + first * sin
    if is_neox_style:
        rotated = torch.cat((rotated_first, rotated_second), dim=-1)
    else:
        rotated = torch.stack((rotated_first, rotated_second), dim=-1).flatten(-2)
    return rotated.to(value.dtype)


def apply_partial_rope(
    value: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
    is_neox_style: bool,
) -> torch.Tensor:
    if rotary_dim and rotary_dim < value.shape[-1]:
        rotated = apply_rope(value[..., :rotary_dim], cos, sin, is_neox_style)
        return torch.cat((rotated, value[..., rotary_dim:]), dim=-1)
    return apply_rope(value, cos, sin, is_neox_style)


def quantize(value: torch.Tensor, scale: float, cache_dtype: torch.dtype) -> torch.Tensor:
    if value.dtype == cache_dtype:
        return value.to(cache_dtype)
    return (value.float() / scale).to(cache_dtype)


def is_sentinel(region: torch.Tensor) -> bool:
    return bool(torch.equal(region, torch.full_like(region, 57)))


def error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual_f32 = actual.float()
    expected_f32 = expected.float()
    absolute = (actual_f32 - expected_f32).abs()
    relative = absolute / expected_f32.abs().clamp_min(1e-6)
    return {
        "max_abs_error": absolute.max().item(),
        "max_rel_error": relative.max().item(),
    }


def run_operator(
    *,
    qkv: torch.Tensor,
    qw: torch.Tensor,
    kw: torch.Tensor,
    cos_sin: torch.Tensor,
    positions: torch.Tensor,
    num_tokens: int,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_size: int,
    is_neox_style: bool,
    eps: float,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    cache_dtype: torch.dtype,
    block_size: int,
    rotary_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_out = torch.empty(
        (num_tokens, num_heads_q, head_size), dtype=qkv.dtype, device=qkv.device
    )
    k_out = torch.empty(
        (num_tokens, num_heads_k, head_size), dtype=cache_dtype, device=qkv.device
    )
    v_out = torch.empty(
        (num_tokens, num_heads_v, head_size), dtype=cache_dtype, device=qkv.device
    )
    operator_args = [
        qkv,
        qw,
        kw,
        cos_sin,
        positions,
        num_tokens,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_size,
        is_neox_style,
        eps,
        q_out,
        k_cache,
        v_cache,
        slot_mapping,
        k_scale,
        v_scale,
        k_out,
        v_out,
        True,
        False,
        block_size,
        16 // k_cache.element_size(),
        rotary_dim,
    ]
    if SUPPORTS_V_NORM:
        operator_args.append(False)
    aiter.fused_qk_norm_rope_cache_pts_quant_shuffle(*operator_args)
    torch.cuda.synchronize()
    return q_out, k_out, v_out


def validate_case(
    *,
    cache_dtype: torch.dtype,
    head_size: int,
    num_heads_q: int,
    num_heads_kv: int,
    num_tokens: int,
    block_size: int,
    num_blocks: int,
    rotary_dim: int,
    is_neox_style: bool,
    eps: float,
    seed: int,
) -> dict[str, object]:
    torch.manual_seed(seed)
    device = "cuda"
    qkv_dtype = torch.bfloat16
    total_heads = num_heads_q + 2 * num_heads_kv
    qkv = torch.randn(
        (num_tokens, total_heads * head_size), dtype=qkv_dtype, device=device
    )
    qw = torch.randn(head_size, dtype=qkv_dtype, device=device)
    kw = torch.randn(head_size, dtype=qkv_dtype, device=device)
    max_position = 512
    cos_sin = torch.randn(
        (max_position, rotary_dim), dtype=qkv_dtype, device=device
    )
    positions = torch.randint(
        0, max_position, (num_tokens,), dtype=torch.int64, device=device
    )
    slot_mapping = torch.randperm(
        num_blocks * block_size, device=device
    )[:num_tokens].to(torch.int64)
    k_scale = torch.tensor(1.0, dtype=torch.float32, device=device)
    v_scale = torch.tensor(1.0, dtype=torch.float32, device=device)

    q_size = num_heads_q * head_size
    k_size = num_heads_kv * head_size
    v_size = num_heads_kv * head_size
    q_source, k_source, v_source = qkv.split([q_size, k_size, v_size], dim=-1)
    q_reference = rms_norm(
        q_source.view(num_tokens, num_heads_q, head_size), qw, eps
    )
    k_reference = rms_norm(
        k_source.view(num_tokens, num_heads_kv, head_size), kw, eps
    )
    v_reference = v_source.view(num_tokens, num_heads_kv, head_size)
    cos, sin = cos_sin[positions].chunk(2, dim=-1)
    q_reference = apply_partial_rope(
        q_reference, cos, sin, rotary_dim, is_neox_style
    )
    k_reference = apply_partial_rope(
        k_reference, cos, sin, rotary_dim, is_neox_style
    )
    k_reference_quant = quantize(k_reference, 1.0, cache_dtype)
    v_reference_quant = quantize(v_reference, 1.0, cache_dtype)

    contiguous_k_storage = torch.full(
        (num_blocks + 1, block_size, num_heads_kv, head_size),
        57,
        dtype=cache_dtype,
        device=device,
    )
    contiguous_v_storage = torch.full(
        (num_blocks + 1, block_size, num_heads_kv, head_size),
        57,
        dtype=cache_dtype,
        device=device,
    )
    contiguous_k = contiguous_k_storage[:num_blocks]
    contiguous_v = contiguous_v_storage[:num_blocks]

    packed_padding = 3
    packed_storage = torch.full(
        (num_blocks, num_heads_kv, block_size, 2 * head_size + packed_padding),
        57,
        dtype=cache_dtype,
        device=device,
    )
    packed_k = packed_storage[..., :head_size].permute(0, 2, 1, 3)
    packed_v = packed_storage[..., head_size : 2 * head_size].permute(0, 2, 1, 3)

    common = {
        "qw": qw,
        "kw": kw,
        "cos_sin": cos_sin,
        "positions": positions,
        "num_tokens": num_tokens,
        "num_heads_q": num_heads_q,
        "num_heads_k": num_heads_kv,
        "num_heads_v": num_heads_kv,
        "head_size": head_size,
        "is_neox_style": is_neox_style,
        "eps": eps,
        "slot_mapping": slot_mapping,
        "k_scale": k_scale,
        "v_scale": v_scale,
        "cache_dtype": cache_dtype,
        "block_size": block_size,
        "rotary_dim": rotary_dim,
    }
    contiguous_result = run_operator(
        qkv=qkv.clone(),
        k_cache=contiguous_k,
        v_cache=contiguous_v,
        **common,
    )
    packed_result = run_operator(
        qkv=qkv.clone(),
        k_cache=packed_k,
        v_cache=packed_v,
        **common,
    )

    block_ids = slot_mapping // block_size
    token_ids = slot_mapping % block_size
    contiguous_cache_k = contiguous_k[block_ids, token_ids]
    contiguous_cache_v = contiguous_v[block_ids, token_ids]
    packed_cache_k = packed_k[block_ids, token_ids]
    packed_cache_v = packed_v[block_ids, token_ids]

    used_slots = torch.zeros(
        (num_blocks, block_size), dtype=torch.bool, device=device
    )
    used_slots[block_ids, token_ids] = True
    contiguous_sentinel_ok = bool(
        is_sentinel(contiguous_k[~used_slots])
        and is_sentinel(contiguous_v[~used_slots])
        and is_sentinel(contiguous_k_storage[num_blocks:])
        and is_sentinel(contiguous_v_storage[num_blocks:])
    )
    packed_sentinel_ok = bool(
        is_sentinel(packed_k[~used_slots])
        and is_sentinel(packed_v[~used_slots])
        and is_sentinel(packed_storage[..., 2 * head_size :])
    )

    view_parity_exact = bool(
        torch.equal(contiguous_result[0], packed_result[0])
        and torch.equal(contiguous_result[1], packed_result[1])
        and torch.equal(contiguous_result[2], packed_result[2])
        and torch.equal(contiguous_cache_k, packed_cache_k)
        and torch.equal(contiguous_cache_v, packed_cache_v)
    )

    independent_rtol = 1e-2
    independent_atol = 0.05
    if cache_dtype != qkv_dtype:
        independent_rtol = 0.0
        independent_atol = 0.0625

    comparisons = {}
    for view_name, (q_out, k_out, v_out, cache_k, cache_v) in {
        "contiguous": (
            contiguous_result[0],
            contiguous_result[1],
            contiguous_result[2],
            contiguous_cache_k,
            contiguous_cache_v,
        ),
        "packed_strided": (
            packed_result[0],
            packed_result[1],
            packed_result[2],
            packed_cache_k,
            packed_cache_v,
        ),
    }.items():
        comparisons[view_name] = {}
        for output_name, actual, expected in (
            ("q_out", q_out, q_reference),
            ("k_out", k_out, k_reference_quant),
            ("v_out", v_out, v_reference_quant),
            ("k_cache", cache_k, k_reference_quant),
            ("v_cache", cache_v, v_reference_quant),
        ):
            metrics = error_metrics(actual, expected)
            metrics["pass"] = bool(
                torch.allclose(
                    actual.float(),
                    expected.float(),
                    rtol=independent_rtol,
                    atol=independent_atol,
                )
            )
            comparisons[view_name][output_name] = metrics

    return {
        "cache_dtype": str(cache_dtype),
        "head_size": head_size,
        "num_heads_q": num_heads_q,
        "num_heads_kv": num_heads_kv,
        "num_tokens": num_tokens,
        "block_size": block_size,
        "num_blocks": num_blocks,
        "rotary_dim": rotary_dim,
        "is_neox_style": is_neox_style,
        "eps": eps,
        "independent_rtol": independent_rtol,
        "independent_atol": independent_atol,
        "contiguous_strides": {
            "k": list(contiguous_k.stride()),
            "v": list(contiguous_v.stride()),
        },
        "packed_strides": {
            "k": list(packed_k.stride()),
            "v": list(packed_v.stride()),
        },
        "view_parity_exact": view_parity_exact,
        "contiguous_sentinel_ok": contiguous_sentinel_ok,
        "packed_sentinel_ok": packed_sentinel_ok,
        "comparisons": comparisons,
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This validation requires an available ROCm/HIP GPU")
    if torch.cuda.get_device_capability(0) != (9, 4):
        raise RuntimeError(
            f"Expected gfx942, got {torch.cuda.get_device_capability(0)}"
        )

    cases = [
        {
            "cache_dtype": torch.bfloat16,
            "head_size": 128,
            "num_heads_q": 32,
            "num_heads_kv": 8,
            "num_tokens": 17,
            "block_size": 16,
            "num_blocks": 4,
            "rotary_dim": 64,
            "is_neox_style": True,
            "eps": 1e-6,
            "seed": 1234,
        },
        {
            "cache_dtype": torch.bfloat16,
            "head_size": 256,
            "num_heads_q": 32,
            "num_heads_kv": 8,
            "num_tokens": 19,
            "block_size": 16,
            "num_blocks": 4,
            "rotary_dim": 128,
            "is_neox_style": False,
            "eps": 1e-6,
            "seed": 2345,
        },
        {
            "cache_dtype": torch.float8_e4m3fnuz,
            "head_size": 128,
            "num_heads_q": 32,
            "num_heads_kv": 8,
            "num_tokens": 17,
            "block_size": 16,
            "num_blocks": 4,
            "rotary_dim": 64,
            "is_neox_style": True,
            "eps": 1e-6,
            "seed": 3456,
        },
    ]
    results = [validate_case(**case) for case in cases]
    overall_pass = all(
        case["view_parity_exact"]
        and case["contiguous_sentinel_ok"]
        and case["packed_sentinel_ok"]
        and all(
            metric["pass"]
            for view in case["comparisons"].values()
            for metric in view.values()
        )
        for case in results
    )
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    report = {
        "overall_pass": overall_pass,
        "git_commit": git_commit,
        "gpu": {
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
        },
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "aiter_python_path": aiter.__file__,
        "operator_python_module": str(
            REPO_ROOT / "aiter/ops/fused_qk_norm_rope_cache_quant.py"
        ),
        "operator_native_module": str(
            REPO_ROOT
            / "aiter/jit/module_fused_qk_norm_rope_cache_quant_shuffle.so"
        ),
        "operator": "fused_qk_norm_rope_cache_pts_quant_shuffle",
        "cases": results,
    }
    RESULT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not overall_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
