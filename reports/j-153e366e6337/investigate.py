#!/usr/bin/env python3
"""Bounded numerical check for BF16 accumulation in RMSNorm and unquantized MoE."""

import json
import os
import platform
import sys
import time

import torch

import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe, torch_moe
from aiter.ops.shuffle import shuffle_weight


SEED = 0
EPS = 1e-6
VOCAB = 32000
DT = torch.bfloat16


def configure_torch():
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        torch.backends.rocm.matmul.allow_tf32 = False
    except AttributeError:
        pass


def rmsnorm_fp32(input_tensor, weight, epsilon):
    input_fp32 = input_tensor.float()
    variance = input_fp32.pow(2).mean(-1, keepdim=True)
    return (input_fp32 * torch.rsqrt(variance + epsilon)) * weight.float()


def fp32_to_bf16_truncate(input_fp32):
    bits = input_fp32.contiguous().view(torch.int32)
    return (bits >> 16).to(torch.int16).view(torch.bfloat16)


def moe_fp32(hidden_states, w1, w2, topk_weight, topk_ids):
    hidden_fp32 = hidden_states.float()
    w1_fp32 = w1.float()
    w2_fp32 = w2.float()
    output = torch.zeros_like(hidden_fp32)
    for token in range(hidden_fp32.shape[0]):
        accumulator = torch.zeros(
            hidden_fp32.shape[1], device=hidden_fp32.device, dtype=torch.float32
        )
        for route in range(topk_ids.shape[1]):
            expert = int(topk_ids[token, route])
            gate_up = hidden_fp32[token] @ w1_fp32[expert].t()
            gate, up = gate_up.chunk(2, dim=-1)
            activated = torch.nn.functional.silu(gate) * up
            accumulator += float(topk_weight[token, route]) * (
                activated @ w2_fp32[expert].t()
            )
        output[token] = accumulator
    return output


def moe_two_stage_emulated(hidden_states, w1, w2, topk_weight, topk_ids, route_accumulator):
    """Emulate CK's FP32 GEMMs with its BF16 intermediate and selectable route accumulator."""
    hidden_fp32 = hidden_states.float()
    w1_fp32 = w1.float()
    w2_fp32 = w2.float()
    tokens = hidden_fp32.shape[0]
    topk = topk_ids.shape[1]
    intermediate = w2.shape[2]

    inter_bf16 = torch.empty(
        (tokens, topk, intermediate), device=hidden_fp32.device, dtype=DT
    )
    for token in range(tokens):
        for route in range(topk):
            expert = int(topk_ids[token, route])
            gate_up = hidden_fp32[token] @ w1_fp32[expert].t()
            gate, up = gate_up.chunk(2, dim=-1)
            activated = torch.nn.functional.silu(gate) * up
            inter_bf16[token, route] = activated.to(DT)

    if route_accumulator == torch.float32:
        output_fp32 = torch.zeros_like(hidden_fp32)
        for token in range(tokens):
            for route in range(topk):
                expert = int(topk_ids[token, route])
                route_output = inter_bf16[token, route].float() @ w2_fp32[expert].t()
                output_fp32[token] += float(topk_weight[token, route]) * route_output
        return output_fp32.to(DT)

    route_outputs = torch.empty(
        (tokens, topk, hidden_fp32.shape[1]), device=hidden_fp32.device, dtype=DT
    )
    for token in range(tokens):
        for route in range(topk):
            expert = int(topk_ids[token, route])
            route_fp32 = inter_bf16[token, route].float() @ w2_fp32[expert].t()
            route_outputs[token, route] = (
                float(topk_weight[token, route]) * route_fp32
            ).to(DT)
    return route_outputs.sum(dim=1).to(DT)


def error_metrics(actual_fp32, reference_fp32):
    difference = (actual_fp32 - reference_fp32).abs()
    reference_abs = reference_fp32.abs()
    relative = difference / reference_abs.clamp_min(1e-30)
    bf16_ulp = reference_abs.clamp_min(1e-30) * (2.0**-8)
    return {
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "max_rel": float(relative.max()),
        "mean_rel": float(relative.mean()),
        "max_bf16_ulp": float((difference / bf16_ulp).max()),
        "mean_bf16_ulp": float((difference / bf16_ulp).mean()),
        "exact_elements": int((difference == 0).sum()),
        "elements": int(difference.numel()),
    }


def argmax_projection(left_fp32, right_fp32, vocab=VOCAB, scale=0.03):
    hidden = left_fp32.shape[-1]
    generator = torch.Generator(device=left_fp32.device)
    generator.manual_seed(SEED + 1234)
    lm_head = torch.randn(
        (hidden, vocab), device=left_fp32.device, dtype=torch.float32, generator=generator
    ) * scale

    def project(values):
        normalized = values / values.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        return normalized @ lm_head

    left_logits = project(left_fp32)
    right_logits = project(right_fp32)
    left_top2 = left_logits.topk(2, dim=-1).values
    right_top2 = right_logits.topk(2, dim=-1).values
    return {
        "flips": int((left_logits.argmax(-1) != right_logits.argmax(-1)).sum()),
        "rows": int(left_fp32.shape[0]),
        "left_mean_top12_gap": float(left_top2.diff(dim=-1).abs().mean()),
        "right_mean_top12_gap": float(right_top2.diff(dim=-1).abs().mean()),
    }


def time_callable(callable_, warmup=5, iterations=20):
    for _ in range(warmup):
        callable_()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        callable_()
    end.record()
    torch.cuda.synchronize()
    return {
        "iterations": iterations,
        "mean_ms": float(start.elapsed_time(end) / iterations),
    }


def make_rmsnorm_cases(device):
    rows, hidden = 4096, 4096
    weight = torch.randn(hidden, device=device, dtype=DT)
    random_input = torch.randn(rows, hidden, device=device, dtype=DT)
    large_input = random_input * 1e17
    cancellation = torch.randn(rows, hidden, device=device, dtype=DT) * 1e17
    cancellation[:, 1::2] *= -1
    dynamic_range = torch.randn(rows, hidden, device=device, dtype=DT) * 1e-20
    dynamic_range[:, 0] = 1e17
    return [
        ("random_bf16", random_input, weight),
        ("large_magnitude_1e17", large_input, weight),
        ("alternating_cancellation_1e17", cancellation, weight),
        ("dynamic_range_1e17_vs_1e-20", dynamic_range, weight),
    ]


def run_rmsnorm(device):
    results = []
    for name, input_tensor, weight in make_rmsnorm_cases(device):
        actual = aiter.rms_norm(input_tensor, weight, EPS)
        reference_fp32 = rmsnorm_fp32(input_tensor, weight, EPS)
        reference_bf16 = reference_fp32.to(DT)
        reference_truncated = fp32_to_bf16_truncate(reference_fp32)
        timing = time_callable(lambda: aiter.rms_norm(input_tensor, weight, EPS))
        results.append(
            {
                "case": name,
                "shape": list(input_tensor.shape),
                "vs_fp32": error_metrics(actual.float(), reference_fp32),
                "vs_bf16_rounded_fp32": error_metrics(
                    actual.float(), reference_bf16.float()
                ),
                "vs_truncated_fp32": error_metrics(
                    actual.float(), reference_truncated.float()
                ),
                "argmax_vs_fp32": argmax_projection(actual.float(), reference_fp32),
                "argmax_vs_bf16_rounded_fp32": argmax_projection(
                    actual.float(), reference_bf16.float()
                ),
                "argmax_vs_truncated_fp32": argmax_projection(
                    actual.float(), reference_truncated.float()
                ),
                "timing_ms": timing,
            }
        )
    return results


def run_moe(device):
    tokens, hidden, intermediate, experts, topk = 512, 1024, 2048, 8, 2
    hidden_states = torch.randn(tokens, hidden, device=device, dtype=DT) * 0.1
    w1 = (
        torch.randn(experts, 2 * intermediate, hidden, device=device, dtype=DT)
        * hidden**-0.5
    )
    w2 = (
        torch.randn(experts, hidden, intermediate, device=device, dtype=DT)
        * intermediate**-0.5
    )
    gate = (
        torch.randn(hidden, experts, device=device, dtype=torch.float32)
        * hidden**-0.5
        * 0.05
    )
    router_logits = hidden_states.float() @ gate
    probabilities = torch.softmax(router_logits, dim=-1)
    topk_weight, topk_ids = torch.topk(probabilities, topk, dim=-1)
    topk_weight = (topk_weight / topk_weight.sum(-1, keepdim=True)).to(torch.float32)
    topk_ids = topk_ids.to(torch.int32)

    w1_shuffled = shuffle_weight(w1, layout=(16, 16))
    w2_shuffled = shuffle_weight(w2, layout=(16, 16))
    actual = fused_moe(
        hidden_states,
        w1_shuffled,
        w2_shuffled,
        topk_weight,
        topk_ids,
        activation=ActivationType.Silu,
        quant_type=QuantType.No,
    )
    patched = fused_moe(
        hidden_states,
        w1_shuffled,
        w2_shuffled,
        topk_weight,
        topk_ids,
        activation=ActivationType.Silu,
        quant_type=QuantType.No,
        fp32_route_accumulation=True,
    )
    reference_fp32 = moe_fp32(hidden_states, w1, w2, topk_weight, topk_ids)
    reference_bf16 = reference_fp32.to(DT)
    two_stage_fp32_routes = moe_two_stage_emulated(
        hidden_states, w1, w2, topk_weight, topk_ids, torch.float32
    )
    two_stage_bf16_routes = moe_two_stage_emulated(
        hidden_states, w1, w2, topk_weight, topk_ids, DT
    )
    torch_reference = torch_moe(
        hidden_states,
        w1,
        w2,
        topk_weight,
        topk_ids,
        activation=ActivationType.Silu,
    ).float()
    timing = time_callable(
        lambda: fused_moe(
            hidden_states,
            w1_shuffled,
            w2_shuffled,
            topk_weight,
            topk_ids,
            activation=ActivationType.Silu,
            quant_type=QuantType.No,
        )
    )
    patched_timing = time_callable(
        lambda: fused_moe(
            hidden_states,
            w1_shuffled,
            w2_shuffled,
            topk_weight,
            topk_ids,
            activation=ActivationType.Silu,
            quant_type=QuantType.No,
            fp32_route_accumulation=True,
        )
    )
    return {
        "case": "near_uniform_router_silu_bf16",
        "shape": {
            "tokens": tokens,
            "hidden": hidden,
            "intermediate": intermediate,
            "experts": experts,
            "topk": topk,
        },
        "vs_fp32": error_metrics(actual.float(), reference_fp32),
        "vs_bf16_rounded_fp32": error_metrics(actual.float(), reference_bf16.float()),
        "vs_two_stage_fp32_route_accum": error_metrics(
            actual.float(), two_stage_fp32_routes.float()
        ),
        "vs_two_stage_bf16_route_accum": error_metrics(
            actual.float(), two_stage_bf16_routes.float()
        ),
        "patched_vs_default": error_metrics(patched.float(), actual.float()),
        "patched_vs_two_stage_fp32_route_accum": error_metrics(
            patched.float(), two_stage_fp32_routes.float()
        ),
        "patched_vs_independent_fp32": error_metrics(
            patched.float(), reference_fp32
        ),
        "two_stage_fp32_routes_vs_independent_fp32": error_metrics(
            two_stage_fp32_routes.float(), reference_fp32
        ),
        "torch_moe_vs_independent_fp32": error_metrics(
            torch_reference, reference_fp32
        ),
        "argmax_vs_fp32": argmax_projection(actual.float(), reference_fp32),
        "argmax_vs_bf16_rounded_fp32": argmax_projection(
            actual.float(), reference_bf16.float()
        ),
        "argmax_vs_two_stage_fp32_route_accum": argmax_projection(
            actual.float(), two_stage_fp32_routes.float()
        ),
        "argmax_vs_two_stage_bf16_route_accum": argmax_projection(
            actual.float(), two_stage_bf16_routes.float()
        ),
        "argmax_patched_vs_default": argmax_projection(
            patched.float(), actual.float()
        ),
        "argmax_patched_vs_two_stage_fp32_route_accum": argmax_projection(
            patched.float(), two_stage_fp32_routes.float()
        ),
        "argmax_patched_vs_independent_fp32": argmax_projection(
            patched.float(), reference_fp32
        ),
        "timing_ms": timing,
        "patched_timing_ms": patched_timing,
    }


def main():
    configure_torch()
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise SystemExit("No CUDA/HIP device is available")
    device = torch.device("cuda:0")
    report = {
        "schema_version": 1,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "aiter_file": aiter.__file__,
        "device_name": torch.cuda.get_device_name(device),
        "device_capability": list(torch.cuda.get_device_capability(device)),
        "gpu_arch": os.environ.get("GPU_ARCHS", "native"),
        "rmsnorm": run_rmsnorm(device),
        "fused_moe": run_moe(device),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
