import argparse
import json
import os
import time

import torch

import aiter
import aiter.fused_moe as fused_moe_module
from aiter import dtypes, pertoken_quant
from aiter.fused_moe import fused_moe
from aiter.jit.utils.chip_info import get_cu_num, get_gfx
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import checkAllclose


def quantize_weight_128x128(weight, quant_dtype):
    expert_count, rows, cols = weight.shape
    blocks = weight.view(
        expert_count,
        rows // 128,
        128,
        cols // 128,
        128,
    ).permute(0, 1, 3, 2, 4)
    quantized, scale = pertoken_quant(
        blocks.reshape(expert_count, -1, 128 * 128),
        quant_dtype=quant_dtype,
    )
    quantized = quantized.view(
        expert_count,
        rows // 128,
        cols // 128,
        128,
        128,
    ).permute(0, 1, 3, 2, 4)
    return (
        quantized.reshape(weight.shape).contiguous(),
        scale.view(expert_count, rows // 128, cols // 128).contiguous(),
    )


def dequantize_weight_128x128(weight, scale):
    rows, cols = weight.shape
    return (
        weight.float()
        .view(rows // 128, 128, cols // 128, 128)
        * scale.view(rows // 128, 1, cols // 128, 1)
    ).reshape(rows, cols)


def dequantize_rows_1x128(quantized, scale):
    rows, cols = quantized.shape
    return (
        quantized.float()
        .view(rows, cols // 128, 128)
        * scale.view(rows, cols // 128, 1)
    ).reshape(rows, cols)


def independent_reference(
    activation_quantized,
    activation_scale,
    w1_quantized,
    w1_scale,
    w2_quantized,
    w2_scale,
    topk_weight,
    topk_ids,
    expert_count,
    inter_dim,
    model_dim,
):
    token_count, topk = topk_ids.shape
    activation_dequantized = dequantize_rows_1x128(
        activation_quantized, activation_scale
    )
    reference = torch.zeros(
        (token_count, model_dim), dtype=torch.float32, device=activation_quantized.device
    )

    for expert_id in range(expert_count):
        token_slots = topk_ids == expert_id
        if not token_slots.any():
            continue
        token_indices, slot_indices = token_slots.nonzero(as_tuple=True)
        expert_activation = activation_dequantized[token_indices]
        expert_w1 = dequantize_weight_128x128(
            w1_quantized[expert_id], w1_scale[expert_id]
        )
        gate_up = expert_activation @ expert_w1.transpose(0, 1)
        gate, up = gate_up.split([inter_dim, inter_dim], dim=-1)
        hidden = torch.nn.functional.silu(gate) * up

        expert_w2 = dequantize_weight_128x128(
            w2_quantized[expert_id], w2_scale[expert_id]
        )
        expert_output = hidden @ expert_w2.transpose(0, 1)
        route_weight = topk_weight[token_indices, slot_indices].float().unsqueeze(-1)
        reference.index_add_(
            0,
            token_indices,
            expert_output * route_weight,
        )

    return reference


def capture_dispatch():
    original_lookup = fused_moe_module.get_2stage_cfgs
    dispatch = []

    def wrapped_lookup(*args, **kwargs):
        metadata = original_lookup(*args, **kwargs)
        stage1 = getattr(metadata.stage1, "func", metadata.stage1)
        stage2 = getattr(metadata.stage2, "func", metadata.stage2)
        dispatch.append(
            {
                "run_1stage": bool(metadata.run_1stage),
                "block_m": int(metadata.block_m),
                "ksplit": int(metadata.ksplit),
                "stage1": getattr(stage1, "__name__", str(stage1)),
                "stage1_kernel": getattr(metadata.stage1, "keywords", {}).get(
                    "kernelName", ""
                ),
                "stage2": None if metadata.stage2 is None else getattr(
                    stage2, "__name__", str(stage2)
                ),
                "stage2_kernel": ""
                if metadata.stage2 is None
                else getattr(metadata.stage2, "keywords", {}).get("kernelName", ""),
            }
        )
        return metadata

    fused_moe_module.get_2stage_cfgs = wrapped_lookup
    return dispatch, original_lookup


def capture_launched_kernel():
    original_kernel = aiter.fmoe_fp8_blockscale_g1u1
    launched = []

    def wrapped_kernel(*args, **kwargs):
        kernel_names = [argument for argument in args if isinstance(argument, str)]
        if "kernelName" in kwargs:
            kernel_names.append(kwargs["kernelName"])
        launched.append(kernel_names[-1] if kernel_names else "")
        return original_kernel(*args, **kwargs)

    aiter.fmoe_fp8_blockscale_g1u1 = wrapped_kernel
    return launched, original_kernel


def run_case(token_count, inter_dim, model_dim, expert_count, topk, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device = torch.device("cuda:0")

    activation = (torch.randn(token_count, model_dim, device=device) / 10).to(
        dtypes.bf16
    )
    w1 = (torch.randn(expert_count, inter_dim * 2, model_dim, device=device) / 10).to(
        dtypes.bf16
    )
    w2 = (torch.randn(expert_count, model_dim, inter_dim, device=device) / 10).to(
        dtypes.bf16
    )

    activation_quantized, activation_scale = pertoken_quant(
        activation.view(token_count, model_dim // 128, 128),
        quant_dtype=dtypes.fp8,
    )
    activation_quantized = activation_quantized.reshape(token_count, model_dim)
    activation_scale = activation_scale.squeeze(-1)

    w1_quantized, w1_scale = quantize_weight_128x128(w1, dtypes.fp8)
    w2_quantized, w2_scale = quantize_weight_128x128(w2, dtypes.fp8)
    w1_kernel = shuffle_weight(w1_quantized, layout=(16, 16))
    w2_kernel = shuffle_weight(w2_quantized, layout=(16, 16))

    routing_scores = torch.randn(token_count, expert_count, device=device)
    topk_weight, topk_ids = torch.topk(
        torch.softmax(routing_scores.float(), dim=-1), topk, dim=-1
    )
    topk_ids = topk_ids.to(torch.int32)

    dispatch, original_lookup = capture_dispatch()
    launched_kernels, original_kernel = capture_launched_kernel()
    started = time.monotonic()
    try:
        output = fused_moe(
            activation,
            w1_kernel,
            w2_kernel,
            topk_weight,
            topk_ids,
            activation=aiter.ActivationType.Silu,
            quant_type=aiter.QuantType.per_128x128,
            doweight_stage1=False,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
        )
        torch.cuda.synchronize()
    finally:
        aiter.fmoe_fp8_blockscale_g1u1 = original_kernel
        fused_moe_module.get_2stage_cfgs = original_lookup
    kernel_elapsed_ms = (time.monotonic() - started) * 1000.0

    reference = independent_reference(
        activation_quantized,
        activation_scale,
        w1_quantized,
        w1_scale,
        w2_quantized,
        w2_scale,
        topk_weight,
        topk_ids,
        expert_count,
        inter_dim,
        model_dim,
    )
    output_float = output.float()
    difference = output_float - reference
    output_double = output_float.double().flatten()
    reference_double = reference.double().flatten()
    logits_diff = 1 - (
        2
        * (output_double * reference_double).sum()
        / (output_double * output_double + reference_double * reference_double).sum()
    ).item()
    gate_error = checkAllclose(
        reference,
        output_float,
        msg=(
            f"M={token_count} N={inter_dim} K={model_dim} "
            f"E={expert_count} topk={topk}"
        ),
    )
    result = {
        "M": token_count,
        "N": inter_dim,
        "K": model_dim,
        "E": expert_count,
        "topk": topk,
        "seed": seed,
        "gfx": get_gfx(),
        "cu_num": get_cu_num(),
        "dispatch": dispatch,
        "launched_kernels": launched_kernels,
        "kernel_elapsed_ms": kernel_elapsed_ms,
        "output_dtype": str(output.dtype),
        "reference_dtype": str(reference.dtype),
        "output_finite": bool(torch.isfinite(output_float).all().item()),
        "reference_finite": bool(torch.isfinite(reference).all().item()),
        "max_abs_error": float(difference.abs().max().item()),
        "mean_abs_error": float(difference.abs().mean().item()),
        "relative_l2_error": float(
            (torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(reference)).item()
        ),
        "cosine_similarity": float(
            torch.nn.functional.cosine_similarity(
                output_float.flatten(), reference.flatten(), dim=0
            ).item()
        ),
        "logits_diff": float(logits_diff),
        "existing_logit_diff_gate_pass": bool(
            not (gate_error != 0 and logits_diff > 0.01)
        ),
        "checkAllclose_default_error": float(gate_error),
        "checkAllclose_default_pass": bool(gate_error == 0),
    }
    print("BOUNDARY_RESULT " + json.dumps(result, sort_keys=True))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, required=True)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--K", type=int, required=True)
    parser.add_argument("--E", type=int, default=32)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()
    run_case(args.M, args.N, args.K, args.E, args.topk, args.seed)


if __name__ == "__main__":
    main()
