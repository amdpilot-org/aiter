#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import torch

from aiter import dtypes, per_tensor_quant
from aiter.ops.mha import _flash_attn_forward, _flash_attn_varlen_forward
from aiter.test_mha_common import attention_ref, generate_qkv, generate_random_padding_mask


def finite_max_diff(actual, expected):
    finite = torch.isfinite(expected)
    if not finite.any():
        return 0.0
    return (actual[finite] - expected[finite]).abs().max().item()


def lse_variant(logits, mask=None):
    if mask is not None:
        logits = logits.masked_fill(~mask, float("-inf"))
    return torch.logsumexp(logits, dim=-1)


def summarize(args, production_lse, direct_lse, reference_lse, output, direct_output, metadata):
    finite = torch.isfinite(direct_lse)
    diff = (production_lse[finite] - direct_lse[finite]).abs()
    max_diff = diff.max().item() if diff.numel() else 0.0
    worst = int(diff.argmax().item()) if diff.numel() else 0
    production_flat = production_lse.reshape(-1)
    direct_flat = direct_lse.reshape(-1)
    result = {
        "case": args.path,
        "sequence_length": args.seq,
        "causal": args.causal,
        "dtype": str(metadata["q_quant"].dtype),
        "softmax_scale": metadata["softmax_scale"],
        "descales": {
            "q": metadata["q_descale"].item(),
            "k": metadata["k_descale"].item(),
            "v": metadata["v_descale"].item(),
        },
        "shapes": {
            "q_quant": list(metadata["q_quant"].shape),
            "k_quant": list(metadata["k_quant"].shape),
            "v_quant": list(metadata["v_quant"].shape),
            "production_lse": list(production_lse.shape),
            "direct_lse": list(direct_lse.shape),
        },
        "cu_seqlens_q": metadata.get("cu_seqlens_q", torch.empty(0, dtype=torch.int32)).tolist(),
        "cu_seqlens_k": metadata.get("cu_seqlens_k", torch.empty(0, dtype=torch.int32)).tolist(),
        "production_lse_dtype": str(production_lse.dtype),
        "direct_lse_dtype": str(direct_lse.dtype),
        "neginf_equal": bool(torch.equal(torch.isneginf(production_lse), torch.isneginf(direct_lse))),
        "neginf_count": int(torch.isneginf(direct_lse).sum().item()),
        "production_vs_direct_lse_max_abs": max_diff,
        "reference_vs_direct_lse_max_abs": finite_max_diff(reference_lse, direct_lse),
        "output_max_abs": (output.float() - direct_output).abs().max().item(),
        "output_gate_0_055": bool((output.float() - direct_output).abs().max().item() < 0.055),
        "lse_gate_0_01": bool(max_diff < 0.01),
        "worst_flat_index": worst,
        "worst_production_lse": production_flat[worst].item() if worst < production_flat.numel() else None,
        "worst_direct_lse": direct_flat[worst].item() if worst < direct_flat.numel() else None,
    }
    for name, value in metadata["diagnostics"].items():
        result[f"{name}_vs_direct_lse_max_abs"] = finite_max_diff(value, direct_lse)
    return result


def run_fixed(args):
    torch.random.manual_seed(0)
    q = torch.rand(1, args.seq, 8, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.rand(1, args.seq, 8, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.rand(1, args.seq, 8, 128, device="cuda", dtype=torch.bfloat16)
    q_quant, q_descale = per_tensor_quant(q, quant_dtype=dtypes.fp8)
    k_quant, k_descale = per_tensor_quant(k, quant_dtype=dtypes.fp8)
    v_quant, v_descale = per_tensor_quant(v, quant_dtype=dtypes.fp8)
    scale = 128 ** -0.5
    output, lse, _, _ = _flash_attn_forward(
        q_quant, k_quant, v_quant, 0.0, scale, causal=args.causal,
        window_size_left=-1, window_size_right=-1, sink_size=0, bias=None,
        alibi_slopes=None, q_descale=q_descale, k_descale=k_descale,
        v_descale=v_descale, return_lse=True, return_softmax=False,
    )
    q_f = q_quant.float() * q_descale
    k_f = k_quant.float() * k_descale
    v_f = v_quant.float() * v_descale
    raw_logits = torch.einsum("bthd,bshd->bhts", q_f, k_f)
    logits = raw_logits * scale
    row = torch.arange(args.seq, device="cuda")[:, None]
    col = torch.arange(args.seq, device="cuda")[None, :]
    causal_mask = col <= row
    direct_lse = lse_variant(logits, causal_mask if args.causal else None)
    direct_output = torch.einsum(
        "bhts,bshd->bthd", torch.softmax(logits.masked_fill(~causal_mask, float("-inf")) if args.causal else logits, dim=-1), v_f
    )
    _, _, reference_lse = attention_ref(
        q_f, k_f, v_f, causal=args.causal, window_size=(-1, -1), upcast=True
    )
    diagnostics = {
        "unscaled": lse_variant(raw_logits, causal_mask if args.causal else None),
        "scale_squared": lse_variant(raw_logits * scale * scale, causal_mask if args.causal else None),
        "noncausal": lse_variant(logits),
        "strictly_causal": lse_variant(logits, col < row),
    }
    metadata = {
        "q_quant": q_quant, "k_quant": k_quant, "v_quant": v_quant,
        "q_descale": q_descale, "k_descale": k_descale, "v_descale": v_descale,
        "softmax_scale": scale, "diagnostics": diagnostics,
    }
    result = summarize(args, lse, direct_lse, reference_lse, output, direct_output, metadata)
    tensors = {
        "q_quant": q_quant.cpu(), "k_quant": k_quant.cpu(), "v_quant": v_quant.cpu(),
        "q_descale": q_descale.cpu(), "k_descale": k_descale.cpu(), "v_descale": v_descale.cpu(),
        "production_lse": lse.cpu(), "direct_lse": direct_lse.cpu(),
        "reference_lse": reference_lse.cpu(), "production_output": output.cpu(),
        "direct_output": direct_output.cpu(), "causal_mask": causal_mask.cpu(),
    }
    return result, tensors


def run_varlen(args):
    torch.random.manual_seed(0)
    q_pad = torch.rand(1, args.seq, 8, 128, device="cuda", dtype=torch.bfloat16)
    k_pad = torch.rand(1, args.seq, 8, 128, device="cuda", dtype=torch.bfloat16)
    v_pad = torch.rand(1, args.seq, 8, 128, device="cuda", dtype=torch.bfloat16)
    query_padding_mask = generate_random_padding_mask(args.seq, 1, "cuda", mode="random")
    key_padding_mask = generate_random_padding_mask(args.seq, 1, "cuda", mode="random")
    q, k, v, cu_q, cu_k, max_q, max_k, _, _, _, _, _, _ = generate_qkv(
        q_pad, k_pad, v_pad, query_padding_mask, key_padding_mask, kvpacked=False
    )
    q_quant, q_descale = per_tensor_quant(q, quant_dtype=dtypes.fp8)
    k_quant, k_descale = per_tensor_quant(k, quant_dtype=dtypes.fp8)
    v_quant, v_descale = per_tensor_quant(v, quant_dtype=dtypes.fp8)
    scale = 128 ** -0.5
    output, lse, _, _ = _flash_attn_varlen_forward(
        q_quant, k_quant, v_quant, cu_q, cu_k, None, None, max_q, max_k, 0,
        0.0, scale, causal=args.causal, logits_soft_cap=0.0,
        window_size_left=-1, window_size_right=-1, sink_size=0, bias=None,
        alibi_slopes=None, q_descale=q_descale, k_descale=k_descale,
        v_descale=v_descale, return_lse=True, return_softmax=False,
    )
    q_f = q_quant.float() * q_descale
    k_f = k_quant.float() * k_descale
    v_f = v_quant.float() * v_descale
    direct_lse_parts = []
    direct_output_parts = []
    reference_lse_parts = []
    unscaled_parts = []
    scale_squared_parts = []
    noncausal_parts = []
    strictly_causal_parts = []
    causal_masks = []
    for q_start, q_end, k_start, k_end in zip(cu_q[:-1], cu_q[1:], cu_k[:-1], cu_k[1:]):
        q_seg = q_f[q_start:q_end]
        k_seg = k_f[k_start:k_end]
        v_seg = v_f[k_start:k_end]
        sq = int(q_end - q_start)
        sk = int(k_end - k_start)
        raw_logits = torch.einsum("thd,shd->hts", q_seg, k_seg)
        logits = raw_logits * scale
        row = torch.arange(sq, device="cuda")[:, None]
        col = torch.arange(sk, device="cuda")[None, :]
        causal_mask = col <= row + (sk - sq)
        masked_logits = logits.masked_fill(~causal_mask, float("-inf")) if args.causal else logits
        direct_lse_parts.append(torch.logsumexp(masked_logits, dim=-1))
        attention = torch.softmax(masked_logits, dim=-1)
        fully_masked = torch.isneginf(direct_lse_parts[-1])
        attention = attention.masked_fill(fully_masked[:, :, None], 0.0)
        direct_output_parts.append(torch.einsum("hts,shd->thd", attention, v_seg))
        _, _, ref_part = attention_ref(
            q_seg.unsqueeze(0), k_seg.unsqueeze(0), v_seg.unsqueeze(0),
            causal=args.causal, window_size=(-1, -1), upcast=True,
        )
        reference_lse_parts.append(ref_part.squeeze(0))
        active_mask = causal_mask if args.causal else None
        unscaled_parts.append(lse_variant(raw_logits, active_mask))
        scale_squared_parts.append(lse_variant(raw_logits * scale * scale, active_mask))
        noncausal_parts.append(lse_variant(logits))
        strictly_causal_parts.append(lse_variant(logits, col < row + (sk - sq)))
        causal_masks.append(causal_mask)
    direct_lse = torch.cat(direct_lse_parts, dim=1)
    direct_output = torch.cat(direct_output_parts, dim=0)
    reference_lse = torch.cat(reference_lse_parts, dim=1)
    diagnostics = {
        "unscaled": torch.cat(unscaled_parts, dim=1),
        "scale_squared": torch.cat(scale_squared_parts, dim=1),
        "noncausal": torch.cat(noncausal_parts, dim=1),
        "strictly_causal": torch.cat(strictly_causal_parts, dim=1),
    }
    metadata = {
        "q_quant": q_quant, "k_quant": k_quant, "v_quant": v_quant,
        "q_descale": q_descale, "k_descale": k_descale, "v_descale": v_descale,
        "softmax_scale": scale, "diagnostics": diagnostics,
        "cu_seqlens_q": cu_q.cpu(), "cu_seqlens_k": cu_k.cpu(),
    }
    result = summarize(args, lse, direct_lse, reference_lse, output, direct_output, metadata)
    tensors = {
        "q_quant": q_quant.detach().cpu(), "k_quant": k_quant.detach().cpu(),
        "v_quant": v_quant.detach().cpu(), "q_descale": q_descale.detach().cpu(),
        "k_descale": k_descale.detach().cpu(), "v_descale": v_descale.detach().cpu(),
        "cu_seqlens_q": cu_q.detach().cpu(), "cu_seqlens_k": cu_k.detach().cpu(),
        "production_lse": lse.detach().cpu(), "direct_lse": direct_lse.detach().cpu(),
        "reference_lse": reference_lse.detach().cpu(), "production_output": output.detach().cpu(),
        "direct_output": direct_output.detach().cpu(),
        "causal_mask": torch.cat(causal_masks, dim=0).detach().cpu(),
    }
    return result, tensors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", choices=("fixed", "varlen"), required=True)
    parser.add_argument("--seq", type=int, required=True)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--tensor-out", type=Path)
    args = parser.parse_args()
    result, tensors = run_fixed(args) if args.path == "fixed" else run_varlen(args)
    if args.tensor_out:
        args.tensor_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensors, args.tensor_out)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
