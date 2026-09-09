#!/usr/bin/env python3
"""Reproduce gfx942 ASM MLA decode split-KV failures under graph replay."""

import argparse
import json
import os
import sys

import torch

REPO_ROOT = os.environ.get(
    "AITER_SOURCE_DIR",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
sys.path.insert(0, REPO_ROOT)

import aiter
from aiter.mla import mla_decode_fwd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("eager", "graph"), required=True)
    parser.add_argument("--splits", type=int, required=True)
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--cache-churn", action="store_true")
    parser.add_argument("--out-threshold", type=float, default=5e-4)
    parser.add_argument("--lse-threshold", type=float, default=1e-4)
    return parser.parse_args()


def make_inputs():
    torch.manual_seed(0)
    batch, heads = 1, 16
    latent_dim, rope_dim = 512, 64
    head_dim = latent_dim + rope_dim
    topk, context = 2048, 65536
    device = "cuda"

    query = torch.randn(
        (batch, heads, head_dim), dtype=torch.bfloat16, device=device
    )
    kv = torch.randn((context, head_dim), dtype=torch.bfloat16, device=device)
    kv_buffer = kv.view(context, 1, 1, head_dim)
    kv_indices = torch.randperm(context, device=device)[:topk].to(torch.int32)
    qo_indptr = torch.tensor([0, 1], dtype=torch.int32, device=device)
    kv_indptr = torch.tensor([0, topk], dtype=torch.int32, device=device)
    kv_last_page_lens = torch.ones(batch, dtype=torch.int32, device=device)
    scale = head_dim**-0.5

    selected = kv[kv_indices.long()].float()
    scores = torch.einsum("bhd,td->bht", query.float(), selected) * scale
    ref_lse = scores.logsumexp(dim=-1)
    weights = torch.softmax(scores, dim=-1)
    ref_out = torch.einsum("bht,td->bhd", weights, selected[:, :latent_dim])

    assert torch.isfinite(query).all()
    assert torch.isfinite(kv).all()
    assert torch.isfinite(ref_out).all()
    assert torch.isfinite(ref_lse).all()

    return (
        query,
        kv_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        scale,
        ref_out,
        ref_lse,
    )


def run_eager(inputs, output, lse, splits):
    call_mla(inputs, output, lse, splits)
    torch.cuda.synchronize()


def call_mla(inputs, output, lse, splits):
    (
        query,
        kv_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        scale,
        _,
        _,
    ) = inputs
    _, returned_lse = mla_decode_fwd(
        query,
        kv_buffer,
        output,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        max_seqlen_q=1,
        page_size=1,
        sm_scale=scale,
        num_kv_splits=splits,
        return_lse=True,
    )
    lse.copy_(returned_lse)


def run_graph(inputs, output, lse, splits, cache_churn):
    warm_output = torch.empty_like(output)
    warm_lse = torch.empty_like(lse)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run_eager(inputs, warm_output, warm_lse, splits)
    torch.cuda.current_stream().wait_stream(stream)

    pool = torch.cuda.graph_pool_handle()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=pool):
        call_mla(inputs, output, lse, splits)

    if cache_churn:
        total_kv = inputs[4].shape[0]
        for offset in range(160):
            aiter.mla.get_meta_param(
                None,
                1,
                total_kv + offset + 1,
                16,
                1,
                torch.bfloat16,
            )

        _, captured_indptr = aiter.mla.get_meta_param(
            None, 1, total_kv, 16, 1, torch.bfloat16
        )
        scribble = torch.empty_like(captured_indptr)
        scribble_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(scribble_graph, pool=pool):
            scribble.copy_(
                torch.full_like(scribble, -777, dtype=torch.int32)
            )
        scribble_graph.replay()
        torch.cuda.synchronize()

    output.fill_(float("nan"))
    lse.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        print(json.dumps({"error": "CUDA unavailable"}), file=sys.stderr)
        return 2
    if torch.cuda.get_device_capability(0) != (9, 4):
        print(json.dumps({"error": "expected gfx942"}), file=sys.stderr)
        return 2

    original_stage1 = aiter.mla_decode_stage1_asm_fwd
    stage1_calls = 0

    def stage1_spy(*call_args, **call_kwargs):
        nonlocal stage1_calls
        stage1_calls += 1
        return original_stage1(*call_args, **call_kwargs)

    aiter.mla_decode_stage1_asm_fwd = stage1_spy
    inputs = make_inputs()
    output = torch.empty_like(inputs[7])
    lse = torch.empty_like(inputs[8])

    try:
        if args.mode == "eager":
            output.fill_(float("nan"))
            lse.fill_(float("nan"))
            run_eager(inputs, output, lse, args.splits)
        else:
            run_graph(inputs, output, lse, args.splits, args.cache_churn)
    finally:
        aiter.mla_decode_stage1_asm_fwd = original_stage1

    if stage1_calls == 0:
        print(json.dumps({"error": "ASM stage1 was not invoked"}), file=sys.stderr)
        return 2

    out_diff = (output.float() - inputs[7]).abs()
    lse_diff = (lse.float() - inputs[8]).abs()
    result = {
        "mode": args.mode,
        "splits": args.splits,
        "trial": args.trial,
        "cache_churn": args.cache_churn,
        "asm_stage1_calls": stage1_calls,
        "out_max_abs_diff": float(out_diff.max()),
        "out_nan_count": int(torch.isnan(output).sum()),
        "out_numel": output.numel(),
        "lse_max_abs_diff": float(lse_diff.max()),
        "lse_nan_count": int(torch.isnan(lse).sum()),
        "lse_numel": lse.numel(),
        "out_threshold": args.out_threshold,
        "lse_threshold": args.lse_threshold,
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
    }
    passed = (
        result["out_nan_count"] == 0
        and result["out_max_abs_diff"] <= args.out_threshold
        and result["lse_nan_count"] == 0
        and result["lse_max_abs_diff"] <= args.lse_threshold
    )
    result["passed"] = passed
    print(json.dumps(result, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
