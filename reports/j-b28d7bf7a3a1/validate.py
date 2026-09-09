#!/usr/bin/env python3
import argparse
import json
import os
import sys

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--aiter-root",
        default="/job/aiter",
        help="AITER source checkout (default: /job/aiter)",
    )
    parser.add_argument(
        "--jit-dir",
        default="/job/.cache/aiter-jit",
        help="Job-private AITER JIT cache (default: /job/.cache/aiter-jit)",
    )
    return parser.parse_args()


def reference(q, k, v, scale, causal, gqa_ratio):
    qf = q.float()
    kf = k.float()
    vf = v.float()
    if gqa_ratio != 1:
        kf = kf.repeat_interleave(gqa_ratio, dim=2)
        vf = vf.repeat_interleave(gqa_ratio, dim=2)

    scores = torch.einsum("bthd,bshd->bhts", qf * scale, kf)
    if causal:
        query_index = torch.arange(q.shape[1], device=q.device).view(1, 1, -1, 1)
        key_index = torch.arange(k.shape[1], device=k.device).view(1, 1, 1, -1)
        scores = scores.masked_fill(key_index > query_index, float("-inf"))

    probabilities = torch.softmax(scores, dim=-1)
    output = torch.einsum("bhts,bshd->bthd", probabilities, vf)
    lse = torch.logsumexp(scores, dim=-1)
    return output, lse


def run_case(fmha_v3_fwd, name, dtype, dq, dv, causal, hq, hk, fp8):
    torch.manual_seed(1234)
    device = "cuda:0"
    batch_size = 2
    seqlen_q = 256
    seqlen_k = 256
    q = torch.randn(batch_size, seqlen_q, hq, dq, device=device)
    k = torch.randn(batch_size, seqlen_k, hk, dq, device=device)
    v = torch.randn(batch_size, seqlen_k, hk, dv, device=device)
    if fp8:
        q = q.to(dtype)
        k = k.to(dtype)
        v = v.to(dtype)
        descale = torch.ones(1, device=device, dtype=torch.float32)
        descales = (descale, descale, descale)
    else:
        q = q.to(dtype)
        k = k.to(dtype)
        v = v.to(dtype)
        descales = (None, None, None)

    try:
        out, lse, _, _ = fmha_v3_fwd(
            q,
            k,
            v,
            0.0,
            dq**-0.5,
            causal,
            -1,
            -1,
            True,
            False,
            1 if fp8 else 0,
            None,
            None,
            None,
            *descales,
            None,
        )
        torch.cuda.synchronize()
    except RuntimeError as error:
        return {
            "case": name,
            "status": "rejected",
            "error": str(error),
        }

    ref_out, ref_lse = reference(q, k, v, dq**-0.5, causal, hq // hk)
    output_error = (out.float() - ref_out).abs()
    lse_error = (lse - ref_lse).abs()
    output_tolerance = 0.06 if fp8 else 0.02
    lse_tolerance = 0.0001
    passed = (
        output_error.max().item() <= output_tolerance
        and lse_error.max().item() <= lse_tolerance
    )
    return {
        "case": name,
        "status": "passed" if passed else "failed",
        "output_dtype": str(out.dtype),
        "output_shape": list(out.shape),
        "max_abs_error": output_error.max().item(),
        "mean_abs_error": output_error.mean().item(),
        "max_lse_error": lse_error.max().item(),
        "output_tolerance": output_tolerance,
        "lse_tolerance": lse_tolerance,
    }


def main():
    args = parse_args()
    os.environ.setdefault("AITER_JIT_DIR", args.jit_dir)
    sys.path.insert(0, args.aiter_root)

    from aiter import dtypes
    from aiter.ops.mha import fmha_v3_fwd

    cases = [
        ("bf16-128x128-noncausal", torch.bfloat16, 128, 128, False, 4, 4, False),
        ("bf16-128x128-causal", torch.bfloat16, 128, 128, True, 4, 4, False),
        ("bf16-192x128-noncausal-gqa", torch.bfloat16, 192, 128, False, 4, 2, False),
        ("bf16-192x128-causal-gqa", torch.bfloat16, 192, 128, True, 4, 2, False),
        ("bf16-64x64-unsupported", torch.bfloat16, 64, 64, False, 4, 4, False),
        ("fp16-128x128-unsupported", torch.float16, 128, 128, False, 4, 4, False),
        ("fp16-192x128-unsupported", torch.float16, 192, 128, False, 4, 2, False),
        ("fp8-128x128-regression", dtypes.fp8, 128, 128, False, 4, 4, True),
    ]

    results = []
    expected_rejections = {
        "bf16-64x64-unsupported": "invalid argument for fmha_fwd",
        "fp16-128x128-unsupported": "fmha_v3_fwd only supports bf16 and fp8_e4m3 data type",
        "fp16-192x128-unsupported": "fmha_v3_fwd only supports bf16 and fp8_e4m3 data type",
    }
    for case in cases:
        result = run_case(fmha_v3_fwd, *case)
        if result["case"] in expected_rejections:
            expected = expected_rejections[result["case"]]
            if result["status"] != "rejected" or expected not in result["error"]:
                result["status"] = "failed"
        results.append(result)
        print(json.dumps(result), flush=True)

    failed = [result for result in results if result["status"] == "failed"]
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
