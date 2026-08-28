#!/usr/bin/env python3
# Reproduction for gfx950 gemm_a8w8_blockscale_bpreshuffle v1 accumulator nondeterminism.
#
# Tensors mirror op_tests/test_gemm_a8w8_blockscale.py: FP8 x[M,K]/weight[N,K],
# fp32 block scales (block (128,128)), transposed x_scale,
# shuffle_weight(weight, layout=(16,16)), BF16 output. Calls the CK JIT module
# (gemm_a8w8_blockscale_bpreshuffle_ck) directly -> C++ heuristic v1 dispatch.
#
# Env: HIP_VISIBLE_DEVICES=0, fresh AITER_JIT_DIR,
#      AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE=<empty csv> (minimal build),
#      AITER_USE_SYSTEM_TRITON=1 (if triton<3.6; gemm path does not use gluon).
#
# Usage:
#   AITER_SRC=/path/to/aiter HIP_VISIBLE_DEVICES=0 AITER_JIT_DIR=/tmp/jit \
#     AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE=/tmp/empty.csv \
#     python repro.py --warmups 10 --iters 1000
import os, sys, hashlib, argparse

sys.path.insert(0, os.environ.get("AITER_SRC", "."))

import torch
import aiter
from aiter import dtypes
from aiter.ops.shuffle import shuffle_weight
from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale_bpreshuffle_ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=8192)
    ap.add_argument("--N", type=int, default=512)
    ap.add_argument("--K", type=int, default=4096)
    ap.add_argument("--warmups", type=int, default=10)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    M, N, K = args.M, args.N, args.K
    torch.manual_seed(args.seed)

    block_shape_n, block_shape_k = 128, 128
    scale_n = (N + block_shape_n - 1) // block_shape_n
    scale_k = (K + block_shape_k - 1) // block_shape_k

    x = (torch.rand((M, K), dtype=dtypes.fp32, device="cuda") / 10).to(dtypes.fp8)
    weight = (torch.rand((N, K), dtype=dtypes.fp32, device="cuda") / 10).to(dtypes.fp8)
    x_scale = torch.rand([M, scale_k], dtype=dtypes.fp32, device="cuda")
    w_scale = torch.rand([scale_n, scale_k], dtype=dtypes.fp32, device="cuda")

    x_scale_t = x_scale.transpose(0, 1).contiguous().view(*x_scale.shape)
    wshuf = shuffle_weight(weight, layout=(16, 16))

    Y = torch.empty(M, N, dtype=dtypes.bf16, device="cuda")

    for _ in range(args.warmups):  # first call triggers the JIT build
        gemm_a8w8_blockscale_bpreshuffle_ck(x, wshuf, x_scale_t, w_scale, Y)
        torch.cuda.synchronize()

    hashes = {}
    for _ in range(args.iters):
        gemm_a8w8_blockscale_bpreshuffle_ck(x, wshuf, x_scale_t, w_scale, Y)
        torch.cuda.synchronize()
        h = hashlib.sha256(Y.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
        hashes[h] = hashes.get(h, 0) + 1

    print(f"RESULT M={M} N={N} K={K} seed={args.seed} "
          f"unique_hashes={len(hashes)}/{args.iters}", flush=True)
    for h, c in sorted(hashes.items(), key=lambda kv: -kv[1])[:5]:
        print(f"  {h} count={c}", flush=True)


if __name__ == "__main__":
    main()
