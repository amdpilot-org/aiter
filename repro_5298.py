import argparse

import torch

from aiter.ops.triton.attention.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--next-n", type=int, default=5)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--context", type=int, default=131072)
    parser.add_argument("--chunk-k", type=int, default=64)
    parser.add_argument("--kv-block-size", type=int, default=16)
    args = parser.parse_args()

    block = args.kv_block_size
    num_blocks = (args.context + block - 1) // block
    q = torch.randn(
        args.batch,
        args.next_n,
        args.heads,
        args.dim,
        device="cuda",
        dtype=torch.bfloat16,
    ).to(torch.float8_e4m3fn)
    kv = torch.randn(
        num_blocks,
        block,
        1,
        args.dim + 4,
        device="cuda",
        dtype=torch.bfloat16,
    ).to(torch.uint8)
    weights = torch.randn(
        args.batch * args.next_n, args.heads, device="cuda", dtype=torch.float32
    )
    context_lens = torch.full(
        (args.batch,), args.context, device="cuda", dtype=torch.int32
    )
    block_table = torch.arange(
        num_blocks, device="cuda", dtype=torch.int32
    ).repeat(args.batch, 1)
    out = torch.empty(
        args.batch * args.next_n,
        args.context,
        device="cuda",
        dtype=torch.float32,
    )

    deepgemm_fp8_paged_mqa_logits(
        q,
        kv,
        weights,
        out,
        context_lens,
        block_table,
        args.context,
        Preshuffle=True,
        KVBlockSize=block,
        ChunkK=args.chunk_k,
        WavePerEU=8,
    )
    torch.cuda.synchronize()
    print(f"completed ChunkK={args.chunk_k} finite={torch.isfinite(out).all().item()}")


if __name__ == "__main__":
    main()
