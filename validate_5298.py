import argparse
import statistics
import time

import torch

from aiter.ops.triton.attention.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits
from aiter.ops.shuffle import shuffle_weight


def make_inputs(batch, next_n, heads, dim, context_lens, block_size=16):
    max_context = max(context_lens)
    num_blocks = (max_context + block_size - 1) // block_size
    q = torch.randn(
        batch, next_n, heads, dim, device="cuda", dtype=torch.bfloat16
    ).to(torch.float8_e4m3fn)
    values = torch.randn(
        num_blocks, block_size, dim, device="cuda", dtype=torch.bfloat16
    ).to(torch.float8_e4m3fn)
    shuffled_values = shuffle_weight(values)
    scales = torch.rand(num_blocks, block_size, device="cuda") + 0.5
    kv_flat = torch.empty(
        num_blocks, block_size * (dim + 4), device="cuda", dtype=torch.uint8
    )
    kv_flat[:, : block_size * dim] = shuffled_values.view(
        num_blocks, block_size * dim
    ).view(torch.uint8)
    kv_flat[:, block_size * dim :] = scales.view(
        num_blocks, block_size
    ).view(torch.uint8)
    kv = kv_flat.view(num_blocks, block_size, 1, dim + 4)
    weights = torch.randn(
        batch * next_n, heads, device="cuda", dtype=torch.float32
    )
    context_lens_tensor = torch.tensor(
        context_lens, device="cuda", dtype=torch.int32
    )
    block_table = torch.arange(
        num_blocks, device="cuda", dtype=torch.int32
    ).repeat(batch, 1)
    return q, kv, weights, context_lens_tensor, block_table, max_context, values, scales


def torch_reference(inputs, next_n, heads, dim, context_lens):
    q, _, weights, _, block_table, _, values, scales = inputs
    block_size = values.shape[1]
    batch = len(context_lens)
    reference = torch.empty(
        batch * next_n, max(context_lens), device="cuda", dtype=torch.float32
    )
    for batch_idx, context_len in enumerate(context_lens):
        num_blocks = (context_len + block_size - 1) // block_size
        block_ids = block_table[batch_idx, :num_blocks].long()
        keys = values[block_ids].reshape(-1, dim)[:context_len].float()
        key_scales = scales[block_ids].reshape(-1)[:context_len].float()
        positions = torch.arange(context_len, device="cuda")
        for next_idx in range(next_n):
            scores = q[batch_idx, next_idx].float() @ keys.T
            scores = torch.relu(scores * key_scales[None, :])
            scores = scores * weights[batch_idx * next_n + next_idx, :, None]
            logits = scores.sum(dim=0)
            valid = positions <= context_len - next_n + next_idx
            logits = torch.where(valid, logits, torch.tensor(float("-inf"), device="cuda"))
            reference[batch_idx * next_n + next_idx, :context_len] = logits
    return reference


def run_kernel(inputs, out, chunk_k, wave_per_eu=8):
    q, kv, weights, context_lens, block_table, max_context, *_ = inputs
    block_size = kv.shape[1]
    out.fill_(float("nan"))
    deepgemm_fp8_paged_mqa_logits(
        q,
        kv,
        weights,
        out,
        context_lens,
        block_table,
        max_context,
        Preshuffle=True,
        KVBlockSize=block_size,
        ChunkK=chunk_k,
        WavePerEU=wave_per_eu,
    )
    torch.cuda.synchronize()


def compare(out, reference, context_lens, next_n, label):
    max_abs = 0.0
    for batch_idx, context_len in enumerate(context_lens):
        for next_idx in range(next_n):
            row = batch_idx * next_n + next_idx
            expected = reference[row, :context_len]
            actual = out[row, :context_len]
            valid = torch.isfinite(expected)
            assert torch.equal(actual[~valid], expected[~valid]), label
            difference = (actual[valid] - expected[valid]).abs()
            if difference.numel():
                max_abs = max(max_abs, difference.max().item())
            torch.testing.assert_close(
                actual[valid],
                expected[valid],
                rtol=2e-2,
                atol=2e-2,
                msg=lambda message: f"{label}: {message}",
            )
    print(f"PASS {label}: max_abs={max_abs:.6g}")


def correctness_case(
    context_lens, batch, next_n, heads, dim, chunk_sizes, block_size
):
    inputs = make_inputs(batch, next_n, heads, dim, context_lens, block_size)
    reference = torch_reference(inputs, next_n, heads, dim, context_lens)
    out = torch.empty(
        batch * next_n,
        max(context_lens),
        device="cuda",
        dtype=torch.float32,
    )
    for chunk_k in chunk_sizes:
        run_kernel(inputs, out, chunk_k)
        compare(
            out,
            reference,
            context_lens,
            next_n,
            f"batch={batch} next_n={next_n} heads={heads} context={context_lens} "
            f"KVBlockSize={block_size} ChunkK={chunk_k}",
        )


def latency_case(block_size):
    batch, next_n, heads, dim, context = 8, 5, 32, 128, 131072
    inputs = make_inputs(
        batch, next_n, heads, dim, [context] * batch, block_size
    )
    out = torch.empty(
        batch * next_n, context, device="cuda", dtype=torch.float32
    )
    configurations = [
        (64, 8),
        (64, 4),
        (64, 2),
        (128, 8),
        (256, 8),
        (512, 8),
        (256, 2),
    ]
    for chunk_k, wave_per_eu in configurations:
        for _ in range(8):
            run_kernel(inputs, out, chunk_k, wave_per_eu)
        samples = []
        for _ in range(3):
            torch.cuda.synchronize()
            start = time.perf_counter()
            deepgemm_fp8_paged_mqa_logits(
                inputs[0],
                inputs[1],
                inputs[2],
                out,
                inputs[3],
                inputs[4],
                context,
                Preshuffle=True,
                KVBlockSize=block_size,
                ChunkK=chunk_k,
                WavePerEU=wave_per_eu,
            )
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) * 1e3)
        print(
            f"LATENCY ChunkK={chunk_k} WavePerEU={wave_per_eu}: "
            f"median={statistics.median(samples):.4f} ms samples="
            + ",".join(f"{sample:.4f}" for sample in samples)
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-large", action="store_true")
    parser.add_argument("--kv-block-sizes", default="16")
    args = parser.parse_args()

    block_sizes = [int(value) for value in args.kv_block_sizes.split(",")]
    chunk_sizes = (64, 128, 256, 512)
    for block_size in block_sizes:
        if not args.skip_large:
            correctness_case(
                [131072] * 8, 8, 5, 32, 128, chunk_sizes, block_size
            )
        for context in (1, 15, 16, 17, 63, 64, 65, 127, 128, 129):
            correctness_case(
                [context], 1, 1, 32, 128, chunk_sizes, block_size
            )
        correctness_case(
            [17] * 2, 2, 1, 32, 128, chunk_sizes, block_size
        )
        correctness_case(
            [65] * 3, 3, 2, 32, 128, chunk_sizes, block_size
        )
        latency_case(block_size)


if __name__ == "__main__":
    main()
