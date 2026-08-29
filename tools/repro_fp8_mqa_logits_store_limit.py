"""Reproduce the 2 GiB store boundary in fp8_mqa_logits.

``fp8_mqa_logits`` selects its logits store path with:

``logits.numel() * logits.element_size() < 2 * 1024 * 1024 * 1024``

The output is float32, so the boundary is ``num_q * num_k * 4 < 2 GiB``.
Below that size, the kernel uses ``buffer_store`` through a 32-bit byte offset.
At or above it, the kernel takes the plain ``gl.store`` fallback.

Measured on 8xMI355X (gfx950) with images
``rocm/sgl-dev:v0.5.18-rocm720-mi35x-20260827`` and
``...-rocm724-mi35x-20260828``, using ``num_heads=32`` and ``head_dim=128``:

| num_q x num_k | logits bytes | selected path | result |
| --- | --- | --- | --- |
| 23,170 x 23,170 | 2,147,395,600 | ``buffer_store`` | returns normally |
| 23,171 x 23,171 | 2,147,580,964 | ``gl.store`` | LLVM assertion abort |
| 16,384 x 32,767 | 2,147,418,112 | ``buffer_store`` | returns normally |
| 16,384 x 32,768 | 2,147,483,648 | ``gl.store`` | LLVM assertion abort |

The over-limit assertion is:

``llvm/ADT/Sequence.h:275: Begin must be less or equal to End.``

An over-limit size aborts the whole process inside the Triton compiler instead
of raising a Python exception. Do not wrap the call in ``try``/``except`` and
conclude that the failure is handled.

With ``num_heads=64``, the same over-2-GiB shapes compile and return because
the launcher selects ``BLOCK_M=1``. With ``num_heads <= 32`` and
``seq_len > 4096``, it selects ``BLOCK_M=2``; only the multi-row store branch
was observed to be affected. The responsible constexpr was not isolated further.
"""

import argparse

import torch

from aiter.ops.triton.attention.fp8_mqa_logits import fp8_mqa_logits
from aiter.ops.triton.utils.types import e4m3_dtype

BUFFER_LIMIT_BYTES = 2 * 1024 * 1024 * 1024
LOGITS_ELEMENT_SIZE = torch.finfo(torch.float32).bits // 8


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce the fp8_mqa_logits 2 GiB logits store boundary",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("num_q", type=int, help="number of query rows")
    parser.add_argument("num_k", type=int, help="number of key/value rows")
    parser.add_argument(
        "num_heads", nargs="?", type=int, default=32, help="query head count"
    )
    args = parser.parse_args()

    if args.num_q <= 0 or args.num_k <= 0:
        parser.error("num_q and num_k must be positive")
    if args.num_heads <= 0 or args.num_heads & (args.num_heads - 1):
        parser.error("num_heads must be a positive power of 2")

    head_dim = 128
    logits_bytes = args.num_q * args.num_k * LOGITS_ELEMENT_SIZE
    store_path = "buffer_store" if logits_bytes < BUFFER_LIMIT_BYTES else "gl.store"
    print(f"logits shape: ({args.num_q}, {args.num_k}), dtype: torch.float32")
    print(f"num_heads: {args.num_heads}, head_dim: {head_dim}")
    print(f"logits bytes: {logits_bytes:,}")
    print(f"store byte-offset limit: {BUFFER_LIMIT_BYTES:,}")
    print(f"bytes below limit: {BUFFER_LIMIT_BYTES - logits_bytes:,}")
    print(f"selected store path: {store_path}")
    if store_path == "gl.store":
        print("WARNING: this size can abort the process during Triton compilation.")
        print("The abort is not a Python exception and cannot be caught safely.")
    print("calling fp8_mqa_logits...", flush=True)

    if not torch.cuda.is_available():
        raise SystemExit("This reproducer requires a CUDA/ROCm GPU.")

    device = "cuda"
    print(f"device: {torch.cuda.get_device_name()}")
    with torch.inference_mode():
        torch.manual_seed(0)
        q = torch.randn(
            args.num_q, args.num_heads, head_dim, device=device, dtype=torch.float32
        ).to(e4m3_dtype)
        kv = torch.randn(args.num_k, head_dim, device=device, dtype=torch.float32).to(
            e4m3_dtype
        )
        kv_scales = torch.ones(args.num_k, device=device, dtype=torch.float32)
        weights = torch.randn(
            args.num_q, args.num_heads, device=device, dtype=torch.float32
        )
        cu_starts = torch.zeros(args.num_q, dtype=torch.int32, device=device)
        cu_ends = torch.clamp(
            torch.arange(1, args.num_q + 1, dtype=torch.int32, device=device),
            max=args.num_k,
        )

        logits = fp8_mqa_logits(
            q, kv, kv_scales, weights, cu_starts, cu_ends, clean_logits=True
        )
        torch.cuda.synchronize()
        print(f"fp8_mqa_logits returned logits with shape {tuple(logits.shape)}")


if __name__ == "__main__":
    main()
