# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import functools

import torch
from torch import Tensor

from ..jit.core import compile_ops


@compile_ops(
    "module_gemm_a16w16_asm",
    fc_name="gemm_a16w16_asm",
    ffi_type="ctypes",
)
def _gemm_a16w16_asm(
    A: Tensor,
    B: Tensor,
    out: Tensor,
    semaphore: Tensor,
    bias: Tensor | None = None,
    splitK: int | None = None,
    kernelName: str | None = None,
    bpreshuffle: bool = False,
) -> None: ...


# Semaphore workspace shape for ASM SplitK kernels.
# The kernel indexes into a flat array of size rows*cols; candidates whose
# grid (gdx*gdy) exceeds this limit must be skipped to avoid out-of-bounds writes.
_SEMA_SHAPE = (16, 64)
ASM_SPLITK_MAX_GRID = _SEMA_SHAPE[0] * _SEMA_SHAPE[1]

_BUFFER_BYTE_LIMIT = (1 << 32) - 1
_CHUNK_ROW_ALIGNMENT = 256


@functools.lru_cache(maxsize=64)
def _get_semaphore_workspace_keyed(device: torch.device, stream_id: int) -> Tensor:
    return torch.zeros(_SEMA_SHAPE, dtype=torch.uint32, device=device)


def get_semaphore_workspace(device: torch.device) -> Tensor:
    """Return a per-(device, stream) zero-initialized semaphore workspace.

    SplitK a16w16 ASM kernels use an atomic-counter protocol where the last
    workgroup performs the reduction phase. Concurrent launches on different
    streams must not share the same atomic counter, or the counts get mixed
    and the reduction phase never fires (deadlock).

    Reuse across launches on the same stream relies on the kernel resetting
    the counter to zero after the reduction completes; do not call this from
    callers that violate that invariant.

    Workspace size is small (~4 KB) and stream count per process is typically
    < 8, so the LRU cap of 64 leaves plenty of headroom before any in-flight
    workspace risks being evicted.
    """
    stream = torch.cuda.current_stream(device)
    return _get_semaphore_workspace_keyed(device, stream.cuda_stream)


def _buffer_extent_bytes(tensor: Tensor) -> int:
    rows, cols = tensor.shape
    if rows <= 0 or cols <= 0:
        return 0
    row_stride = tensor.stride(0) if rows > 1 else 0
    col_stride = tensor.stride(1) if cols > 1 else 0
    return (
        (rows - 1) * row_stride + (cols - 1) * col_stride + 1
    ) * tensor.element_size()


def _rows_within_buffer_limit(tensor: Tensor, limit: int) -> int:
    rows, cols = tensor.shape
    if rows <= 0 or cols <= 0:
        return 0
    row_stride = tensor.stride(0) if rows > 1 else 0
    col_stride = tensor.stride(1) if cols > 1 else 0
    element_limit = limit // tensor.element_size()
    fixed_elements = (cols - 1) * col_stride + 1
    if fixed_elements > element_limit:
        return 0
    if row_stride == 0:
        return rows
    return (element_limit - fixed_elements) // row_stride + 1


def _validate_asm_gemm_layout(
    A: Tensor, B: Tensor, out: Tensor, bias: Tensor | None = None
) -> None:
    if A.ndim != 2 or B.ndim != 2 or out.ndim != 2:
        raise ValueError("gemm_a16w16_asm expects 2-D A, B, and out tensors")
    if A.shape[0] != out.shape[0] or A.shape[1] != B.shape[1]:
        raise ValueError(
            f"incompatible gemm shapes: A={tuple(A.shape)}, B={tuple(B.shape)}, "
            f"out={tuple(out.shape)}"
        )
    if B.shape[0] != out.shape[1]:
        raise ValueError(
            f"incompatible gemm shapes: B={tuple(B.shape)}, out={tuple(out.shape)}"
        )
    if bias is not None and (bias.ndim != 1 or bias.shape[0] != B.shape[0]):
        raise ValueError(
            f"gemm_a16w16_asm bias must have shape ({B.shape[0]},), "
            f"got {tuple(bias.shape)}"
        )
    if bias is not None and not bias.is_contiguous():
        raise ValueError("gemm_a16w16_asm bias must be contiguous")
    for name, tensor in (("A", A), ("B", B)):
        if tensor.shape[1] > 1 and tensor.stride(1) != 1:
            raise ValueError(
                f"gemm_a16w16_asm {name} must have unit stride along its last dimension"
            )
    if not out.is_contiguous():
        raise ValueError("gemm_a16w16_asm out must be contiguous")


def gemm_a16w16_asm(
    A: Tensor,
    B: Tensor,
    out: Tensor,
    bias: Tensor | None = None,
    splitK: int | None = None,
    kernelName: str | None = None,
    bpreshuffle: bool = False,
):
    _validate_asm_gemm_layout(A, B, out, bias)
    if A.shape[0] == 0:
        return out

    if _buffer_extent_bytes(B) > _BUFFER_BYTE_LIMIT:
        raise ValueError(
            "gemm_a16w16_asm B exceeds the kernel's 32-bit buffer-addressing limit; "
            "M chunking cannot repair a weight overflow"
        )

    needs_chunking = (
        _buffer_extent_bytes(A) > _BUFFER_BYTE_LIMIT
        or _buffer_extent_bytes(out) > _BUFFER_BYTE_LIMIT
    )
    if needs_chunking:
        chunk_rows = min(
            _rows_within_buffer_limit(A, _BUFFER_BYTE_LIMIT),
            _rows_within_buffer_limit(out, _BUFFER_BYTE_LIMIT),
        )
        chunk_rows -= chunk_rows % _CHUNK_ROW_ALIGNMENT
        if chunk_rows == 0:
            raise ValueError(
                "gemm_a16w16_asm cannot split A/out into chunks within the "
                "kernel's 32-bit buffer-addressing limit"
            )
    else:
        chunk_rows = A.shape[0]

    if splitK is None or splitK > 1:
        sema = get_semaphore_workspace(out.device)
    else:
        sema = torch.empty((0,), dtype=torch.uint32, device=out.device)

    for row_start in range(0, A.shape[0], chunk_rows):
        row_stop = min(row_start + chunk_rows, A.shape[0])
        _gemm_a16w16_asm(
            A[row_start:row_stop],
            B,
            out[row_start:row_stop],
            sema,
            bias,
            splitK,
            kernelName,
            bpreshuffle,
        )
    return out
