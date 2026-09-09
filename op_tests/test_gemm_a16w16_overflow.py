# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops import gemm_op_a16w16


def _assert_close(out, ref):
    torch.testing.assert_close(out, ref, rtol=3e-2, atol=0.25)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/HIP")
def test_chunks_input_overflow(monkeypatch):
    monkeypatch.setattr(gemm_op_a16w16, "_BUFFER_BYTE_LIMIT", 1 << 20)
    m, k, n = 8192, 128, 128
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)

    gemm_op_a16w16.gemm_a16w16_asm(a, b, out)
    torch.cuda.synchronize()
    _assert_close(out, torch.nn.functional.linear(a, b))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/HIP")
def test_chunks_output_overflow(monkeypatch):
    monkeypatch.setattr(gemm_op_a16w16, "_BUFFER_BYTE_LIMIT", 1 << 20)
    m, k, n = 4096, 128, 1024
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16) * 0.01
    out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)

    gemm_op_a16w16.gemm_a16w16_asm(a, b, out, bias=bias)
    torch.cuda.synchronize()
    _assert_close(out, torch.nn.functional.linear(a, b, bias))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/HIP")
def test_rejects_weight_overflow(monkeypatch):
    monkeypatch.setattr(gemm_op_a16w16, "_BUFFER_BYTE_LIMIT", 1 << 10)
    a = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(256, 128, device="cuda", dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="B exceeds"):
        gemm_op_a16w16.gemm_a16w16_asm(a, b, out)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/HIP")
def test_rejects_noncontiguous_output():
    a = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(256, 256, device="cuda", dtype=torch.bfloat16)[:, :128]

    with pytest.raises(ValueError, match="out must be contiguous"):
        gemm_op_a16w16.gemm_a16w16_asm(a, b, out)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/HIP")
def test_accepts_padded_input_row_stride():
    m, k, n = 512, 128, 128
    a = torch.randn(m, 2 * k, device="cuda", dtype=torch.bfloat16)[:, :k]
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)

    gemm_op_a16w16.gemm_a16w16_asm(a, b, out)
    torch.cuda.synchronize()
    _assert_close(out, torch.nn.functional.linear(a, b))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/HIP")
def test_accepts_padded_weight_row_stride():
    k, n = 128, 128
    a = torch.randn(256, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, 2 * k, device="cuda", dtype=torch.bfloat16)[:, :k]
    out = torch.empty(256, n, device="cuda", dtype=torch.bfloat16)

    gemm_op_a16w16.gemm_a16w16_asm(a, b, out)
    torch.cuda.synchronize()
    _assert_close(out, torch.nn.functional.linear(a, b))
