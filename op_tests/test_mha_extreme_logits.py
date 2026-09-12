# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.mha import (
    fmha_fwd_bf16_opus_fwd,
    fmha_fwd_bf16_opus_varlen_fwd,
    mha_fwd_native_splitkv,
)


def _dense_inputs(head_dim_qk, head_dim_v, exponent, seed):
    target = -(2.0**exponent)
    torch.manual_seed(seed)
    q = torch.ones((2, 256, 8, head_dim_qk), device="cuda", dtype=dtypes.bf16)
    k = torch.full(
        (2, 256, 2, head_dim_qk),
        target / head_dim_qk,
        device="cuda",
        dtype=dtypes.bf16,
    )
    v = torch.randn(
        (2, 256, 2, head_dim_v), device="cuda", dtype=dtypes.bf16
    )
    return q, k, v


def _dense_reference(q, k, v, causal, softmax_scale):
    batch, seqlen_q, heads_q, _ = q.shape
    _, seqlen_k, heads_k, _ = k.shape
    head_dim_v = v.shape[-1]
    repeat = heads_q // heads_k
    values = v.float().repeat_interleave(repeat, dim=2)
    raw_score = (q[0, 0, 0].float() * k[0, 0, 0].float()).sum()
    scores = torch.full(
        (seqlen_k,), raw_score * softmax_scale, device=q.device, dtype=torch.float32
    )
    probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32)
    assert torch.allclose(
        probabilities, torch.full_like(probabilities, 1 / seqlen_k)
    )
    if causal:
        cumulative = values.cumsum(dim=1)
        counts = torch.arange(
            1, seqlen_k + 1, device=q.device, dtype=torch.float32
        )
        return (cumulative / counts.view(1, -1, 1, 1)).to(q.dtype)
    mean = values.mean(dim=1, keepdim=True)
    return mean.expand(batch, seqlen_q, heads_q, head_dim_v).to(q.dtype)


@pytest.mark.parametrize("exponent", [101])
def test_flash_attn_func_native_extreme_negative_logits(exponent):
    if get_gfx() != "gfx942":
        pytest.skip("native D64 regression runs on gfx942")

    target = -(2.0**exponent)
    q = torch.ones((1, 64, 1, 64), device="cuda", dtype=dtypes.bf16)
    k = torch.full(
        (1, 64, 1, 64), target / 64, device="cuda", dtype=dtypes.bf16
    )
    v = torch.ones_like(q)
    with torch.no_grad():
        out, lse = aiter.flash_attn_func(
            q,
            k,
            v,
            softmax_scale=64**-0.5,
            causal=False,
            return_lse=True,
            num_splits=2,
        )
        native_out, native_lse = mha_fwd_native_splitkv(
            q,
            k,
            v,
            out=None,
            softmax_scale=64**-0.5,
            causal=False,
            return_lse=True,
            num_splits=2,
        )
    assert torch.equal(out, native_out)
    assert torch.equal(lse, native_lse)
    torch.testing.assert_close(out, torch.ones_like(out), rtol=0, atol=0)
    assert torch.isfinite(lse).all()


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("exponent", [36, 101, 127])
@pytest.mark.parametrize("head_dim_qk", [128, 192])
def test_flash_attn_func_opus_extreme_negative_logits(
    head_dim_qk, exponent, causal, seed, monkeypatch
):
    if get_gfx() != "gfx950":
        pytest.skip("OPUS regression requires gfx950")
    if head_dim_qk == 128:
        monkeypatch.setenv("AITER_ENABLE_FMHA_OPUS", "1")
    else:
        monkeypatch.delenv("AITER_DISABLE_FMHA_OPUS", raising=False)

    q, k, v = _dense_inputs(head_dim_qk, 128, exponent, seed)
    softmax_scale = head_dim_qk**-0.5
    with torch.no_grad():
        out, lse = aiter.flash_attn_func(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            causal=causal,
            return_lse=True,
        )
        direct_out, direct_lse = fmha_fwd_bf16_opus_fwd(
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            causal=causal,
            return_lse=True,
        )
    assert torch.equal(out, direct_out)
    assert torch.equal(lse, direct_lse)
    expected = _dense_reference(q, k, v, causal, softmax_scale)
    torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)
    assert torch.isfinite(lse).all()


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("exponent", [36, 127])
def test_flash_attn_varlen_func_opus_extreme_negative_logits(exponent, causal):
    if get_gfx() != "gfx950":
        pytest.skip("OPUS D192/V128 varlen regression requires gfx950")

    target = -(2.0**exponent)
    lengths = (64, 192)
    total_q = sum(lengths)
    cu_q = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()],
        device="cuda",
        dtype=torch.int32,
    )
    q = torch.ones((total_q, 4, 192), device="cuda", dtype=dtypes.bf16)
    k = torch.full(
        (total_q, 1, 192), target / 192, device="cuda", dtype=dtypes.bf16
    )
    v = torch.ones((total_q, 1, 128), device="cuda", dtype=dtypes.bf16)
    with torch.no_grad():
        out, lse = aiter.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_q,
            cu_q,
            max(lengths),
            max(lengths),
            softmax_scale=192**-0.5,
            causal=causal,
            return_lse=True,
        )
        direct_out, direct_lse = fmha_fwd_bf16_opus_varlen_fwd(
            q,
            k,
            v,
            softmax_scale=192**-0.5,
            causal=causal,
            seqstart_q=cu_q,
            seqstart_k=cu_q,
            max_seqlen_q=max(lengths),
            max_seqlen_k=max(lengths),
            return_lse=True,
        )
    assert torch.equal(out, direct_out)
    assert torch.equal(lse, direct_lse)
    torch.testing.assert_close(out, torch.ones_like(out), rtol=0, atol=0)
    assert torch.isfinite(lse).all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
