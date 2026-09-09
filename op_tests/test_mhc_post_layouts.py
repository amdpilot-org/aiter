# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

import aiter


def _finite_nonzero(shape, dtype, device):
    values = torch.randn(shape, dtype=torch.float32, device=device).clamp(-2.0, 2.0)
    values = torch.where(values.abs() < 0.25, torch.full_like(values, 0.75), values)
    return values.to(dtype)


def _reference(x, residual, post_mix, comb_mix):
    combined = torch.bmm(comb_mix.mT, residual.float())
    return x.float().unsqueeze(-2) * post_mix.unsqueeze(-1) + combined


def _assert_untouched_sentinels(storage, view, sentinel):
    untouched = storage == sentinel
    assert untouched.sum().item() == storage.numel() - view.numel()


@pytest.mark.parametrize(
    ("post_noncontiguous", "comb_noncontiguous", "out_noncontiguous"),
    [
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (True, True, True),
    ],
)
def test_mhc_post_layouts(
    post_noncontiguous, comb_noncontiguous, out_noncontiguous
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    if not hasattr(aiter, "mhc_post"):
        pytest.skip("aiter.mhc_post is unavailable")

    torch.manual_seed(1234)
    device = "cuda"
    m, hc_mult, hidden_size = 37, 4, 768
    dtype = torch.bfloat16
    sentinel = -12345.0

    x = _finite_nonzero((m, hidden_size), dtype, device)
    residual = _finite_nonzero((m, hc_mult, hidden_size), dtype, device)

    if post_noncontiguous:
        post_storage = torch.full(
            (m, hc_mult * 2), sentinel, dtype=torch.float32, device=device
        )
        post_mix = post_storage[:, ::2]
        post_mix.copy_(_finite_nonzero((m, hc_mult), torch.float32, device))
    else:
        post_mix = _finite_nonzero((m, hc_mult), torch.float32, device)

    if comb_noncontiguous:
        comb_storage = torch.full(
            (m, hc_mult, hc_mult * 2), sentinel, dtype=torch.float32, device=device
        )
        comb_mix = comb_storage[:, :, ::2]
        comb_mix.copy_(
            _finite_nonzero((m, hc_mult, hc_mult), torch.float32, device)
        )
    else:
        comb_mix = _finite_nonzero((m, hc_mult, hc_mult), torch.float32, device)

    if out_noncontiguous:
        out_storage = torch.full(
            (m, hc_mult * 2, hidden_size * 2),
            sentinel,
            dtype=dtype,
            device=device,
        )
        out = out_storage[:, ::2, ::2]
    else:
        out_storage = None
        out = torch.empty((m, hc_mult, hidden_size), dtype=dtype, device=device)

    aiter.mhc_post(out, x, residual, post_mix, comb_mix)
    torch.cuda.synchronize()

    expected = _reference(x, residual, post_mix, comb_mix)
    torch.testing.assert_close(out.float(), expected, atol=1e-2, rtol=1e-2)

    if out_storage is not None:
        _assert_untouched_sentinels(out_storage, out, sentinel)
    if post_noncontiguous:
        _assert_untouched_sentinels(post_storage, post_mix, sentinel)
    if comb_noncontiguous:
        _assert_untouched_sentinels(comb_storage, comb_mix, sentinel)
