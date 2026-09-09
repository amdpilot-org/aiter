import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.triton.attention.pa_decode_sparse import pa_decode_sparse
from aiter.ops.triton.gluon.mla_gluon import mla_gluon


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() != "gfx942",
    reason="gfx942 capability contract requires an MI300X-class GPU",
)


def _mla_reference(q_nope, q_pe, kv, page_table, seq_info, scale, use_2d_view):
    batch, heads, nope = q_nope.shape
    q = torch.cat((q_nope, q_pe), dim=-1).float()
    outputs = []
    lse = []
    for batch_idx in range(batch):
        if use_2d_view:
            length = int(seq_info[batch_idx].item())
            token_ids = page_table[batch_idx, :length]
        else:
            start = int(seq_info[batch_idx].item())
            end = int(seq_info[batch_idx + 1].item())
            token_ids = page_table[start:end]
        keys = kv[token_ids].float()
        scores = torch.einsum("hd,kd->hk", q[batch_idx], keys) * scale
        lse.append(scores.logsumexp(-1))
        outputs.append(
            torch.einsum("hk,kd->hd", scores.softmax(-1), keys[:, :nope])
        )
    return torch.stack(outputs).to(q_nope.dtype), torch.stack(lse)


def _make_mla_inputs(use_2d_view):
    torch.manual_seed(0)
    device = torch.device("cuda")
    batch, heads, ctx, nope, rope = 2, 16, 129, 512, 64
    qk_dim = nope + rope
    total_kv = batch * ctx if use_2d_view else 2 * ctx - 1
    q = torch.randn(batch, heads, qk_dim, device=device, dtype=torch.bfloat16) * 0.2
    kv = torch.randn(total_kv, qk_dim, device=device, dtype=torch.bfloat16) * 0.2
    out = torch.empty(batch, heads, nope, device=device, dtype=torch.bfloat16)
    if use_2d_view:
        page_table = torch.arange(
            batch * ctx, device=device, dtype=torch.int32
        ).view(batch, ctx)
        seq_info = torch.tensor([ctx, ctx - 1], device=device, dtype=torch.int32)
    else:
        lengths = torch.tensor([ctx, ctx - 1], device=device, dtype=torch.int32)
        page_table = torch.arange(
            int(lengths.sum().item()), device=device, dtype=torch.int32
        )
        seq_info = torch.zeros(batch + 1, device=device, dtype=torch.int32)
        seq_info[1:] = lengths.cumsum(0)
    return q, kv, out, page_table, seq_info, qk_dim**-0.5


@pytest.mark.parametrize("use_2d_view", [True, False])
@pytest.mark.parametrize("return_lse", [False, True])
def test_mla_gluon_gfx942_fallback_matches_torch(use_2d_view, return_lse):
    q, kv, out, page_table, seq_info, scale = _make_mla_inputs(use_2d_view)
    nope = 512
    result, lse = mla_gluon(
        q[..., :nope],
        q[..., nope:],
        kv,
        out,
        page_table,
        seq_info,
        scale,
        use_2d_view=use_2d_view,
        return_lse=return_lse,
    )
    reference, reference_lse = _mla_reference(
        q[..., :nope], q[..., nope:], kv, page_table, seq_info, scale, use_2d_view
    )
    torch.testing.assert_close(result, reference, atol=2e-2, rtol=2e-2)
    if return_lse:
        assert lse is not None and lse.shape == (2, 1, 16)
        torch.testing.assert_close(
            lse.squeeze(1), reference_lse, atol=2e-2, rtol=2e-2
        )
    else:
        assert lse is None


@pytest.mark.parametrize("heads,kv_dtype", [(64, torch.bfloat16), (16, None)])
def test_mla_gluon_gfx942_rejects_unsupported_contract(heads, kv_dtype):
    device = torch.device("cuda")
    batch, ctx, nope, rope = 1, 128, 512, 64
    qk_dim = nope + rope
    q = torch.randn(batch, heads, qk_dim, device=device, dtype=torch.bfloat16) * 0.2
    kv = torch.randn(ctx, qk_dim, device=device, dtype=torch.bfloat16) * 0.2
    if kv_dtype is None:
        kv = kv.to(torch.float8_e4m3fn)
    out = torch.empty(batch, heads, nope, device=device, dtype=torch.bfloat16)
    page_table = torch.arange(ctx, device=device, dtype=torch.int32).view(batch, ctx)
    seq_info = torch.full((batch,), ctx, device=device, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="mla_gluon requires gfx950"):
        mla_gluon(
            q[..., :nope],
            q[..., nope:],
            kv,
            out,
            page_table,
            seq_info,
            qk_dim**-0.5,
        )


def test_pa_decode_sparse_gfx942_rejects_packed_cache():
    device = torch.device("cuda")
    q = torch.randn(1, 16, 512, device=device, dtype=torch.bfloat16) * 0.2
    cache = torch.zeros(1, 256, 584, device=device, dtype=torch.uint8)
    indices = torch.arange(256, device=device, dtype=torch.int32)
    indptr = torch.tensor([0, 256], device=device, dtype=torch.int32)
    sink = torch.zeros(16, device=device, dtype=torch.float32)
    with pytest.raises(
        RuntimeError, match="packed fp8_ds_mla/bf16 cache requires gfx950"
    ):
        pa_decode_sparse(q, cache, indices, indptr, sink, 512**-0.5)


def test_pa_decode_sparse_gfx942_rejects_extra_loop():
    device = torch.device("cuda")
    q = torch.randn(1, 16, 512, device=device, dtype=torch.bfloat16) * 0.2
    cache = torch.randn(1, 512, device=device, dtype=torch.bfloat16) * 0.2
    indices = torch.arange(1, device=device, dtype=torch.int32)
    indptr = torch.tensor([0, 1], device=device, dtype=torch.int32)
    sink = torch.zeros(16, device=device, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="extra_\\* two-loop requires gfx950"):
        pa_decode_sparse(
            q,
            cache,
            indices,
            indptr,
            sink,
            512**-0.5,
            extra_cache=cache,
            extra_indices=indices,
            extra_indptr=indptr,
        )
