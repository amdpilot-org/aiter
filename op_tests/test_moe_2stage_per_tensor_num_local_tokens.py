import torch

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import (
    fused_moe,
    fused_topk,
    get_2stage_cfgs,
    get_padded_M,
    torch_moe_stage1,
    torch_moe_stage2,
)
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.moe_common import GateMode
from aiter.ops.shuffle import shuffle_weight
from aiter.utility import fp4_utils


def _quant_reference(x, quant_dtype=dtypes.fp8):
    return aiter.per_tensor_quant(x, quant_dtype=quant_dtype)


def _test_quantizer_num_rows():
    torch.manual_seed(7)
    x = torch.randn(8, 64, dtype=dtypes.bf16, device="cuda")
    x[5:] = 1e6
    num_rows = torch.tensor([5], dtype=dtypes.i32, device="cuda")

    actual, actual_scale = aiter.per_tensor_quant_hip(
        x, quant_dtype=dtypes.fp8, num_rows=num_rows
    )
    expected, expected_scale = _quant_reference(x[:5])
    torch.cuda.synchronize()

    assert actual_scale.item() == expected_scale.item()
    assert torch.equal(actual[:5], expected)

    static_actual, static_scale = aiter.per_tensor_quant_hip(
        x,
        scale=actual_scale,
        quant_dtype=dtypes.fp8,
        num_rows=num_rows,
    )
    torch.cuda.synchronize()
    assert static_scale.item() == expected_scale.item()
    assert torch.equal(static_actual[:5], expected)

    factor_actual, factor_scale = aiter.per_tensor_quant_hip(
        x,
        quant_dtype=dtypes.fp8,
        num_rows=torch.tensor([3], dtype=dtypes.i32, device="cuda"),
        num_rows_factor=2,
    )
    factor_expected, factor_expected_scale = _quant_reference(x[:6])
    torch.cuda.synchronize()
    assert factor_scale.item() == factor_expected_scale.item()
    assert torch.equal(factor_actual[:6], factor_expected)


def _test_fused_moe_num_local_tokens():
    torch.manual_seed(7)
    token_num = 4
    valid_tokens = 2
    model_dim = 4096
    inter_dim = 384
    experts = 128
    topk = 8
    dtype = dtypes.bf16
    device = "cuda"

    metadata = get_2stage_cfgs(
        get_padded_M(token_num),
        model_dim,
        inter_dim,
        experts,
        topk,
        dtype,
        dtypes.fp8,
        dtypes.fp8,
        QuantType.per_Tensor,
        True,
        ActivationType.Silu,
        False,
        0,
        0,
        True,
        GateMode.INTERLEAVE.value,
    )
    assert not metadata.run_1stage

    hidden_states = torch.randn(
        (token_num, model_dim), dtype=dtype, device=device
    )
    hidden_states[valid_tokens:] = 1e6
    router_logits = torch.randn(
        (valid_tokens, experts), dtype=dtype, device=device
    )
    topk_weights, topk_ids = fused_topk(
        hidden_states[:valid_tokens], router_logits, topk, True
    )
    topk_weights = torch.cat(
        (
            topk_weights,
            torch.zeros(
                (token_num - valid_tokens, topk),
                dtype=topk_weights.dtype,
                device=device,
            ),
        )
    )
    topk_ids = torch.cat(
        (
            topk_ids,
            torch.zeros(
                (token_num - valid_tokens, topk),
                dtype=topk_ids.dtype,
                device=device,
            ),
        )
    )

    w1 = torch.randn(
        (experts, inter_dim * 2, model_dim), dtype=dtype, device=device
    ) * 0.1
    w2 = torch.randn(
        (experts, model_dim, inter_dim), dtype=dtype, device=device
    ) * 0.1
    w1_q, w1_scale = aiter.pertoken_quant(
        w1.view(experts, -1), quant_dtype=dtypes.fp8
    )
    w2_q, w2_scale = aiter.pertoken_quant(
        w2.view(experts, -1), quant_dtype=dtypes.fp8
    )
    w1_q = w1_q.view(w1.shape)
    w2_q = w2_q.view(w2.shape)

    fused_w1 = shuffle_weight(w1_q, layout=(16, 16))
    fused_w2 = shuffle_weight(w2_q, layout=(16, 16))
    fused_w1_scale = fp4_utils.e8m0_shuffle(w1_scale)
    fused_w2_scale = fp4_utils.e8m0_shuffle(w2_scale)
    num_local_tokens = torch.tensor(
        [valid_tokens], dtype=dtypes.i32, device=device
    )

    actual = fused_moe(
        hidden_states,
        fused_w1,
        fused_w2,
        topk_weights,
        topk_ids,
        quant_type=QuantType.per_Tensor,
        w1_scale=fused_w1_scale,
        w2_scale=fused_w2_scale,
        num_local_tokens=num_local_tokens,
        gate_mode=GateMode.INTERLEAVE.value,
    )

    a1_q, a1_scale = _quant_reference(hidden_states[:valid_tokens])
    stage1_ref = torch_moe_stage1(
        a1_q,
        w1_q,
        w2_q,
        topk_weights[:valid_tokens],
        topk_ids[:valid_tokens],
        dtype=dtype,
        activation=ActivationType.Silu,
        quant_type=QuantType.per_Tensor,
        a1_scale=a1_scale,
        w1_scale=w1_scale,
    )
    a2_q, a2_scale = _quant_reference(stage1_ref)
    expected = torch_moe_stage2(
        a2_q,
        w1_q,
        w2_q,
        topk_weights[:valid_tokens],
        topk_ids[:valid_tokens],
        dtype=dtype,
        quant_type=QuantType.per_Tensor,
        w2_scale=w2_scale,
        a2_scale=a2_scale,
        doweight=True,
    )
    torch.cuda.synchronize()

    actual_valid = actual[:valid_tokens].double()
    expected_valid = expected.double()
    abs_error = (actual_valid - expected_valid).abs()
    relative_error = abs_error / (expected_valid.abs() + 1e-6)
    dot_product = (actual_valid * expected_valid).sum()
    cosine_error = 1.0 - (
        2.0
        * dot_product
        / (
            (actual_valid * actual_valid).sum()
            + (expected_valid * expected_valid).sum()
        )
    )

    print(
        f"per_tensor num_local_tokens: max_abs={abs_error.max().item():.6g} "
        f"mean_rel={relative_error.mean().item():.6g} "
        f"cosine_error={cosine_error.item():.6g}"
    )
    assert relative_error.mean().item() < 0.05
    assert cosine_error.item() < 1e-3


def test_per_tensor_num_local_tokens():
    if get_gfx() != "gfx942":
        print(f"skip per_tensor num_local_tokens test on {get_gfx()}")
        return
    _test_quantizer_num_rows()
    _test_fused_moe_num_local_tokens()


if __name__ == "__main__":
    test_per_tensor_num_local_tokens()
