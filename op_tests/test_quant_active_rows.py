# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter import QuantType, dtypes, get_hip_quant
from aiter.ops.quant import per_tensor_quant_hip


def _poison_padding(input, valid_rows):
    if valid_rows < input.shape[0]:
        padding = input[valid_rows:]
        padding[..., : padding.shape[-1] // 2] = float("nan")
        padding[..., padding.shape[-1] // 2 :] = float("inf")


def _reference(input, valid_rows, quant_dtype, scale=None, num_rows_factor=1):
    active_rows = valid_rows * num_rows_factor
    active = input.view(-1, input.shape[-1])[:active_rows].float()
    dtype_max = (
        torch.finfo(quant_dtype).max
        if quant_dtype == dtypes.fp8
        else torch.iinfo(quant_dtype).max
    )
    if scale is None:
        max_abs = active.abs().max() if active_rows else torch.tensor(0.0)
        scale = max_abs / dtype_max
    expected = torch.clamp(active / scale, -dtype_max, dtype_max).to(quant_dtype)
    return expected, scale.reshape(1)


def _check_case(valid_rows, quant_dtype, static):
    torch.manual_seed(5256)
    input = torch.randn(8, 128, dtype=dtypes.bf16, device="cuda")
    _poison_padding(input, valid_rows)
    num_rows = torch.tensor([valid_rows], dtype=torch.int32, device="cuda")
    scale = None
    if static:
        scale = torch.ones(1, dtype=dtypes.fp32, device="cuda")

    output, output_scale = per_tensor_quant_hip(
        input,
        scale=scale,
        quant_dtype=quant_dtype,
        num_rows=num_rows,
    )
    expected, expected_scale = _reference(input, valid_rows, quant_dtype, scale=scale)
    torch.cuda.synchronize()

    assert torch.allclose(
        output.view(-1, output.shape[-1])[:valid_rows].float(),
        expected.float(),
        atol=1 if quant_dtype == dtypes.i8 else 1e-6,
    )
    assert torch.allclose(output_scale, expected_scale.to(output_scale.device))


def test_per_tensor_active_rows():
    for valid_rows in (0, 3, 8):
        for quant_dtype in (dtypes.fp8, dtypes.i8):
            for static in (False, True):
                _check_case(valid_rows, quant_dtype, static)


def test_per_tensor_num_rows_factor():
    torch.manual_seed(5256)
    input = torch.randn(4, 2, 128, dtype=dtypes.bf16, device="cuda")
    _poison_padding(input, 2)
    num_rows = torch.tensor([2], dtype=torch.int32, device="cuda")
    static_scale = torch.ones(1, dtype=dtypes.fp32, device="cuda")

    for scale in (None, static_scale):
        output, output_scale = per_tensor_quant_hip(
            input,
            scale=scale,
            quant_dtype=dtypes.fp8,
            num_rows=num_rows,
            num_rows_factor=2,
        )
        expected, expected_scale = _reference(
            input,
            2,
            dtypes.fp8,
            scale=scale,
            num_rows_factor=2,
        )
        torch.cuda.synchronize()

        assert torch.allclose(
            output.view(-1, output.shape[-1])[:4].float(), expected.float()
        )
        assert torch.allclose(output_scale, expected_scale.to(output_scale.device))


def test_per_tensor_quant_lookup_active_rows():
    torch.manual_seed(5256)
    input = torch.randn(8, 128, dtype=dtypes.bf16, device="cuda")
    _poison_padding(input, 3)
    num_rows = torch.tensor([3], dtype=torch.int32, device="cuda")

    quant_func = get_hip_quant(QuantType.per_Tensor)
    output, scale = quant_func(
        input,
        quant_dtype=dtypes.fp8,
        num_rows=num_rows,
    )
    expected, expected_scale = _reference(input, 3, dtypes.fp8)
    torch.cuda.synchronize()

    assert torch.allclose(
        output.view(-1, output.shape[-1])[:3].float(),
        expected.float(),
    )
    assert torch.allclose(scale, expected_scale.to(scale.device))


def test_per_tensor_active_rows_validation():
    input = torch.randn(4, 128, dtype=dtypes.bf16, device="cuda")

    with pytest.raises(AssertionError, match="num_rows must be int32"):
        per_tensor_quant_hip(
            input,
            quant_dtype=dtypes.fp8,
            num_rows=torch.tensor([2], dtype=torch.int64, device="cuda"),
        )

    with pytest.raises(AssertionError, match="num_rows must be on the input device"):
        per_tensor_quant_hip(
            input,
            quant_dtype=dtypes.fp8,
            num_rows=torch.tensor([2], dtype=torch.int32, device="cpu"),
        )

    with pytest.raises(AssertionError, match="num_rows must contain exactly one value"):
        per_tensor_quant_hip(
            input,
            quant_dtype=dtypes.fp8,
            num_rows=torch.tensor([2, 3], dtype=torch.int32, device="cuda"),
        )

    with pytest.raises(AssertionError, match="num_rows_factor must be positive"):
        per_tensor_quant_hip(
            input,
            quant_dtype=dtypes.fp8,
            num_rows=torch.tensor([2], dtype=torch.int32, device="cuda"),
            num_rows_factor=0,
        )


def test_per_tensor_graph_replay():
    torch.manual_seed(5256)
    input = torch.randn(8, 128, dtype=dtypes.bf16, device="cuda")
    finite_input = input.clone()
    _poison_padding(input, 2)
    num_rows = torch.tensor([2], dtype=torch.int32, device="cuda")
    static_scale = torch.ones(1, dtype=dtypes.fp32, device="cuda")

    per_tensor_quant_hip(input, quant_dtype=dtypes.fp8, num_rows=num_rows)
    per_tensor_quant_hip(
        input, scale=static_scale, quant_dtype=dtypes.i8, num_rows=num_rows
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        dynamic_output, dynamic_scale = per_tensor_quant_hip(
            input, quant_dtype=dtypes.fp8, num_rows=num_rows
        )
        static_output, _ = per_tensor_quant_hip(
            input,
            scale=static_scale,
            quant_dtype=dtypes.i8,
            num_rows=num_rows,
        )

    for valid_rows in (1, 4, 8):
        input.copy_(finite_input)
        _poison_padding(input, valid_rows)
        num_rows.fill_(valid_rows)
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        expected_dynamic, expected_scale = _reference(input, valid_rows, dtypes.fp8)
        expected_static, _ = _reference(
            input, valid_rows, dtypes.i8, scale=static_scale
        )
        assert torch.allclose(
            dynamic_output.view(-1, 128)[:valid_rows].float(),
            expected_dynamic.float(),
        )
        assert torch.allclose(dynamic_scale, expected_scale.to(dynamic_scale.device))
        assert torch.allclose(
            static_output.view(-1, 128)[:valid_rows].float(),
            expected_static.float(),
            atol=1,
        )


def test_per_tensor_graph_replay_num_rows_factor():
    torch.manual_seed(5256)
    input = torch.randn(8, 2, 128, dtype=dtypes.bf16, device="cuda")
    finite_input = input.clone()
    _poison_padding(input, 2)
    num_rows = torch.tensor([2], dtype=torch.int32, device="cuda")
    static_scale = torch.ones(1, dtype=dtypes.fp32, device="cuda")

    per_tensor_quant_hip(
        input,
        quant_dtype=dtypes.fp8,
        num_rows=num_rows,
        num_rows_factor=2,
    )
    per_tensor_quant_hip(
        input,
        scale=static_scale,
        quant_dtype=dtypes.i8,
        num_rows=num_rows,
        num_rows_factor=2,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        dynamic_output, dynamic_scale = per_tensor_quant_hip(
            input,
            quant_dtype=dtypes.fp8,
            num_rows=num_rows,
            num_rows_factor=2,
        )
        static_output, _ = per_tensor_quant_hip(
            input,
            scale=static_scale,
            quant_dtype=dtypes.i8,
            num_rows=num_rows,
            num_rows_factor=2,
        )

    for valid_rows in (0, 1, 2, 4):
        input.copy_(finite_input)
        _poison_padding(input, valid_rows)
        num_rows.fill_(valid_rows)
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        expected_dynamic, expected_scale = _reference(
            input,
            valid_rows,
            dtypes.fp8,
            num_rows_factor=2,
        )
        expected_static, _ = _reference(
            input,
            valid_rows,
            dtypes.i8,
            scale=static_scale,
            num_rows_factor=2,
        )
        assert torch.allclose(
            dynamic_output.view(-1, 128)[: valid_rows * 2].float(),
            expected_dynamic.float(),
        )
        assert torch.allclose(dynamic_scale, expected_scale.to(dynamic_scale.device))
        assert torch.allclose(
            static_output.view(-1, 128)[: valid_rows * 2].float(),
            expected_static.float(),
            atol=1,
        )


if __name__ == "__main__":
    test_per_tensor_active_rows()
    test_per_tensor_num_rows_factor()
    test_per_tensor_quant_lookup_active_rows()
    test_per_tensor_active_rows_validation()
    test_per_tensor_graph_replay()
    test_per_tensor_graph_replay_num_rows_factor()
