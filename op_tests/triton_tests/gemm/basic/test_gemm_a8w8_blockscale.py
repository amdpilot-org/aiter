# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import copy
import importlib
import sys

import pytest
import torch
import torch.nn.functional as F

from aiter.ops.shuffle import shuffle_weight
from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import (
    gemm_a8w8_blockscale,
    gemm_a8w8_blockscale_preshuffle,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.gemm_config_utils import compute_splitk_params
from aiter.ops.triton.utils.types import get_fp8_dtypes, str_to_torch_dtype

block_shape = (128, 128)
DEVICE_ARCH = arch_info.get_arch()


def run_torch(x, weight, x_scale, w_scale, dtype=torch.bfloat16):
    block_shape_n, block_shape_k = block_shape
    m, k = x.shape
    n = weight.shape[0]
    x_scale = x_scale.repeat_interleave(block_shape_k, dim=1)
    x = x.to(x_scale.dtype) * x_scale[:m, :k]
    x = x.view(m, k)
    w_scale = w_scale.repeat_interleave(block_shape_n, dim=0)
    w_scale = w_scale.repeat_interleave(block_shape_k, dim=1)
    w_scale = w_scale[:n, :k]
    weight = weight.to(w_scale.dtype) * w_scale

    out = F.linear(x.to(torch.float32), weight.to(torch.float32))

    return out.to(dtype)


def run_triton(x, weight, x_scale, w_scale, dtype=torch.bfloat16, y=None, impl=None):
    return impl(x, weight, x_scale, w_scale, dtype, y)


e5m2_type, e4m3_type = get_fp8_dtypes()


def get_x_vals():
    x_vals = [(1024 * v, 1024 * v, 1024 * v) for v in (1, 2, 4, 5, 8)]
    # GPT-OSS-120B attention projections
    x_vals += [(v, 106496, 16384) for v in (256, 4096)]  # LL3 405B FC1
    x_vals += [(v, 9216, 7168) for v in (128, 192, 4096, 8000)]
    x_vals += [(v, 7168, 4608) for v in (128, 192, 4096, 8000)]
    x_vals += [(v, 8192, 512) for v in (128, 192, 4096, 8000)]
    # Small-K shapes that exercise the gluon wind-down's num_k_iter guards
    # (BLOCK_SIZE_K=128; K in {128,192,256,320} -> num_k_iter in {1,2,2,3}).
    # K<BLOCK_SIZE_K isn't supported by the gluon wrapper (GROUP_K assert).
    x_vals += [(512, 512, K) for K in (128, 192, 256, 320)]
    x_vals += [(v, 8192, 1024) for v in (1, 32, 64, 128, 256, 1024)]
    x_vals += [(v, 4096, 8192) for v in (1, 32, 64, 128, 256, 1024)]
    x_vals += [(v, 4096, 4096) for v in (1, 32, 64, 128, 256, 1024)]
    x_vals += [(v, 4096, 2048) for v in (1, 32, 64, 128, 256, 1024)]
    x_vals += [(v, 1536, 4096) for v in (1, 32, 64, 128, 256, 1024)]
    x_vals += [(v, 32768, 1024) for v in (1, 32, 64, 128, 256, 1024)]
    x_vals += [(v, 8192, 1536) for v in (1, 32, 64, 128, 256, 1024)]
    x_vals += [(v, 7168, 4096) for v in (1, 32, 64, 128, 256, 1024)]
    x_vals += [(v, 1536, 7168) for v in (1, 32, 64, 128, 256, 1024)]
    x_vals += [(v, 7168, 768) for v in (1, 32, 64, 128, 256, 1024)]
    x_vals += [(v, 2048, 7168) for v in (1, 32, 64, 128, 256, 1024)]
    x_vals += [(v, 16384, 1536) for v in (1, 32, 64, 128, 256, 1024)]
    x_vals += [(v, 65536, 1536) for v in (1, 32, 64, 128, 256, 1024)]
    x_vals += [(v, 7168, 16384) for v in (1, 32, 64, 128, 256, 1024)]
    x_vals += [(v, 6144, 7168) for v in (1, 32, 64, 128, 256, 1024)]
    x_vals += [(v, 7168, 3072) for v in (1, 32, 64, 128, 256, 1024)]
    return x_vals


def generate_gemm_a8w8_blockscale_inputs(
    M: int,
    N: int,
    K: int,
    block_shape_n: int,
    block_shape_k: int,
    dtype=torch.bfloat16,
    layout: str = "TN",
    output: bool = False,
    shuffle: bool = False,
):
    """
    The GEMM kernel expects:
    - x: (M, K) -> row-major format
    - w: (N, K) -> column-major format
    """
    torch.manual_seed(0)
    scale_n = (N + block_shape_n - 1) // block_shape_n
    scale_k = (K + block_shape_k - 1) // block_shape_k

    if layout[0] == "T":
        x = (torch.rand((M, K), dtype=torch.float16, device="cuda") / 10).to(e4m3_type)
    else:
        x = (
            (torch.rand((K, M), dtype=torch.float16, device="cuda") / 10)
            .to(e4m3_type)
            .T
        )

    if layout[1] == "N":
        weight = (torch.rand((N, K), dtype=torch.float16, device="cuda") / 10).to(
            e4m3_type
        )
    else:
        weight = (
            (torch.rand((K, N), dtype=torch.float16, device="cuda") / 10)
            .to(e4m3_type)
            .T
        )

    x_scale = torch.rand([M, scale_k], dtype=torch.float32, device="cuda")
    w_scale = torch.rand([scale_n, scale_k], dtype=torch.float32, device="cuda")

    if shuffle:
        weight_shuffle_layout = (16, 16)
        weight_shuffled = shuffle_weight(weight, weight_shuffle_layout).reshape(
            weight.shape[0] // weight_shuffle_layout[0],
            weight.shape[1] * weight_shuffle_layout[0],
        )
        x_scale_shuffled = x_scale.transpose(0, 1).contiguous().view(*x_scale.shape)
    else:
        weight_shuffled = weight
        x_scale_shuffled = x_scale

    y = None
    if output:
        y = torch.empty((M, N), dtype=dtype, device="cuda").cuda()

    return x, weight, weight_shuffled, x_scale, x_scale_shuffled, w_scale, y


@pytest.mark.parametrize(
    "requested, actual",
    [(1, 1), (2, 2), (3, 2), (4, 4), (5, 5), (7, 5), (8, 5)],
)
def test_splitk_partitions_align_to_k_tiles(requested, actual):
    config = {"BLOCK_SIZE_K": 128, "NUM_KSPLIT": requested}
    compute_splitk_params(config, 12800)

    assert config["NUM_KSPLIT"] == actual
    assert config["SPLITK_BLOCK_SIZE"] == 12800 // actual


@pytest.mark.parametrize("K", [5120, 12800, 25600, 12864])
@pytest.mark.parametrize("preshuffle", [False, True])
def test_splitk_partition_boundary(K, preshuffle):
    if preshuffle and K % block_shape[1] != 0:
        pytest.skip("The preshuffle kernel requires K to be a multiple of 128.")

    M = 1
    x, weight, weight_triton, x_scale, x_scale_shuffled, w_scale, _ = (
        generate_gemm_a8w8_blockscale_inputs(
            M,
            5120,
            K,
            *block_shape,
            output=False,
            shuffle=preshuffle,
        )
    )
    reference = run_torch(x, weight, x_scale, w_scale)
    config = {
        "BLOCK_SIZE_M": 32 if preshuffle else 128,
        "BLOCK_SIZE_N": 64 if preshuffle else 128,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "NUM_KSPLIT": 8,
        "num_warps": 4,
        "num_stages": 2,
        "waves_per_eu": 2,
        "matrix_instr_nonkdim": 16,
        "cache_modifier": None if preshuffle else ".cg",
    }
    original = copy.deepcopy(config)
    if preshuffle:
        output = gemm_a8w8_blockscale_preshuffle(
            x,
            weight_triton,
            x_scale_shuffled,
            w_scale,
            config=config,
        )
    else:
        output = gemm_a8w8_blockscale(
            x,
            weight,
            x_scale,
            w_scale,
            config=config,
        )

    adjusted = compute_splitk_params(copy.deepcopy(original), K)
    assert config == original
    assert adjusted["SPLITK_BLOCK_SIZE"] % adjusted["BLOCK_SIZE_K"] == 0
    padded_k = (K + adjusted["BLOCK_SIZE_K"] - 1) // adjusted["BLOCK_SIZE_K"]
    padded_k *= adjusted["BLOCK_SIZE_K"]
    assert K <= adjusted["NUM_KSPLIT"] * adjusted["SPLITK_BLOCK_SIZE"] <= padded_k
    torch.testing.assert_close(output, reference, atol=0.01, rtol=1e-2)


def test_splitk_skip_reduce_shape():
    M, K = 1, 12800
    x, weight, _, x_scale, _, w_scale, _ = (
        generate_gemm_a8w8_blockscale_inputs(
            M,
            5120,
            K,
            *block_shape,
            output=False,
            shuffle=False,
        )
    )
    reference = run_torch(x, weight, x_scale, w_scale)
    config = {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "NUM_KSPLIT": 8,
        "num_warps": 4,
        "num_stages": 2,
        "waves_per_eu": 2,
        "matrix_instr_nonkdim": 16,
        "cache_modifier": ".cg",
    }
    output = gemm_a8w8_blockscale(
        x,
        weight,
        x_scale,
        w_scale,
        config=config,
        skip_reduce=True,
    )

    assert output.shape == (5, M, 5120)
    torch.testing.assert_close(
        output.sum(dim=0).to(reference.dtype),
        reference,
        atol=0.01,
        rtol=1e-2,
    )


def test_splitk_graph_replay():
    M, K = 1, 12800
    x, weight, _, x_scale, _, w_scale, _ = (
        generate_gemm_a8w8_blockscale_inputs(
            M,
            5120,
            K,
            *block_shape,
            output=False,
            shuffle=False,
        )
    )
    y = torch.empty((M, 5120), dtype=torch.bfloat16, device="cuda")
    config = {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "NUM_KSPLIT": 8,
        "num_warps": 4,
        "num_stages": 2,
        "waves_per_eu": 2,
        "matrix_instr_nonkdim": 16,
        "cache_modifier": ".cg",
    }
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            gemm_a8w8_blockscale(
                x,
                weight,
                x_scale,
                w_scale,
                dtype=torch.bfloat16,
                y=y,
                config=dict(config),
            )
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = gemm_a8w8_blockscale(
                x,
                weight,
                x_scale,
                w_scale,
                dtype=torch.bfloat16,
                y=y,
                config=dict(config),
            )

    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    x.copy_((torch.rand_like(x, dtype=torch.float16) / 10).to(x.dtype))
    weight.copy_((torch.rand_like(weight, dtype=torch.float16) / 10).to(weight.dtype))
    x_scale.copy_(torch.rand_like(x_scale))
    w_scale.copy_(torch.rand_like(w_scale))
    reference = run_torch(x, weight, x_scale, w_scale)
    graph.replay()
    torch.cuda.synchronize()

    assert config["NUM_KSPLIT"] == 8
    assert output.shape == (M, 5120)
    torch.testing.assert_close(output, reference, atol=0.01, rtol=1e-2)


@pytest.mark.parametrize(
    "dtype, M, N, K, layout, output",
    [
        (dtype, *shape, layout, output)
        for output in [True]
        for dtype in ["bf16"]
        for layout in ["TN"]
        for shape in get_x_vals()
    ],
)
@pytest.mark.parametrize("backend", ["gluon", "triton"])
@pytest.mark.parametrize("shuffle", [True, False])
def test_gemm(dtype, M, N, K, layout, output, backend, shuffle):
    torch.cuda.empty_cache()  # Helps avoid hangs in large tests
    torch.cuda.synchronize()

    block_shape_n, block_shape_k = block_shape

    if backend == "gluon":
        if shuffle:
            if DEVICE_ARCH not in ("gfx1250"):
                pytest.skip("Gluon + shuffle implementation requires gfx1250.")
        elif DEVICE_ARCH not in ("gfx950", "gfx1250"):
            pytest.skip("Gluon implementation requires gfx950 or gfx1250.")

    if shuffle and (N % 16 > 0 or K % 32 > 0):
        pytest.skip(
            "N has to be multiple of 16 and K has to be multiple of 32 for preshuffle cases"
        )

    if backend not in ("gluon",) and K < 512:
        pytest.skip("Small-K shapes exercise gluon-only paths.")

    dtype = str_to_torch_dtype[dtype]
    x, weight, weight_triton, x_scale, x_scale_shuffled, w_scale, y = (
        generate_gemm_a8w8_blockscale_inputs(
            M,
            N,
            K,
            block_shape_n,
            block_shape_k,
            dtype=dtype,
            layout=layout,
            output=output,
            shuffle=shuffle,
        )
    )

    a = run_torch(x, weight, x_scale, w_scale, dtype)

    if shuffle:

        def impl(x, w, xs, ws, dt, y):
            return gemm_a8w8_blockscale_preshuffle(x, w, xs, ws, dt, y, backend=backend)

    else:

        def impl(x, w, xs, ws, dt, y):
            return gemm_a8w8_blockscale(x, w, xs, ws, dt, y, backend=backend)

    b = run_triton(x, weight_triton, x_scale_shuffled, w_scale, dtype, y, impl)

    torch.testing.assert_close(a, b, atol=0.01, rtol=1e-2)


def test_legacy_gluon_import_path_warns():
    """The pre-move path still resolves here, but tells callers to move on."""
    legacy = "aiter.ops.triton.gluon.gemm_a8w8_blockscale"
    sys.modules.pop(legacy, None)

    with pytest.warns(DeprecationWarning, match="has moved to"):
        mod = importlib.import_module(legacy)

    assert (
        mod.gemm_a8w8_blockscale.__module__
        == "aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale"
    )
