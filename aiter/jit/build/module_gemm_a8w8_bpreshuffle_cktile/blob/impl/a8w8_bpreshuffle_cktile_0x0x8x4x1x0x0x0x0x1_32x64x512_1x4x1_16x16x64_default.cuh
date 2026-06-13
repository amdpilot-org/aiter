// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "gemm_a8w8_bpreshuffle_cktile_common.cuh"

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_cktile_0x0x8x4x1x0x0x0x0x1_32x64x512_1x4x1_16x16x64_default(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int KBatch = 1
    )
{
    // The smallest kernel we have available. Works well for memory bound shapes.

    // Check if this input needs to be padded.
    int M = size_to_dim_(XQ.dim() - 1, XQ.sizes());
    int N = WQ.size(0);
    int K = WQ.size(1);
    bool pad = (M % 32 != 0) || (N % 64 != 0) || (K % (512) != 0);
    if (pad)
    {
        // pad
        using FlatmmInstance = CustomConfig<
            DDataType, EDataType,
            0,0, 8,
            4, 1, 0,
            0,  0,  0,
            1,
            32, 64, 512,
            1, 4, 1,
            16, 16, 64,
            ck_tile::GemmPipelineScheduler::Default>;
        // Run kernel instance.
        return gemm_a8w8_bpreshuffle_cktile_impl<DDataType, EDataType, FlatmmInstance>(XQ, WQ, x_scale, w_scale, Y, KBatch);

        // pad
    }
    else
    {
        // no pad
        using FlatmmInstance = CustomConfig<
            DDataType, EDataType,
            0,0, 8,
            4, 1, 0,
            0,  0,  0,
            1,
            32, 64, 512,
            1, 4, 1,
            16, 16, 64,
            ck_tile::GemmPipelineScheduler::Default>;
        // Run kernel instance.
        return gemm_a8w8_bpreshuffle_cktile_impl<DDataType, EDataType, FlatmmInstance>(XQ, WQ, x_scale, w_scale, Y, KBatch);

        // no pad
    }
}

