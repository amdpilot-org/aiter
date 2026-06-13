// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "gemm_a8w8_blockscale_cktile_common.cuh"

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_blockscale_cktile_16x128x256_1x4x1_16x16x64_intrawave_0x1x0_2(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    bool preshuffleB
    )
{
    // Get M, N, K from input tensors.
    int M = XQ.numel() / XQ.size(-1);
    int N = WQ.size(0);
    int K = WQ.size(1);

    // Instantiate tile gemm instance.
    using TileGemmInstance = TileGemmConfig<
            16, 128, 256,
            1, 4, 1,
            16, 16, 64,
            false,
            true,
            false,
            ck_tile::GemmPipelineScheduler::Intrawave,
            2>;

        // Run kernel instance.
        return gemm_a8w8_blockscale_cktile_impl<DDataType, EDataType, TileGemmInstance>(XQ, WQ, x_scale, w_scale, Y, preshuffleB);


}

