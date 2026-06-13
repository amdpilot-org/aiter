// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "impl/a8w8_blockscale_cktile_16x128x256_1x4x1_16x16x64_intrawave_0x1x0_1.cuh"

template torch::Tensor
a8w8_blockscale_cktile_16x128x256_1x4x1_16x16x64_intrawave_0x1x0_1<TILE_FP32, TILE_BF16>(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    bool preshuffleB
    );

