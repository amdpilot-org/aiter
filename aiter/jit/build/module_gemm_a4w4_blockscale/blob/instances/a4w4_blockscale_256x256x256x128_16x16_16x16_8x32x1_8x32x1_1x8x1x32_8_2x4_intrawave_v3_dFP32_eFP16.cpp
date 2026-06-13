// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "impl/a4w4_blockscale_256x256x256x128_16x16_16x16_8x32x1_8x32x1_1x8x1x32_8_2x4_intrawave_v3.cuh"

template torch::Tensor
a4w4_blockscale_256x256x256x128_16x16_16x16_8x32x1_8x32x1_1x8x1x32_8_2x4_intrawave_v3<F16>(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK
    );

