// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include "impl/deepgemm_256x128x128x128_16x16x32_1x4.cuh"
template torch::Tensor
deepgemm_256x128x128x128_16x16x32_1x4<fp16, float, fp16>(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &Y,
    torch::Tensor &grouped_layout,
    std::optional<torch::Tensor> x_scale,
    std::optional<torch::Tensor> w_scale);
