// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "impl/bf16_batched_256x32x96x128_16x16_1x3_16x16x1_16x16x1_1x32x1x8_4x4x1_1x1_intrawave_v3.cuh"

torch::Tensor
bf16_batched_256x32x96x128_16x16_1x3_16x16x1_16x16x1_1x32x1x8_4x4x1_1x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &Y,
    std::optional<torch::Tensor> bias,
    int KBatch);

