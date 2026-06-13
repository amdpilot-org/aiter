// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "impl/a8w8_bpreshuffle_cktile_0x0x8x4x1x0x0x0x0x1_128x128x64_1x4x1_16x16x64_default.cuh"

template torch::Tensor
a8w8_bpreshuffle_cktile_0x0x8x4x1x0x0x0x0x1_128x128x64_1x4x1_16x16x64_default<F32, B16>(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int KBatch
    );

