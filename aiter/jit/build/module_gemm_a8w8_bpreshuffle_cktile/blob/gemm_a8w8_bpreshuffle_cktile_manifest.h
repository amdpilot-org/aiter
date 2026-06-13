#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#ifdef USE_ROCM

#include <cstdlib>

#include <torch/extension.h>

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_cktile_0x0x8x4x1x0x0x0x0x1_128x128x128_1x4x1_16x16x64_default(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int KBatch);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_cktile_0x0x8x4x1x0x0x0x0x1_16x64x512_1x4x1_16x16x64_default(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int KBatch);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_cktile_0x0x8x4x1x0x0x0x0x1_32x64x512_1x4x1_16x16x64_default(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int KBatch);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_cktile_0x0x8x4x1x0x0x0x0x1_64x256x64_1x4x1_16x16x64_default(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int KBatch);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_cktile_0x0x8x4x1x0x0x0x0x1_128x128x64_1x4x1_16x16x64_default(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int KBatch);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_cktile_0x0x8x4x1x0x0x0x0x1_128x64x128_1x4x1_16x16x64_default(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int KBatch);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_cktile_0x0x8x4x1x0x0x0x0x1_64x256x128_1x4x1_16x16x64_default(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int KBatch);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_cktile_0x0x8x4x1x0x0x0x0x1_32x64x512_1x4x1_16x16x64_default(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int KBatch);


#endif // USE_ROCM
