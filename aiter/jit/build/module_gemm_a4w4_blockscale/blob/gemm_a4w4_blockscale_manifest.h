#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#ifdef USE_ROCM

#include <cstdlib>

#include <torch/extension.h>

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x32x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x32x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x32x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x64x512x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x32x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x32x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x32x512x128_16x16_16x16_8x32x1_8x32x1_1x8x1x32_8_2x4_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x32x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x32x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x64x512x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x32x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x32x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x64x512x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x96x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x32x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x32x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x32x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x32x512x128_16x16_16x16_8x32x1_8x32x1_1x8x1x32_8_2x4_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x96x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x32x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x32x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x32x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x32x256x128_16x16_16x16_8x32x1_8x32x1_1x8x1x32_8_2x4_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x32x512x128_16x16_16x16_8x32x1_8x32x1_1x8x1x32_8_2x4_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x96x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x64x512x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x128x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x128x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x128x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x128x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x128x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x128x512x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x64x384x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x256x256x128_16x16_16x16_8x32x1_8x32x1_1x8x1x32_8_2x4_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x256x256x128_16x16_16x16_8x32x1_8x32x1_1x8x1x32_8_2x4_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x256x256x128_16x16_16x16_8x32x1_8x32x1_1x8x1x32_8_2x4_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x128x384x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x96x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x128x384x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x96x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x96x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x96x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x128x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x96x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x96x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x96x384x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x128x384x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x96x512x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x128x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x128x384x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x128x384x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x128x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x128x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x128x384x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x128x384x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x128x384x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x96x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x128x384x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x128x384x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x128x384x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x128x384x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x128x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x128x384x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x128x384x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x128x384x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x128x384x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x128x384x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x96x512x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x128x384x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x128x384x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x128x384x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x128x384x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x128x384x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK);


#endif // USE_ROCM
