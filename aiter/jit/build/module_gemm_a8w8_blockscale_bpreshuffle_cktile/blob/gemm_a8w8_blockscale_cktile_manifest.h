#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#ifdef USE_ROCM

#include <cstdlib>

#include <torch/extension.h>

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_blockscale_cktile_128x128x128_1x4x1_16x16x64_intrawave_0x1x0_1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    bool preshuffleB);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_blockscale_cktile_16x128x256_1x4x1_16x16x64_intrawave_0x1x0_1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    bool preshuffleB);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_blockscale_cktile_16x128x256_1x4x1_16x16x64_intrawave_0x1x0_1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    bool preshuffleB);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_blockscale_cktile_16x128x256_1x4x1_16x16x64_intrawave_0x1x0_1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    bool preshuffleB);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_blockscale_cktile_16x128x256_1x4x1_16x16x64_intrawave_0x1x0_1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    bool preshuffleB);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_blockscale_cktile_16x128x256_1x4x1_16x16x64_intrawave_0x1x0_3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    bool preshuffleB);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_blockscale_cktile_16x128x256_1x4x1_16x16x64_intrawave_0x1x0_1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    bool preshuffleB);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_blockscale_cktile_16x128x256_1x4x1_16x16x64_intrawave_0x1x0_1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    bool preshuffleB);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_blockscale_cktile_16x128x256_1x4x1_16x16x64_intrawave_0x1x0_1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    bool preshuffleB);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_blockscale_cktile_16x128x256_1x4x1_16x16x64_intrawave_0x1x0_1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    bool preshuffleB);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_blockscale_cktile_16x128x256_1x4x1_16x16x64_intrawave_0x1x0_1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    bool preshuffleB);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_blockscale_cktile_16x128x256_1x4x1_16x16x64_intrawave_0x1x0_1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    bool preshuffleB);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_blockscale_cktile_16x128x256_1x4x1_16x16x64_intrawave_0x1x0_4(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    bool preshuffleB);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_blockscale_cktile_16x128x256_1x4x1_16x16x64_intrawave_0x1x0_1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    bool preshuffleB);


#endif // USE_ROCM
