#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
// #ifdef USE_ROCM
#include <cstdlib>
#include <torch/extension.h>

template <typename ABDataType, typename DDataType, typename EDataType>
torch::Tensor
deepgemm_256x128x128x128_16x16x64_1x4(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &Y,
    torch::Tensor &grouped_layout,
    std::optional<torch::Tensor> x_scale,
    std::optional<torch::Tensor> w_scale);

template <typename ABDataType, typename DDataType, typename EDataType>
torch::Tensor
deepgemm_256x128x128x128_16x16x32_1x4(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &Y,
    torch::Tensor &grouped_layout,
    std::optional<torch::Tensor> x_scale,
    std::optional<torch::Tensor> w_scale);

template <typename ABDataType, typename DDataType, typename EDataType>
torch::Tensor
deepgemm_256x32x64x256_16x16x64_1x4(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &Y,
    torch::Tensor &grouped_layout,
    std::optional<torch::Tensor> x_scale,
    std::optional<torch::Tensor> w_scale);

template <typename ABDataType, typename DDataType, typename EDataType>
torch::Tensor
deepgemm_256x32x64x256_16x16x32_1x4(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &Y,
    torch::Tensor &grouped_layout,
    std::optional<torch::Tensor> x_scale,
    std::optional<torch::Tensor> w_scale);

// endif // USE_ROCM
