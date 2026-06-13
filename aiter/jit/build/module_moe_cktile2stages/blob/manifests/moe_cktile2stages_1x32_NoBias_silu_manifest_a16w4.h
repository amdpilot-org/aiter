#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

// #ifdef USE_ROCM

#include <cstdlib>

#include <torch/extension.h>

// template <typename ADataType, typename BDataType, typename DDataType, typename EDataType>
template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
torch::Tensor
moe_cktile2stages_gemm1_256x16x128x256_1x4_16x16x32_2perCU_1x32_silu(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& Y,
    torch::Tensor& sorted_ids,
    torch::Tensor& sorted_expert_ids,
    torch::Tensor& max_token_ids,
    int topk,
    std::optional<int> n_padded_zeros,
    std::optional<int> k_padded_zeros,
    std::optional<torch::Tensor> topk_weight,
    std::optional<torch::Tensor> x_scale,
    std::optional<torch::Tensor> w_scale,
    std::optional<torch::Tensor> exp_bias,
    std::optional<int> activation,
    std::optional<int> k_batch);

// template <typename ADataType, typename BDataType, typename DDataType, typename EDataType>
template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
torch::Tensor
moe_cktile2stages_gemm1_256x32x256x256_1x4_16x16x32_2perCU_1x32_silu(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& Y,
    torch::Tensor& sorted_ids,
    torch::Tensor& sorted_expert_ids,
    torch::Tensor& max_token_ids,
    int topk,
    std::optional<int> n_padded_zeros,
    std::optional<int> k_padded_zeros,
    std::optional<torch::Tensor> topk_weight,
    std::optional<torch::Tensor> x_scale,
    std::optional<torch::Tensor> w_scale,
    std::optional<torch::Tensor> exp_bias,
    std::optional<int> activation,
    std::optional<int> k_batch);

// template <typename ADataType, typename BDataType, typename DDataType, typename EDataType>
template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
torch::Tensor
moe_cktile2stages_gemm1_256x64x256x256_1x4_16x16x32_1perCU_1x32_silu(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& Y,
    torch::Tensor& sorted_ids,
    torch::Tensor& sorted_expert_ids,
    torch::Tensor& max_token_ids,
    int topk,
    std::optional<int> n_padded_zeros,
    std::optional<int> k_padded_zeros,
    std::optional<torch::Tensor> topk_weight,
    std::optional<torch::Tensor> x_scale,
    std::optional<torch::Tensor> w_scale,
    std::optional<torch::Tensor> exp_bias,
    std::optional<int> activation,
    std::optional<int> k_batch);

// template <typename ADataType, typename BDataType, typename DDataType, typename EDataType>
template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
torch::Tensor
moe_cktile2stages_gemm1_256x16x128x256_1x4_16x16x32_1perCU_1x32_silu(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& Y,
    torch::Tensor& sorted_ids,
    torch::Tensor& sorted_expert_ids,
    torch::Tensor& max_token_ids,
    int topk,
    std::optional<int> n_padded_zeros,
    std::optional<int> k_padded_zeros,
    std::optional<torch::Tensor> topk_weight,
    std::optional<torch::Tensor> x_scale,
    std::optional<torch::Tensor> w_scale,
    std::optional<torch::Tensor> exp_bias,
    std::optional<int> activation,
    std::optional<int> k_batch);

// template <typename ADataType, typename BDataType, typename DDataType, typename EDataType>
template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
torch::Tensor
moe_cktile2stages_gemm1_256x16x128x256_1x4_16x16x32_3perCU_1x32_silu(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& Y,
    torch::Tensor& sorted_ids,
    torch::Tensor& sorted_expert_ids,
    torch::Tensor& max_token_ids,
    int topk,
    std::optional<int> n_padded_zeros,
    std::optional<int> k_padded_zeros,
    std::optional<torch::Tensor> topk_weight,
    std::optional<torch::Tensor> x_scale,
    std::optional<torch::Tensor> w_scale,
    std::optional<torch::Tensor> exp_bias,
    std::optional<int> activation,
    std::optional<int> k_batch);

// template <typename ADataType, typename BDataType, typename DDataType, typename EDataType>
template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
torch::Tensor
moe_cktile2stages_gemm1_256x16x128x256_1x4_16x16x32_4perCU_1x32_silu(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& Y,
    torch::Tensor& sorted_ids,
    torch::Tensor& sorted_expert_ids,
    torch::Tensor& max_token_ids,
    int topk,
    std::optional<int> n_padded_zeros,
    std::optional<int> k_padded_zeros,
    std::optional<torch::Tensor> topk_weight,
    std::optional<torch::Tensor> x_scale,
    std::optional<torch::Tensor> w_scale,
    std::optional<torch::Tensor> exp_bias,
    std::optional<int> activation,
    std::optional<int> k_batch);

// template <typename ADataType, typename BDataType, typename DDataType, typename EDataType>
template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
torch::Tensor
moe_cktile2stages_gemm1_256x32x256x256_1x4_16x16x32_1perCU_1x32_silu(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& Y,
    torch::Tensor& sorted_ids,
    torch::Tensor& sorted_expert_ids,
    torch::Tensor& max_token_ids,
    int topk,
    std::optional<int> n_padded_zeros,
    std::optional<int> k_padded_zeros,
    std::optional<torch::Tensor> topk_weight,
    std::optional<torch::Tensor> x_scale,
    std::optional<torch::Tensor> w_scale,
    std::optional<torch::Tensor> exp_bias,
    std::optional<int> activation,
    std::optional<int> k_batch);

// template <typename ADataType, typename BDataType, typename DDataType, typename EDataType>
template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
torch::Tensor
moe_cktile2stages_gemm1_256x32x256x256_1x4_16x16x32_3perCU_1x32_silu(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& Y,
    torch::Tensor& sorted_ids,
    torch::Tensor& sorted_expert_ids,
    torch::Tensor& max_token_ids,
    int topk,
    std::optional<int> n_padded_zeros,
    std::optional<int> k_padded_zeros,
    std::optional<torch::Tensor> topk_weight,
    std::optional<torch::Tensor> x_scale,
    std::optional<torch::Tensor> w_scale,
    std::optional<torch::Tensor> exp_bias,
    std::optional<int> activation,
    std::optional<int> k_batch);

// template <typename ADataType, typename BDataType, typename DDataType, typename EDataType>
template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
torch::Tensor
moe_cktile2stages_gemm1_256x32x256x256_1x4_16x16x32_4perCU_1x32_silu(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& Y,
    torch::Tensor& sorted_ids,
    torch::Tensor& sorted_expert_ids,
    torch::Tensor& max_token_ids,
    int topk,
    std::optional<int> n_padded_zeros,
    std::optional<int> k_padded_zeros,
    std::optional<torch::Tensor> topk_weight,
    std::optional<torch::Tensor> x_scale,
    std::optional<torch::Tensor> w_scale,
    std::optional<torch::Tensor> exp_bias,
    std::optional<int> activation,
    std::optional<int> k_batch);

// template <typename ADataType, typename BDataType, typename DDataType, typename EDataType>
template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
torch::Tensor
moe_cktile2stages_gemm1_256x64x256x256_1x4_16x16x32_2perCU_1x32_silu(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& Y,
    torch::Tensor& sorted_ids,
    torch::Tensor& sorted_expert_ids,
    torch::Tensor& max_token_ids,
    int topk,
    std::optional<int> n_padded_zeros,
    std::optional<int> k_padded_zeros,
    std::optional<torch::Tensor> topk_weight,
    std::optional<torch::Tensor> x_scale,
    std::optional<torch::Tensor> w_scale,
    std::optional<torch::Tensor> exp_bias,
    std::optional<int> activation,
    std::optional<int> k_batch);

// template <typename ADataType, typename BDataType, typename DDataType, typename EDataType>
template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
torch::Tensor
moe_cktile2stages_gemm1_256x64x256x256_1x4_16x16x32_3perCU_1x32_silu(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& Y,
    torch::Tensor& sorted_ids,
    torch::Tensor& sorted_expert_ids,
    torch::Tensor& max_token_ids,
    int topk,
    std::optional<int> n_padded_zeros,
    std::optional<int> k_padded_zeros,
    std::optional<torch::Tensor> topk_weight,
    std::optional<torch::Tensor> x_scale,
    std::optional<torch::Tensor> w_scale,
    std::optional<torch::Tensor> exp_bias,
    std::optional<int> activation,
    std::optional<int> k_batch);

// template <typename ADataType, typename BDataType, typename DDataType, typename EDataType>
template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
torch::Tensor
moe_cktile2stages_gemm1_256x64x256x256_1x4_16x16x32_4perCU_1x32_silu(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& Y,
    torch::Tensor& sorted_ids,
    torch::Tensor& sorted_expert_ids,
    torch::Tensor& max_token_ids,
    int topk,
    std::optional<int> n_padded_zeros,
    std::optional<int> k_padded_zeros,
    std::optional<torch::Tensor> topk_weight,
    std::optional<torch::Tensor> x_scale,
    std::optional<torch::Tensor> w_scale,
    std::optional<torch::Tensor> exp_bias,
    std::optional<int> activation,
    std::optional<int> k_batch);

// template <typename ADataType, typename BDataType, typename DDataType, typename EDataType>
template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
torch::Tensor
moe_cktile2stages_gemm2_256x16x128x256_1x4_16x16x32_2perCU_1x32_MulRoutedWeight(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& Y,
    torch::Tensor& sorted_ids,
    torch::Tensor& sorted_expert_ids,
    torch::Tensor& max_token_ids,
    int topk,
    std::optional<int> n_padded_zeros,
    std::optional<int> k_padded_zeros,
    std::optional<torch::Tensor> topk_weight,
    std::optional<torch::Tensor> x_scale,
    std::optional<torch::Tensor> w_scale,
    std::optional<torch::Tensor> exp_bias,
    std::optional<int> activation,
    std::optional<int> k_batch);

// template <typename ADataType, typename BDataType, typename DDataType, typename EDataType>
template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
torch::Tensor
moe_cktile2stages_gemm2_256x32x256x256_1x4_16x16x32_2perCU_1x32_MulRoutedWeight(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& Y,
    torch::Tensor& sorted_ids,
    torch::Tensor& sorted_expert_ids,
    torch::Tensor& max_token_ids,
    int topk,
    std::optional<int> n_padded_zeros,
    std::optional<int> k_padded_zeros,
    std::optional<torch::Tensor> topk_weight,
    std::optional<torch::Tensor> x_scale,
    std::optional<torch::Tensor> w_scale,
    std::optional<torch::Tensor> exp_bias,
    std::optional<int> activation,
    std::optional<int> k_batch);

// template <typename ADataType, typename BDataType, typename DDataType, typename EDataType>
template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
torch::Tensor
moe_cktile2stages_gemm2_256x64x256x256_1x4_16x16x32_1perCU_1x32_MulRoutedWeight(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& Y,
    torch::Tensor& sorted_ids,
    torch::Tensor& sorted_expert_ids,
    torch::Tensor& max_token_ids,
    int topk,
    std::optional<int> n_padded_zeros,
    std::optional<int> k_padded_zeros,
    std::optional<torch::Tensor> topk_weight,
    std::optional<torch::Tensor> x_scale,
    std::optional<torch::Tensor> w_scale,
    std::optional<torch::Tensor> exp_bias,
    std::optional<int> activation,
    std::optional<int> k_batch);

// template <typename ADataType, typename BDataType, typename DDataType, typename EDataType>
template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
torch::Tensor
moe_cktile2stages_gemm2_256x16x128x256_1x4_16x16x32_1perCU_1x32_MulRoutedWeight(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& Y,
    torch::Tensor& sorted_ids,
    torch::Tensor& sorted_expert_ids,
    torch::Tensor& max_token_ids,
    int topk,
    std::optional<int> n_padded_zeros,
    std::optional<int> k_padded_zeros,
    std::optional<torch::Tensor> topk_weight,
    std::optional<torch::Tensor> x_scale,
    std::optional<torch::Tensor> w_scale,
    std::optional<torch::Tensor> exp_bias,
    std::optional<int> activation,
    std::optional<int> k_batch);

// template <typename ADataType, typename BDataType, typename DDataType, typename EDataType>
template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
torch::Tensor
moe_cktile2stages_gemm2_256x16x128x256_1x4_16x16x32_3perCU_1x32_MulRoutedWeight(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& Y,
    torch::Tensor& sorted_ids,
    torch::Tensor& sorted_expert_ids,
    torch::Tensor& max_token_ids,
    int topk,
    std::optional<int> n_padded_zeros,
    std::optional<int> k_padded_zeros,
    std::optional<torch::Tensor> topk_weight,
    std::optional<torch::Tensor> x_scale,
    std::optional<torch::Tensor> w_scale,
    std::optional<torch::Tensor> exp_bias,
    std::optional<int> activation,
    std::optional<int> k_batch);

// template <typename ADataType, typename BDataType, typename DDataType, typename EDataType>
template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
torch::Tensor
moe_cktile2stages_gemm2_256x16x128x256_1x4_16x16x32_4perCU_1x32_MulRoutedWeight(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& Y,
    torch::Tensor& sorted_ids,
    torch::Tensor& sorted_expert_ids,
    torch::Tensor& max_token_ids,
    int topk,
    std::optional<int> n_padded_zeros,
    std::optional<int> k_padded_zeros,
    std::optional<torch::Tensor> topk_weight,
    std::optional<torch::Tensor> x_scale,
    std::optional<torch::Tensor> w_scale,
    std::optional<torch::Tensor> exp_bias,
    std::optional<int> activation,
    std::optional<int> k_batch);

// template <typename ADataType, typename BDataType, typename DDataType, typename EDataType>
template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
torch::Tensor
moe_cktile2stages_gemm2_256x32x256x256_1x4_16x16x32_1perCU_1x32_MulRoutedWeight(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& Y,
    torch::Tensor& sorted_ids,
    torch::Tensor& sorted_expert_ids,
    torch::Tensor& max_token_ids,
    int topk,
    std::optional<int> n_padded_zeros,
    std::optional<int> k_padded_zeros,
    std::optional<torch::Tensor> topk_weight,
    std::optional<torch::Tensor> x_scale,
    std::optional<torch::Tensor> w_scale,
    std::optional<torch::Tensor> exp_bias,
    std::optional<int> activation,
    std::optional<int> k_batch);

// template <typename ADataType, typename BDataType, typename DDataType, typename EDataType>
template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
torch::Tensor
moe_cktile2stages_gemm2_256x32x256x256_1x4_16x16x32_3perCU_1x32_MulRoutedWeight(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& Y,
    torch::Tensor& sorted_ids,
    torch::Tensor& sorted_expert_ids,
    torch::Tensor& max_token_ids,
    int topk,
    std::optional<int> n_padded_zeros,
    std::optional<int> k_padded_zeros,
    std::optional<torch::Tensor> topk_weight,
    std::optional<torch::Tensor> x_scale,
    std::optional<torch::Tensor> w_scale,
    std::optional<torch::Tensor> exp_bias,
    std::optional<int> activation,
    std::optional<int> k_batch);

// template <typename ADataType, typename BDataType, typename DDataType, typename EDataType>
template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
torch::Tensor
moe_cktile2stages_gemm2_256x32x256x256_1x4_16x16x32_4perCU_1x32_MulRoutedWeight(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& Y,
    torch::Tensor& sorted_ids,
    torch::Tensor& sorted_expert_ids,
    torch::Tensor& max_token_ids,
    int topk,
    std::optional<int> n_padded_zeros,
    std::optional<int> k_padded_zeros,
    std::optional<torch::Tensor> topk_weight,
    std::optional<torch::Tensor> x_scale,
    std::optional<torch::Tensor> w_scale,
    std::optional<torch::Tensor> exp_bias,
    std::optional<int> activation,
    std::optional<int> k_batch);

// template <typename ADataType, typename BDataType, typename DDataType, typename EDataType>
template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
torch::Tensor
moe_cktile2stages_gemm2_256x64x256x256_1x4_16x16x32_2perCU_1x32_MulRoutedWeight(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& Y,
    torch::Tensor& sorted_ids,
    torch::Tensor& sorted_expert_ids,
    torch::Tensor& max_token_ids,
    int topk,
    std::optional<int> n_padded_zeros,
    std::optional<int> k_padded_zeros,
    std::optional<torch::Tensor> topk_weight,
    std::optional<torch::Tensor> x_scale,
    std::optional<torch::Tensor> w_scale,
    std::optional<torch::Tensor> exp_bias,
    std::optional<int> activation,
    std::optional<int> k_batch);

// template <typename ADataType, typename BDataType, typename DDataType, typename EDataType>
template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
torch::Tensor
moe_cktile2stages_gemm2_256x64x256x256_1x4_16x16x32_3perCU_1x32_MulRoutedWeight(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& Y,
    torch::Tensor& sorted_ids,
    torch::Tensor& sorted_expert_ids,
    torch::Tensor& max_token_ids,
    int topk,
    std::optional<int> n_padded_zeros,
    std::optional<int> k_padded_zeros,
    std::optional<torch::Tensor> topk_weight,
    std::optional<torch::Tensor> x_scale,
    std::optional<torch::Tensor> w_scale,
    std::optional<torch::Tensor> exp_bias,
    std::optional<int> activation,
    std::optional<int> k_batch);

// template <typename ADataType, typename BDataType, typename DDataType, typename EDataType>
template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
torch::Tensor
moe_cktile2stages_gemm2_256x64x256x256_1x4_16x16x32_4perCU_1x32_MulRoutedWeight(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& Y,
    torch::Tensor& sorted_ids,
    torch::Tensor& sorted_expert_ids,
    torch::Tensor& max_token_ids,
    int topk,
    std::optional<int> n_padded_zeros,
    std::optional<int> k_padded_zeros,
    std::optional<torch::Tensor> topk_weight,
    std::optional<torch::Tensor> x_scale,
    std::optional<torch::Tensor> w_scale,
    std::optional<torch::Tensor> exp_bias,
    std::optional<int> activation,
    std::optional<int> k_batch);


// endif // USE_ROCM
