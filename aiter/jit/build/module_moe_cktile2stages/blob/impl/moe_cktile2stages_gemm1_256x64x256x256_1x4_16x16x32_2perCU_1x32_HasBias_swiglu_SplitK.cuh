// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include "moe_cktile2stages_common.cuh"

template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
torch::Tensor
moe_cktile2stages_gemm1_256x64x256x256_1x4_16x16x32_2perCU_1x32_HasBias_swiglu_SplitK(
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
    std::optional<int> k_batch)
{
    // The smallest kernel we have available. Works well for memory bound shapes.
    int NumTokens = XQ.size(0);
    int M = sorted_ids.size(0);
    int N = WQ.size(1);
    int K = XQ.size(-1);
    int E = WQ.size(0);
    int KBatch = k_batch.has_value() ? k_batch.value() : 1;
    int stride_A = K;
    int stride_B = K;
    int stride_C = KBatch > 1 ? N : N / 2; //gemm1 gate+up need / 2.
    void *sorted_weights_ptr = topk_weight.has_value() ? topk_weight.value().data_ptr() : nullptr;

    auto per_a_scale_dev_ptr = ck_tile::FlatmmScalePointer<1, 32, ck_tile::e8m0_t>{x_scale.has_value() ? static_cast<ck_tile::e8m0_t*>(x_scale.value().data_ptr()) : nullptr};
    auto per_b_scale_dev_ptr = ck_tile::FlatmmScalePointer<1, 32, ck_tile::e8m0_t>{static_cast<ck_tile::e8m0_t*>(w_scale.value().data_ptr())};
    auto exp_bias_dev_ptr = ck_tile::FlatmmScalePointer<1>{static_cast<float*>(exp_bias.has_value() ? exp_bias.value().data_ptr() : nullptr)};
    ck_tile::MoeFlatmmHostArgs<decltype(per_a_scale_dev_ptr),
                               decltype(per_b_scale_dev_ptr),
                               decltype(exp_bias_dev_ptr)> kernel_args{
                reinterpret_cast<const ck_tile::index_t*>(sorted_ids.data_ptr()),
                sorted_weights_ptr,
                reinterpret_cast<const ck_tile::index_t*>(sorted_expert_ids.data_ptr()),
                reinterpret_cast<const ck_tile::index_t*>(max_token_ids.data_ptr()),
                reinterpret_cast<const void*>(XQ.data_ptr()),
                reinterpret_cast<const void*>(WQ.data_ptr()),
                reinterpret_cast<void*>(Y.data_ptr()),
                NumTokens,
                E,
                topk,
                KBatch, // k_batch
                M,
                N,
                K,
                stride_A,
                stride_B,
                stride_C,
                n_padded_zeros.has_value() ? n_padded_zeros.value() : 0,
                k_padded_zeros.has_value() ? k_padded_zeros.value() : 0,
                per_a_scale_dev_ptr,
                per_b_scale_dev_ptr,
                exp_bias_dev_ptr
    };
    using TileConfig = MoeFlatmmConfig<ADataType,
        64,
        256,
        256,
        1,
        4,
        16,
        16,
        32,
        2>;
    // Run kernel instance.
    auto stream_config = ck_stream_config{at::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream()};
    moe_gemm<TileConfig,
                ADataType,
                BDataType,
                ck_tile::tuple<>,
                AccDataType,
                CDataType,
                row_major,
                col_major,
                ck_tile::tuple<>,
                row_major,
                ck_tile::MoeFlatmmKind::kFFN_gemm1_split_k,
                ck_tile::element_wise::PassThrough,
                2
                >(kernel_args, stream_config);

    return Y;
}

