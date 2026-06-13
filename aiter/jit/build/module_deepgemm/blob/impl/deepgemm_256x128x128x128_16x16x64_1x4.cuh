// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include "deepgemm_common.cuh"
template <typename ABDataType, typename AccDataType, typename CDataType>
torch::Tensor
deepgemm_256x128x128x128_16x16x64_1x4(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &Y,
    torch::Tensor &group_layout,
    std::optional<torch::Tensor> x_scale,
    std::optional<torch::Tensor> w_scale)
{
    // The smallest kernel we have available. Works well for memory bound shapes.
    int group_count = Y.size(0);
    int M = XQ.size(1);
    int N = Y.size(2);
    int K = XQ.size(2);
    int Stride_A = K;
    int Stride_B = K;
    int Stride_C = N;
    if (x_scale != std::nullopt && w_scale != std::nullopt )
        {{
            auto per_a_scale_dev_ptr = ck_tile::FlatmmScalePointer<1>{static_cast<float*>(x_scale.value().data_ptr())};
            auto per_b_scale_dev_ptr = ck_tile::FlatmmScalePointer<1>{static_cast<float*>(w_scale.value().data_ptr())};
            ck_tile::MaskedGroupedFlatmmHostArgs<decltype(per_a_scale_dev_ptr), decltype(per_b_scale_dev_ptr)> kernel_args{
                reinterpret_cast<ck_tile::index_t *>(group_layout.data_ptr()),
                group_count,
                M,
                N,
                K,
                reinterpret_cast<const void*>(XQ.data_ptr()),
                Stride_A,
                reinterpret_cast<const void*>(WQ.data_ptr()),
                Stride_B,
                {},{},
                reinterpret_cast<void*>(Y.data_ptr()),
                Stride_C,
                1, //KBatch
                per_a_scale_dev_ptr,
                per_b_scale_dev_ptr
            };
            using TileConfig = MGroupedFlatmmConfig<ABDataType,
                128,
                128,
                128,
                1,
                4,
                16,
                16,
                64>;
            // Run kernel instance.
            auto stream_config = ck_stream_config{at::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream()};
            grouped_flatmm<TileConfig, ABDataType, ABDataType, ck_tile::tuple<>, AccDataType, CDataType, row_major, col_major, ck_tile::tuple<>, row_major, false, ck_tile::element_wise::PassThrough>(kernel_args, stream_config);
        }}
        else
        {{
            auto per_a_scale_dev_ptr = ck_tile::FlatmmScalePointer<-1>{nullptr};
            auto per_b_scale_dev_ptr = ck_tile::FlatmmScalePointer<-1>{nullptr};
            ck_tile::MaskedGroupedFlatmmHostArgs<decltype(per_a_scale_dev_ptr), decltype(per_b_scale_dev_ptr)> kernel_args{
                reinterpret_cast<ck_tile::index_t *>(group_layout.data_ptr()),
                group_count,
                M,
                N,
                K,
                reinterpret_cast<const void*>(XQ.data_ptr()),
                Stride_A,
                reinterpret_cast<const void*>(WQ.data_ptr()),
                Stride_B,
                {},{},
                reinterpret_cast<void*>(Y.data_ptr()),
                Stride_C,
                1, //KBatch
                per_a_scale_dev_ptr,
                per_b_scale_dev_ptr
            };
            using TileConfig = MGroupedFlatmmConfig<ABDataType,
                128,
                128,
                128,
                1,
                4,
                16,
                16,
                32>;
            // Run kernel instance.
            auto stream_config = ck_stream_config{at::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream()};
            grouped_flatmm<TileConfig, ABDataType, ABDataType, ck_tile::tuple<>, AccDataType, CDataType, row_major, col_major, ck_tile::tuple<>, row_major, false, ck_tile::element_wise::PassThrough>(kernel_args, stream_config);
        }}

    return Y;
}
