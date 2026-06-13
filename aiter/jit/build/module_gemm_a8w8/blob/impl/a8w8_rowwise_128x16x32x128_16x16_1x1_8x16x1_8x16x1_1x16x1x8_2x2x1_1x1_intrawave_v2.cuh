// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "gemm_a8w8_common.cuh"

template <typename ABDataType, typename DDataType, typename EDataType = DDataType>
torch::Tensor
a8w8_rowwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_2x2x1_1x1_intrawave_v2(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    std::optional<torch::Tensor> bias,
    int KBatch)
{
    // The smallest kernel we have available. Works well for memory bound shapes.

    // Check if this input needs to be padded.
    int M = size_to_dim_(XQ.dim() - 1, XQ.sizes());
    int N = WQ.size(0);
    int K = WQ.size(1);
    bool pad = (M % 16 != 0) || (N % 32 != 0) || (K % (128 * KBatch) != 0);
    using AccDataType = std::conditional_t<ck::is_same_v<ABDataType, I8>, I32, F32>;
    if (pad)
    {
        // pad
        if (bias != std::nullopt)
        {
            using DeviceGemmInstance = DeviceGemmHelper<
                ABDataType,
                AccDataType,
                DDataType, EDataType,
                MultiplyMultiplyAdd<AccDataType, DDataType, EDataType>,
                128,
                16,
                32,
                128,
                16,
                16,
                1,
                1,
                S<8, 16, 1>,
                S<8, 16, 1>,
                S<1, 16, 1, 8>,
                S<2, 2, 1, 2>,
                1,
                1,
                ck::BlockGemmPipelineScheduler::Intrawave,
                ck::BlockGemmPipelineVersion::v2,
                ck::tensor_operation::device::GemmSpecialization::MNKPadding>;
            // Run kernel instance.
            return gemm_a8w8_rowwise_impl<ABDataType, DDataType, EDataType, true, DeviceGemmInstance>(XQ, WQ, x_scale, w_scale, Y, bias, KBatch);
        }
        else
        {
            using DeviceGemmInstance = DeviceGemmHelper<
                ABDataType,
                AccDataType,
                DDataType, EDataType,
                RowwiseScale<AccDataType, DDataType, EDataType>,
                128,
                16,
                32,
                128,
                16,
                16,
                1,
                1,
                S<8, 16, 1>,
                S<8, 16, 1>,
                S<1, 16, 1, 8>,
                S<2, 2, 1>,
                1,
                1,
                ck::BlockGemmPipelineScheduler::Intrawave,
                ck::BlockGemmPipelineVersion::v2,
                ck::tensor_operation::device::GemmSpecialization::MNKPadding>;
            // Run kernel instance.
            return gemm_a8w8_rowwise_impl<ABDataType, DDataType, EDataType, false, DeviceGemmInstance>(XQ, WQ, x_scale, w_scale, Y, bias, KBatch);
        }

        // pad
    }
    else
    {
        // no pad
        if (bias != std::nullopt)
        {
            using DeviceGemmInstance = DeviceGemmHelper<
                ABDataType,
                AccDataType,
                DDataType, EDataType,
                MultiplyMultiplyAdd<AccDataType, DDataType, EDataType>,
                128,
                16,
                32,
                128,
                16,
                16,
                1,
                1,
                S<8, 16, 1>,
                S<8, 16, 1>,
                S<1, 16, 1, 8>,
                S<2, 2, 1, 2>,
                1,
                1,
                ck::BlockGemmPipelineScheduler::Intrawave,
                ck::BlockGemmPipelineVersion::v2,
                ck::tensor_operation::device::GemmSpecialization::Default>;
            // Run kernel instance.
            return gemm_a8w8_rowwise_impl<ABDataType, DDataType, EDataType, true, DeviceGemmInstance>(XQ, WQ, x_scale, w_scale, Y, bias, KBatch);
        }
        else
        {
            using DeviceGemmInstance = DeviceGemmHelper<
                ABDataType,
                AccDataType,
                DDataType, EDataType,
                RowwiseScale<AccDataType, DDataType, EDataType>,
                128,
                16,
                32,
                128,
                16,
                16,
                1,
                1,
                S<8, 16, 1>,
                S<8, 16, 1>,
                S<1, 16, 1, 8>,
                S<2, 2, 1>,
                1,
                1,
                ck::BlockGemmPipelineScheduler::Intrawave,
                ck::BlockGemmPipelineVersion::v2,
                ck::tensor_operation::device::GemmSpecialization::Default>;
            // Run kernel instance.
            return gemm_a8w8_rowwise_impl<ABDataType, DDataType, EDataType, false, DeviceGemmInstance>(XQ, WQ, x_scale, w_scale, Y, bias, KBatch);
        }

        // no pad
    }
}

