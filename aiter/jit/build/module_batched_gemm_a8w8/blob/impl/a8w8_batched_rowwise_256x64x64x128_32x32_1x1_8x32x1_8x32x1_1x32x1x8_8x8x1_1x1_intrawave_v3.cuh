// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "batched_gemm_a8w8_common.cuh"

template <typename DDataType, typename EDataType = DDataType>
torch::Tensor
a8w8_batched_rowwise_256x64x64x128_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3(
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
    int B = XQ.size(0);
    int M = XQ.size(1);
    int N = WQ.size(1);
    int K = WQ.size(2);
    bool pad = (M % 64 != 0) || (N % 64 != 0) || (K % (128 * KBatch) != 0);
    if (pad)
    {
        // pad
        
        {
           using DeviceGemmInstance = DeviceGemmHelper<
                DDataType, EDataType,
                256,
                64,
                64,
                128,
                32,
                32,
                1,
                1,
                S<8, 32, 1>,
                S<8, 32, 1>,
                S<1, 32, 1, 8>,
                S<8, 8, 1>,
                1,
                1,
                ck::BlockGemmPipelineScheduler::Intrawave,
                ck::BlockGemmPipelineVersion::v3,
                ck::tensor_operation::device::GemmSpecialization::MNKPadding>;
            // Run kernel instance.
            return batched_gemm_a8w8_rowwise_impl<DDataType, EDataType, DeviceGemmInstance>(XQ, WQ, x_scale, w_scale, Y, bias, KBatch);
        }

        // pad
    }
    else
    {
        // no pad
        
        {
           using DeviceGemmInstance = DeviceGemmHelper<
                DDataType, EDataType,
                256,
                64,
                64,
                128,
                32,
                32,
                1,
                1,
                S<8, 32, 1>,
                S<8, 32, 1>,
                S<1, 32, 1, 8>,
                S<8, 8, 1>,
                1,
                1,
                ck::BlockGemmPipelineScheduler::Intrawave,
                ck::BlockGemmPipelineVersion::v3,
                ck::tensor_operation::device::GemmSpecialization::Default>;
            // Run kernel instance.
            return batched_gemm_a8w8_rowwise_impl<DDataType, EDataType, DeviceGemmInstance>(XQ, WQ, x_scale, w_scale, Y, bias, KBatch);
        }

        // no pad
    }
}

