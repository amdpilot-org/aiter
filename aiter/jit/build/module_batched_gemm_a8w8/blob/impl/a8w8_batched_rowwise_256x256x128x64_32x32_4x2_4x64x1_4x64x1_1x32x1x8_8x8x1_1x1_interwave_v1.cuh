// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "batched_gemm_a8w8_common.cuh"

template <typename DDataType, typename EDataType = DDataType>
torch::Tensor
a8w8_batched_rowwise_256x256x128x64_32x32_4x2_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_interwave_v1(
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
    bool pad = (M % 256 != 0) || (N % 128 != 0) || (K % (64 * KBatch) != 0);
    if (pad)
    {
        // pad
        
        {
           using DeviceGemmInstance = DeviceGemmHelper<
                DDataType, EDataType,
                256,
                256,
                128,
                64,
                32,
                32,
                4,
                2,
                S<4, 64, 1>,
                S<4, 64, 1>,
                S<1, 32, 1, 8>,
                S<8, 8, 1>,
                1,
                1,
                ck::BlockGemmPipelineScheduler::Interwave,
                ck::BlockGemmPipelineVersion::v1,
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
                256,
                128,
                64,
                32,
                32,
                4,
                2,
                S<4, 64, 1>,
                S<4, 64, 1>,
                S<1, 32, 1, 8>,
                S<8, 8, 1>,
                1,
                1,
                ck::BlockGemmPipelineScheduler::Interwave,
                ck::BlockGemmPipelineVersion::v1,
                ck::tensor_operation::device::GemmSpecialization::Default>;
            // Run kernel instance.
            return batched_gemm_a8w8_rowwise_impl<DDataType, EDataType, DeviceGemmInstance>(XQ, WQ, x_scale, w_scale, Y, bias, KBatch);
        }

        // no pad
    }
}

