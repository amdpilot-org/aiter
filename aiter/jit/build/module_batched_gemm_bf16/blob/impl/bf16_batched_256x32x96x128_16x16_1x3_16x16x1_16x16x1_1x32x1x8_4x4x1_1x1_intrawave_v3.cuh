// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "batched_gemm_bf16_common.cuh"

torch::Tensor
bf16_batched_256x32x96x128_16x16_1x3_16x16x1_16x16x1_1x32x1x8_4x4x1_1x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &Y,
    std::optional<torch::Tensor> bias,
    int KBatch)
{
    // The smallest kernel we have available. Works well for memory bound shapes.

#if 0
    // Check if this input needs to be padded.
    int B = XQ.size(0);
    int M = XQ.size(1);
    int N = WQ.size(1);
    int K = WQ.size(2);
    bool pad = (M % 32 != 0) || (N % 96 != 0) || (K % (128 * KBatch) != 0);
#else
    // Disable padding for packed tensor
    bool pad = false;
#endif
    if (pad)
    {
        // pad
        
        {
           using DeviceGemmInstance = DeviceGemmHelper<
                256,
                32,
                96,
                128,
                16,
                16,
                1,
                3,
                S<16, 16, 1>,
                S<16, 16, 1>,
                S<1, 32, 1, 8>,
                S<4, 4, 1>,
                1,
                1,
                ck::BlockGemmPipelineScheduler::Intrawave,
                ck::BlockGemmPipelineVersion::v3,
                ck::tensor_operation::device::GemmSpecialization::MNKPadding>;
            // Run kernel instance.
            return batched_gemm_bf16_impl<DeviceGemmInstance>(XQ, WQ, Y, bias, KBatch);
        }

        // pad
    }
    else
    {
        // no pad
        
        {
           using DeviceGemmInstance = DeviceGemmHelper<
                256,
                32,
                96,
                128,
                16,
                16,
                1,
                3,
                S<16, 16, 1>,
                S<16, 16, 1>,
                S<1, 32, 1, 8>,
                S<4, 4, 1>,
                1,
                1,
                ck::BlockGemmPipelineScheduler::Intrawave,
                ck::BlockGemmPipelineVersion::v3,
                ck::tensor_operation::device::GemmSpecialization::Default>;
            // Run kernel instance.
            return batched_gemm_bf16_impl<DeviceGemmInstance>(XQ, WQ, Y, bias, KBatch);
        }

        // no pad
    }
}

