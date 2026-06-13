// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "batched_gemm_bf16_common.cuh"

torch::Tensor
bf16_batched_64x16x16x64_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4x4x1_1x1_interwave_v2(
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
    bool pad = (M % 16 != 0) || (N % 16 != 0) || (K % (64 * KBatch) != 0);
#else
    // Disable padding for packed tensor
    bool pad = false;
#endif
    if (pad)
    {
        // pad
        
        {
           using DeviceGemmInstance = DeviceGemmHelper<
                64,
                16,
                16,
                64,
                16,
                16,
                1,
                1,
                S<8, 8, 1>,
                S<8, 8, 1>,
                S<1, 16, 1, 4>,
                S<4, 4, 1>,
                1,
                1,
                ck::BlockGemmPipelineScheduler::Interwave,
                ck::BlockGemmPipelineVersion::v2,
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
                64,
                16,
                16,
                64,
                16,
                16,
                1,
                1,
                S<8, 8, 1>,
                S<8, 8, 1>,
                S<1, 16, 1, 4>,
                S<4, 4, 1>,
                1,
                1,
                ck::BlockGemmPipelineScheduler::Interwave,
                ck::BlockGemmPipelineVersion::v2,
                ck::tensor_operation::device::GemmSpecialization::Default>;
            // Run kernel instance.
            return batched_gemm_bf16_impl<DeviceGemmInstance>(XQ, WQ, Y, bias, KBatch);
        }

        // no pad
    }
}

