// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "gemm_a4w4_blockscale_common.cuh"

template <typename CDataType>
torch::Tensor
a4w4_blockscale_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int splitK
    )
{
    // The smallest kernel we have available. Works well for memory bound shapes.

    // Check if this input needs to be padded.
    int M = size_to_dim_(XQ.dim() - 1, XQ.sizes());
    int N = WQ.size(0);
    int K = WQ.size(1);
    // TODO - padding
    using DeviceGemmInstance = DeviceGemmHelperF4BlockScale<
            MFMA,
            CDataType,
            256,
            128, 128, 128,
            16, 16,
            16, 16,
            8, 2,
            S<8, 32, 1>,
            S<8, 32, 1>,
            2,
            2,
            S<1, 32, 1, 8>,
            8,
            ck::BlockGemmPipelineScheduler::Intrawave,
            ck::BlockGemmPipelineVersion::v3,
            ck::tensor_operation::device::GemmSpecialization::Default>;
        // Run kernel instance.
        return gemm_a4w4_blockscale_impl<CDataType, DeviceGemmInstance>(XQ, WQ, x_scale, w_scale, Y, splitK);

}

