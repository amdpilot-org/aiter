// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "gemm_a8w8_blockscale_common.cuh"

enum class GemmSpecialization {
    Default    = 0,
    MPadding   = 1,
    NPadding   = 2,
    KPadding   = 3,
    MNPadding  = 4,
    MKPadding  = 5,
    NKPadding  = 6,
    MNKPadding = 7
};

static const std::unordered_map<std::string, GemmSpecialization> g_gemm_spec_names{
    {"", GemmSpecialization::Default},
    {"M", GemmSpecialization::MPadding},
    {"N", GemmSpecialization::NPadding},
    {"K", GemmSpecialization::KPadding},
    {"MN", GemmSpecialization::MNPadding},
    {"MK", GemmSpecialization::MKPadding},
    {"NK", GemmSpecialization::NKPadding},
    {"MNK", GemmSpecialization::MNKPadding}
};

static GemmSpecialization GetGemmSpec(const int64_t m,
                               const int64_t n,
                               const int64_t k,
                               const int64_t m_per_block,
                               const int64_t n_per_block,
                               const int64_t k_per_block)
{
    auto IntegerDivideCeil = [](int x, int y) {
        return (x + y - size_t{1}) / y;
    };

    std::string spec = "";
    if (IntegerDivideCeil(m, m_per_block) * m_per_block - m != 0)
        spec += "M";
    if (IntegerDivideCeil(n, n_per_block) * n_per_block - n != 0)
        spec += "N";
    if (IntegerDivideCeil(k, k_per_block) * k_per_block - k != 0)
        spec += "K";

    return g_gemm_spec_names.at(spec);
}

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_blockscale_1x128x128_256x128x64x128_16x16_32x32_2x1_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y
    )
{
    // Get M, N, K from input tensors.
    int M = XQ.numel() / XQ.size(-1);
    int N = WQ.size(0);
    int K = WQ.size(1);

    // Get whether this input needs to be padded.
    auto gemm_spec = GetGemmSpec(M, N, K, 128, 64, 128);


    if(gemm_spec == GemmSpecialization::Default)
    {
        // Default
        using LegacyGemmInstance = DeviceLegacyGemmHelperF8BlockScale<
            DDataType, EDataType,
            256,
            1, 128, 128,
            128, 64, 128,
            16, 16,
            32, 32,
            2, 1,
            S<8, 32, 1>,
            S<8, 32, 1>,
            1,
            1,
            S<1, 32, 1, 8>,
            S<8>,
            ck::BlockGemmPipelineScheduler::Intrawave,
            ck::BlockGemmPipelineVersion::v3,
            ck::tensor_operation::device::GemmSpecialization::Default>;

        // Run kernel instance.
        return gemm_a8w8_blockscale_impl<DDataType, EDataType, LegacyGemmInstance>(XQ, WQ, x_scale, w_scale, Y);

    } else if(gemm_spec == GemmSpecialization::MPadding)
    {
        // MNK Padding
        using LegacyGemmInstance = DeviceLegacyGemmHelperF8BlockScale<
            DDataType, EDataType,
            256,
            1, 128, 128,
            128, 64, 128,
            16, 16,
            32, 32,
            2, 1,
            S<8, 32, 1>,
            S<8, 32, 1>,
            1,
            1,
            S<1, 32, 1, 8>,
            S<8>,
            ck::BlockGemmPipelineScheduler::Intrawave,
            ck::BlockGemmPipelineVersion::v3,
            ck::tensor_operation::device::GemmSpecialization::MPadding>;

        // Run kernel instance.
        return gemm_a8w8_blockscale_impl<DDataType, EDataType, LegacyGemmInstance>(XQ, WQ, x_scale, w_scale, Y);

    } else if(gemm_spec == GemmSpecialization::NPadding)
    {
        // N Padding
        using LegacyGemmInstance = DeviceLegacyGemmHelperF8BlockScale<
            DDataType, EDataType,
            256,
            1, 128, 128,
            128, 64, 128,
            16, 16,
            32, 32,
            2, 1,
            S<8, 32, 1>,
            S<8, 32, 1>,
            1,
            1,
            S<1, 32, 1, 8>,
            S<8>,
            ck::BlockGemmPipelineScheduler::Intrawave,
            ck::BlockGemmPipelineVersion::v3,
            ck::tensor_operation::device::GemmSpecialization::NPadding>;

        // Run kernel instance.
        return gemm_a8w8_blockscale_impl<DDataType, EDataType, LegacyGemmInstance>(XQ, WQ, x_scale, w_scale, Y);

    } else if(gemm_spec == GemmSpecialization::KPadding)
    {
        // K Padding
        using LegacyGemmInstance = DeviceLegacyGemmHelperF8BlockScale<
            DDataType, EDataType,
            256,
            1, 128, 128,
            128, 64, 128,
            16, 16,
            32, 32,
            2, 1,
            S<8, 32, 1>,
            S<8, 32, 1>,
            1,
            1,
            S<1, 32, 1, 8>,
            S<8>,
            ck::BlockGemmPipelineScheduler::Intrawave,
            ck::BlockGemmPipelineVersion::v3,
            ck::tensor_operation::device::GemmSpecialization::KPadding>;

        // Run kernel instance.
        return gemm_a8w8_blockscale_impl<DDataType, EDataType, LegacyGemmInstance>(XQ, WQ, x_scale, w_scale, Y);

    } else if(gemm_spec == GemmSpecialization::MNPadding)
    {
        // MN Padding
        using LegacyGemmInstance = DeviceLegacyGemmHelperF8BlockScale<
            DDataType, EDataType,
            256,
            1, 128, 128,
            128, 64, 128,
            16, 16,
            32, 32,
            2, 1,
            S<8, 32, 1>,
            S<8, 32, 1>,
            1,
            1,
            S<1, 32, 1, 8>,
            S<8>,
            ck::BlockGemmPipelineScheduler::Intrawave,
            ck::BlockGemmPipelineVersion::v3,
            ck::tensor_operation::device::GemmSpecialization::MNPadding>;

        // Run kernel instance.
        return gemm_a8w8_blockscale_impl<DDataType, EDataType, LegacyGemmInstance>(XQ, WQ, x_scale, w_scale, Y);

    } else if(gemm_spec == GemmSpecialization::MKPadding)
    {
        // MK Padding
        using LegacyGemmInstance = DeviceLegacyGemmHelperF8BlockScale<
            DDataType, EDataType,
            256,
            1, 128, 128,
            128, 64, 128,
            16, 16,
            32, 32,
            2, 1,
            S<8, 32, 1>,
            S<8, 32, 1>,
            1,
            1,
            S<1, 32, 1, 8>,
            S<8>,
            ck::BlockGemmPipelineScheduler::Intrawave,
            ck::BlockGemmPipelineVersion::v3,
            ck::tensor_operation::device::GemmSpecialization::MKPadding>;

        // Run kernel instance.
        return gemm_a8w8_blockscale_impl<DDataType, EDataType, LegacyGemmInstance>(XQ, WQ, x_scale, w_scale, Y);

    } else if(gemm_spec == GemmSpecialization::NKPadding)
    {
        // NK Padding
        using LegacyGemmInstance = DeviceLegacyGemmHelperF8BlockScale<
            DDataType, EDataType,
            256,
            1, 128, 128,
            128, 64, 128,
            16, 16,
            32, 32,
            2, 1,
            S<8, 32, 1>,
            S<8, 32, 1>,
            1,
            1,
            S<1, 32, 1, 8>,
            S<8>,
            ck::BlockGemmPipelineScheduler::Intrawave,
            ck::BlockGemmPipelineVersion::v3,
            ck::tensor_operation::device::GemmSpecialization::NKPadding>;

        // Run kernel instance.
        return gemm_a8w8_blockscale_impl<DDataType, EDataType, LegacyGemmInstance>(XQ, WQ, x_scale, w_scale, Y);

    } else if(gemm_spec == GemmSpecialization::MNKPadding)
    {
        // MNK Padding
        using LegacyGemmInstance = DeviceLegacyGemmHelperF8BlockScale<
            DDataType, EDataType,
            256,
            1, 128, 128,
            128, 64, 128,
            16, 16,
            32, 32,
            2, 1,
            S<8, 32, 1>,
            S<8, 32, 1>,
            1,
            1,
            S<1, 32, 1, 8>,
            S<8>,
            ck::BlockGemmPipelineScheduler::Intrawave,
            ck::BlockGemmPipelineVersion::v3,
            ck::tensor_operation::device::GemmSpecialization::MNKPadding>;

        // Run kernel instance.
        return gemm_a8w8_blockscale_impl<DDataType, EDataType, LegacyGemmInstance>(XQ, WQ, x_scale, w_scale, Y);

    } else
    {
        throw std::runtime_error("Unsupported GemmSpecialization!");
    }
}

