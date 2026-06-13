#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include "moe_cktile2stages.h"
#include "moe_cktile2stages_heuristic_dispatch_common.h"

template <>
struct moe_gemm1_heuristic_dispatcher<ck_tile::bf16_t, ck_tile::pk_fp4_t, float, ck_tile::bf16_t, 2, true, true>
{
    static MoeKernel dispatch(int M, int N, int K, int block_m)
    {
        // Apply shape heuristics to find a suitable kernel implementation.
        if (block_m == 16)
        {
            return moe_cktile2stages_gemm1_256x16x128x256_1x4_16x16x32_2perCU_1x32_HasBias_swiglu_SplitK<ck_tile::bf16_t, ck_tile::pk_fp4_t, float, ck_tile::bf16_t>;
        }
        else if (block_m == 32)
        {
            return moe_cktile2stages_gemm1_256x32x256x256_1x4_16x16x32_2perCU_1x32_HasBias_swiglu_SplitK<ck_tile::bf16_t, ck_tile::pk_fp4_t, float, ck_tile::bf16_t>;
        }
        else if (block_m == 64)
        {
            return moe_cktile2stages_gemm1_256x64x256x256_1x4_16x16x32_1perCU_1x32_HasBias_swiglu_SplitK<ck_tile::bf16_t, ck_tile::pk_fp4_t, float, ck_tile::bf16_t>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe_geem1 heuristic dispatch: ",
                block_m);
        }
    }
};

template <>
struct moe_gemm2_heuristic_dispatcher<ck_tile::bf16_t, ck_tile::pk_fp4_t, float, ck_tile::bf16_t, 2, true, true>
{
    static MoeKernel dispatch(int M, int N, int K, int block_m)
    {
        // Apply shape heuristics to find a suitable kernel implementation.
        if (block_m == 16)
        {
            return moe_cktile2stages_gemm2_256x16x128x256_1x4_16x16x32_2perCU_1x32_MulRoutedWeight_HasBias<ck_tile::bf16_t, ck_tile::pk_fp4_t, float, ck_tile::bf16_t>;
        }
        else if (block_m == 32)
        {
            return moe_cktile2stages_gemm2_256x32x256x256_1x4_16x16x32_2perCU_1x32_MulRoutedWeight_HasBias<ck_tile::bf16_t, ck_tile::pk_fp4_t, float, ck_tile::bf16_t>;
        }
        else if (block_m == 64)
        {
            return moe_cktile2stages_gemm2_256x64x256x256_1x4_16x16x32_1perCU_1x32_MulRoutedWeight_HasBias<ck_tile::bf16_t, ck_tile::pk_fp4_t, float, ck_tile::bf16_t>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe_gemm2 heuristic dispatch: ",
                block_m);
        }
    }
};
