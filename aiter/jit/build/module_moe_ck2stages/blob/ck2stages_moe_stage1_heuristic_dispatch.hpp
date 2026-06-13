#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include "gemm_moe_ck2stages.h"

MoeKernel moe_stage1_heuristic_dispatch(int block_m, int inter_dim, at::ScalarType x_dtype, at::ScalarType w_dtype, at::ScalarType y_dtype, int act_op, int quant, bool mul_routed_weight_stage, bool is_shuffled)
{{

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<F8>{}(w_dtype)
        && dtype_checker<F16>{}(y_dtype)
        && 1 == act_op
        && true == mul_routed_weight_stage
        && 1 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 16, 64, 256/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 32, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 64)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 64, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V3, 256, 64, 128, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
        }
        else if (block_m == 128)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 128, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V3, 256, 128, 128, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
        }
        else if (block_m == 256)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 256, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V3, 256, 256, 128, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<F8>{}(w_dtype)
        && dtype_checker<F16>{}(y_dtype)
        && 1 == act_op
        && true == mul_routed_weight_stage
        && 1 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 16, 64, 256/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 32, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 64)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 64, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V3, 256, 64, 128, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
        }
        else if (block_m == 128)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 128, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V3, 256, 128, 128, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
        }
        else if (block_m == 256)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 256, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V3, 256, 256, 128, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<F8>{}(w_dtype)
        && dtype_checker<F16>{}(y_dtype)
        && 1 == act_op
        && true == mul_routed_weight_stage
        && 2 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 16, 64, 256/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 32, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 64)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 64, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V3, 256, 64, 128, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
        }
        else if (block_m == 128)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 128, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V3, 256, 128, 128, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
        }
        else if (block_m == 256)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 256, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V3, 256, 256, 128, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<F8>{}(w_dtype)
        && dtype_checker<F16>{}(y_dtype)
        && 1 == act_op
        && true == mul_routed_weight_stage
        && 2 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 16, 64, 256/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 32, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 64)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 64, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V3, 256, 64, 128, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
        }
        else if (block_m == 128)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 128, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V3, 256, 128, 128, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
        }
        else if (block_m == 256)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 256, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V3, 256, 256, 128, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<F8>{}(w_dtype)
        && dtype_checker<F16>{}(y_dtype)
        && 1 == act_op
        && false == mul_routed_weight_stage
        && 1 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 16, 64, 256/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 32, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 64)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 64, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V3, 256, 64, 128, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
        }
        else if (block_m == 128)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 128, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V3, 256, 128, 128, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
        }
        else if (block_m == 256)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 256, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V3, 256, 256, 128, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<F8>{}(w_dtype)
        && dtype_checker<F16>{}(y_dtype)
        && 1 == act_op
        && false == mul_routed_weight_stage
        && 1 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 16, 64, 256/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 32, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 64)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 64, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V3, 256, 64, 128, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
        }
        else if (block_m == 128)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 128, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V3, 256, 128, 128, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
        }
        else if (block_m == 256)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 256, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V3, 256, 256, 128, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<F8>{}(w_dtype)
        && dtype_checker<F16>{}(y_dtype)
        && 1 == act_op
        && false == mul_routed_weight_stage
        && 2 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 16, 64, 256/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 32, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 64)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 64, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V3, 256, 64, 128, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
        }
        else if (block_m == 128)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 128, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V3, 256, 128, 128, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
        }
        else if (block_m == 256)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 256, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V3, 256, 256, 128, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<F8>{}(w_dtype)
        && dtype_checker<F16>{}(y_dtype)
        && 1 == act_op
        && false == mul_routed_weight_stage
        && 2 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 16, 64, 256/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 32, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 64)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 64, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V3, 256, 64, 128, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
        }
        else if (block_m == 128)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 128, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V3, 256, 128, 128, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
        }
        else if (block_m == 256)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 256, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V3, 256, 256, 128, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<F8>{}(w_dtype)
        && dtype_checker<F16>{}(y_dtype)
        && 0 == act_op
        && true == mul_routed_weight_stage
        && 1 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 16, 64, 256/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 32, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 64)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 64, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V3, 256, 64, 128, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
        }
        else if (block_m == 128)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 128, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V3, 256, 128, 128, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
        }
        else if (block_m == 256)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 256, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V3, 256, 256, 128, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<F8>{}(w_dtype)
        && dtype_checker<F16>{}(y_dtype)
        && 0 == act_op
        && true == mul_routed_weight_stage
        && 1 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 16, 64, 256/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 32, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 64)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 64, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V3, 256, 64, 128, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
        }
        else if (block_m == 128)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 128, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V3, 256, 128, 128, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
        }
        else if (block_m == 256)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 256, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V3, 256, 256, 128, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<F8>{}(w_dtype)
        && dtype_checker<F16>{}(y_dtype)
        && 0 == act_op
        && true == mul_routed_weight_stage
        && 2 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 16, 64, 256/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 32, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 64)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 64, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V3, 256, 64, 128, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
        }
        else if (block_m == 128)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 128, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V3, 256, 128, 128, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
        }
        else if (block_m == 256)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 256, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V3, 256, 256, 128, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<F8>{}(w_dtype)
        && dtype_checker<F16>{}(y_dtype)
        && 0 == act_op
        && true == mul_routed_weight_stage
        && 2 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 16, 64, 256/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 32, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 64)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 64, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V3, 256, 64, 128, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
        }
        else if (block_m == 128)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 128, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V3, 256, 128, 128, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
        }
        else if (block_m == 256)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 256, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V3, 256, 256, 128, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<F8>{}(w_dtype)
        && dtype_checker<F16>{}(y_dtype)
        && 0 == act_op
        && false == mul_routed_weight_stage
        && 1 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 16, 64, 256/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 32, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 64)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 64, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V3, 256, 64, 128, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
        }
        else if (block_m == 128)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 128, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V3, 256, 128, 128, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
        }
        else if (block_m == 256)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 256, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V3, 256, 256, 128, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<F8>{}(w_dtype)
        && dtype_checker<F16>{}(y_dtype)
        && 0 == act_op
        && false == mul_routed_weight_stage
        && 1 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 16, 64, 256/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 32, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 64)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 64, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V3, 256, 64, 128, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
        }
        else if (block_m == 128)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 128, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V3, 256, 128, 128, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
        }
        else if (block_m == 256)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 256, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V3, 256, 256, 128, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<F8>{}(w_dtype)
        && dtype_checker<F16>{}(y_dtype)
        && 0 == act_op
        && false == mul_routed_weight_stage
        && 2 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 16, 64, 256/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 32, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 64)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 64, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V3, 256, 64, 128, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
        }
        else if (block_m == 128)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 128, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V3, 256, 128, 128, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
        }
        else if (block_m == 256)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 256, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V3, 256, 256, 128, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<F8>{}(w_dtype)
        && dtype_checker<F16>{}(y_dtype)
        && 0 == act_op
        && false == mul_routed_weight_stage
        && 2 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 16, 64, 256/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 32, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 64)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 64, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V3, 256, 64, 128, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
        }
        else if (block_m == 128)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 128, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V3, 256, 128, 128, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
        }
        else if (block_m == 256)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V1, 256, 256, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScale, V3, 256, 256, 128, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<F8>{}(w_dtype)
        && dtype_checker<B16>{}(y_dtype)
        && 1 == act_op
        && true == mul_routed_weight_stage
        && 1 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 16, 64, 256/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 32, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 64)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 64, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V3, 256, 64, 128, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
        }
        else if (block_m == 128)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 128, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V3, 256, 128, 128, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
        }
        else if (block_m == 256)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 256, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V3, 256, 256, 128, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<F8>{}(w_dtype)
        && dtype_checker<B16>{}(y_dtype)
        && 1 == act_op
        && true == mul_routed_weight_stage
        && 1 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 16, 64, 256/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 32, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 64)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 64, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V3, 256, 64, 128, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
        }
        else if (block_m == 128)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 128, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V3, 256, 128, 128, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
        }
        else if (block_m == 256)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 256, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V3, 256, 256, 128, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<F8>{}(w_dtype)
        && dtype_checker<B16>{}(y_dtype)
        && 1 == act_op
        && true == mul_routed_weight_stage
        && 2 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 16, 64, 256/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 32, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 64)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 64, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V3, 256, 64, 128, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
        }
        else if (block_m == 128)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 128, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V3, 256, 128, 128, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
        }
        else if (block_m == 256)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 256, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V3, 256, 256, 128, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<F8>{}(w_dtype)
        && dtype_checker<B16>{}(y_dtype)
        && 1 == act_op
        && true == mul_routed_weight_stage
        && 2 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 16, 64, 256/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 32, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 64)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 64, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V3, 256, 64, 128, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
        }
        else if (block_m == 128)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 128, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V3, 256, 128, 128, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
        }
        else if (block_m == 256)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 256, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V3, 256, 256, 128, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<F8>{}(w_dtype)
        && dtype_checker<B16>{}(y_dtype)
        && 1 == act_op
        && false == mul_routed_weight_stage
        && 1 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 16, 64, 256/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 32, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 64)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 64, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V3, 256, 64, 128, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
        }
        else if (block_m == 128)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 128, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V3, 256, 128, 128, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
        }
        else if (block_m == 256)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 256, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V3, 256, 256, 128, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<F8>{}(w_dtype)
        && dtype_checker<B16>{}(y_dtype)
        && 1 == act_op
        && false == mul_routed_weight_stage
        && 1 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 16, 64, 256/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 32, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 64)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 64, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V3, 256, 64, 128, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
        }
        else if (block_m == 128)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 128, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V3, 256, 128, 128, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
        }
        else if (block_m == 256)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 256, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V3, 256, 256, 128, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<F8>{}(w_dtype)
        && dtype_checker<B16>{}(y_dtype)
        && 1 == act_op
        && false == mul_routed_weight_stage
        && 2 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 16, 64, 256/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 32, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 64)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 64, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V3, 256, 64, 128, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
        }
        else if (block_m == 128)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 128, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V3, 256, 128, 128, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
        }
        else if (block_m == 256)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 256, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V3, 256, 256, 128, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<F8>{}(w_dtype)
        && dtype_checker<B16>{}(y_dtype)
        && 1 == act_op
        && false == mul_routed_weight_stage
        && 2 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 16, 64, 256/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 32, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 64)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 64, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V3, 256, 64, 128, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
        }
        else if (block_m == 128)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 128, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V3, 256, 128, 128, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
        }
        else if (block_m == 256)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 256, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V3, 256, 256, 128, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<F8>{}(w_dtype)
        && dtype_checker<B16>{}(y_dtype)
        && 0 == act_op
        && true == mul_routed_weight_stage
        && 1 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 16, 64, 256/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 32, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 64)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 64, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V3, 256, 64, 128, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
        }
        else if (block_m == 128)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 128, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V3, 256, 128, 128, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
        }
        else if (block_m == 256)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 256, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V3, 256, 256, 128, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<F8>{}(w_dtype)
        && dtype_checker<B16>{}(y_dtype)
        && 0 == act_op
        && true == mul_routed_weight_stage
        && 1 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 16, 64, 256/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 32, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 64)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 64, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V3, 256, 64, 128, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
        }
        else if (block_m == 128)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 128, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V3, 256, 128, 128, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
        }
        else if (block_m == 256)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 256, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V3, 256, 256, 128, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<F8>{}(w_dtype)
        && dtype_checker<B16>{}(y_dtype)
        && 0 == act_op
        && true == mul_routed_weight_stage
        && 2 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 16, 64, 256/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 32, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 64)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 64, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V3, 256, 64, 128, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
        }
        else if (block_m == 128)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 128, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V3, 256, 128, 128, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
        }
        else if (block_m == 256)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 256, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V3, 256, 256, 128, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<F8>{}(w_dtype)
        && dtype_checker<B16>{}(y_dtype)
        && 0 == act_op
        && true == mul_routed_weight_stage
        && 2 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 16, 64, 256/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 32, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 64)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 64, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V3, 256, 64, 128, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
        }
        else if (block_m == 128)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 128, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V3, 256, 128, 128, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
        }
        else if (block_m == 256)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 256, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V3, 256, 256, 128, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<F8>{}(w_dtype)
        && dtype_checker<B16>{}(y_dtype)
        && 0 == act_op
        && false == mul_routed_weight_stage
        && 1 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 16, 64, 256/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 32, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 64)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 64, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V3, 256, 64, 128, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
        }
        else if (block_m == 128)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 128, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V3, 256, 128, 128, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
        }
        else if (block_m == 256)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 256, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V3, 256, 256, 128, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<F8>{}(w_dtype)
        && dtype_checker<B16>{}(y_dtype)
        && 0 == act_op
        && false == mul_routed_weight_stage
        && 1 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 16, 64, 256/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 32, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 64)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 64, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V3, 256, 64, 128, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
        }
        else if (block_m == 128)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 128, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V3, 256, 128, 128, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
        }
        else if (block_m == 256)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 256, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V3, 256, 256, 128, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<F8>{}(w_dtype)
        && dtype_checker<B16>{}(y_dtype)
        && 0 == act_op
        && false == mul_routed_weight_stage
        && 2 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 16, 64, 256/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 32, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 64)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 64, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V3, 256, 64, 128, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
        }
        else if (block_m == 128)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 128, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V3, 256, 128, 128, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
        }
        else if (block_m == 256)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 256, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V3, 256, 256, 128, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<F8>{}(w_dtype)
        && dtype_checker<B16>{}(y_dtype)
        && 0 == act_op
        && false == mul_routed_weight_stage
        && 2 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 16, 64, 256/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 32, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 64)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 64, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V3, 256, 64, 128, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
        }
        else if (block_m == 128)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 128, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V3, 256, 128, 128, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
        }
        else if (block_m == 256)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V1, 256, 256, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScale, V3, 256, 256, 128, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<I8>{}(x_dtype)
        && dtype_checker<I8>{}(w_dtype)
        && dtype_checker<F16>{}(y_dtype)
        && 1 == act_op
        && true == mul_routed_weight_stage
        && 1 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 16, 64, 256/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 32, 64, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 64)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 64, 64, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V3, 256, 64, 128, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
        }
        else if (block_m == 128)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 128, 64, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V3, 256, 128, 128, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
        }
        else if (block_m == 256)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 256, 64, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V3, 256, 256, 128, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<I8>{}(x_dtype)
        && dtype_checker<I8>{}(w_dtype)
        && dtype_checker<F16>{}(y_dtype)
        && 1 == act_op
        && true == mul_routed_weight_stage
        && 1 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 16, 64, 256/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 32, 64, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 64)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 64, 64, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V3, 256, 64, 128, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
        }
        else if (block_m == 128)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 128, 64, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V3, 256, 128, 128, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
        }
        else if (block_m == 256)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 256, 64, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V3, 256, 256, 128, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<I8>{}(x_dtype)
        && dtype_checker<I8>{}(w_dtype)
        && dtype_checker<F16>{}(y_dtype)
        && 1 == act_op
        && true == mul_routed_weight_stage
        && 2 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 16, 64, 256/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 32, 64, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 64)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 64, 64, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V3, 256, 64, 128, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
        }
        else if (block_m == 128)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 128, 64, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V3, 256, 128, 128, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
        }
        else if (block_m == 256)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 256, 64, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V3, 256, 256, 128, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<I8>{}(x_dtype)
        && dtype_checker<I8>{}(w_dtype)
        && dtype_checker<F16>{}(y_dtype)
        && 1 == act_op
        && true == mul_routed_weight_stage
        && 2 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 16, 64, 256/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 32, 64, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 64)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 64, 64, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V3, 256, 64, 128, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
        }
        else if (block_m == 128)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 128, 64, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V3, 256, 128, 128, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
        }
        else if (block_m == 256)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 256, 64, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V3, 256, 256, 128, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<I8>{}(x_dtype)
        && dtype_checker<I8>{}(w_dtype)
        && dtype_checker<F16>{}(y_dtype)
        && 1 == act_op
        && false == mul_routed_weight_stage
        && 1 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 16, 64, 256/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 32, 64, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 64)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 64, 64, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V3, 256, 64, 128, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
        }
        else if (block_m == 128)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 128, 64, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V3, 256, 128, 128, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
        }
        else if (block_m == 256)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 256, 64, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V3, 256, 256, 128, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<I8>{}(x_dtype)
        && dtype_checker<I8>{}(w_dtype)
        && dtype_checker<F16>{}(y_dtype)
        && 1 == act_op
        && false == mul_routed_weight_stage
        && 1 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 16, 64, 256/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 32, 64, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 64)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 64, 64, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V3, 256, 64, 128, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
        }
        else if (block_m == 128)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 128, 64, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V3, 256, 128, 128, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
        }
        else if (block_m == 256)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 256, 64, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V3, 256, 256, 128, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<I8>{}(x_dtype)
        && dtype_checker<I8>{}(w_dtype)
        && dtype_checker<F16>{}(y_dtype)
        && 1 == act_op
        && false == mul_routed_weight_stage
        && 2 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 16, 64, 256/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 32, 64, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 64)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 64, 64, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V3, 256, 64, 128, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
        }
        else if (block_m == 128)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 128, 64, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V3, 256, 128, 128, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
        }
        else if (block_m == 256)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 256, 64, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V3, 256, 256, 128, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<I8>{}(x_dtype)
        && dtype_checker<I8>{}(w_dtype)
        && dtype_checker<F16>{}(y_dtype)
        && 1 == act_op
        && false == mul_routed_weight_stage
        && 2 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 16, 64, 256/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 32, 64, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 64)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 64, 64, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V3, 256, 64, 128, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
        }
        else if (block_m == 128)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 128, 64, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V3, 256, 128, 128, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
        }
        else if (block_m == 256)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 256, 64, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V3, 256, 256, 128, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<I8>{}(x_dtype)
        && dtype_checker<I8>{}(w_dtype)
        && dtype_checker<F16>{}(y_dtype)
        && 0 == act_op
        && true == mul_routed_weight_stage
        && 1 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 16, 64, 256/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 32, 64, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 64)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 64, 64, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V3, 256, 64, 128, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
        }
        else if (block_m == 128)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 128, 64, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V3, 256, 128, 128, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
        }
        else if (block_m == 256)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 256, 64, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V3, 256, 256, 128, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<I8>{}(x_dtype)
        && dtype_checker<I8>{}(w_dtype)
        && dtype_checker<F16>{}(y_dtype)
        && 0 == act_op
        && true == mul_routed_weight_stage
        && 1 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 16, 64, 256/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 32, 64, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 64)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 64, 64, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V3, 256, 64, 128, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
        }
        else if (block_m == 128)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 128, 64, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V3, 256, 128, 128, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
        }
        else if (block_m == 256)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 256, 64, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V3, 256, 256, 128, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<I8>{}(x_dtype)
        && dtype_checker<I8>{}(w_dtype)
        && dtype_checker<F16>{}(y_dtype)
        && 0 == act_op
        && true == mul_routed_weight_stage
        && 2 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 16, 64, 256/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 32, 64, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 64)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 64, 64, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V3, 256, 64, 128, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
        }
        else if (block_m == 128)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 128, 64, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V3, 256, 128, 128, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
        }
        else if (block_m == 256)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 256, 64, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V3, 256, 256, 128, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<I8>{}(x_dtype)
        && dtype_checker<I8>{}(w_dtype)
        && dtype_checker<F16>{}(y_dtype)
        && 0 == act_op
        && true == mul_routed_weight_stage
        && 2 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 16, 64, 256/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 32, 64, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 64)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 64, 64, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V3, 256, 64, 128, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
        }
        else if (block_m == 128)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 128, 64, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V3, 256, 128, 128, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
        }
        else if (block_m == 256)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 256, 64, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V3, 256, 256, 128, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<I8>{}(x_dtype)
        && dtype_checker<I8>{}(w_dtype)
        && dtype_checker<F16>{}(y_dtype)
        && 0 == act_op
        && false == mul_routed_weight_stage
        && 1 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 16, 64, 256/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 32, 64, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 64)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 64, 64, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V3, 256, 64, 128, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
        }
        else if (block_m == 128)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 128, 64, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V3, 256, 128, 128, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
        }
        else if (block_m == 256)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 256, 64, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V3, 256, 256, 128, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<I8>{}(x_dtype)
        && dtype_checker<I8>{}(w_dtype)
        && dtype_checker<F16>{}(y_dtype)
        && 0 == act_op
        && false == mul_routed_weight_stage
        && 1 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 16, 64, 256/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 32, 64, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 64)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 64, 64, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V3, 256, 64, 128, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
        }
        else if (block_m == 128)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 128, 64, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V3, 256, 128, 128, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
        }
        else if (block_m == 256)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 256, 64, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V3, 256, 256, 128, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<I8>{}(x_dtype)
        && dtype_checker<I8>{}(w_dtype)
        && dtype_checker<F16>{}(y_dtype)
        && 0 == act_op
        && false == mul_routed_weight_stage
        && 2 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 16, 64, 256/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 32, 64, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 64)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 64, 64, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V3, 256, 64, 128, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
        }
        else if (block_m == 128)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 128, 64, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V3, 256, 128, 128, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
        }
        else if (block_m == 256)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 256, 64, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V3, 256, 256, 128, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<I8>{}(x_dtype)
        && dtype_checker<I8>{}(w_dtype)
        && dtype_checker<F16>{}(y_dtype)
        && 0 == act_op
        && false == mul_routed_weight_stage
        && 2 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 16, 64, 256/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 32, 64, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 64)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 64, 64, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V3, 256, 64, 128, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
        }
        else if (block_m == 128)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 128, 64, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V3, 256, 128, 128, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
        }
        else if (block_m == 256)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V1, 256, 256, 64, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, F16, MulABScale, V3, 256, 256, 128, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<I8>{}(x_dtype)
        && dtype_checker<I8>{}(w_dtype)
        && dtype_checker<B16>{}(y_dtype)
        && 1 == act_op
        && true == mul_routed_weight_stage
        && 1 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 16, 64, 256/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 32, 64, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 64)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 64, 64, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V3, 256, 64, 128, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
        }
        else if (block_m == 128)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 128, 64, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V3, 256, 128, 128, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
        }
        else if (block_m == 256)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 256, 64, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V3, 256, 256, 128, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<I8>{}(x_dtype)
        && dtype_checker<I8>{}(w_dtype)
        && dtype_checker<B16>{}(y_dtype)
        && 1 == act_op
        && true == mul_routed_weight_stage
        && 1 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 16, 64, 256/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 32, 64, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 64)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 64, 64, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V3, 256, 64, 128, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
        }
        else if (block_m == 128)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 128, 64, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V3, 256, 128, 128, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
        }
        else if (block_m == 256)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 256, 64, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V3, 256, 256, 128, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<I8>{}(x_dtype)
        && dtype_checker<I8>{}(w_dtype)
        && dtype_checker<B16>{}(y_dtype)
        && 1 == act_op
        && true == mul_routed_weight_stage
        && 2 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 16, 64, 256/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 32, 64, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 64)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 64, 64, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V3, 256, 64, 128, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
        }
        else if (block_m == 128)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 128, 64, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V3, 256, 128, 128, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
        }
        else if (block_m == 256)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 256, 64, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V3, 256, 256, 128, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<I8>{}(x_dtype)
        && dtype_checker<I8>{}(w_dtype)
        && dtype_checker<B16>{}(y_dtype)
        && 1 == act_op
        && true == mul_routed_weight_stage
        && 2 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 16, 64, 256/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 32, 64, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 64)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 64, 64, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V3, 256, 64, 128, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
        }
        else if (block_m == 128)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 128, 64, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V3, 256, 128, 128, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
        }
        else if (block_m == 256)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 256, 64, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V3, 256, 256, 128, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<I8>{}(x_dtype)
        && dtype_checker<I8>{}(w_dtype)
        && dtype_checker<B16>{}(y_dtype)
        && 1 == act_op
        && false == mul_routed_weight_stage
        && 1 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 16, 64, 256/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 32, 64, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 64)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 64, 64, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V3, 256, 64, 128, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
        }
        else if (block_m == 128)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 128, 64, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V3, 256, 128, 128, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
        }
        else if (block_m == 256)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 256, 64, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V3, 256, 256, 128, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<I8>{}(x_dtype)
        && dtype_checker<I8>{}(w_dtype)
        && dtype_checker<B16>{}(y_dtype)
        && 1 == act_op
        && false == mul_routed_weight_stage
        && 1 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 16, 64, 256/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 32, 64, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 64)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 64, 64, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V3, 256, 64, 128, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
        }
        else if (block_m == 128)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 128, 64, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V3, 256, 128, 128, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
        }
        else if (block_m == 256)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 256, 64, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V3, 256, 256, 128, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<I8>{}(x_dtype)
        && dtype_checker<I8>{}(w_dtype)
        && dtype_checker<B16>{}(y_dtype)
        && 1 == act_op
        && false == mul_routed_weight_stage
        && 2 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 16, 64, 256/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 32, 64, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 64)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 64, 64, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V3, 256, 64, 128, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
        }
        else if (block_m == 128)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 128, 64, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V3, 256, 128, 128, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
        }
        else if (block_m == 256)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 256, 64, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V3, 256, 256, 128, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<I8>{}(x_dtype)
        && dtype_checker<I8>{}(w_dtype)
        && dtype_checker<B16>{}(y_dtype)
        && 1 == act_op
        && false == mul_routed_weight_stage
        && 2 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 16, 64, 256/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 32, 64, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 64)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 64, 64, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V3, 256, 64, 128, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
        }
        else if (block_m == 128)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 128, 64, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V3, 256, 128, 128, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
        }
        else if (block_m == 256)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 256, 64, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V3, 256, 256, 128, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<I8>{}(x_dtype)
        && dtype_checker<I8>{}(w_dtype)
        && dtype_checker<B16>{}(y_dtype)
        && 0 == act_op
        && true == mul_routed_weight_stage
        && 1 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 16, 64, 256/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 32, 64, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 64)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 64, 64, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V3, 256, 64, 128, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
        }
        else if (block_m == 128)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 128, 64, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V3, 256, 128, 128, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
        }
        else if (block_m == 256)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 256, 64, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V3, 256, 256, 128, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<I8>{}(x_dtype)
        && dtype_checker<I8>{}(w_dtype)
        && dtype_checker<B16>{}(y_dtype)
        && 0 == act_op
        && true == mul_routed_weight_stage
        && 1 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 16, 64, 256/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 32, 64, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 64)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 64, 64, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V3, 256, 64, 128, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
        }
        else if (block_m == 128)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 128, 64, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V3, 256, 128, 128, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
        }
        else if (block_m == 256)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 256, 64, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V3, 256, 256, 128, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<I8>{}(x_dtype)
        && dtype_checker<I8>{}(w_dtype)
        && dtype_checker<B16>{}(y_dtype)
        && 0 == act_op
        && true == mul_routed_weight_stage
        && 2 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 16, 64, 256/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 32, 64, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 64)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 64, 64, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V3, 256, 64, 128, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
        }
        else if (block_m == 128)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 128, 64, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V3, 256, 128, 128, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
        }
        else if (block_m == 256)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 256, 64, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V3, 256, 256, 128, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<I8>{}(x_dtype)
        && dtype_checker<I8>{}(w_dtype)
        && dtype_checker<B16>{}(y_dtype)
        && 0 == act_op
        && true == mul_routed_weight_stage
        && 2 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 16, 64, 256/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 32, 64, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 64)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 64, 64, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V3, 256, 64, 128, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
        }
        else if (block_m == 128)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 128, 64, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V3, 256, 128, 128, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
        }
        else if (block_m == 256)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 256, 64, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V3, 256, 256, 128, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<I8>{}(x_dtype)
        && dtype_checker<I8>{}(w_dtype)
        && dtype_checker<B16>{}(y_dtype)
        && 0 == act_op
        && false == mul_routed_weight_stage
        && 1 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 16, 64, 256/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 32, 64, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 64)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 64, 64, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V3, 256, 64, 128, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
        }
        else if (block_m == 128)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 128, 64, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V3, 256, 128, 128, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
        }
        else if (block_m == 256)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 256, 64, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V3, 256, 256, 128, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<I8>{}(x_dtype)
        && dtype_checker<I8>{}(w_dtype)
        && dtype_checker<B16>{}(y_dtype)
        && 0 == act_op
        && false == mul_routed_weight_stage
        && 1 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 16, 64, 256/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 32, 64, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 64)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 64, 64, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V3, 256, 64, 128, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
        }
        else if (block_m == 128)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 128, 64, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V3, 256, 128, 128, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
        }
        else if (block_m == 256)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 256, 64, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V3, 256, 256, 128, 128/sizeof(I8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<I8>{}(x_dtype)
        && dtype_checker<I8>{}(w_dtype)
        && dtype_checker<B16>{}(y_dtype)
        && 0 == act_op
        && false == mul_routed_weight_stage
        && 2 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 16, 64, 256/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 32, 64, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 64)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 64, 64, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V3, 256, 64, 128, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
        }
        else if (block_m == 128)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 128, 64, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V3, 256, 128, 128, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
        }
        else if (block_m == 256)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 256, 64, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V3, 256, 256, 128, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<I8>{}(x_dtype)
        && dtype_checker<I8>{}(w_dtype)
        && dtype_checker<B16>{}(y_dtype)
        && 0 == act_op
        && false == mul_routed_weight_stage
        && 2 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 16, 64, 256/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 32, 64, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 64)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 64, 64, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V3, 256, 64, 128, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
        }
        else if (block_m == 128)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 128, 64, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V3, 256, 128, 128, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
        }
        else if (block_m == 256)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V1, 256, 256, 64, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<I8, I8, I32, B16, MulABScale, V3, 256, 256, 128, 128/sizeof(I8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<I4>{}(w_dtype)
        && dtype_checker<F16>{}(y_dtype)
        && 1 == act_op
        && true == mul_routed_weight_stage
        && 1 == quant)
    {
        if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, F16, MulABScaleWint4, V1, 256, 32, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, F16, MulABScaleWint4, V1, 256, 64, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 128)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, F16, MulABScaleWint4, V1, 256, 128, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 256)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, F16, MulABScaleWint4, V1, 256, 256, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<I4>{}(w_dtype)
        && dtype_checker<F16>{}(y_dtype)
        && 1 == act_op
        && true == mul_routed_weight_stage
        && 1 == quant)
    {
        if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, F16, MulABScaleWint4, V1, 256, 32, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, F16, MulABScaleWint4, V1, 256, 64, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 128)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, F16, MulABScaleWint4, V1, 256, 128, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 256)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, F16, MulABScaleWint4, V1, 256, 256, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<I4>{}(w_dtype)
        && dtype_checker<F16>{}(y_dtype)
        && 1 == act_op
        && true == mul_routed_weight_stage
        && 2 == quant)
    {
        if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, F16, MulABScaleWint4, V1, 256, 32, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, F16, MulABScaleWint4, V1, 256, 64, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 128)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, F16, MulABScaleWint4, V1, 256, 128, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 256)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, F16, MulABScaleWint4, V1, 256, 256, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<I4>{}(w_dtype)
        && dtype_checker<F16>{}(y_dtype)
        && 1 == act_op
        && true == mul_routed_weight_stage
        && 2 == quant)
    {
        if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, F16, MulABScaleWint4, V1, 256, 32, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, F16, MulABScaleWint4, V1, 256, 64, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 128)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, F16, MulABScaleWint4, V1, 256, 128, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 256)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, F16, MulABScaleWint4, V1, 256, 256, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<I4>{}(w_dtype)
        && dtype_checker<F16>{}(y_dtype)
        && 1 == act_op
        && false == mul_routed_weight_stage
        && 1 == quant)
    {
        if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, F16, MulABScaleWint4, V1, 256, 32, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, F16, MulABScaleWint4, V1, 256, 64, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 128)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, F16, MulABScaleWint4, V1, 256, 128, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 256)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, F16, MulABScaleWint4, V1, 256, 256, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<I4>{}(w_dtype)
        && dtype_checker<F16>{}(y_dtype)
        && 1 == act_op
        && false == mul_routed_weight_stage
        && 1 == quant)
    {
        if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, F16, MulABScaleWint4, V1, 256, 32, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, F16, MulABScaleWint4, V1, 256, 64, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 128)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, F16, MulABScaleWint4, V1, 256, 128, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 256)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, F16, MulABScaleWint4, V1, 256, 256, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<I4>{}(w_dtype)
        && dtype_checker<F16>{}(y_dtype)
        && 1 == act_op
        && false == mul_routed_weight_stage
        && 2 == quant)
    {
        if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, F16, MulABScaleWint4, V1, 256, 32, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, F16, MulABScaleWint4, V1, 256, 64, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 128)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, F16, MulABScaleWint4, V1, 256, 128, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 256)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, F16, MulABScaleWint4, V1, 256, 256, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<I4>{}(w_dtype)
        && dtype_checker<F16>{}(y_dtype)
        && 1 == act_op
        && false == mul_routed_weight_stage
        && 2 == quant)
    {
        if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, F16, MulABScaleWint4, V1, 256, 32, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, F16, MulABScaleWint4, V1, 256, 64, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 128)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, F16, MulABScaleWint4, V1, 256, 128, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 256)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, F16, MulABScaleWint4, V1, 256, 256, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<I4>{}(w_dtype)
        && dtype_checker<F16>{}(y_dtype)
        && 0 == act_op
        && true == mul_routed_weight_stage
        && 1 == quant)
    {
        if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, F16, MulABScaleWint4, V1, 256, 32, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, F16, MulABScaleWint4, V1, 256, 64, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 128)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, F16, MulABScaleWint4, V1, 256, 128, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 256)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, F16, MulABScaleWint4, V1, 256, 256, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<I4>{}(w_dtype)
        && dtype_checker<F16>{}(y_dtype)
        && 0 == act_op
        && true == mul_routed_weight_stage
        && 1 == quant)
    {
        if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, F16, MulABScaleWint4, V1, 256, 32, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, F16, MulABScaleWint4, V1, 256, 64, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 128)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, F16, MulABScaleWint4, V1, 256, 128, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 256)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, F16, MulABScaleWint4, V1, 256, 256, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<I4>{}(w_dtype)
        && dtype_checker<F16>{}(y_dtype)
        && 0 == act_op
        && true == mul_routed_weight_stage
        && 2 == quant)
    {
        if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, F16, MulABScaleWint4, V1, 256, 32, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, F16, MulABScaleWint4, V1, 256, 64, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 128)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, F16, MulABScaleWint4, V1, 256, 128, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 256)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, F16, MulABScaleWint4, V1, 256, 256, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<I4>{}(w_dtype)
        && dtype_checker<F16>{}(y_dtype)
        && 0 == act_op
        && true == mul_routed_weight_stage
        && 2 == quant)
    {
        if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, F16, MulABScaleWint4, V1, 256, 32, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, F16, MulABScaleWint4, V1, 256, 64, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 128)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, F16, MulABScaleWint4, V1, 256, 128, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 256)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, F16, MulABScaleWint4, V1, 256, 256, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<I4>{}(w_dtype)
        && dtype_checker<F16>{}(y_dtype)
        && 0 == act_op
        && false == mul_routed_weight_stage
        && 1 == quant)
    {
        if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, F16, MulABScaleWint4, V1, 256, 32, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, F16, MulABScaleWint4, V1, 256, 64, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 128)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, F16, MulABScaleWint4, V1, 256, 128, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 256)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, F16, MulABScaleWint4, V1, 256, 256, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<I4>{}(w_dtype)
        && dtype_checker<F16>{}(y_dtype)
        && 0 == act_op
        && false == mul_routed_weight_stage
        && 1 == quant)
    {
        if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, F16, MulABScaleWint4, V1, 256, 32, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, F16, MulABScaleWint4, V1, 256, 64, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 128)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, F16, MulABScaleWint4, V1, 256, 128, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 256)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, F16, MulABScaleWint4, V1, 256, 256, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<I4>{}(w_dtype)
        && dtype_checker<F16>{}(y_dtype)
        && 0 == act_op
        && false == mul_routed_weight_stage
        && 2 == quant)
    {
        if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, F16, MulABScaleWint4, V1, 256, 32, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, F16, MulABScaleWint4, V1, 256, 64, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 128)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, F16, MulABScaleWint4, V1, 256, 128, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 256)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, F16, MulABScaleWint4, V1, 256, 256, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<I4>{}(w_dtype)
        && dtype_checker<F16>{}(y_dtype)
        && 0 == act_op
        && false == mul_routed_weight_stage
        && 2 == quant)
    {
        if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, F16, MulABScaleWint4, V1, 256, 32, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, F16, MulABScaleWint4, V1, 256, 64, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 128)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, F16, MulABScaleWint4, V1, 256, 128, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 256)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, F16, MulABScaleWint4, V1, 256, 256, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<I4>{}(w_dtype)
        && dtype_checker<B16>{}(y_dtype)
        && 1 == act_op
        && true == mul_routed_weight_stage
        && 1 == quant)
    {
        if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, B16, MulABScaleWint4, V1, 256, 32, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, B16, MulABScaleWint4, V1, 256, 64, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 128)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, B16, MulABScaleWint4, V1, 256, 128, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 256)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, B16, MulABScaleWint4, V1, 256, 256, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<I4>{}(w_dtype)
        && dtype_checker<B16>{}(y_dtype)
        && 1 == act_op
        && true == mul_routed_weight_stage
        && 1 == quant)
    {
        if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, B16, MulABScaleWint4, V1, 256, 32, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, B16, MulABScaleWint4, V1, 256, 64, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 128)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, B16, MulABScaleWint4, V1, 256, 128, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 256)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, B16, MulABScaleWint4, V1, 256, 256, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<I4>{}(w_dtype)
        && dtype_checker<B16>{}(y_dtype)
        && 1 == act_op
        && true == mul_routed_weight_stage
        && 2 == quant)
    {
        if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, B16, MulABScaleWint4, V1, 256, 32, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, B16, MulABScaleWint4, V1, 256, 64, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 128)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, B16, MulABScaleWint4, V1, 256, 128, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 256)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, B16, MulABScaleWint4, V1, 256, 256, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<I4>{}(w_dtype)
        && dtype_checker<B16>{}(y_dtype)
        && 1 == act_op
        && true == mul_routed_weight_stage
        && 2 == quant)
    {
        if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, B16, MulABScaleWint4, V1, 256, 32, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, B16, MulABScaleWint4, V1, 256, 64, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 128)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, B16, MulABScaleWint4, V1, 256, 128, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 256)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, B16, MulABScaleWint4, V1, 256, 256, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<I4>{}(w_dtype)
        && dtype_checker<B16>{}(y_dtype)
        && 1 == act_op
        && false == mul_routed_weight_stage
        && 1 == quant)
    {
        if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, B16, MulABScaleWint4, V1, 256, 32, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, B16, MulABScaleWint4, V1, 256, 64, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 128)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, B16, MulABScaleWint4, V1, 256, 128, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 256)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, B16, MulABScaleWint4, V1, 256, 256, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<I4>{}(w_dtype)
        && dtype_checker<B16>{}(y_dtype)
        && 1 == act_op
        && false == mul_routed_weight_stage
        && 1 == quant)
    {
        if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, B16, MulABScaleWint4, V1, 256, 32, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, B16, MulABScaleWint4, V1, 256, 64, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 128)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, B16, MulABScaleWint4, V1, 256, 128, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 256)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, B16, MulABScaleWint4, V1, 256, 256, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<I4>{}(w_dtype)
        && dtype_checker<B16>{}(y_dtype)
        && 1 == act_op
        && false == mul_routed_weight_stage
        && 2 == quant)
    {
        if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, B16, MulABScaleWint4, V1, 256, 32, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, B16, MulABScaleWint4, V1, 256, 64, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 128)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, B16, MulABScaleWint4, V1, 256, 128, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 256)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, B16, MulABScaleWint4, V1, 256, 256, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<I4>{}(w_dtype)
        && dtype_checker<B16>{}(y_dtype)
        && 1 == act_op
        && false == mul_routed_weight_stage
        && 2 == quant)
    {
        if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, B16, MulABScaleWint4, V1, 256, 32, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, B16, MulABScaleWint4, V1, 256, 64, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 128)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, B16, MulABScaleWint4, V1, 256, 128, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 256)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, B16, MulABScaleWint4, V1, 256, 256, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<I4>{}(w_dtype)
        && dtype_checker<B16>{}(y_dtype)
        && 0 == act_op
        && true == mul_routed_weight_stage
        && 1 == quant)
    {
        if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, B16, MulABScaleWint4, V1, 256, 32, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, B16, MulABScaleWint4, V1, 256, 64, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 128)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, B16, MulABScaleWint4, V1, 256, 128, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 256)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, B16, MulABScaleWint4, V1, 256, 256, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<I4>{}(w_dtype)
        && dtype_checker<B16>{}(y_dtype)
        && 0 == act_op
        && true == mul_routed_weight_stage
        && 1 == quant)
    {
        if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, B16, MulABScaleWint4, V1, 256, 32, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, B16, MulABScaleWint4, V1, 256, 64, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 128)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, B16, MulABScaleWint4, V1, 256, 128, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 256)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, B16, MulABScaleWint4, V1, 256, 256, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<I4>{}(w_dtype)
        && dtype_checker<B16>{}(y_dtype)
        && 0 == act_op
        && true == mul_routed_weight_stage
        && 2 == quant)
    {
        if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, B16, MulABScaleWint4, V1, 256, 32, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, B16, MulABScaleWint4, V1, 256, 64, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 128)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, B16, MulABScaleWint4, V1, 256, 128, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 256)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, B16, MulABScaleWint4, V1, 256, 256, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<I4>{}(w_dtype)
        && dtype_checker<B16>{}(y_dtype)
        && 0 == act_op
        && true == mul_routed_weight_stage
        && 2 == quant)
    {
        if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, B16, MulABScaleWint4, V1, 256, 32, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, B16, MulABScaleWint4, V1, 256, 64, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 128)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, B16, MulABScaleWint4, V1, 256, 128, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 256)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, B16, MulABScaleWint4, V1, 256, 256, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<I4>{}(w_dtype)
        && dtype_checker<B16>{}(y_dtype)
        && 0 == act_op
        && false == mul_routed_weight_stage
        && 1 == quant)
    {
        if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, B16, MulABScaleWint4, V1, 256, 32, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, B16, MulABScaleWint4, V1, 256, 64, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 128)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, B16, MulABScaleWint4, V1, 256, 128, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 256)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, B16, MulABScaleWint4, V1, 256, 256, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<I4>{}(w_dtype)
        && dtype_checker<B16>{}(y_dtype)
        && 0 == act_op
        && false == mul_routed_weight_stage
        && 1 == quant)
    {
        if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, B16, MulABScaleWint4, V1, 256, 32, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, B16, MulABScaleWint4, V1, 256, 64, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 128)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, B16, MulABScaleWint4, V1, 256, 128, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 256)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, B16, MulABScaleWint4, V1, 256, 256, 64, 128/sizeof(F8), 1, 4, false, 1 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<I4>{}(w_dtype)
        && dtype_checker<B16>{}(y_dtype)
        && 0 == act_op
        && false == mul_routed_weight_stage
        && 2 == quant)
    {
        if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, B16, MulABScaleWint4, V1, 256, 32, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, B16, MulABScaleWint4, V1, 256, 64, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 128)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, B16, MulABScaleWint4, V1, 256, 128, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 256)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, B16, MulABScaleWint4, V1, 256, 256, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<I4>{}(w_dtype)
        && dtype_checker<B16>{}(y_dtype)
        && 0 == act_op
        && false == mul_routed_weight_stage
        && 2 == quant)
    {
        if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, B16, MulABScaleWint4, V1, 256, 32, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, B16, MulABScaleWint4, V1, 256, 64, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 128)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, B16, MulABScaleWint4, V1, 256, 128, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 256)
        {
            return ck_moe_stage1_gemm<F8, I4, F32, B16, MulABScaleWint4, V1, 256, 256, 64, 128/sizeof(F8), 1, 4, false, 2 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

#if defined(__Float4_e2m1fn_x2)
    if (dtype_checker<FP4X2>{}(x_dtype)
        && dtype_checker<FP4X2>{}(w_dtype)
        && dtype_checker<F16>{}(y_dtype)
        && 1 == act_op
        && true == mul_routed_weight_stage
        && 3 == quant
        && true == is_shuffled)
    {
        if (block_m == 32)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, F16, MulABScaleShuffled, V3, 64, 32, 32, 128/sizeof(FP4X2), 1, 1, false, 3 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, F16, MulABScaleShuffled, V3, 256, 64, 128, 128/sizeof(FP4X2), 1, 4, false, 3 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 128)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, F16, MulABScaleShuffled, V3, 256, 128, 128, 128/sizeof(FP4X2), 1, 4, false, 3 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }
#endif


#if defined(__Float4_e2m1fn_x2)
    if (dtype_checker<FP4X2>{}(x_dtype)
        && dtype_checker<FP4X2>{}(w_dtype)
        && dtype_checker<F16>{}(y_dtype)
        && 1 == act_op
        && true == mul_routed_weight_stage
        && 3 == quant
        && false == is_shuffled)
    {
        if (block_m == 32)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, F16, MulABScale, V3, 256, 32, 128, 128/sizeof(FP4X2), 1, 4, false, 3 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, F16, MulABScale, V3, 256, 64, 64, 128/sizeof(FP4X2), 2, 2, false, 3 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 128)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, F16, MulABScale, V3, 256, 128, 64, 128/sizeof(FP4X2), 2, 2, false, 3 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }
#endif


#if defined(__Float4_e2m1fn_x2)
    if (dtype_checker<FP4X2>{}(x_dtype)
        && dtype_checker<FP4X2>{}(w_dtype)
        && dtype_checker<F16>{}(y_dtype)
        && 1 == act_op
        && true == mul_routed_weight_stage
        && 3 == quant
        && true == is_shuffled)
    {
        if (block_m == 32)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, F16, MulABScaleShuffled, V3, 64, 32, 32, 128/sizeof(FP4X2), 1, 1, false, 3 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, F16, MulABScaleShuffled, V3, 256, 64, 128, 128/sizeof(FP4X2), 1, 4, false, 3 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 128)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, F16, MulABScaleShuffled, V3, 256, 128, 128, 128/sizeof(FP4X2), 1, 4, false, 3 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }
#endif


#if defined(__Float4_e2m1fn_x2)
    if (dtype_checker<FP4X2>{}(x_dtype)
        && dtype_checker<FP4X2>{}(w_dtype)
        && dtype_checker<F16>{}(y_dtype)
        && 1 == act_op
        && true == mul_routed_weight_stage
        && 3 == quant
        && false == is_shuffled)
    {
        if (block_m == 32)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, F16, MulABScale, V3, 256, 32, 128, 128/sizeof(FP4X2), 1, 4, false, 3 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, F16, MulABScale, V3, 256, 64, 64, 128/sizeof(FP4X2), 2, 2, false, 3 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 128)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, F16, MulABScale, V3, 256, 128, 64, 128/sizeof(FP4X2), 2, 2, false, 3 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }
#endif


#if defined(__Float4_e2m1fn_x2)
    if (dtype_checker<FP4X2>{}(x_dtype)
        && dtype_checker<FP4X2>{}(w_dtype)
        && dtype_checker<F16>{}(y_dtype)
        && 1 == act_op
        && false == mul_routed_weight_stage
        && 3 == quant
        && true == is_shuffled)
    {
        if (block_m == 32)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, F16, MulABScaleShuffled, V3, 64, 32, 32, 128/sizeof(FP4X2), 1, 1, false, 3 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, F16, MulABScaleShuffled, V3, 256, 64, 128, 128/sizeof(FP4X2), 1, 4, false, 3 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 128)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, F16, MulABScaleShuffled, V3, 256, 128, 128, 128/sizeof(FP4X2), 1, 4, false, 3 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }
#endif


#if defined(__Float4_e2m1fn_x2)
    if (dtype_checker<FP4X2>{}(x_dtype)
        && dtype_checker<FP4X2>{}(w_dtype)
        && dtype_checker<F16>{}(y_dtype)
        && 1 == act_op
        && false == mul_routed_weight_stage
        && 3 == quant
        && false == is_shuffled)
    {
        if (block_m == 32)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, F16, MulABScale, V3, 256, 32, 128, 128/sizeof(FP4X2), 1, 4, false, 3 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, F16, MulABScale, V3, 256, 64, 64, 128/sizeof(FP4X2), 2, 2, false, 3 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 128)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, F16, MulABScale, V3, 256, 128, 64, 128/sizeof(FP4X2), 2, 2, false, 3 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }
#endif


#if defined(__Float4_e2m1fn_x2)
    if (dtype_checker<FP4X2>{}(x_dtype)
        && dtype_checker<FP4X2>{}(w_dtype)
        && dtype_checker<F16>{}(y_dtype)
        && 1 == act_op
        && false == mul_routed_weight_stage
        && 3 == quant
        && true == is_shuffled)
    {
        if (block_m == 32)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, F16, MulABScaleShuffled, V3, 64, 32, 32, 128/sizeof(FP4X2), 1, 1, false, 3 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, F16, MulABScaleShuffled, V3, 256, 64, 128, 128/sizeof(FP4X2), 1, 4, false, 3 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 128)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, F16, MulABScaleShuffled, V3, 256, 128, 128, 128/sizeof(FP4X2), 1, 4, false, 3 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }
#endif


#if defined(__Float4_e2m1fn_x2)
    if (dtype_checker<FP4X2>{}(x_dtype)
        && dtype_checker<FP4X2>{}(w_dtype)
        && dtype_checker<F16>{}(y_dtype)
        && 1 == act_op
        && false == mul_routed_weight_stage
        && 3 == quant
        && false == is_shuffled)
    {
        if (block_m == 32)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, F16, MulABScale, V3, 256, 32, 128, 128/sizeof(FP4X2), 1, 4, false, 3 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, F16, MulABScale, V3, 256, 64, 64, 128/sizeof(FP4X2), 2, 2, false, 3 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 128)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, F16, MulABScale, V3, 256, 128, 64, 128/sizeof(FP4X2), 2, 2, false, 3 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }
#endif


#if defined(__Float4_e2m1fn_x2)
    if (dtype_checker<FP4X2>{}(x_dtype)
        && dtype_checker<FP4X2>{}(w_dtype)
        && dtype_checker<F16>{}(y_dtype)
        && 0 == act_op
        && true == mul_routed_weight_stage
        && 3 == quant
        && true == is_shuffled)
    {
        if (block_m == 32)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, F16, MulABScaleShuffled, V3, 64, 32, 32, 128/sizeof(FP4X2), 1, 1, false, 3 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, F16, MulABScaleShuffled, V3, 256, 64, 128, 128/sizeof(FP4X2), 1, 4, false, 3 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 128)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, F16, MulABScaleShuffled, V3, 256, 128, 128, 128/sizeof(FP4X2), 1, 4, false, 3 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }
#endif


#if defined(__Float4_e2m1fn_x2)
    if (dtype_checker<FP4X2>{}(x_dtype)
        && dtype_checker<FP4X2>{}(w_dtype)
        && dtype_checker<F16>{}(y_dtype)
        && 0 == act_op
        && true == mul_routed_weight_stage
        && 3 == quant
        && false == is_shuffled)
    {
        if (block_m == 32)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, F16, MulABScale, V3, 256, 32, 128, 128/sizeof(FP4X2), 1, 4, false, 3 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, F16, MulABScale, V3, 256, 64, 64, 128/sizeof(FP4X2), 2, 2, false, 3 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 128)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, F16, MulABScale, V3, 256, 128, 64, 128/sizeof(FP4X2), 2, 2, false, 3 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }
#endif


#if defined(__Float4_e2m1fn_x2)
    if (dtype_checker<FP4X2>{}(x_dtype)
        && dtype_checker<FP4X2>{}(w_dtype)
        && dtype_checker<F16>{}(y_dtype)
        && 0 == act_op
        && true == mul_routed_weight_stage
        && 3 == quant
        && true == is_shuffled)
    {
        if (block_m == 32)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, F16, MulABScaleShuffled, V3, 64, 32, 32, 128/sizeof(FP4X2), 1, 1, false, 3 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, F16, MulABScaleShuffled, V3, 256, 64, 128, 128/sizeof(FP4X2), 1, 4, false, 3 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 128)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, F16, MulABScaleShuffled, V3, 256, 128, 128, 128/sizeof(FP4X2), 1, 4, false, 3 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }
#endif


#if defined(__Float4_e2m1fn_x2)
    if (dtype_checker<FP4X2>{}(x_dtype)
        && dtype_checker<FP4X2>{}(w_dtype)
        && dtype_checker<F16>{}(y_dtype)
        && 0 == act_op
        && true == mul_routed_weight_stage
        && 3 == quant
        && false == is_shuffled)
    {
        if (block_m == 32)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, F16, MulABScale, V3, 256, 32, 128, 128/sizeof(FP4X2), 1, 4, false, 3 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, F16, MulABScale, V3, 256, 64, 64, 128/sizeof(FP4X2), 2, 2, false, 3 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 128)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, F16, MulABScale, V3, 256, 128, 64, 128/sizeof(FP4X2), 2, 2, false, 3 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }
#endif


#if defined(__Float4_e2m1fn_x2)
    if (dtype_checker<FP4X2>{}(x_dtype)
        && dtype_checker<FP4X2>{}(w_dtype)
        && dtype_checker<F16>{}(y_dtype)
        && 0 == act_op
        && false == mul_routed_weight_stage
        && 3 == quant
        && true == is_shuffled)
    {
        if (block_m == 32)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, F16, MulABScaleShuffled, V3, 64, 32, 32, 128/sizeof(FP4X2), 1, 1, false, 3 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, F16, MulABScaleShuffled, V3, 256, 64, 128, 128/sizeof(FP4X2), 1, 4, false, 3 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 128)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, F16, MulABScaleShuffled, V3, 256, 128, 128, 128/sizeof(FP4X2), 1, 4, false, 3 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }
#endif


#if defined(__Float4_e2m1fn_x2)
    if (dtype_checker<FP4X2>{}(x_dtype)
        && dtype_checker<FP4X2>{}(w_dtype)
        && dtype_checker<F16>{}(y_dtype)
        && 0 == act_op
        && false == mul_routed_weight_stage
        && 3 == quant
        && false == is_shuffled)
    {
        if (block_m == 32)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, F16, MulABScale, V3, 256, 32, 128, 128/sizeof(FP4X2), 1, 4, false, 3 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, F16, MulABScale, V3, 256, 64, 64, 128/sizeof(FP4X2), 2, 2, false, 3 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 128)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, F16, MulABScale, V3, 256, 128, 64, 128/sizeof(FP4X2), 2, 2, false, 3 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }
#endif


#if defined(__Float4_e2m1fn_x2)
    if (dtype_checker<FP4X2>{}(x_dtype)
        && dtype_checker<FP4X2>{}(w_dtype)
        && dtype_checker<F16>{}(y_dtype)
        && 0 == act_op
        && false == mul_routed_weight_stage
        && 3 == quant
        && true == is_shuffled)
    {
        if (block_m == 32)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, F16, MulABScaleShuffled, V3, 64, 32, 32, 128/sizeof(FP4X2), 1, 1, false, 3 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, F16, MulABScaleShuffled, V3, 256, 64, 128, 128/sizeof(FP4X2), 1, 4, false, 3 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 128)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, F16, MulABScaleShuffled, V3, 256, 128, 128, 128/sizeof(FP4X2), 1, 4, false, 3 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }
#endif


#if defined(__Float4_e2m1fn_x2)
    if (dtype_checker<FP4X2>{}(x_dtype)
        && dtype_checker<FP4X2>{}(w_dtype)
        && dtype_checker<F16>{}(y_dtype)
        && 0 == act_op
        && false == mul_routed_weight_stage
        && 3 == quant
        && false == is_shuffled)
    {
        if (block_m == 32)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, F16, MulABScale, V3, 256, 32, 128, 128/sizeof(FP4X2), 1, 4, false, 3 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, F16, MulABScale, V3, 256, 64, 64, 128/sizeof(FP4X2), 2, 2, false, 3 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 128)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, F16, MulABScale, V3, 256, 128, 64, 128/sizeof(FP4X2), 2, 2, false, 3 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }
#endif


#if defined(__Float4_e2m1fn_x2)
    if (dtype_checker<FP4X2>{}(x_dtype)
        && dtype_checker<FP4X2>{}(w_dtype)
        && dtype_checker<B16>{}(y_dtype)
        && 1 == act_op
        && true == mul_routed_weight_stage
        && 3 == quant
        && true == is_shuffled)
    {
        if (block_m == 32)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, B16, MulABScaleShuffled, V3, 64, 32, 32, 128/sizeof(FP4X2), 1, 1, false, 3 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, B16, MulABScaleShuffled, V3, 256, 64, 128, 128/sizeof(FP4X2), 1, 4, false, 3 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 128)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, B16, MulABScaleShuffled, V3, 256, 128, 128, 128/sizeof(FP4X2), 1, 4, false, 3 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }
#endif


#if defined(__Float4_e2m1fn_x2)
    if (dtype_checker<FP4X2>{}(x_dtype)
        && dtype_checker<FP4X2>{}(w_dtype)
        && dtype_checker<B16>{}(y_dtype)
        && 1 == act_op
        && true == mul_routed_weight_stage
        && 3 == quant
        && false == is_shuffled)
    {
        if (block_m == 32)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, B16, MulABScale, V3, 256, 32, 128, 128/sizeof(FP4X2), 1, 4, false, 3 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, B16, MulABScale, V3, 256, 64, 64, 128/sizeof(FP4X2), 2, 2, false, 3 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 128)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, B16, MulABScale, V3, 256, 128, 64, 128/sizeof(FP4X2), 2, 2, false, 3 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }
#endif


#if defined(__Float4_e2m1fn_x2)
    if (dtype_checker<FP4X2>{}(x_dtype)
        && dtype_checker<FP4X2>{}(w_dtype)
        && dtype_checker<B16>{}(y_dtype)
        && 1 == act_op
        && true == mul_routed_weight_stage
        && 3 == quant
        && true == is_shuffled)
    {
        if (block_m == 32)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, B16, MulABScaleShuffled, V3, 64, 32, 32, 128/sizeof(FP4X2), 1, 1, false, 3 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, B16, MulABScaleShuffled, V3, 256, 64, 128, 128/sizeof(FP4X2), 1, 4, false, 3 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 128)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, B16, MulABScaleShuffled, V3, 256, 128, 128, 128/sizeof(FP4X2), 1, 4, false, 3 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }
#endif


#if defined(__Float4_e2m1fn_x2)
    if (dtype_checker<FP4X2>{}(x_dtype)
        && dtype_checker<FP4X2>{}(w_dtype)
        && dtype_checker<B16>{}(y_dtype)
        && 1 == act_op
        && true == mul_routed_weight_stage
        && 3 == quant
        && false == is_shuffled)
    {
        if (block_m == 32)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, B16, MulABScale, V3, 256, 32, 128, 128/sizeof(FP4X2), 1, 4, false, 3 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, B16, MulABScale, V3, 256, 64, 64, 128/sizeof(FP4X2), 2, 2, false, 3 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 128)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, B16, MulABScale, V3, 256, 128, 64, 128/sizeof(FP4X2), 2, 2, false, 3 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }
#endif


#if defined(__Float4_e2m1fn_x2)
    if (dtype_checker<FP4X2>{}(x_dtype)
        && dtype_checker<FP4X2>{}(w_dtype)
        && dtype_checker<B16>{}(y_dtype)
        && 1 == act_op
        && false == mul_routed_weight_stage
        && 3 == quant
        && true == is_shuffled)
    {
        if (block_m == 32)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, B16, MulABScaleShuffled, V3, 64, 32, 32, 128/sizeof(FP4X2), 1, 1, false, 3 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, B16, MulABScaleShuffled, V3, 256, 64, 128, 128/sizeof(FP4X2), 1, 4, false, 3 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 128)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, B16, MulABScaleShuffled, V3, 256, 128, 128, 128/sizeof(FP4X2), 1, 4, false, 3 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }
#endif


#if defined(__Float4_e2m1fn_x2)
    if (dtype_checker<FP4X2>{}(x_dtype)
        && dtype_checker<FP4X2>{}(w_dtype)
        && dtype_checker<B16>{}(y_dtype)
        && 1 == act_op
        && false == mul_routed_weight_stage
        && 3 == quant
        && false == is_shuffled)
    {
        if (block_m == 32)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, B16, MulABScale, V3, 256, 32, 128, 128/sizeof(FP4X2), 1, 4, false, 3 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, B16, MulABScale, V3, 256, 64, 64, 128/sizeof(FP4X2), 2, 2, false, 3 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 128)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, B16, MulABScale, V3, 256, 128, 64, 128/sizeof(FP4X2), 2, 2, false, 3 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }
#endif


#if defined(__Float4_e2m1fn_x2)
    if (dtype_checker<FP4X2>{}(x_dtype)
        && dtype_checker<FP4X2>{}(w_dtype)
        && dtype_checker<B16>{}(y_dtype)
        && 1 == act_op
        && false == mul_routed_weight_stage
        && 3 == quant
        && true == is_shuffled)
    {
        if (block_m == 32)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, B16, MulABScaleShuffled, V3, 64, 32, 32, 128/sizeof(FP4X2), 1, 1, false, 3 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, B16, MulABScaleShuffled, V3, 256, 64, 128, 128/sizeof(FP4X2), 1, 4, false, 3 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 128)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, B16, MulABScaleShuffled, V3, 256, 128, 128, 128/sizeof(FP4X2), 1, 4, false, 3 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }
#endif


#if defined(__Float4_e2m1fn_x2)
    if (dtype_checker<FP4X2>{}(x_dtype)
        && dtype_checker<FP4X2>{}(w_dtype)
        && dtype_checker<B16>{}(y_dtype)
        && 1 == act_op
        && false == mul_routed_weight_stage
        && 3 == quant
        && false == is_shuffled)
    {
        if (block_m == 32)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, B16, MulABScale, V3, 256, 32, 128, 128/sizeof(FP4X2), 1, 4, false, 3 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, B16, MulABScale, V3, 256, 64, 64, 128/sizeof(FP4X2), 2, 2, false, 3 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 128)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, B16, MulABScale, V3, 256, 128, 64, 128/sizeof(FP4X2), 2, 2, false, 3 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }
#endif


#if defined(__Float4_e2m1fn_x2)
    if (dtype_checker<FP4X2>{}(x_dtype)
        && dtype_checker<FP4X2>{}(w_dtype)
        && dtype_checker<B16>{}(y_dtype)
        && 0 == act_op
        && true == mul_routed_weight_stage
        && 3 == quant
        && true == is_shuffled)
    {
        if (block_m == 32)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, B16, MulABScaleShuffled, V3, 64, 32, 32, 128/sizeof(FP4X2), 1, 1, false, 3 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, B16, MulABScaleShuffled, V3, 256, 64, 128, 128/sizeof(FP4X2), 1, 4, false, 3 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 128)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, B16, MulABScaleShuffled, V3, 256, 128, 128, 128/sizeof(FP4X2), 1, 4, false, 3 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }
#endif


#if defined(__Float4_e2m1fn_x2)
    if (dtype_checker<FP4X2>{}(x_dtype)
        && dtype_checker<FP4X2>{}(w_dtype)
        && dtype_checker<B16>{}(y_dtype)
        && 0 == act_op
        && true == mul_routed_weight_stage
        && 3 == quant
        && false == is_shuffled)
    {
        if (block_m == 32)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, B16, MulABScale, V3, 256, 32, 128, 128/sizeof(FP4X2), 1, 4, false, 3 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, B16, MulABScale, V3, 256, 64, 64, 128/sizeof(FP4X2), 2, 2, false, 3 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 128)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, B16, MulABScale, V3, 256, 128, 64, 128/sizeof(FP4X2), 2, 2, false, 3 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }
#endif


#if defined(__Float4_e2m1fn_x2)
    if (dtype_checker<FP4X2>{}(x_dtype)
        && dtype_checker<FP4X2>{}(w_dtype)
        && dtype_checker<B16>{}(y_dtype)
        && 0 == act_op
        && true == mul_routed_weight_stage
        && 3 == quant
        && true == is_shuffled)
    {
        if (block_m == 32)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, B16, MulABScaleShuffled, V3, 64, 32, 32, 128/sizeof(FP4X2), 1, 1, false, 3 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, B16, MulABScaleShuffled, V3, 256, 64, 128, 128/sizeof(FP4X2), 1, 4, false, 3 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 128)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, B16, MulABScaleShuffled, V3, 256, 128, 128, 128/sizeof(FP4X2), 1, 4, false, 3 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }
#endif


#if defined(__Float4_e2m1fn_x2)
    if (dtype_checker<FP4X2>{}(x_dtype)
        && dtype_checker<FP4X2>{}(w_dtype)
        && dtype_checker<B16>{}(y_dtype)
        && 0 == act_op
        && true == mul_routed_weight_stage
        && 3 == quant
        && false == is_shuffled)
    {
        if (block_m == 32)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, B16, MulABScale, V3, 256, 32, 128, 128/sizeof(FP4X2), 1, 4, false, 3 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, B16, MulABScale, V3, 256, 64, 64, 128/sizeof(FP4X2), 2, 2, false, 3 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 128)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, B16, MulABScale, V3, 256, 128, 64, 128/sizeof(FP4X2), 2, 2, false, 3 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }
#endif


#if defined(__Float4_e2m1fn_x2)
    if (dtype_checker<FP4X2>{}(x_dtype)
        && dtype_checker<FP4X2>{}(w_dtype)
        && dtype_checker<B16>{}(y_dtype)
        && 0 == act_op
        && false == mul_routed_weight_stage
        && 3 == quant
        && true == is_shuffled)
    {
        if (block_m == 32)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, B16, MulABScaleShuffled, V3, 64, 32, 32, 128/sizeof(FP4X2), 1, 1, false, 3 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, B16, MulABScaleShuffled, V3, 256, 64, 128, 128/sizeof(FP4X2), 1, 4, false, 3 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 128)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, B16, MulABScaleShuffled, V3, 256, 128, 128, 128/sizeof(FP4X2), 1, 4, false, 3 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }
#endif


#if defined(__Float4_e2m1fn_x2)
    if (dtype_checker<FP4X2>{}(x_dtype)
        && dtype_checker<FP4X2>{}(w_dtype)
        && dtype_checker<B16>{}(y_dtype)
        && 0 == act_op
        && false == mul_routed_weight_stage
        && 3 == quant
        && false == is_shuffled)
    {
        if (block_m == 32)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, B16, MulABScale, V3, 256, 32, 128, 128/sizeof(FP4X2), 1, 4, false, 3 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, B16, MulABScale, V3, 256, 64, 64, 128/sizeof(FP4X2), 2, 2, false, 3 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 128)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, B16, MulABScale, V3, 256, 128, 64, 128/sizeof(FP4X2), 2, 2, false, 3 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }
#endif


#if defined(__Float4_e2m1fn_x2)
    if (dtype_checker<FP4X2>{}(x_dtype)
        && dtype_checker<FP4X2>{}(w_dtype)
        && dtype_checker<B16>{}(y_dtype)
        && 0 == act_op
        && false == mul_routed_weight_stage
        && 3 == quant
        && true == is_shuffled)
    {
        if (block_m == 32)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, B16, MulABScaleShuffled, V3, 64, 32, 32, 128/sizeof(FP4X2), 1, 1, false, 3 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, B16, MulABScaleShuffled, V3, 256, 64, 128, 128/sizeof(FP4X2), 1, 4, false, 3 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 128)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, B16, MulABScaleShuffled, V3, 256, 128, 128, 128/sizeof(FP4X2), 1, 4, false, 3 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }
#endif


#if defined(__Float4_e2m1fn_x2)
    if (dtype_checker<FP4X2>{}(x_dtype)
        && dtype_checker<FP4X2>{}(w_dtype)
        && dtype_checker<B16>{}(y_dtype)
        && 0 == act_op
        && false == mul_routed_weight_stage
        && 3 == quant
        && false == is_shuffled)
    {
        if (block_m == 32)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, B16, MulABScale, V3, 256, 32, 128, 128/sizeof(FP4X2), 1, 4, false, 3 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, B16, MulABScale, V3, 256, 64, 64, 128/sizeof(FP4X2), 2, 2, false, 3 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 128)
        {
            return ck_moe_stage1_gemm<FP4X2, FP4X2, F32, B16, MulABScale, V3, 256, 128, 64, 128/sizeof(FP4X2), 2, 2, false, 3 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }
#endif


    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<F8>{}(w_dtype)
        && dtype_checker<F16>{}(y_dtype)
        && 1 == act_op
        && true == mul_routed_weight_stage
        && 4 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScaleExpertWeightA8W8blkscale, V1, 256, 16, 128, 256/sizeof(F8), 1, 4, false, 4 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScaleExpertWeightA8W8blkscale, V1, 256, 32, 128, 128/sizeof(F8), 1, 4, false, 4 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScaleExpertWeightA8W8blkscale, V3, 256, 64, 128, 128/sizeof(F8), 1, 4, false, 4 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<F8>{}(w_dtype)
        && dtype_checker<F32>{}(y_dtype)
        && 1 == act_op
        && true == mul_routed_weight_stage
        && 4 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, F32, MulABScaleExpertWeightA8W8blkscaleSplitk, V1, 256, 16, 128, 256/sizeof(F8), 1, 4, false, 4 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, F32, MulABScaleExpertWeightA8W8blkscaleSplitk, V1, 256, 32, 128, 128/sizeof(F8), 1, 4, false, 4 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, F32, MulABScaleExpertWeightA8W8blkscaleSplitk, V3, 256, 64, 128, 128/sizeof(F8), 1, 4, false, 4 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<F8>{}(w_dtype)
        && dtype_checker<F16>{}(y_dtype)
        && 1 == act_op
        && false == mul_routed_weight_stage
        && 4 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScaleExpertWeightA8W8blkscale, V1, 256, 16, 128, 256/sizeof(F8), 1, 4, false, 4 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScaleExpertWeightA8W8blkscale, V1, 256, 32, 128, 128/sizeof(F8), 1, 4, false, 4 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScaleExpertWeightA8W8blkscale, V3, 256, 64, 128, 128/sizeof(F8), 1, 4, false, 4 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<F8>{}(w_dtype)
        && dtype_checker<F32>{}(y_dtype)
        && 1 == act_op
        && false == mul_routed_weight_stage
        && 4 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, F32, MulABScaleExpertWeightA8W8blkscaleSplitk, V1, 256, 16, 128, 256/sizeof(F8), 1, 4, false, 4 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, F32, MulABScaleExpertWeightA8W8blkscaleSplitk, V1, 256, 32, 128, 128/sizeof(F8), 1, 4, false, 4 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, F32, MulABScaleExpertWeightA8W8blkscaleSplitk, V3, 256, 64, 128, 128/sizeof(F8), 1, 4, false, 4 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<F8>{}(w_dtype)
        && dtype_checker<F16>{}(y_dtype)
        && 0 == act_op
        && true == mul_routed_weight_stage
        && 4 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScaleExpertWeightA8W8blkscale, V1, 256, 16, 128, 256/sizeof(F8), 1, 4, false, 4 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScaleExpertWeightA8W8blkscale, V1, 256, 32, 128, 128/sizeof(F8), 1, 4, false, 4 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScaleExpertWeightA8W8blkscale, V3, 256, 64, 128, 128/sizeof(F8), 1, 4, false, 4 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<F8>{}(w_dtype)
        && dtype_checker<F32>{}(y_dtype)
        && 0 == act_op
        && true == mul_routed_weight_stage
        && 4 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, F32, MulABScaleExpertWeightA8W8blkscaleSplitk, V1, 256, 16, 128, 256/sizeof(F8), 1, 4, false, 4 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, F32, MulABScaleExpertWeightA8W8blkscaleSplitk, V1, 256, 32, 128, 128/sizeof(F8), 1, 4, false, 4 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, F32, MulABScaleExpertWeightA8W8blkscaleSplitk, V3, 256, 64, 128, 128/sizeof(F8), 1, 4, false, 4 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<F8>{}(w_dtype)
        && dtype_checker<F16>{}(y_dtype)
        && 0 == act_op
        && false == mul_routed_weight_stage
        && 4 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScaleExpertWeightA8W8blkscale, V1, 256, 16, 128, 256/sizeof(F8), 1, 4, false, 4 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScaleExpertWeightA8W8blkscale, V1, 256, 32, 128, 128/sizeof(F8), 1, 4, false, 4 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, F16, MulABScaleExpertWeightA8W8blkscale, V3, 256, 64, 128, 128/sizeof(F8), 1, 4, false, 4 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<F8>{}(w_dtype)
        && dtype_checker<F32>{}(y_dtype)
        && 0 == act_op
        && false == mul_routed_weight_stage
        && 4 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, F32, MulABScaleExpertWeightA8W8blkscaleSplitk, V1, 256, 16, 128, 256/sizeof(F8), 1, 4, false, 4 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, F32, MulABScaleExpertWeightA8W8blkscaleSplitk, V1, 256, 32, 128, 128/sizeof(F8), 1, 4, false, 4 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, F32, MulABScaleExpertWeightA8W8blkscaleSplitk, V3, 256, 64, 128, 128/sizeof(F8), 1, 4, false, 4 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<F8>{}(w_dtype)
        && dtype_checker<B16>{}(y_dtype)
        && 1 == act_op
        && true == mul_routed_weight_stage
        && 4 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScaleExpertWeightA8W8blkscale, V1, 256, 16, 128, 256/sizeof(F8), 1, 4, false, 4 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScaleExpertWeightA8W8blkscale, V1, 256, 32, 128, 128/sizeof(F8), 1, 4, false, 4 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScaleExpertWeightA8W8blkscale, V3, 256, 64, 128, 128/sizeof(F8), 1, 4, false, 4 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<F8>{}(w_dtype)
        && dtype_checker<F32>{}(y_dtype)
        && 1 == act_op
        && true == mul_routed_weight_stage
        && 4 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, F32, MulABScaleExpertWeightA8W8blkscaleSplitk, V1, 256, 16, 128, 256/sizeof(F8), 1, 4, false, 4 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, F32, MulABScaleExpertWeightA8W8blkscaleSplitk, V1, 256, 32, 128, 128/sizeof(F8), 1, 4, false, 4 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, F32, MulABScaleExpertWeightA8W8blkscaleSplitk, V3, 256, 64, 128, 128/sizeof(F8), 1, 4, false, 4 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<F8>{}(w_dtype)
        && dtype_checker<B16>{}(y_dtype)
        && 1 == act_op
        && false == mul_routed_weight_stage
        && 4 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScaleExpertWeightA8W8blkscale, V1, 256, 16, 128, 256/sizeof(F8), 1, 4, false, 4 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScaleExpertWeightA8W8blkscale, V1, 256, 32, 128, 128/sizeof(F8), 1, 4, false, 4 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScaleExpertWeightA8W8blkscale, V3, 256, 64, 128, 128/sizeof(F8), 1, 4, false, 4 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<F8>{}(w_dtype)
        && dtype_checker<F32>{}(y_dtype)
        && 1 == act_op
        && false == mul_routed_weight_stage
        && 4 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, F32, MulABScaleExpertWeightA8W8blkscaleSplitk, V1, 256, 16, 128, 256/sizeof(F8), 1, 4, false, 4 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, F32, MulABScaleExpertWeightA8W8blkscaleSplitk, V1, 256, 32, 128, 128/sizeof(F8), 1, 4, false, 4 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, F32, MulABScaleExpertWeightA8W8blkscaleSplitk, V3, 256, 64, 128, 128/sizeof(F8), 1, 4, false, 4 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<F8>{}(w_dtype)
        && dtype_checker<B16>{}(y_dtype)
        && 0 == act_op
        && true == mul_routed_weight_stage
        && 4 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScaleExpertWeightA8W8blkscale, V1, 256, 16, 128, 256/sizeof(F8), 1, 4, false, 4 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScaleExpertWeightA8W8blkscale, V1, 256, 32, 128, 128/sizeof(F8), 1, 4, false, 4 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScaleExpertWeightA8W8blkscale, V3, 256, 64, 128, 128/sizeof(F8), 1, 4, false, 4 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<F8>{}(w_dtype)
        && dtype_checker<F32>{}(y_dtype)
        && 0 == act_op
        && true == mul_routed_weight_stage
        && 4 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, F32, MulABScaleExpertWeightA8W8blkscaleSplitk, V1, 256, 16, 128, 256/sizeof(F8), 1, 4, false, 4 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, F32, MulABScaleExpertWeightA8W8blkscaleSplitk, V1, 256, 32, 128, 128/sizeof(F8), 1, 4, false, 4 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, F32, MulABScaleExpertWeightA8W8blkscaleSplitk, V3, 256, 64, 128, 128/sizeof(F8), 1, 4, false, 4 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<F8>{}(w_dtype)
        && dtype_checker<B16>{}(y_dtype)
        && 0 == act_op
        && false == mul_routed_weight_stage
        && 4 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScaleExpertWeightA8W8blkscale, V1, 256, 16, 128, 256/sizeof(F8), 1, 4, false, 4 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScaleExpertWeightA8W8blkscale, V1, 256, 32, 128, 128/sizeof(F8), 1, 4, false, 4 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, B16, MulABScaleExpertWeightA8W8blkscale, V3, 256, 64, 128, 128/sizeof(F8), 1, 4, false, 4 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F8>{}(x_dtype)
        && dtype_checker<F8>{}(w_dtype)
        && dtype_checker<F32>{}(y_dtype)
        && 0 == act_op
        && false == mul_routed_weight_stage
        && 4 == quant)
    {
        if (block_m == 16)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, F32, MulABScaleExpertWeightA8W8blkscaleSplitk, V1, 256, 16, 128, 256/sizeof(F8), 1, 4, false, 4 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, F32, MulABScaleExpertWeightA8W8blkscaleSplitk, V1, 256, 32, 128, 128/sizeof(F8), 1, 4, false, 4 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 64)
        {
            return ck_moe_stage1_gemm<F8, F8, F32, F32, MulABScaleExpertWeightA8W8blkscaleSplitk, V3, 256, 64, 128, 128/sizeof(F8), 1, 4, false, 4 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F16>{}(x_dtype)
        && dtype_checker<F16>{}(w_dtype)
        && dtype_checker<F16>{}(y_dtype)
        && 1 == act_op
        && true == mul_routed_weight_stage
        && 0 == quant)
    {
        if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F16, F16, F32, F16, TypeCastExpertWeight, V1, 256, 32, 64, 128/sizeof(F16), 1, 4, false, 0 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 64)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F16, F16, F32, F16, TypeCastExpertWeight, V1, 256, 64, 64, 128/sizeof(F16), 1, 4, false, 0 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<F16, F16, F32, F16, TypeCastExpertWeight, V3, 256, 64, 128, 128/sizeof(F16), 1, 4, false, 0 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
        }
        else if (block_m == 128)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F16, F16, F32, F16, TypeCastExpertWeight, V1, 256, 128, 64, 128/sizeof(F16), 1, 4, false, 0 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<F16, F16, F32, F16, TypeCastExpertWeight, V3, 256, 128, 128, 128/sizeof(F16), 1, 4, false, 0 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
        }
        else if (block_m == 256)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F16, F16, F32, F16, TypeCastExpertWeight, V1, 256, 256, 64, 128/sizeof(F16), 1, 4, false, 0 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<F16, F16, F32, F16, TypeCastExpertWeight, V3, 256, 256, 128, 128/sizeof(F16), 1, 4, false, 0 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F16>{}(x_dtype)
        && dtype_checker<F16>{}(w_dtype)
        && dtype_checker<F16>{}(y_dtype)
        && 1 == act_op
        && false == mul_routed_weight_stage
        && 0 == quant)
    {
        if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F16, F16, F32, F16, TypeCast, V1, 256, 32, 64, 128/sizeof(F16), 1, 4, false, 0 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 64)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F16, F16, F32, F16, TypeCast, V1, 256, 64, 64, 128/sizeof(F16), 1, 4, false, 0 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<F16, F16, F32, F16, TypeCast, V3, 256, 64, 128, 128/sizeof(F16), 1, 4, false, 0 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
        }
        else if (block_m == 128)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F16, F16, F32, F16, TypeCast, V1, 256, 128, 64, 128/sizeof(F16), 1, 4, false, 0 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<F16, F16, F32, F16, TypeCast, V3, 256, 128, 128, 128/sizeof(F16), 1, 4, false, 0 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
        }
        else if (block_m == 256)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F16, F16, F32, F16, TypeCast, V1, 256, 256, 64, 128/sizeof(F16), 1, 4, false, 0 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<F16, F16, F32, F16, TypeCast, V3, 256, 256, 128, 128/sizeof(F16), 1, 4, false, 0 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F16>{}(x_dtype)
        && dtype_checker<F16>{}(w_dtype)
        && dtype_checker<F16>{}(y_dtype)
        && 0 == act_op
        && true == mul_routed_weight_stage
        && 0 == quant)
    {
        if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F16, F16, F32, F16, TypeCastExpertWeight, V1, 256, 32, 64, 128/sizeof(F16), 1, 4, false, 0 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 64)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F16, F16, F32, F16, TypeCastExpertWeight, V1, 256, 64, 64, 128/sizeof(F16), 1, 4, false, 0 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<F16, F16, F32, F16, TypeCastExpertWeight, V3, 256, 64, 128, 128/sizeof(F16), 1, 4, false, 0 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
        }
        else if (block_m == 128)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F16, F16, F32, F16, TypeCastExpertWeight, V1, 256, 128, 64, 128/sizeof(F16), 1, 4, false, 0 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<F16, F16, F32, F16, TypeCastExpertWeight, V3, 256, 128, 128, 128/sizeof(F16), 1, 4, false, 0 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
        }
        else if (block_m == 256)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F16, F16, F32, F16, TypeCastExpertWeight, V1, 256, 256, 64, 128/sizeof(F16), 1, 4, false, 0 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<F16, F16, F32, F16, TypeCastExpertWeight, V3, 256, 256, 128, 128/sizeof(F16), 1, 4, false, 0 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<F16>{}(x_dtype)
        && dtype_checker<F16>{}(w_dtype)
        && dtype_checker<F16>{}(y_dtype)
        && 0 == act_op
        && false == mul_routed_weight_stage
        && 0 == quant)
    {
        if (block_m == 32)
        {
            return ck_moe_stage1_gemm<F16, F16, F32, F16, TypeCast, V1, 256, 32, 64, 128/sizeof(F16), 1, 4, false, 0 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 64)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F16, F16, F32, F16, TypeCast, V1, 256, 64, 64, 128/sizeof(F16), 1, 4, false, 0 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<F16, F16, F32, F16, TypeCast, V3, 256, 64, 128, 128/sizeof(F16), 1, 4, false, 0 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
        }
        else if (block_m == 128)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F16, F16, F32, F16, TypeCast, V1, 256, 128, 64, 128/sizeof(F16), 1, 4, false, 0 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<F16, F16, F32, F16, TypeCast, V3, 256, 128, 128, 128/sizeof(F16), 1, 4, false, 0 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
        }
        else if (block_m == 256)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<F16, F16, F32, F16, TypeCast, V1, 256, 256, 64, 128/sizeof(F16), 1, 4, false, 0 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<F16, F16, F32, F16, TypeCast, V3, 256, 256, 128, 128/sizeof(F16), 1, 4, false, 0 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<B16>{}(x_dtype)
        && dtype_checker<B16>{}(w_dtype)
        && dtype_checker<B16>{}(y_dtype)
        && 1 == act_op
        && true == mul_routed_weight_stage
        && 0 == quant)
    {
        if (block_m == 32)
        {
            return ck_moe_stage1_gemm<B16, B16, F32, B16, TypeCastExpertWeight, V1, 256, 32, 64, 128/sizeof(B16), 1, 4, false, 0 == static_cast<int>(QuantType::per_Tensor), true, 1>;
        }
        else if (block_m == 64)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<B16, B16, F32, B16, TypeCastExpertWeight, V1, 256, 64, 64, 128/sizeof(B16), 1, 4, false, 0 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<B16, B16, F32, B16, TypeCastExpertWeight, V3, 256, 64, 128, 128/sizeof(B16), 1, 4, false, 0 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
        }
        else if (block_m == 128)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<B16, B16, F32, B16, TypeCastExpertWeight, V1, 256, 128, 64, 128/sizeof(B16), 1, 4, false, 0 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<B16, B16, F32, B16, TypeCastExpertWeight, V3, 256, 128, 128, 128/sizeof(B16), 1, 4, false, 0 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
        }
        else if (block_m == 256)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<B16, B16, F32, B16, TypeCastExpertWeight, V1, 256, 256, 64, 128/sizeof(B16), 1, 4, false, 0 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<B16, B16, F32, B16, TypeCastExpertWeight, V3, 256, 256, 128, 128/sizeof(B16), 1, 4, false, 0 == static_cast<int>(QuantType::per_Tensor), true, 1>;
            }
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<B16>{}(x_dtype)
        && dtype_checker<B16>{}(w_dtype)
        && dtype_checker<B16>{}(y_dtype)
        && 1 == act_op
        && false == mul_routed_weight_stage
        && 0 == quant)
    {
        if (block_m == 32)
        {
            return ck_moe_stage1_gemm<B16, B16, F32, B16, TypeCast, V1, 256, 32, 64, 128/sizeof(B16), 1, 4, false, 0 == static_cast<int>(QuantType::per_Tensor), false, 1>;
        }
        else if (block_m == 64)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<B16, B16, F32, B16, TypeCast, V1, 256, 64, 64, 128/sizeof(B16), 1, 4, false, 0 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<B16, B16, F32, B16, TypeCast, V3, 256, 64, 128, 128/sizeof(B16), 1, 4, false, 0 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
        }
        else if (block_m == 128)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<B16, B16, F32, B16, TypeCast, V1, 256, 128, 64, 128/sizeof(B16), 1, 4, false, 0 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<B16, B16, F32, B16, TypeCast, V3, 256, 128, 128, 128/sizeof(B16), 1, 4, false, 0 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
        }
        else if (block_m == 256)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<B16, B16, F32, B16, TypeCast, V1, 256, 256, 64, 128/sizeof(B16), 1, 4, false, 0 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
            else
            {
                return ck_moe_stage1_gemm<B16, B16, F32, B16, TypeCast, V3, 256, 256, 128, 128/sizeof(B16), 1, 4, false, 0 == static_cast<int>(QuantType::per_Tensor), false, 1>;
            }
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<B16>{}(x_dtype)
        && dtype_checker<B16>{}(w_dtype)
        && dtype_checker<B16>{}(y_dtype)
        && 0 == act_op
        && true == mul_routed_weight_stage
        && 0 == quant)
    {
        if (block_m == 32)
        {
            return ck_moe_stage1_gemm<B16, B16, F32, B16, TypeCastExpertWeight, V1, 256, 32, 64, 128/sizeof(B16), 1, 4, false, 0 == static_cast<int>(QuantType::per_Tensor), true, 0>;
        }
        else if (block_m == 64)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<B16, B16, F32, B16, TypeCastExpertWeight, V1, 256, 64, 64, 128/sizeof(B16), 1, 4, false, 0 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<B16, B16, F32, B16, TypeCastExpertWeight, V3, 256, 64, 128, 128/sizeof(B16), 1, 4, false, 0 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
        }
        else if (block_m == 128)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<B16, B16, F32, B16, TypeCastExpertWeight, V1, 256, 128, 64, 128/sizeof(B16), 1, 4, false, 0 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<B16, B16, F32, B16, TypeCastExpertWeight, V3, 256, 128, 128, 128/sizeof(B16), 1, 4, false, 0 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
        }
        else if (block_m == 256)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<B16, B16, F32, B16, TypeCastExpertWeight, V1, 256, 256, 64, 128/sizeof(B16), 1, 4, false, 0 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<B16, B16, F32, B16, TypeCastExpertWeight, V3, 256, 256, 128, 128/sizeof(B16), 1, 4, false, 0 == static_cast<int>(QuantType::per_Tensor), true, 0>;
            }
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    if (dtype_checker<B16>{}(x_dtype)
        && dtype_checker<B16>{}(w_dtype)
        && dtype_checker<B16>{}(y_dtype)
        && 0 == act_op
        && false == mul_routed_weight_stage
        && 0 == quant)
    {
        if (block_m == 32)
        {
            return ck_moe_stage1_gemm<B16, B16, F32, B16, TypeCast, V1, 256, 32, 64, 128/sizeof(B16), 1, 4, false, 0 == static_cast<int>(QuantType::per_Tensor), false, 0>;
        }
        else if (block_m == 64)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<B16, B16, F32, B16, TypeCast, V1, 256, 64, 64, 128/sizeof(B16), 1, 4, false, 0 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<B16, B16, F32, B16, TypeCast, V3, 256, 64, 128, 128/sizeof(B16), 1, 4, false, 0 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
        }
        else if (block_m == 128)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<B16, B16, F32, B16, TypeCast, V1, 256, 128, 64, 128/sizeof(B16), 1, 4, false, 0 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<B16, B16, F32, B16, TypeCast, V3, 256, 128, 128, 128/sizeof(B16), 1, 4, false, 0 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
        }
        else if (block_m == 256)
        {
            if (inter_dim <= 192)
            {
                return ck_moe_stage1_gemm<B16, B16, F32, B16, TypeCast, V1, 256, 256, 64, 128/sizeof(B16), 1, 4, false, 0 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
            else
            {
                return ck_moe_stage1_gemm<B16, B16, F32, B16, TypeCast, V3, 256, 256, 128, 128/sizeof(B16), 1, 4, false, 0 == static_cast<int>(QuantType::per_Tensor), false, 0>;
            }
        }
        else
        {
            TORCH_CHECK(
                false,
                "Unsupported block_m value for moe heuristic dispatch: ",
                block_m);
        }
    }

    TORCH_CHECK(
        false,
        "Unsupported kernel config for moe heuristic dispatch");
}}

