// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "gemm_moe_ck2stages_common.cuh"

using A0DataType = B16;
using B0DataType = B16;
using AccDataType = F32;
using EDataType = B16;
using CDEElementOp = TypeCastExpertWeight;
const bool Nswizzle = false;
const bool PerTensorQuant = 0 == static_cast<int>(QuantType::per_Tensor);
const bool MulRoutedWeight = true;
const int ActOP = 0;
CK_MOE_STAGE1_GEMM_DEFINE(256, 64, 64, 128, 1, 4, V1)
