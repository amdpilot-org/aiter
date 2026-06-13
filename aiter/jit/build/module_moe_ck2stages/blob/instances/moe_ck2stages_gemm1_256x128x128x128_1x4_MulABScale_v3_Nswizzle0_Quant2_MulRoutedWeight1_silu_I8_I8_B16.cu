// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "gemm_moe_ck2stages_common.cuh"

using A0DataType = I8;
using B0DataType = I8;
using AccDataType = I32;
using EDataType = B16;
using CDEElementOp = MulABScale;
const bool Nswizzle = false;
const bool PerTensorQuant = 2 == static_cast<int>(QuantType::per_Tensor);
const bool MulRoutedWeight = true;
const int ActOP = 1;
CK_MOE_STAGE1_GEMM_DEFINE(256, 128, 128, 128, 1, 4, V3)
