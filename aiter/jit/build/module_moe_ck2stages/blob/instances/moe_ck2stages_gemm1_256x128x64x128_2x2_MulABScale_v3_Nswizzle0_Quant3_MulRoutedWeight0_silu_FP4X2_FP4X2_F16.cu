#ifndef __gfx942__
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "gemm_moe_ck2stages_common_mxfp4_bns.cuh"

using A0DataType = FP4X2;
using B0DataType = FP4X2;
using AccDataType = F32;
using EDataType = F16;
using CDEElementOp = MulABScale;
const bool Nswizzle = false;
const bool PerTensorQuant = 3 == static_cast<int>(QuantType::per_Tensor);
const bool MulRoutedWeight = false;
const int ActOP = 1;
CK_MOE_STAGE1_GEMM_DEFINE(256, 128, 64, 128, 2, 2, V3)

#endif
