// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "gemm_moe_ck2stages_common.cuh"

using A0DataType = F8;
using B0DataType = F8;
using AccDataType = F32;
using EDataType = F16;
using CDEElementOp = MulABScaleExpertWeight;
const bool Nswizzle = false;
const bool PerTensorQuant = 1 == static_cast<int>(QuantType::per_Tensor);
const bool MulRoutedWeight = false;
const int ActOP = 0;
CK_MOE_STAGE2_GEMM_DEFINE(256, 64, 128, 128, 1, 4, V3)
