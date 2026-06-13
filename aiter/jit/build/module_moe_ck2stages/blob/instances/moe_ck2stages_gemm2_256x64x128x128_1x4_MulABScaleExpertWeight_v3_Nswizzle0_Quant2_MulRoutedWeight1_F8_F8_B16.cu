// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "gemm_moe_ck2stages_common.cuh"

using A0DataType = F8;
using B0DataType = F8;
using AccDataType = F32;
using EDataType = B16;
using CDEElementOp = MulABScaleExpertWeight;
const bool Nswizzle = false;
const bool PerTensorQuant = 2 == static_cast<int>(QuantType::per_Tensor);
const bool MulRoutedWeight = true;
const int ActOP = 0;
CK_MOE_STAGE2_GEMM_DEFINE(256, 64, 128, 128, 1, 4, V3)
