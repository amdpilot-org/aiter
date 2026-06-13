#ifndef __gfx942__
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "gemm_moe_ck2stages_common_mxfp4.cuh"

using A0DataType = FP4X2;
using B0DataType = FP4X2;
using AccDataType = F32;
using EDataType = B16;
using CDEElementOp = MulABScaleExpertWeightShuffled;
const bool Nswizzle = false;
const bool PerTensorQuant = 3 == static_cast<int>(QuantType::per_Tensor);
const bool MulRoutedWeight = true;
const int ActOP = 0;
CK_MOE_STAGE2_GEMM_DEFINE(64, 128, 128, 128, 1, 1, V3)

#endif
