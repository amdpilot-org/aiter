
// SPDX-License-Identifier: MIT
// Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.

#include "binary_op_api_common.hpp"

// Explicit instantiation
template void binary_op_impl<aiter::MulOp, int, float>(torch::Tensor&, torch::Tensor&, torch::Tensor&);
