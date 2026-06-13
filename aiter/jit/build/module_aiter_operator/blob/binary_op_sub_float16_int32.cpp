
// SPDX-License-Identifier: MIT
// Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.

#include "binary_op_api_common.hpp"

// Explicit instantiation
template void binary_op_impl<aiter::SubOp, torch::Half, int>(torch::Tensor&, torch::Tensor&, torch::Tensor&);
