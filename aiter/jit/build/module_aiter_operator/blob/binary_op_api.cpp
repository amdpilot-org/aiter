
// SPDX-License-Identifier: MIT
// Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.

#include "binary_op_api_common.hpp"
void binary_op_dispatch(const std::string& op_type,
                       torch::Tensor &input,
                       torch::Tensor &other,
                       torch::Tensor &output) {
    // Dispatch based on operator and input types
    if(op_type == "add") {
        if(input.scalar_type() == torch::kFloat32 && other.scalar_type() == torch::kFloat32) {
            binary_op_impl<aiter::AddOp, float, float>(input, other, output);
        }
        else if(input.scalar_type() == torch::kFloat32 && other.scalar_type() == torch::kBFloat16) {
            binary_op_impl<aiter::AddOp, float, torch::BFloat16>(input, other, output);
        }
        else if(input.scalar_type() == torch::kFloat32 && other.scalar_type() == torch::kHalf) {
            binary_op_impl<aiter::AddOp, float, torch::Half>(input, other, output);
        }
        else if(input.scalar_type() == torch::kFloat32 && other.scalar_type() == torch::kInt32) {
            binary_op_impl<aiter::AddOp, float, int>(input, other, output);
        }
        else if(input.scalar_type() == torch::kBFloat16 && other.scalar_type() == torch::kFloat32) {
            binary_op_impl<aiter::AddOp, torch::BFloat16, float>(input, other, output);
        }
        else if(input.scalar_type() == torch::kBFloat16 && other.scalar_type() == torch::kBFloat16) {
            binary_op_impl<aiter::AddOp, torch::BFloat16, torch::BFloat16>(input, other, output);
        }
        else if(input.scalar_type() == torch::kBFloat16 && other.scalar_type() == torch::kHalf) {
            binary_op_impl<aiter::AddOp, torch::BFloat16, torch::Half>(input, other, output);
        }
        else if(input.scalar_type() == torch::kBFloat16 && other.scalar_type() == torch::kInt32) {
            binary_op_impl<aiter::AddOp, torch::BFloat16, int>(input, other, output);
        }
        else if(input.scalar_type() == torch::kHalf && other.scalar_type() == torch::kFloat32) {
            binary_op_impl<aiter::AddOp, torch::Half, float>(input, other, output);
        }
        else if(input.scalar_type() == torch::kHalf && other.scalar_type() == torch::kBFloat16) {
            binary_op_impl<aiter::AddOp, torch::Half, torch::BFloat16>(input, other, output);
        }
        else if(input.scalar_type() == torch::kHalf && other.scalar_type() == torch::kHalf) {
            binary_op_impl<aiter::AddOp, torch::Half, torch::Half>(input, other, output);
        }
        else if(input.scalar_type() == torch::kHalf && other.scalar_type() == torch::kInt32) {
            binary_op_impl<aiter::AddOp, torch::Half, int>(input, other, output);
        }
        else if(input.scalar_type() == torch::kInt32 && other.scalar_type() == torch::kFloat32) {
            binary_op_impl<aiter::AddOp, int, float>(input, other, output);
        }
        else if(input.scalar_type() == torch::kInt32 && other.scalar_type() == torch::kBFloat16) {
            binary_op_impl<aiter::AddOp, int, torch::BFloat16>(input, other, output);
        }
        else if(input.scalar_type() == torch::kInt32 && other.scalar_type() == torch::kHalf) {
            binary_op_impl<aiter::AddOp, int, torch::Half>(input, other, output);
        }
        else if(input.scalar_type() == torch::kInt32 && other.scalar_type() == torch::kInt32) {
            binary_op_impl<aiter::AddOp, int, int>(input, other, output);
        }

    }
    else if(op_type == "sub") {
        if(input.scalar_type() == torch::kFloat32 && other.scalar_type() == torch::kFloat32) {
            binary_op_impl<aiter::SubOp, float, float>(input, other, output);
        }
        else if(input.scalar_type() == torch::kFloat32 && other.scalar_type() == torch::kBFloat16) {
            binary_op_impl<aiter::SubOp, float, torch::BFloat16>(input, other, output);
        }
        else if(input.scalar_type() == torch::kFloat32 && other.scalar_type() == torch::kHalf) {
            binary_op_impl<aiter::SubOp, float, torch::Half>(input, other, output);
        }
        else if(input.scalar_type() == torch::kFloat32 && other.scalar_type() == torch::kInt32) {
            binary_op_impl<aiter::SubOp, float, int>(input, other, output);
        }
        else if(input.scalar_type() == torch::kBFloat16 && other.scalar_type() == torch::kFloat32) {
            binary_op_impl<aiter::SubOp, torch::BFloat16, float>(input, other, output);
        }
        else if(input.scalar_type() == torch::kBFloat16 && other.scalar_type() == torch::kBFloat16) {
            binary_op_impl<aiter::SubOp, torch::BFloat16, torch::BFloat16>(input, other, output);
        }
        else if(input.scalar_type() == torch::kBFloat16 && other.scalar_type() == torch::kHalf) {
            binary_op_impl<aiter::SubOp, torch::BFloat16, torch::Half>(input, other, output);
        }
        else if(input.scalar_type() == torch::kBFloat16 && other.scalar_type() == torch::kInt32) {
            binary_op_impl<aiter::SubOp, torch::BFloat16, int>(input, other, output);
        }
        else if(input.scalar_type() == torch::kHalf && other.scalar_type() == torch::kFloat32) {
            binary_op_impl<aiter::SubOp, torch::Half, float>(input, other, output);
        }
        else if(input.scalar_type() == torch::kHalf && other.scalar_type() == torch::kBFloat16) {
            binary_op_impl<aiter::SubOp, torch::Half, torch::BFloat16>(input, other, output);
        }
        else if(input.scalar_type() == torch::kHalf && other.scalar_type() == torch::kHalf) {
            binary_op_impl<aiter::SubOp, torch::Half, torch::Half>(input, other, output);
        }
        else if(input.scalar_type() == torch::kHalf && other.scalar_type() == torch::kInt32) {
            binary_op_impl<aiter::SubOp, torch::Half, int>(input, other, output);
        }
        else if(input.scalar_type() == torch::kInt32 && other.scalar_type() == torch::kFloat32) {
            binary_op_impl<aiter::SubOp, int, float>(input, other, output);
        }
        else if(input.scalar_type() == torch::kInt32 && other.scalar_type() == torch::kBFloat16) {
            binary_op_impl<aiter::SubOp, int, torch::BFloat16>(input, other, output);
        }
        else if(input.scalar_type() == torch::kInt32 && other.scalar_type() == torch::kHalf) {
            binary_op_impl<aiter::SubOp, int, torch::Half>(input, other, output);
        }
        else if(input.scalar_type() == torch::kInt32 && other.scalar_type() == torch::kInt32) {
            binary_op_impl<aiter::SubOp, int, int>(input, other, output);
        }

    }
    else if(op_type == "mul") {
        if(input.scalar_type() == torch::kFloat32 && other.scalar_type() == torch::kFloat32) {
            binary_op_impl<aiter::MulOp, float, float>(input, other, output);
        }
        else if(input.scalar_type() == torch::kFloat32 && other.scalar_type() == torch::kBFloat16) {
            binary_op_impl<aiter::MulOp, float, torch::BFloat16>(input, other, output);
        }
        else if(input.scalar_type() == torch::kFloat32 && other.scalar_type() == torch::kHalf) {
            binary_op_impl<aiter::MulOp, float, torch::Half>(input, other, output);
        }
        else if(input.scalar_type() == torch::kFloat32 && other.scalar_type() == torch::kInt32) {
            binary_op_impl<aiter::MulOp, float, int>(input, other, output);
        }
        else if(input.scalar_type() == torch::kBFloat16 && other.scalar_type() == torch::kFloat32) {
            binary_op_impl<aiter::MulOp, torch::BFloat16, float>(input, other, output);
        }
        else if(input.scalar_type() == torch::kBFloat16 && other.scalar_type() == torch::kBFloat16) {
            binary_op_impl<aiter::MulOp, torch::BFloat16, torch::BFloat16>(input, other, output);
        }
        else if(input.scalar_type() == torch::kBFloat16 && other.scalar_type() == torch::kHalf) {
            binary_op_impl<aiter::MulOp, torch::BFloat16, torch::Half>(input, other, output);
        }
        else if(input.scalar_type() == torch::kBFloat16 && other.scalar_type() == torch::kInt32) {
            binary_op_impl<aiter::MulOp, torch::BFloat16, int>(input, other, output);
        }
        else if(input.scalar_type() == torch::kHalf && other.scalar_type() == torch::kFloat32) {
            binary_op_impl<aiter::MulOp, torch::Half, float>(input, other, output);
        }
        else if(input.scalar_type() == torch::kHalf && other.scalar_type() == torch::kBFloat16) {
            binary_op_impl<aiter::MulOp, torch::Half, torch::BFloat16>(input, other, output);
        }
        else if(input.scalar_type() == torch::kHalf && other.scalar_type() == torch::kHalf) {
            binary_op_impl<aiter::MulOp, torch::Half, torch::Half>(input, other, output);
        }
        else if(input.scalar_type() == torch::kHalf && other.scalar_type() == torch::kInt32) {
            binary_op_impl<aiter::MulOp, torch::Half, int>(input, other, output);
        }
        else if(input.scalar_type() == torch::kInt32 && other.scalar_type() == torch::kFloat32) {
            binary_op_impl<aiter::MulOp, int, float>(input, other, output);
        }
        else if(input.scalar_type() == torch::kInt32 && other.scalar_type() == torch::kBFloat16) {
            binary_op_impl<aiter::MulOp, int, torch::BFloat16>(input, other, output);
        }
        else if(input.scalar_type() == torch::kInt32 && other.scalar_type() == torch::kHalf) {
            binary_op_impl<aiter::MulOp, int, torch::Half>(input, other, output);
        }
        else if(input.scalar_type() == torch::kInt32 && other.scalar_type() == torch::kInt32) {
            binary_op_impl<aiter::MulOp, int, int>(input, other, output);
        }

    }
    else if(op_type == "div") {
        if(input.scalar_type() == torch::kFloat32 && other.scalar_type() == torch::kFloat32) {
            binary_op_impl<aiter::DivOp, float, float>(input, other, output);
        }
        else if(input.scalar_type() == torch::kFloat32 && other.scalar_type() == torch::kBFloat16) {
            binary_op_impl<aiter::DivOp, float, torch::BFloat16>(input, other, output);
        }
        else if(input.scalar_type() == torch::kFloat32 && other.scalar_type() == torch::kHalf) {
            binary_op_impl<aiter::DivOp, float, torch::Half>(input, other, output);
        }
        else if(input.scalar_type() == torch::kFloat32 && other.scalar_type() == torch::kInt32) {
            binary_op_impl<aiter::DivOp, float, int>(input, other, output);
        }
        else if(input.scalar_type() == torch::kBFloat16 && other.scalar_type() == torch::kFloat32) {
            binary_op_impl<aiter::DivOp, torch::BFloat16, float>(input, other, output);
        }
        else if(input.scalar_type() == torch::kBFloat16 && other.scalar_type() == torch::kBFloat16) {
            binary_op_impl<aiter::DivOp, torch::BFloat16, torch::BFloat16>(input, other, output);
        }
        else if(input.scalar_type() == torch::kBFloat16 && other.scalar_type() == torch::kHalf) {
            binary_op_impl<aiter::DivOp, torch::BFloat16, torch::Half>(input, other, output);
        }
        else if(input.scalar_type() == torch::kBFloat16 && other.scalar_type() == torch::kInt32) {
            binary_op_impl<aiter::DivOp, torch::BFloat16, int>(input, other, output);
        }
        else if(input.scalar_type() == torch::kHalf && other.scalar_type() == torch::kFloat32) {
            binary_op_impl<aiter::DivOp, torch::Half, float>(input, other, output);
        }
        else if(input.scalar_type() == torch::kHalf && other.scalar_type() == torch::kBFloat16) {
            binary_op_impl<aiter::DivOp, torch::Half, torch::BFloat16>(input, other, output);
        }
        else if(input.scalar_type() == torch::kHalf && other.scalar_type() == torch::kHalf) {
            binary_op_impl<aiter::DivOp, torch::Half, torch::Half>(input, other, output);
        }
        else if(input.scalar_type() == torch::kHalf && other.scalar_type() == torch::kInt32) {
            binary_op_impl<aiter::DivOp, torch::Half, int>(input, other, output);
        }
        else if(input.scalar_type() == torch::kInt32 && other.scalar_type() == torch::kFloat32) {
            binary_op_impl<aiter::DivOp, int, float>(input, other, output);
        }
        else if(input.scalar_type() == torch::kInt32 && other.scalar_type() == torch::kBFloat16) {
            binary_op_impl<aiter::DivOp, int, torch::BFloat16>(input, other, output);
        }
        else if(input.scalar_type() == torch::kInt32 && other.scalar_type() == torch::kHalf) {
            binary_op_impl<aiter::DivOp, int, torch::Half>(input, other, output);
        }
        else if(input.scalar_type() == torch::kInt32 && other.scalar_type() == torch::kInt32) {
            binary_op_impl<aiter::DivOp, int, int>(input, other, output);
        }

    }

    //AT_ERROR("Unsupported operator or dtype combination");
}
