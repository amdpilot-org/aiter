
// SPDX-License-Identifier: MIT
// Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#include <torch/extension.h>
#include "binary_operator.cuh"

template <typename Op, typename T0, typename T1>
void binary_op_impl(torch::Tensor &input, torch::Tensor &other, torch::Tensor &output)
{
  const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(input));
  int dim = input.dim();

  bool is_support = false;
  bool order_flag = true;
  int pattern = 0;
  constexpr uint32_t PATTERN_TRANSPOSE = 1;
  constexpr uint32_t PATTERN_BROADCAST_0 = 2;   // (m, n, k), (1, n, k)
  constexpr uint32_t PATTERN_BROADCAST_1 = 3;   // (m, n, k), (m, 1, k)
  constexpr uint32_t PATTERN_CONTIGUOUS = 4;
  constexpr uint32_t PATTERN_BROADCAST_2 = 5;   // (m, n, k), (m, n, 1)
  constexpr uint32_t PATTERN_BROADCAST_3 = 6;   // (m, n, k), (   n, 1)
  constexpr uint32_t PATTERN_BROADCAST_4 = 7;   // (m, n, k), (m, 1, 1)

  // contiguous case
  if (!is_support)
  {
    is_support = true;
    is_support &= (input.dim() == other.dim());
    is_support &= input.is_contiguous() == other.is_contiguous();
    is_support &= input.is_contiguous() == true;
    if (input.dim() == 1)
    {
      is_support &= input.numel() % 128 == 0;
    }
    for (int i = 0; i < input.dim() && is_support; ++i)
    {
      is_support &= (input.size(i) == other.size(i));
    }
    pattern = is_support ? PATTERN_CONTIGUOUS : 0;
  }

  if (!is_support && (dim == 3 || other.dim() == 3))
  {
    // transpose case
    if (input.is_contiguous() != other.is_contiguous())
    {
      auto tensor_not_conti = input.is_contiguous() ? other : input;
      order_flag = !input.is_contiguous() ? true : false;
      is_support = true;
      // avoid broadcast
      is_support &= input.dim() == other.dim();
      is_support &= input.size(0) == other.size(0);
      is_support &= input.size(1) == other.size(1);
      is_support &= input.size(2) == other.size(2);
      is_support &= tensor_not_conti.stride(1) == 1;
      pattern = is_support ? PATTERN_TRANSPOSE : 0;
    }
    // broadcast case
    else if (input.is_contiguous() && other.is_contiguous())
    {
      is_support = false;
      // input tensor dim and other tensor dim both equal to 3
      if (input.dim() == other.dim())
      {
        // broadcast at dim0 or dim1 or dim2
        auto broadcast_3d_case = [&] (int bcast_dim)
        {
          constexpr int bcast_pattern[3] = {PATTERN_BROADCAST_0, PATTERN_BROADCAST_1, PATTERN_BROADCAST_2};
          if (!is_support && (input.size(bcast_dim) == 1 || other.size(bcast_dim)) && input.size(bcast_dim) != other.size(bcast_dim))
          {
            is_support = true;
            for (int i = 0; i < 3; ++i)
            {
              if (bcast_dim != i) is_support &= input.size(i) == other.size(i);
            }
            is_support &= input.size(bcast_dim) == 1 ? other.size(bcast_dim) != 1 : true;
            pattern = is_support ? bcast_pattern[bcast_dim] : 0;
            order_flag = input.size(bcast_dim) != 1 ? true : false;
            // if (bcast_dim == 1) order_flag = !order_flag;
          }
        };
        // (m, n, k), (1, n, k) or (1, n, k), (m, n, k)
        broadcast_3d_case(0);
        // (m, n, k), (m, 1, k) or (m, 1, k), (m, n, k)
        broadcast_3d_case(1);
        // (m, n, k), (m, n, 1) or (m, n, 1), (m, n, k)
        broadcast_3d_case(2);


        bool first_dim_eq = input.size(0) == other.size(0);
        bool input_bcast_last2dim = input.size(1) == 1 && input.size(2) == 1;
        bool other_bcast_last2dim = other.size(1) == 1 && other.size(2) == 1;
        if (first_dim_eq && (input_bcast_last2dim || other_bcast_last2dim)) // broadcast in last 2 dim
        {
          is_support = true;
          order_flag = other_bcast_last2dim ? true : false;
          pattern = PATTERN_BROADCAST_4;
        }
      }
      // (m, n, k), (n, 1) or (n, 1), (m, n, k)
      else if (input.dim() == 2 || other.dim() == 2)
      {
        is_support = true;
        if (input.dim() == 2)
        {
          is_support &= input.size(0) == other.size(1);
          is_support &= input.size(1) == 1;
          pattern = is_support ? PATTERN_BROADCAST_3 : 0;
          order_flag = false;
        }
        else
        {
          is_support &= other.size(0) == input.size(1);
          is_support &= other.size(1) == 1;
          pattern = is_support ? PATTERN_BROADCAST_3 : 0;
          order_flag = true;
        }
      }
      // (m, n, k), (k) or (k), (m, n, k)
      // (m, n, k), (1) or (1), (m, n, k)
      else if (input.dim() == 1 || other.dim() == 1)
      {
        if (other.dim() == 1)
        {
          if (other.size(0) == 1 || (other.size(0) == input.size(2) && input.size(2) % (128 / input.element_size()) == 0))
          {
            if (input.numel() % (256 * 8 * 16 / input.element_size()) == 0)
            {
              is_support = true;
              pattern = PATTERN_BROADCAST_0;
              order_flag = true;
            }
          }
        }
        else
        {
          if (input.size(0) == 1 || (input.size(0) == other.size(2) && other.size(2) % (128 / other.element_size()) == 0))
          {
            if (other.numel() % (256 * 8 * 16 / other.element_size()) == 0)
            {
              is_support = true;
              pattern = PATTERN_BROADCAST_0;
              order_flag = false;
            }
          }
        }
      }
    }
  }

  if (!is_support && input.dim() != 3 && other.dim() != 3)
  {
    if (input.dim() == other.dim())
    {
      std::vector<int> bcast_dim_index = {};
      for (int i = 0; i < input.dim(); ++i)
      {
        // broadcast condition
        if (input.size(i) != other.size(i) && (input.size(i) == 1 || other.size(i) == 1))
        {
          bcast_dim_index.push_back(i);
        }
      }
      if (bcast_dim_index.size() == 1 && bcast_dim_index[0] != 0 && bcast_dim_index[0] != input.dim() - 1)
      {
        is_support = true;
        pattern = PATTERN_BROADCAST_1;
        order_flag = other.size(bcast_dim_index[0]) == 1 ? true : false;
      }
    }
  }

  // hip does not support double
  if (input.dtype() == torch::kDouble || other.dtype() == torch::kDouble)
  {
    is_support = false;
  }

  if (is_support)
  {
    switch (pattern)
    {
      case PATTERN_TRANSPOSE:
        binary_operation_process<1, Op, T0, T1>(input, other, output, order_flag);
        break;
      case PATTERN_BROADCAST_0:
        binary_operation_process<2, Op, T0, T1>(input, other, output, order_flag);
        break;
      case PATTERN_BROADCAST_1:
        binary_operation_process<3, Op, T0, T1>(input, other, output, order_flag);
        break;
      case PATTERN_CONTIGUOUS:
        binary_operation_process<4, Op, T0, T1>(input, other, output, order_flag);
        break;
      case PATTERN_BROADCAST_2:
        binary_operation_process<5, Op, T0, T1>(input, other, output, order_flag);
        break;
      case PATTERN_BROADCAST_3:
        binary_operation_process<6, Op, T0, T1>(input, other, output, order_flag);
        break;
      case PATTERN_BROADCAST_4:
        binary_operation_process<7, Op, T0, T1>(input, other, output, order_flag);
        break;
      default:
        return ;
    }
    return ;
  }
  else
  {
  output = aiter::aten_compute<Op>(input, other);
  return ;
  }
}
