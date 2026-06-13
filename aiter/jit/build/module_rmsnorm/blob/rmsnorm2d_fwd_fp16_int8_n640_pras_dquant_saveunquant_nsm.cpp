
// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "rmsnorm2d_fwd_api_common.hpp"

// clang-format off
//                                                                                    rm  rn  tm    tn  vn    pd     rms     2p
template float rmsnorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float, ck_tile::fp16_t,  1,  5,  4,   64,  2, true , false, true , false,    1,    2,    0>>(const S&, A);
template float rmsnorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float, ck_tile::fp16_t,  1,  5,  4,  128,  1, true , false, true , false,    1,    2,    0>>(const S&, A);

// clang-format on

