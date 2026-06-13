
// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "rmsnorm2d_fwd_api_common.hpp"

// clang-format off
//                                                                                    rm  rn  tm    tn  vn    pd     rms     2p
template float rmsnorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float, ck_tile::fp16_t,  1,  1,  4,   64,  4, true , false, false, false,    1,    2,    0>>(const S&, A);
template float rmsnorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float, ck_tile::fp16_t,  1,  2,  4,   64,  2, true , false, false, false,    1,    2,    0>>(const S&, A);
template float rmsnorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::fp16_t, float, float, ck_tile::fp16_t,  1,  4,  4,   64,  1, true , false, false, false,    1,    2,    0>>(const S&, A);

// clang-format on

