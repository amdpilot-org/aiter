
// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "rmsnorm2d_fwd_api_common.hpp"

// clang-format off
//                                                                                    rm  rn  tm    tn  vn    pd     rms     2p
template float rmsnorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float, ck_tile::bf16_t,  1,  4,  1,  256,  8, true , false, true , false,    1,    2,    1>>(const S&, A);
template float rmsnorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float, ck_tile::bf16_t,  1,  4,  1,  512,  4, true , false, true , false,    1,    2,    1>>(const S&, A);
template float rmsnorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float, ck_tile::bf16_t,  1,  4,  1, 1024,  2, true , false, true , false,    1,    2,    1>>(const S&, A);
template float rmsnorm2d_fwd_<traits_<ck_tile::bf16_t, ck_tile::bf16_t, float, float, ck_tile::bf16_t,  1,  8,  1, 1024,  1, true , false, true , false,    1,    2,    1>>(const S&, A);

// clang-format on

