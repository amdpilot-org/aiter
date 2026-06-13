
// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "layernorm2d_fwd_api_common.hpp"

// clang-format off
//                                      prec_i           prec_o           prec_sy           rm  rn  tm    tn  vn  pd     mv    rpcf   welford   2p   xbias   add  sweep
template float layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  3,  1,  128,  8, true , false, true , false, false,    1,    0,    1>>(const S&, A);
template float layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  3,  1,  256,  4, true , false, true , false, false,    1,    0,    1>>(const S&, A);
template float layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  6,  1,  256,  2, true , false, true , false, false,    1,    0,    1>>(const S&, A);
template float layernorm2d_fwd_<traits_<ck_tile::fp16_t, ck_tile::int8_t, float, float,  1,  3,  1, 1024,  1, true , false, true , false, false,    1,    0,    1>>(const S&, A);

// clang-format on

