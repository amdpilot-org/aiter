#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#ifdef USE_ROCM

#define GENERATE_LOOKUP_TABLE(DTYPE, ETYPE)                                                                                      \
   {                                                                                                                             \
       {{8, 16, 4096},                                                                                                       \
        a8w8_blockscale_cktile_16x128x256_1x4x1_16x16x64_intrawave_0x1x0_1<DTYPE, ETYPE>},                       \
       {{32, 16, 4096},                                                                                                       \
        a8w8_blockscale_cktile_16x128x256_1x4x1_16x16x64_intrawave_0x1x0_1<DTYPE, ETYPE>},                       \
       {{64, 16, 4096},                                                                                                       \
        a8w8_blockscale_cktile_16x128x256_1x4x1_16x16x64_intrawave_0x1x0_1<DTYPE, ETYPE>},                       \
       {{128, 16, 4096},                                                                                                       \
        a8w8_blockscale_cktile_16x128x256_1x4x1_16x16x64_intrawave_0x1x0_1<DTYPE, ETYPE>},                       \
       {{512, 16, 4096},                                                                                                       \
        a8w8_blockscale_cktile_16x128x256_1x4x1_16x16x64_intrawave_0x1x0_3<DTYPE, ETYPE>},                       \
       {{1024, 16, 4096},                                                                                                       \
        a8w8_blockscale_cktile_16x128x256_1x4x1_16x16x64_intrawave_0x1x0_1<DTYPE, ETYPE>},                       \
       {{2048, 16, 4096},                                                                                                       \
        a8w8_blockscale_cktile_16x128x256_1x4x1_16x16x64_intrawave_0x1x0_1<DTYPE, ETYPE>},                       \
       {{4096, 16, 4096},                                                                                                       \
        a8w8_blockscale_cktile_16x128x256_1x4x1_16x16x64_intrawave_0x1x0_1<DTYPE, ETYPE>},                       \
       {{4, 32, 4096},                                                                                                       \
        a8w8_blockscale_cktile_16x128x256_1x4x1_16x16x64_intrawave_0x1x0_1<DTYPE, ETYPE>},                       \
       {{1024, 32, 4096},                                                                                                       \
        a8w8_blockscale_cktile_16x128x256_1x4x1_16x16x64_intrawave_0x1x0_1<DTYPE, ETYPE>},                       \
       {{8192, 32, 4096},                                                                                                       \
        a8w8_blockscale_cktile_16x128x256_1x4x1_16x16x64_intrawave_0x1x0_1<DTYPE, ETYPE>},                       \
       {{1, 2048, 512},                                                                                                       \
        a8w8_blockscale_cktile_16x128x256_1x4x1_16x16x64_intrawave_0x1x0_4<DTYPE, ETYPE>},                       \
       {{64, 2240, 7168},                                                                                                       \
        a8w8_blockscale_cktile_16x128x256_1x4x1_16x16x64_intrawave_0x1x0_1<DTYPE, ETYPE>},                       \
   }

#endif // USE_ROCM
