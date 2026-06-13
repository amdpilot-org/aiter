#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#ifdef USE_ROCM

#define GENERATE_LOOKUP_TABLE(DTYPE, ETYPE)                                                                                      \
   {                                                                                                                             \
       {{2048, 512, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16384, 512, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{20480, 512, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 576, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 576, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 576, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 576, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 576, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 576, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1024, 576, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1536, 576, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2048, 576, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4096, 576, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8192, 576, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16384, 576, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{20480, 576, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2048, 1536, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4096, 1536, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8192, 1536, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16384, 1536, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{20480, 1536, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{20480, 3072, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 4096, 1280},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{20480, 4096, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{20480, 4608, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 7168, 256},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x128_8x16_16x16_16x16x1_8x32x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 7168, 256},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 7168, 256},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 7168, 256},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 7168, 256},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 7168, 256},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1024, 7168, 256},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1536, 7168, 256},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2048, 7168, 256},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4096, 7168, 256},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8192, 7168, 256},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16384, 7168, 256},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{20480, 7168, 256},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{20480, 7168, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{20480, 7168, 2304},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 256, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 256, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 256, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 256, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 256, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 256, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 256, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 256, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 256, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 256, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1024, 256, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2048, 256, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4096, 256, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8192, 256, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16384, 256, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{32768, 256, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 512, 256},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x128_8x16_16x16_16x16x1_8x32x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 512, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 512, 1024},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 512, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 512, 256},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 512, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 512, 1024},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 512, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 512, 256},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x128x128_8x16_16x16_16x16x1_8x32x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 512, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 512, 1024},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 512, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 512, 256},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 512, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 512, 1024},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 512, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 512, 256},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 512, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 512, 1024},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 512, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 512, 256},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 512, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 512, 1024},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 512, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 512, 256},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 512, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 512, 1024},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 512, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 512, 256},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 512, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 512, 1024},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 512, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 512, 256},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 512, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 512, 1024},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 512, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 512, 256},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 512, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 512, 1024},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 512, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1024, 512, 256},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1024, 512, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1024, 512, 1024},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1024, 512, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2048, 512, 256},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x128x128_8x16_16x16_16x16x1_8x32x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2048, 512, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2048, 512, 1024},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2048, 512, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4096, 512, 256},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4096, 512, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4096, 512, 1024},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4096, 512, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8192, 512, 256},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8192, 512, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{8192, 512, 1024},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8192, 512, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16384, 512, 256},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16384, 512, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16384, 512, 1024},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16384, 512, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32768, 512, 256},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32768, 512, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32768, 512, 1024},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32768, 512, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 1024, 256},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x128_8x16_16x16_16x16x1_8x32x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 1024, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 1024, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 1024, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 1024, 256},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x128_8x16_16x16_16x16x1_8x32x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 1024, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 1024, 1024},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 1024, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 1024, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 1024, 256},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x128_8x16_16x16_16x16x1_8x32x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 1024, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 1024, 1024},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 1024, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 1024, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 1024, 256},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 1024, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 1024, 1024},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 1024, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 1024, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 1024, 256},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 1024, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 1024, 1024},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 1024, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 1024, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 1024, 256},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 1024, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 1024, 1024},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 1024, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 1024, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 1024, 256},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x128_8x16_16x16_16x16x1_8x32x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 1024, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 1024, 1024},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 1024, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 1024, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 1024, 256},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 1024, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 1024, 1024},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 1024, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 1024, 256},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 1024, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 1024, 1024},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 1024, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 1024, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 1024, 256},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 1024, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 1024, 1024},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 1024, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 1024, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1024, 1024, 256},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1024, 1024, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1024, 1024, 1024},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1024, 1024, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1024, 1024, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2048, 1024, 256},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2048, 1024, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2048, 1024, 1024},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2048, 1024, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2048, 1024, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4096, 1024, 256},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4096, 1024, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4096, 1024, 1024},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4096, 1024, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4096, 1024, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8192, 1024, 256},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8192, 1024, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8192, 1024, 1024},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8192, 1024, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8192, 1024, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16384, 1024, 256},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16384, 1024, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16384, 1024, 1024},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16384, 1024, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16384, 1024, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32768, 1024, 256},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32768, 1024, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32768, 1024, 1024},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32768, 1024, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32768, 1024, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 1152, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 1152, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 1152, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 1152, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 1152, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 1152, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 1152, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 1152, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 1152, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 1152, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1024, 1152, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2048, 1152, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4096, 1152, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8192, 1152, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16384, 1152, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32768, 1152, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 1536, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 1536, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 1536, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 1536, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 1536, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 1536, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 1536, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 1536, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 1536, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 1536, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1024, 1536, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2048, 1536, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4096, 1536, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8192, 1536, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16384, 1536, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32768, 1536, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 2048, 1024},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 2048, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 2048, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 2048, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 2048, 1024},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 2048, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 2048, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 2048, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 2048, 1024},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 2048, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 2048, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 2048, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 2048, 1024},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 2048, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 2048, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 2048, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 2048, 1024},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 2048, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 2048, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 2048, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 2048, 1024},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 2048, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 2048, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 2048, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 2048, 1024},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 2048, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 2048, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 2048, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 2048, 1024},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 2048, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 2048, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 2048, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 2048, 1024},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 2048, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 2048, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 2048, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 2048, 1024},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 2048, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 2048, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1024, 2048, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1024, 2048, 1024},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1024, 2048, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1024, 2048, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2048, 2048, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2048, 2048, 1024},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2048, 2048, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2048, 2048, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4096, 2048, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4096, 2048, 1024},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4096, 2048, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4096, 2048, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8192, 2048, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8192, 2048, 1024},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8192, 2048, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8192, 2048, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16384, 2048, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16384, 2048, 1024},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16384, 2048, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16384, 2048, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32768, 2048, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32768, 2048, 1024},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32768, 2048, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32768, 2048, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 2176, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 2176, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 2176, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 2176, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 2176, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 2176, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 2176, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 2176, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 2176, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 2176, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1024, 2176, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2048, 2176, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4096, 2176, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8192, 2176, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16384, 2176, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{32768, 2176, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1, 2304, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 2304, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 2304, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 2304, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 2304, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 2304, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 2304, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 2304, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 2304, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 2304, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1024, 2304, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{2048, 2304, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8192, 2304, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16384, 2304, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32768, 2304, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 2560, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 2560, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 2560, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 2560, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 2560, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 2560, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 2560, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 2560, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 2560, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 2560, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1024, 2560, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2048, 2560, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8192, 2560, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16384, 2560, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32768, 2560, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 3072, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 3072, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 3072, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 3072, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 3072, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 3072, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 3072, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 3072, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 3072, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 3072, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1024, 3072, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2048, 3072, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4096, 3072, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8192, 3072, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16384, 3072, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32768, 3072, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 4096, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 4096, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 4096, 1024},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 4096, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 4096, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 4096, 1024},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 4096, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 4096, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 4096, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 4096, 1024},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 4096, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 4096, 1024},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 4096, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 4096, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1024, 4096, 1024},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1024, 4096, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2048, 4096, 1024},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2048, 4096, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4096, 4096, 1024},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4096, 4096, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8192, 4096, 1024},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8192, 4096, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16384, 4096, 1024},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16384, 4096, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32768, 4096, 1024},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32768, 4096, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 4352, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 4352, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 4352, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 4352, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 4352, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 4352, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 4352, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 4352, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 4352, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 4352, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1024, 4352, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4096, 4352, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16384, 4352, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 4608, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 4608, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 4608, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 4608, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 4608, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 4608, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 4608, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 4608, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 4608, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 4608, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 4608, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 4608, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 4608, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 4608, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 4608, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 4608, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 4608, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 4608, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{512, 4608, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1024, 4608, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1024, 4608, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4096, 4608, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4096, 4608, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8192, 4608, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16384, 4608, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32768, 4608, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 5120, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 5120, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 5120, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 5120, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 5120, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 5120, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 5120, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 5120, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 5120, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 5120, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1024, 5120, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{8192, 5120, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32768, 5120, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 6144, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 6144, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 6144, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 6144, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 6144, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 6144, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 6144, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 6144, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 6144, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 6144, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1024, 6144, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2048, 6144, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4096, 6144, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8192, 6144, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16384, 6144, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32768, 6144, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 8704, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 8704, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 8704, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 8704, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 8704, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 8704, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 8704, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 8704, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 8704, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 8704, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2048, 8704, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8192, 8704, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 9216, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 9216, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 9216, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 9216, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 9216, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 9216, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 9216, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 9216, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 9216, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{512, 9216, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2048, 9216, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4096, 9216, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8192, 9216, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16384, 9216, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32768, 9216, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 10240, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 10240, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 10240, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 10240, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 10240, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 10240, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 10240, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 10240, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 10240, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 10240, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4096, 10240, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 12288, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 12288, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 12288, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 12288, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 12288, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 12288, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 12288, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 12288, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 12288, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 12288, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1024, 12288, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2048, 12288, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4096, 12288, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8192, 12288, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16384, 12288, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32768, 12288, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 17408, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 17408, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 17408, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 17408, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 17408, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 17408, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 17408, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 17408, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{256, 17408, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1024, 17408, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4096, 17408, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16384, 17408, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 20480, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 20480, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 20480, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 20480, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 20480, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 20480, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 20480, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 20480, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{256, 20480, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 4096, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 7168, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 7168, 2304},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 2112, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 2240, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 8192, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 11264, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 4096, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 7168, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 2112, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 2240, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 8192, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 11264, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{48, 3072, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{48, 4096, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{48, 7168, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{48, 7168, 2304},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{48, 2112, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{48, 2240, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{48, 8192, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{48, 11264, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 3072, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 4096, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 7168, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 7168, 2304},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 2112, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 2240, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 8192, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 11264, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{80, 3072, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{80, 4096, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{80, 7168, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{80, 7168, 2304},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{80, 2112, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{80, 2240, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{80, 8192, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{80, 11264, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{96, 3072, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{96, 4096, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{96, 7168, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{96, 7168, 2304},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{96, 2112, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{96, 2240, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{96, 8192, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{96, 11264, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{112, 3072, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{112, 4096, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{112, 7168, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{112, 7168, 2304},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{112, 2112, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{112, 2240, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{112, 8192, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{112, 11264, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 3072, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 4096, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 7168, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 7168, 2304},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 2112, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 2240, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 8192, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 11264, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{144, 3072, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{144, 4096, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{144, 2112, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{144, 2240, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{160, 3072, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{160, 4096, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{160, 2112, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{160, 2240, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{176, 3072, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{176, 4096, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{176, 2112, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{176, 2240, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{176, 11264, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{192, 3072, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{192, 4096, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{192, 2112, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{192, 2240, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{192, 8192, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{192, 11264, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{208, 3072, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{208, 4096, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{208, 7168, 2304},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{208, 2112, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{208, 2240, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{208, 11264, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{224, 3072, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{224, 4096, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{224, 7168, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{224, 2112, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{224, 2240, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{224, 8192, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{224, 11264, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{240, 3072, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{240, 4096, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{240, 7168, 2304},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{240, 2112, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{240, 2240, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{240, 11264, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 3072, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 4096, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 7168, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 2112, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 2240, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 11264, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{288, 3072, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{288, 4096, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{288, 2112, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{288, 2240, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{288, 11264, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{320, 3072, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{320, 4096, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{320, 2112, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{320, 2240, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{320, 11264, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{352, 3072, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{352, 4096, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{352, 7168, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{352, 7168, 2304},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{352, 2112, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{352, 2240, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{352, 11264, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{384, 3072, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{384, 4096, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{384, 2112, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{384, 2240, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{384, 11264, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{416, 3072, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{416, 4096, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{416, 7168, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{416, 2112, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{416, 2240, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{416, 8192, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{416, 11264, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{448, 3072, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{448, 4096, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{448, 7168, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{448, 7168, 2304},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{448, 2112, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{448, 2240, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{448, 8192, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{448, 11264, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{480, 3072, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{480, 4096, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{480, 7168, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{480, 2112, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{480, 2240, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{480, 8192, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{480, 11264, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 3072, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 4096, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 7168, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 2112, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 2240, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 8192, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 11264, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{544, 3072, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{544, 4096, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{544, 7168, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{544, 7168, 2304},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{544, 2112, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{544, 2240, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{544, 8192, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{544, 11264, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{576, 3072, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{576, 4096, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{576, 7168, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{576, 7168, 2304},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{576, 2112, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{576, 2240, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{576, 8192, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{576, 11264, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{608, 3072, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{608, 4096, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{608, 7168, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{608, 7168, 2304},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{608, 2112, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{608, 2240, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{608, 8192, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{608, 11264, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{640, 3072, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{640, 4096, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{640, 7168, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{640, 7168, 2304},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{640, 2112, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{640, 2240, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{640, 8192, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{640, 11264, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{672, 3072, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{672, 4096, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{672, 7168, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{672, 7168, 2304},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{672, 2112, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{672, 2240, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{672, 8192, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{672, 11264, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{704, 4096, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{704, 7168, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{704, 7168, 2304},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{704, 2112, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{704, 2240, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{704, 8192, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{704, 11264, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{736, 4096, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{736, 7168, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{736, 4608, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{736, 7168, 2304},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{736, 2112, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{736, 2240, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{736, 8192, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{736, 11264, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{768, 3072, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{768, 4096, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{768, 7168, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{768, 7168, 2304},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{768, 2112, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{768, 2240, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{768, 8192, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{768, 11264, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{800, 4096, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{800, 7168, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{800, 7168, 2304},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{800, 2112, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{800, 2240, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{800, 8192, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{800, 11264, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{832, 3072, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{832, 4096, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{832, 7168, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{832, 7168, 2304},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{832, 2112, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{832, 2240, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{832, 8192, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{832, 11264, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{864, 3072, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{864, 4096, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{864, 7168, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{864, 7168, 2304},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{864, 2112, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{864, 2240, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{864, 8192, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{864, 11264, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{896, 4096, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{896, 7168, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{896, 4608, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{896, 7168, 2304},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{896, 2112, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{896, 2240, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{896, 8192, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{896, 11264, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{928, 4096, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{928, 7168, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{928, 4608, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{928, 7168, 2304},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{928, 2112, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{928, 2240, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{928, 8192, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{928, 11264, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{960, 4096, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{960, 7168, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{960, 4608, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{960, 7168, 2304},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{960, 2112, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{960, 2240, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{960, 8192, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{960, 11264, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{992, 3072, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{992, 4096, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{992, 7168, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{992, 4608, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{992, 7168, 2304},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{992, 2112, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{992, 2240, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{992, 8192, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{992, 11264, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1024, 3072, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1024, 4096, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1024, 7168, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1024, 4608, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1024, 7168, 2304},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1024, 2112, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1024, 2240, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1024, 8192, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1024, 11264, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1088, 3072, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1088, 4096, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1088, 7168, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1088, 4608, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1088, 7168, 2304},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1088, 2112, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1088, 2240, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1088, 8192, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1088, 11264, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1152, 3072, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1152, 4096, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1152, 7168, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1152, 4608, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1152, 7168, 2304},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1152, 2112, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1152, 2240, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1152, 8192, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1152, 11264, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1216, 3072, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1216, 4096, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1216, 7168, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1216, 4608, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1216, 7168, 2304},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1216, 2112, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1216, 2240, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1216, 8192, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1216, 11264, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1280, 3072, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1280, 4096, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1280, 7168, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1280, 4608, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1280, 7168, 2304},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1280, 2112, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1280, 2240, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1280, 8192, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1280, 11264, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1344, 3072, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1344, 4096, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1344, 7168, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1344, 4608, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1344, 7168, 2304},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1344, 2112, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1344, 2240, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1344, 8192, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1344, 11264, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1408, 3072, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1408, 4096, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1408, 7168, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1408, 4608, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1408, 7168, 2304},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1408, 2112, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1408, 2240, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1408, 8192, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1408, 11264, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1472, 3072, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1472, 4096, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1472, 7168, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1472, 4608, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1472, 7168, 2304},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1472, 2112, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1472, 2240, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1472, 8192, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1472, 11264, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1536, 3072, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1536, 4096, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1536, 7168, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1536, 4608, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1536, 7168, 2304},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1536, 2112, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1536, 2240, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1536, 8192, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1536, 11264, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1600, 3072, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1600, 4096, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1600, 7168, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1600, 4608, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1600, 7168, 2304},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1600, 2112, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1600, 2240, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1600, 8192, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1600, 11264, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1664, 3072, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1664, 4096, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1664, 7168, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1664, 4608, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1664, 7168, 2304},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1664, 2112, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1664, 2240, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1664, 8192, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1664, 11264, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1728, 3072, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1728, 4096, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1728, 7168, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1728, 4608, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1728, 7168, 2304},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1728, 2112, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1728, 2240, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1728, 8192, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1728, 11264, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1792, 3072, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1792, 4096, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1792, 7168, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1792, 4608, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1792, 7168, 2304},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1792, 2112, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1792, 2240, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1792, 8192, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1792, 11264, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1856, 3072, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1856, 4096, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1856, 7168, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1856, 4608, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1856, 7168, 2304},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1856, 2112, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1856, 2240, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1856, 8192, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1856, 11264, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1920, 3072, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1920, 4096, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1920, 7168, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1920, 4608, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1920, 7168, 2304},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1920, 2112, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1920, 2240, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1920, 8192, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1920, 11264, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1984, 3072, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1984, 4096, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1984, 7168, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1984, 4608, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1984, 7168, 2304},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1984, 2112, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1984, 2240, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1984, 8192, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1984, 11264, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2048, 3072, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2048, 4096, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2048, 7168, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2048, 4608, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2048, 7168, 2304},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2048, 2112, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{2048, 2240, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{2048, 8192, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2048, 11264, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4096, 3072, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4096, 4096, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4096, 7168, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4096, 4608, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4096, 7168, 2304},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4096, 2112, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{4096, 2240, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{4096, 8192, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4096, 11264, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8192, 3072, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8192, 4096, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8192, 7168, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8192, 4608, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8192, 7168, 2304},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8192, 128, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8192, 2112, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{8192, 2240, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{8192, 8192, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8192, 11264, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16384, 3072, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16384, 4096, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16384, 7168, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16384, 4608, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16384, 7168, 2304},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16384, 2112, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{16384, 2240, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{16384, 8192, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16384, 11264, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32768, 3072, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32768, 4096, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32768, 7168, 2048},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32768, 4608, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32768, 7168, 2304},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32768, 2112, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{32768, 2240, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{32768, 8192, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32768, 11264, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 3072, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 3072, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{48, 3072, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 3072, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{80, 3072, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{96, 3072, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{112, 3072, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 3072, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{144, 3072, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{160, 3072, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{176, 3072, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{192, 3072, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{208, 3072, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{224, 3072, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{240, 3072, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 3072, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{288, 3072, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{320, 3072, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{352, 3072, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{384, 3072, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{416, 3072, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{448, 3072, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{480, 3072, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 3072, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{544, 3072, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{576, 3072, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{608, 3072, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{640, 3072, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{672, 3072, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{704, 3072, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{736, 3072, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{768, 3072, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{800, 3072, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{832, 3072, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{864, 3072, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{896, 3072, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{928, 3072, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{960, 3072, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{992, 3072, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1024, 3072, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1088, 3072, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1152, 3072, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1216, 3072, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1280, 3072, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1344, 3072, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1408, 3072, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1472, 3072, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1536, 3072, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1600, 3072, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1664, 3072, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1728, 3072, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1792, 3072, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1856, 3072, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1920, 3072, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1984, 3072, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2048, 3072, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4096, 3072, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8192, 3072, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16384, 3072, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32768, 3072, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 4096, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 4096, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{48, 4096, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 4096, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{80, 4096, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{96, 4096, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{112, 4096, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 4096, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{144, 4096, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{160, 4096, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{176, 4096, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{192, 4096, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{208, 4096, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{224, 4096, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{240, 4096, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 4096, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{288, 4096, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{320, 4096, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{352, 4096, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{384, 4096, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{416, 4096, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{448, 4096, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{480, 4096, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 4096, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{544, 4096, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{576, 4096, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{608, 4096, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{640, 4096, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{672, 4096, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{704, 4096, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{736, 4096, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{768, 4096, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{800, 4096, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{832, 4096, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{864, 4096, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{896, 4096, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{928, 4096, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{960, 4096, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{992, 4096, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1024, 4096, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1088, 4096, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1152, 4096, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1216, 4096, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1280, 4096, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1344, 4096, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1408, 4096, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1472, 4096, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1536, 4096, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1600, 4096, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1664, 4096, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1728, 4096, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1792, 4096, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1856, 4096, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1920, 4096, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1984, 4096, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2048, 4096, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4096, 4096, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8192, 4096, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16384, 4096, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32768, 4096, 3072},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1152, 128, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{208, 3072, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{288, 3072, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{480, 3072, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 3072, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{544, 3072, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{576, 3072, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{768, 3072, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{832, 3072, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{960, 3072, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1024, 3072, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1152, 3072, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1280, 3072, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1344, 3072, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1408, 3072, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1472, 3072, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1664, 3072, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1792, 3072, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1920, 3072, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2048, 3072, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8192, 3072, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32768, 3072, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 4096, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{144, 4096, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{208, 4096, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{288, 4096, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{320, 4096, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{576, 4096, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{640, 4096, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{704, 4096, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{768, 4096, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{832, 4096, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{896, 4096, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1024, 4096, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1088, 4096, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1280, 4096, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1408, 4096, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1600, 4096, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1664, 4096, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1792, 4096, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2048, 4096, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8192, 4096, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16384, 4096, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32768, 4096, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{416, 4608, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 4608, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{576, 4608, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{704, 4608, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{832, 4608, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 7168, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 7168, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 7168, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 7168, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 7168, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 7168, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{160, 7168, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{224, 7168, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 7168, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{576, 7168, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{640, 7168, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{704, 7168, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{832, 7168, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{896, 7168, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1088, 7168, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1152, 7168, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1280, 7168, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1472, 7168, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1536, 7168, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1728, 7168, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1792, 7168, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1984, 7168, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2048, 7168, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 8192, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{160, 8192, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{224, 8192, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{352, 8192, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{384, 8192, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{416, 8192, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{480, 8192, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 8192, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{608, 8192, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{704, 8192, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{800, 8192, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{832, 8192, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{864, 8192, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{992, 8192, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1024, 8192, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1088, 8192, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1280, 8192, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1344, 8192, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1600, 8192, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1664, 8192, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1728, 8192, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1792, 8192, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1856, 8192, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1984, 8192, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2048, 8192, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4096, 8192, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8192, 8192, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16384, 8192, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 11264, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x128_8x16_16x16_16x16x1_8x32x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 11264, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{96, 11264, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{192, 11264, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{208, 11264, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{224, 11264, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 11264, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{320, 11264, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{384, 11264, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{416, 11264, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{448, 11264, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 11264, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{576, 11264, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1216, 11264, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1472, 11264, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1600, 11264, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1728, 11264, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1856, 11264, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 36864, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 36864, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 36864, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 36864, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x256x128_8x16_16x16_16x16x1_8x32x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 36864, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 36864, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 36864, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 36864, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 36864, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1024, 36864, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2048, 36864, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4096, 36864, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{6144, 36864, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{10240, 36864, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{12288, 36864, 7168},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 128, 6144},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 128, 6144},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 128, 6144},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 128, 6144},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 128, 6144},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 128, 6144},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{48, 128, 6144},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 128, 6144},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 128, 6144},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 128, 6144},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 128, 6144},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1024, 128, 6144},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2048, 128, 6144},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4096, 128, 6144},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8192, 128, 6144},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16384, 128, 6144},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 2624, 6144},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 2624, 6144},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 2624, 6144},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 2624, 6144},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 2624, 6144},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 2624, 6144},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{48, 2624, 6144},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 2624, 6144},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 2624, 6144},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 2624, 6144},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 2624, 6144},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1024, 2624, 6144},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 3072, 6144},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 3072, 6144},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 3072, 6144},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 3072, 6144},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 3072, 6144},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 3072, 6144},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{48, 3072, 6144},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 3072, 6144},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 3072, 6144},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 3072, 6144},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 3072, 6144},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1024, 3072, 6144},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2048, 3072, 6144},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4096, 3072, 6144},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8192, 3072, 6144},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 3584, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 3584, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 3584, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 3584, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 3584, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 3584, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{48, 3584, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x128_8x16_16x16_16x16x1_8x32x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 3584, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 3584, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 3584, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 3584, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1024, 3584, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2048, 3584, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4096, 3584, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8192, 3584, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16384, 3584, 512},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 6144, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 6144, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 6144, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 6144, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 6144, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 6144, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{48, 6144, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 6144, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 6144, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 6144, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 6144, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1024, 6144, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2048, 6144, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4096, 6144, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8192, 6144, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16384, 6144, 1536},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 1280, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 1280, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 1280, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 1280, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 1280, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 1280, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 1280, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 1280, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 1280, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 1280, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1024, 1280, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2048, 1280, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4096, 1280, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 2304, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 2304, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 2304, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 2304, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 2304, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 2304, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 2304, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 2304, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 2304, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 2304, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1024, 2304, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{2048, 2304, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 4096, 1024},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 4096, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 4096, 8192},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 4096, 1024},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 4096, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 4096, 8192},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 4096, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 4096, 8192},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 4096, 1024},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 4096, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 4096, 8192},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 4096, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 4096, 8192},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 4096, 1024},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 4096, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 4096, 8192},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 4096, 1024},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 4096, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 4096, 8192},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 4096, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 4096, 8192},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 4096, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 4096, 8192},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 4096, 1024},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{512, 4096, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 4096, 8192},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1024, 4096, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1024, 4096, 8192},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2048, 4096, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4096, 4096, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4096, 4096, 8192},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 4608, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 9216, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 9216, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 9216, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 9216, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 9216, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 9216, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 9216, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 9216, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 9216, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 9216, 4096},                                                                                                       \
        a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
   }

#endif // USE_ROCM
