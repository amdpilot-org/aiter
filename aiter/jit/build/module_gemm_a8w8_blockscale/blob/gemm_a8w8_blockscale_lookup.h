#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#ifdef USE_ROCM

#define GENERATE_LOOKUP_TABLE(DTYPE, ETYPE)                                                                                      \
   {                                                                                                                             \
       {{8192, 512, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{16384, 512, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x128_16x16_16x16_2x1_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 1024, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 4096, 1280},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x128x256_16x16_16x16_1x2_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 128, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 512, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 1024, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 2112, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 2240, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 3072, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 4096, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x64x128_16x16_32x32_2x1_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1, 4096, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 4608, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 7168, 256},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 7168, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x256_16x16_16x16_2x1_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 7168, 2048},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 7168, 2304},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 7168, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 7168, 4608},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 7168, 16384},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 7168, 18432},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 8192, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x256_16x16_16x16_2x1_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 8192, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 9216, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 11264, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 12288, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 14336, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 16384, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 20480, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x128x256_16x16_16x16_1x2_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 24576, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 32768, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 32768, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 36864, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 128, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 512, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 1024, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 2112, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 2240, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 3072, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 4096, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 4096, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 4608, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 7168, 256},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 7168, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x256_16x16_16x16_2x1_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 7168, 2048},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 7168, 2304},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 7168, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 7168, 4608},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 7168, 16384},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 7168, 18432},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 8192, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x256_16x16_16x16_2x1_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 8192, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 9216, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 11264, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 12288, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 14336, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 16384, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 20480, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x128x256_16x16_16x16_1x2_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 24576, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 32768, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 32768, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 36864, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 128, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 512, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 1024, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 2112, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 2240, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 3072, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 4096, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{4, 4096, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 4608, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 7168, 256},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 7168, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x256_16x16_16x16_2x1_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 7168, 2048},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 7168, 2304},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 7168, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 7168, 4608},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 7168, 16384},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 7168, 18432},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 8192, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x256_16x16_16x16_2x1_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 8192, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 9216, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 11264, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 12288, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 14336, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 16384, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 20480, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x128x256_16x16_16x16_1x2_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 24576, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 32768, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 32768, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 36864, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 128, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 512, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 1024, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 2112, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 2240, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 3072, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 4096, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 4096, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 4608, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 7168, 256},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 7168, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x256_16x16_16x16_2x1_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 7168, 2048},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 7168, 2304},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 7168, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 7168, 4608},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 7168, 16384},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 7168, 18432},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 8192, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x256_16x16_16x16_2x1_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 8192, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 9216, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 11264, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 12288, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 14336, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 16384, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 20480, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x128x256_16x16_16x16_1x2_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 24576, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 32768, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 32768, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 36864, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 64, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 128, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 512, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 576, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 1024, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 1536, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 2112, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 2240, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 3072, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 4096, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x128_16x16_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{16, 4096, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 4608, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 7168, 256},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 7168, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 7168, 2048},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 7168, 2304},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 7168, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 7168, 4608},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 7168, 16384},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 7168, 18432},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 8192, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 8192, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 9216, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 11264, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 12288, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 14336, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 16384, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 20480, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x128x256_16x16_16x16_1x2_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 24576, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 32768, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x128x256_16x16_16x16_1x2_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 32768, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 36864, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 64, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 128, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 512, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 576, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 1024, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 1536, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 2112, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 2240, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 3072, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 4096, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x256_16x16_16x16_2x1_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 4096, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 4608, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 7168, 256},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x256_16x16_16x16_2x1_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 7168, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x256_16x16_16x16_2x1_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 7168, 2048},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 7168, 2304},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 7168, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 7168, 4608},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 7168, 16384},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 7168, 18432},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 8192, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x256_16x16_16x16_2x1_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 8192, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 9216, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x128x256_16x16_16x16_1x2_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 11264, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 12288, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 14336, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x128x256_16x16_16x16_1x2_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 16384, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x256_16x16_16x16_2x1_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 20480, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x128x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 24576, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x128x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 32768, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x256_16x16_16x16_2x1_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 32768, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x128x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 36864, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x256x128_16x16_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 64, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 128, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 512, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 576, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 1024, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 1536, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 2112, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 2240, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 3072, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 4096, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 4096, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 4608, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x128x256_16x16_16x16_1x2_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 7168, 256},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{64, 7168, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x256_16x16_16x16_2x1_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 7168, 2048},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x128x256_16x16_16x16_1x2_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 7168, 2304},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x128x256_16x16_16x16_1x2_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 7168, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x128x256_16x16_16x16_1x2_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 7168, 4608},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x128x256_16x16_16x16_1x2_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 7168, 16384},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 7168, 18432},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 8192, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x256_16x16_16x16_2x1_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 8192, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 9216, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 11264, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 12288, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 14336, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 16384, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 20480, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 24576, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 32768, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x128x128_16x16_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{64, 32768, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 36864, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x256x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{96, 128, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{96, 512, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{96, 1024, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{96, 2112, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{96, 2240, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{96, 3072, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{96, 4096, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x256_16x16_16x16_2x1_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{96, 4096, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x128x256_16x16_16x16_1x2_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{96, 4608, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x128x256_16x16_16x16_1x2_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{96, 7168, 256},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{96, 7168, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x256_16x16_16x16_2x1_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{96, 7168, 2048},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{96, 7168, 2304},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{96, 7168, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{96, 7168, 4608},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{96, 7168, 16384},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{96, 7168, 18432},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{96, 8192, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x256_16x16_16x16_2x1_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{96, 8192, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{96, 9216, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x128x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{96, 11264, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{96, 12288, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{96, 14336, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{96, 16384, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x256_16x16_16x16_2x1_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{96, 20480, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x128x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{96, 24576, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{96, 32768, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x256_16x16_16x16_2x1_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{96, 32768, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{96, 36864, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x128x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 64, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 128, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 512, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 576, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 1024, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 1536, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 2112, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 2240, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 3072, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x128x256_16x16_16x16_1x2_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 4096, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x256_16x16_16x16_2x1_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 4096, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x128x256_16x16_16x16_1x2_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 4608, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 7168, 256},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 7168, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x256_16x16_16x16_2x1_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 7168, 2048},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 7168, 2304},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 7168, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 7168, 4608},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 7168, 16384},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 7168, 18432},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 8192, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x256_16x16_16x16_2x1_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 8192, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 9216, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 11264, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 12288, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 14336, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 16384, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x256x128_16x16_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 20480, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 24576, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 32768, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x128x128_16x16_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{128, 32768, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{128, 36864, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{160, 128, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{160, 512, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{160, 1024, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{160, 2112, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{160, 2240, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{160, 3072, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x128x256_16x16_16x16_1x2_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{160, 4096, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x256_16x16_16x16_2x1_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{160, 4096, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{160, 4608, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{160, 7168, 256},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x256_16x16_16x16_2x1_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{160, 7168, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x256_16x16_16x16_2x1_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{160, 7168, 2048},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{160, 7168, 2304},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{160, 7168, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{160, 7168, 4608},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{160, 7168, 16384},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{160, 7168, 18432},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{160, 8192, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x256_16x16_16x16_2x1_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{160, 8192, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{160, 9216, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{160, 11264, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x128x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{160, 12288, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x128x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{160, 14336, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{160, 16384, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x256_16x16_16x16_2x1_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{160, 20480, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x128x128_16x16_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{160, 24576, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{160, 32768, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{160, 32768, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x128x128_16x16_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{160, 36864, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x128_16x16_16x16_2x1_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{192, 128, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{192, 512, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{192, 1024, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{192, 2112, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{192, 2240, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{192, 3072, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{192, 4096, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x256_16x16_16x16_2x1_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{192, 4096, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{192, 4608, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{192, 7168, 256},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x256_16x16_16x16_2x1_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{192, 7168, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x256_16x16_16x16_2x1_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{192, 7168, 2048},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{192, 7168, 2304},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{192, 7168, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{192, 7168, 4608},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{192, 7168, 16384},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{192, 7168, 18432},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{192, 8192, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x256_16x16_16x16_2x1_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{192, 8192, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{192, 9216, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{192, 11264, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{192, 12288, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{192, 14336, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{192, 16384, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{192, 20480, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x128x128_16x16_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{192, 24576, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{192, 32768, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x128x256_16x16_32x32_2x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{192, 32768, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x128x128_16x16_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{192, 36864, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x256x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{224, 128, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{224, 512, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{224, 1024, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{224, 2112, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{224, 2240, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{224, 3072, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{224, 4096, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x256_16x16_16x16_2x1_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{224, 4096, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{224, 4608, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x128x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{224, 7168, 256},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{224, 7168, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{224, 7168, 2048},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{224, 7168, 2304},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{224, 7168, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{224, 7168, 4608},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{224, 7168, 16384},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{224, 7168, 18432},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{224, 8192, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x128x256_16x16_32x32_2x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{224, 8192, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{224, 9216, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x128x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{224, 11264, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{224, 12288, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{224, 14336, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{224, 16384, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{224, 20480, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{224, 24576, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x128x128_16x16_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{224, 32768, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{224, 32768, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{224, 36864, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{256, 64, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 128, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 512, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 576, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 1024, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 1536, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x128x256_16x16_16x16_1x2_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 2112, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 2240, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 3072, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 4096, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x256_16x16_16x16_2x1_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 4096, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 4608, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 7168, 256},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x128_8x16_16x16_1x1_16x16x1_8x32x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 7168, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 7168, 2048},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 7168, 2304},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 7168, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 7168, 4608},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 7168, 16384},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 7168, 18432},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 8192, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 8192, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 9216, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{256, 11264, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 12288, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{256, 14336, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{256, 16384, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{256, 20480, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 24576, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x128x128_16x16_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{256, 32768, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{256, 32768, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{256, 36864, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{288, 128, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{288, 512, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{288, 1024, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{288, 2112, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{288, 2240, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{288, 3072, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x128x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{288, 4096, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x256_16x16_16x16_2x1_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{288, 4096, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{288, 4608, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{288, 7168, 256},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x256_16x16_16x16_2x1_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{288, 7168, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x128x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{288, 7168, 2048},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x128x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{288, 7168, 2304},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x128x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{288, 7168, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x128x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{288, 7168, 4608},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x128x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{288, 7168, 16384},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x128x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{288, 7168, 18432},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x128x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{288, 8192, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x256_16x16_16x16_2x1_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{288, 8192, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{288, 9216, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{288, 11264, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x128x128_16x16_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{288, 12288, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x128x128_16x16_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{288, 14336, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{288, 16384, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{288, 20480, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{288, 24576, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x256x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{288, 32768, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x128x256_16x16_32x32_2x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{288, 36864, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x256x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{320, 128, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{320, 512, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{320, 1024, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{320, 2112, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{320, 2240, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{320, 3072, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x128x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{320, 4096, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x256_16x16_16x16_2x1_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{320, 4096, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{320, 4608, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{320, 7168, 256},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{320, 7168, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x256_16x16_16x16_2x1_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{320, 7168, 2048},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{320, 7168, 2304},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{320, 7168, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{320, 7168, 4608},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{320, 7168, 16384},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x128x128_16x16_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{320, 7168, 18432},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x128x128_16x16_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{320, 8192, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x256_16x16_16x16_2x1_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{320, 8192, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{320, 9216, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{320, 11264, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x128x128_16x16_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{320, 12288, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x256x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{320, 14336, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{320, 16384, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{320, 20480, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{320, 24576, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x256x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{320, 32768, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x128x128_16x16_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{320, 36864, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{352, 128, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{352, 512, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{352, 1024, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{352, 2112, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{352, 2240, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{352, 3072, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{352, 4096, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x128x256_16x16_32x32_2x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{352, 4096, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{352, 4608, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{352, 7168, 256},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x256_16x16_16x16_2x1_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{352, 7168, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x256_16x16_16x16_2x1_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{352, 7168, 2048},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{352, 7168, 2304},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{352, 7168, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{352, 7168, 4608},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{352, 7168, 16384},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{352, 7168, 18432},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{352, 8192, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{352, 8192, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{352, 9216, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{352, 11264, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{352, 12288, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{352, 14336, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x128x128_16x16_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{352, 16384, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{352, 20480, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{352, 32768, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{384, 128, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{384, 512, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{384, 1024, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{384, 2112, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{384, 2240, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{384, 3072, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{384, 4096, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x256_16x16_16x16_2x1_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{384, 4096, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{384, 4608, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{384, 7168, 256},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{384, 7168, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{384, 7168, 2048},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{384, 7168, 2304},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{384, 7168, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{384, 7168, 4608},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{384, 7168, 16384},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{384, 7168, 18432},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{384, 8192, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{384, 8192, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{384, 9216, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{384, 11264, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{384, 12288, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{384, 14336, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x128x128_16x16_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{384, 16384, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x128x256_16x16_32x32_2x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{384, 20480, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{384, 24576, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x128_16x16_16x16_2x1_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{384, 32768, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{384, 36864, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{416, 128, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{416, 512, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{416, 1024, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{416, 2112, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{416, 2240, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{416, 3072, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{416, 4096, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x128x256_16x16_32x32_2x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{416, 4096, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{416, 4608, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{416, 7168, 256},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x256_16x16_16x16_2x1_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{416, 7168, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x256_16x16_16x16_2x1_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{416, 7168, 2048},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{416, 7168, 2304},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{416, 7168, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{416, 7168, 4608},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{416, 7168, 16384},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{416, 7168, 18432},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{416, 8192, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{416, 8192, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{416, 9216, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x256x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{416, 11264, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{416, 12288, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x128x128_16x16_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{416, 14336, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{416, 16384, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{416, 24576, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{416, 32768, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{416, 32768, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{448, 128, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{448, 512, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{448, 1024, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{448, 2112, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{448, 2240, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{448, 3072, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{448, 4096, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x256_16x16_16x16_2x1_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{448, 4096, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{448, 4608, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{448, 7168, 256},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{448, 7168, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x128x128_16x16_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{448, 7168, 2048},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{448, 7168, 2304},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{448, 7168, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{448, 7168, 4608},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{448, 7168, 16384},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{448, 7168, 18432},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{448, 8192, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{448, 8192, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{448, 9216, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x256x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{448, 11264, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{448, 12288, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x128x128_16x16_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{448, 14336, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{448, 16384, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{448, 20480, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x128x128_16x16_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{448, 24576, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{448, 32768, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{448, 32768, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{448, 36864, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{480, 128, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{480, 512, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{480, 1024, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x128x256_16x16_16x16_1x2_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{480, 2112, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{480, 2240, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{480, 3072, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{480, 4096, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{480, 4096, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x128x128_16x16_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{480, 4608, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{480, 7168, 256},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x256_16x16_16x16_2x1_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{480, 7168, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{480, 7168, 2048},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{480, 7168, 2304},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{480, 7168, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{480, 7168, 4608},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{480, 7168, 16384},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{480, 7168, 18432},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{480, 8192, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{480, 8192, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{480, 9216, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{480, 11264, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x128x128_16x16_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{480, 12288, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{480, 14336, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{480, 16384, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{480, 24576, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{480, 32768, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{480, 32768, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{480, 36864, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x128x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 64, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 128, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 512, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 576, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 1024, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 1536, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 2112, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 2240, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 3072, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 4096, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 4096, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 4608, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 7168, 256},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x128x128_16x16_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 7168, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{512, 7168, 2048},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{512, 7168, 2304},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{512, 7168, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{512, 7168, 4608},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{512, 7168, 16384},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{512, 7168, 18432},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{512, 8192, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{512, 8192, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{512, 9216, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 11264, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{512, 12288, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x128x128_16x16_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{512, 14336, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{512, 16384, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{512, 24576, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{512, 32768, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{512, 32768, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{512, 36864, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1024, 64, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1024, 128, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1024, 512, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x128x256_16x16_16x16_1x2_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1024, 576, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1024, 1024, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1024, 1536, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1024, 2112, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1024, 2240, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1024, 3072, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1024, 4096, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1024, 4096, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1024, 4608, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x128_16x16_16x16_2x1_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1024, 7168, 256},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x128x128_16x16_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1024, 7168, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1024, 7168, 2048},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1024, 7168, 2304},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1024, 7168, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1024, 7168, 4608},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1024, 7168, 16384},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1024, 7168, 18432},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1024, 8192, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1024, 8192, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1024, 11264, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1024, 12288, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1024, 14336, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1024, 16384, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1024, 20480, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1024, 24576, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1024, 32768, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1024, 32768, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1024, 36864, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1536, 512, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1536, 576, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1536, 1536, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1536, 3072, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1536, 4096, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1536, 4608, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x128_16x16_16x16_2x1_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1536, 7168, 256},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x128x128_16x16_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1536, 7168, 2048},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x128_16x16_16x16_2x1_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2048, 64, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2048, 128, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2048, 512, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2048, 576, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2048, 1024, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2048, 1536, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{2048, 2112, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x128x128_16x16_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{2048, 2240, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x128x128_16x16_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{2048, 3072, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{2048, 4096, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{2048, 4096, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{2048, 7168, 256},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x128x128_16x16_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2048, 7168, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{2048, 7168, 2048},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{2048, 7168, 2304},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{2048, 7168, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{2048, 7168, 4608},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{2048, 7168, 16384},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{2048, 7168, 18432},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{2048, 8192, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{2048, 8192, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{2048, 12288, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{2048, 14336, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{2048, 16384, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{2048, 20480, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{2048, 24576, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{2048, 32768, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{2048, 32768, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{4096, 64, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4096, 128, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4096, 512, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4096, 576, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4096, 1024, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{4096, 1536, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x128_16x16_16x16_2x1_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4096, 3072, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{4096, 4096, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{4096, 4096, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{4096, 7168, 256},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x128x128_16x16_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4096, 7168, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{4096, 7168, 2048},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{4096, 7168, 2304},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x128_16x16_16x16_2x1_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4096, 7168, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{4096, 7168, 4608},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{4096, 7168, 16384},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x128_16x16_16x16_2x1_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4096, 7168, 18432},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{4096, 8192, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{4096, 8192, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{4096, 11264, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{4096, 12288, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{4096, 14336, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{4096, 16384, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{4096, 20480, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{4096, 24576, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{4096, 32768, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x256x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4096, 32768, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{6144, 128, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{6144, 512, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{6144, 1024, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{6144, 2112, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x256_16x16_16x16_2x1_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{6144, 2240, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x256_16x16_16x16_2x1_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{6144, 3072, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{6144, 4096, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{6144, 4096, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x128_16x16_16x16_2x1_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{6144, 4608, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{6144, 7168, 256},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x128x256_16x16_32x32_2x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{6144, 7168, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{6144, 7168, 2048},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{6144, 7168, 2304},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{6144, 7168, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{6144, 8192, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{6144, 8192, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{6144, 11264, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{6144, 12288, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{6144, 14336, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{6144, 16384, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{6144, 20480, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{6144, 24576, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{6144, 32768, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{6144, 32768, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{8192, 64, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8192, 128, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8192, 576, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{8192, 1024, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{8192, 1536, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x128_16x16_16x16_2x1_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8192, 3072, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{8192, 4096, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{8192, 7168, 256},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x128x128_16x16_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8192, 7168, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{8192, 7168, 2048},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{8192, 7168, 2304},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x128_16x16_16x16_2x1_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8192, 7168, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{8192, 8192, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{8192, 8192, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{8192, 11264, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{8192, 12288, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{8192, 14336, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{8192, 16384, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{8192, 20480, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{8192, 24576, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{8192, 32768, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{8192, 32768, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{10240, 128, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{10240, 512, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{10240, 2240, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x128_16x16_16x16_2x1_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{10240, 3072, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{10240, 4096, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{10240, 4096, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{10240, 7168, 256},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{10240, 7168, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{10240, 7168, 2048},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{10240, 7168, 2304},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{10240, 7168, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x128_16x16_16x16_2x1_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{10240, 8192, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{10240, 8192, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{10240, 9216, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x128_16x16_16x16_2x1_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{10240, 11264, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{10240, 12288, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{10240, 14336, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{10240, 16384, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{10240, 20480, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{10240, 24576, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{10240, 32768, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{10240, 32768, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{12288, 128, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{12288, 512, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{12288, 3072, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{12288, 4096, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{12288, 4096, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x128_16x16_16x16_2x1_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{12288, 4608, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x128_16x16_16x16_2x1_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{12288, 7168, 256},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x128_8x16_16x16_1x1_16x16x1_8x32x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{12288, 7168, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{12288, 7168, 2048},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{12288, 7168, 2304},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{12288, 7168, 4608},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x128_16x16_16x16_2x1_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{12288, 7168, 18432},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x128_16x16_16x16_2x1_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{12288, 8192, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{12288, 8192, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{12288, 9216, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x128_16x16_16x16_2x1_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{12288, 11264, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{12288, 12288, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{12288, 14336, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{12288, 16384, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{12288, 20480, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{12288, 24576, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{12288, 32768, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{12288, 32768, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{14336, 128, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{14336, 512, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{14336, 1024, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{14336, 3072, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{14336, 4096, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{14336, 7168, 256},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x128x256_16x16_32x32_2x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{14336, 7168, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{14336, 7168, 2048},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{14336, 7168, 2304},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{14336, 7168, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x128_16x16_16x16_2x1_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{14336, 8192, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{14336, 8192, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{14336, 9216, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x128_16x16_16x16_2x1_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{14336, 11264, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{14336, 12288, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{14336, 14336, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{14336, 16384, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{14336, 20480, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{14336, 24576, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{14336, 32768, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{14336, 32768, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{14336, 36864, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x128_16x16_16x16_2x1_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16384, 64, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16384, 128, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16384, 576, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{16384, 1024, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{16384, 1536, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x128_16x16_16x16_2x1_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16384, 2112, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x128_16x16_16x16_2x1_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16384, 2240, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x128_16x16_16x16_2x1_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16384, 3072, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{16384, 4096, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{16384, 6144, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{16384, 7168, 256},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x128x128_16x16_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16384, 7168, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{16384, 7168, 2048},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{16384, 7168, 2304},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x128_16x16_16x16_2x1_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16384, 7168, 18432},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x128_16x16_16x16_2x1_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16384, 8192, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{16384, 8192, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{16384, 11264, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{16384, 12288, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x256x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16384, 14336, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{16384, 16384, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{16384, 20480, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{16384, 24576, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{16384, 32768, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{16384, 32768, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{20480, 576, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x256_16x16_16x16_2x1_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{20480, 1536, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x128_16x16_16x16_2x1_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{20480, 3072, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{20480, 4096, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{20480, 7168, 256},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x128x128_16x16_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{20480, 7168, 2048},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{20480, 7168, 2304},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x128_16x16_16x16_2x1_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32768, 64, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32768, 128, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{32768, 512, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{32768, 576, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x128_16x16_16x16_2x1_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32768, 1536, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x128_16x16_16x16_2x1_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32768, 3072, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{32768, 4096, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{32768, 6144, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{32768, 7168, 256},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x128x256_16x16_32x32_2x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32768, 7168, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{32768, 7168, 2048},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{32768, 7168, 2304},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{32768, 8192, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{32768, 8192, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{32768, 11264, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{32768, 12288, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{32768, 14336, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{32768, 16384, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{32768, 20480, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{32768, 24576, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{32768, 32768, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{32768, 32768, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{65536, 64, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{65536, 128, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{65536, 2112, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x128_16x16_16x16_2x1_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{65536, 2240, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x128_16x16_16x16_2x1_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{65536, 3072, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{65536, 4096, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{65536, 6144, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{65536, 7168, 256},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x128x256_16x16_32x32_2x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{65536, 7168, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{65536, 7168, 2048},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{65536, 7168, 2304},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{65536, 7168, 4608},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x128_16x16_16x16_2x1_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{65536, 7168, 9216},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x128_16x16_16x16_2x1_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{65536, 8192, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{65536, 8192, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{65536, 11264, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{65536, 12288, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{65536, 14336, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{65536, 16384, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{65536, 20480, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{65536, 24576, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{98304, 8192, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{98304, 8192, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{98304, 11264, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{98304, 12288, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{98304, 14336, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{98304, 16384, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{98304, 20480, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{131072, 128, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{131072, 512, 7168},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x128_16x16_16x16_2x1_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{131072, 3072, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{131072, 4096, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{131072, 6144, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{131072, 7168, 256},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x128x256_16x16_32x32_2x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{131072, 7168, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{131072, 7168, 2048},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{131072, 7168, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x128x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{131072, 8192, 512},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{131072, 8192, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{131072, 12288, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{131072, 14336, 1536},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{16, 256, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 256, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{48, 256, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 256, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{768, 256, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{832, 256, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{864, 256, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{896, 256, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{928, 256, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{960, 256, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{992, 256, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1024, 256, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1216, 256, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x128x256_16x16_16x16_1x2_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1280, 256, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1536, 256, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x128x256_16x16_16x16_1x2_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2048, 256, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x128x256_16x16_16x16_1x2_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4096, 256, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8192, 256, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x128x128_16x16_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{16384, 256, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{16, 1024, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 1024, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{48, 1024, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 1024, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{768, 1024, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{832, 1024, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{864, 1024, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{896, 1024, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{928, 1024, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{960, 1024, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{992, 1024, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1024, 1024, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1216, 1024, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x128x128_16x16_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1280, 1024, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x128_16x16_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1536, 1024, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x128x128_16x16_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{2048, 1024, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x128x128_16x16_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{4096, 1024, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{8192, 1024, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{16384, 1024, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{16, 1536, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 1536, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{48, 1536, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 1536, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{768, 1536, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x128_16x16_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{832, 1536, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x128x128_16x16_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{864, 1536, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x128x128_16x16_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{896, 1536, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x128x128_16x16_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{928, 1536, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x128x128_16x16_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{960, 1536, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x128x128_16x16_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{992, 1536, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x128x128_16x16_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1024, 1536, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x128x128_16x16_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1216, 1536, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x128x128_16x16_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1280, 1536, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x128x128_16x16_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1536, 1536, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x128x128_16x16_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{2048, 1536, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{4096, 1536, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{8192, 1536, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{16384, 1536, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x128_16x16_16x16_2x1_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 2560, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 2560, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{48, 2560, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 2560, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{768, 2560, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x128x128_16x16_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{832, 2560, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x128x128_16x16_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{864, 2560, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x128x128_16x16_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{896, 2560, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x128x128_16x16_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{928, 2560, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{960, 2560, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{992, 2560, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1024, 2560, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1216, 2560, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1280, 2560, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1536, 2560, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{2048, 2560, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x128x128_16x16_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{4096, 2560, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x128_16x16_16x16_2x1_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8192, 2560, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x128_16x16_16x16_2x1_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16384, 2560, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x128_16x16_16x16_2x1_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 4096, 128},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 4096, 128},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x128_16x16_16x16_2x1_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{48, 4096, 128},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x128x256_16x16_16x16_1x2_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 4096, 128},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{768, 4096, 128},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x128_8x16_16x16_1x1_16x16x1_8x32x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{832, 4096, 128},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x128x256_16x16_16x16_1x2_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{864, 4096, 128},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x128x128_16x16_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{896, 4096, 128},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x128x256_16x16_16x16_1x2_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{928, 4096, 128},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x128x256_16x16_16x16_1x2_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{960, 4096, 128},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x256x128_8x16_16x16_1x4_16x16x1_8x32x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{992, 4096, 128},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x128x128_16x16_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1024, 4096, 128},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x256x128_8x16_16x16_1x4_16x16x1_8x32x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1216, 4096, 128},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x128x256_16x16_16x16_1x2_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1280, 4096, 128},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x128x256_16x16_16x16_1x2_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1536, 4096, 128},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x128x128_16x16_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2048, 4096, 128},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x128x128_16x16_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4096, 4096, 128},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x128x128_16x16_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8192, 4096, 128},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x128x128_16x16_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16384, 4096, 128},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x128x128_16x16_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 256, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 1024, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 1536, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 2560, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 4096, 128},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x128x256_16x16_16x16_1x2_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 1280, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 1280, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 1280, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 1280, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 1280, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 1280, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 1280, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 1280, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 1280, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 1280, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1024, 1280, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x128x128_16x16_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{2048, 1280, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{4096, 1280, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{16384, 1280, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x128_16x16_16x16_2x1_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32768, 1280, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x128_16x16_16x16_2x1_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 2304, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 2304, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 2304, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 2304, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 2304, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 2304, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 2304, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 2304, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 2304, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 2304, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x128x128_16x16_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1024, 2304, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{2048, 2304, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{8192, 2304, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x128_16x16_16x16_2x1_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16384, 2304, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x128_16x16_16x16_2x1_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 4096, 1024},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 4096, 2048},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 4096, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 4096, 8192},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 4096, 1024},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 4096, 2048},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 4096, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 4096, 8192},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 4096, 1024},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 4096, 2048},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 4096, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 4096, 8192},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 4096, 1024},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 4096, 2048},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 4096, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 4096, 8192},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 4096, 1024},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 4096, 2048},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 4096, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 4096, 8192},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 4096, 1024},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 4096, 2048},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 4096, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 4096, 8192},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 4096, 1024},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 4096, 2048},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 4096, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 4096, 8192},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 4096, 1024},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x128x256_16x16_16x16_1x2_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 4096, 2048},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x128x256_16x16_16x16_1x2_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 4096, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 4096, 8192},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 4096, 1024},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 4096, 2048},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 4096, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 4096, 8192},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 4096, 1024},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x128x128_16x16_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{512, 4096, 2048},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x128x128_16x16_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{512, 4096, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x128x128_16x16_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{512, 4096, 8192},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x128x128_16x16_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1024, 4096, 1024},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1024, 4096, 2048},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1024, 4096, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1024, 4096, 8192},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{2048, 4096, 1024},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{2048, 4096, 2048},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{2048, 4096, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x128_16x16_16x16_2x1_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2048, 4096, 8192},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x128_16x16_16x16_2x1_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4096, 4096, 1024},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{4096, 4096, 2048},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x128_16x16_16x16_2x1_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4096, 4096, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x128_16x16_16x16_2x1_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4096, 4096, 8192},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x128_16x16_16x16_2x1_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8192, 4096, 1024},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x128_16x16_16x16_2x1_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8192, 4096, 2048},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x128_16x16_16x16_2x1_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8192, 4096, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x128_16x16_16x16_2x1_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16384, 4096, 1024},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x128_16x16_16x16_2x1_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16384, 4096, 2048},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x128_16x16_16x16_2x1_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16384, 4096, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x128_16x16_16x16_2x1_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32768, 4096, 1024},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x128_16x16_16x16_2x1_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32768, 4096, 2048},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x128_16x16_16x16_2x1_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 4608, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 4608, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 4608, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 4608, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 4608, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 4608, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 4608, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 4608, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 4608, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x128x128_16x16_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{512, 4608, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1024, 4608, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x128_16x16_16x16_2x1_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2048, 4608, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x128_16x16_16x16_2x1_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4096, 4608, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x128_16x16_16x16_2x1_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8192, 4608, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x128_16x16_16x16_2x1_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32768, 4608, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x128_16x16_16x16_2x1_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 9216, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 9216, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 9216, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 9216, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 9216, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 9216, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x128x256_16x16_16x16_1x2_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 9216, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_1x1_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 9216, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x128x128_16x16_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{256, 9216, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{512, 9216, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x128_16x16_16x16_2x1_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2048, 9216, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x128_16x16_16x16_2x1_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4096, 9216, 4096},                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x128_16x16_16x16_2x1_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
   }

#endif // USE_ROCM
