#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#ifdef USE_ROCM

#define GENERATE_LOOKUP_TABLE(CTYPE)                                                                                      \
   {                                                                                                                             \
       {{1, 18432, 16384},                                                                                                       \
        a4w4_blockscale_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{1, 51200, 5120},                                                                                                       \
        a4w4_blockscale_256x32x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8_2x2_intrawave_v3<CTYPE>},                       \
       {{1, 53248, 16384},                                                                                                       \
        a4w4_blockscale_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{1, 57344, 8192},                                                                                                       \
        a4w4_blockscale_256x32x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8_2x2_intrawave_v3<CTYPE>},                       \
       {{1, 59136, 8192},                                                                                                       \
        a4w4_blockscale_256x32x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8_2x2_intrawave_v3<CTYPE>},                       \
       {{1, 106496, 16384},                                                                                                       \
        a4w4_blockscale_256x64x512x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{2, 57344, 8192},                                                                                                       \
        a4w4_blockscale_256x32x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8_2x2_intrawave_v3<CTYPE>},                       \
       {{2, 59136, 8192},                                                                                                       \
        a4w4_blockscale_256x32x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8_2x2_intrawave_v3<CTYPE>},                       \
       {{2, 106496, 16384},                                                                                                       \
        a4w4_blockscale_256x32x512x128_16x16_16x16_8x32x1_8x32x1_1x8x1x32_8_2x4_intrawave_v3<CTYPE>},                       \
       {{4, 18432, 16384},                                                                                                       \
        a4w4_blockscale_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{4, 57344, 8192},                                                                                                       \
        a4w4_blockscale_256x32x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8_2x2_intrawave_v3<CTYPE>},                       \
       {{4, 59136, 8192},                                                                                                       \
        a4w4_blockscale_256x32x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8_2x2_intrawave_v3<CTYPE>},                       \
       {{4, 106496, 16384},                                                                                                       \
        a4w4_blockscale_256x64x512x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{8, 57344, 8192},                                                                                                       \
        a4w4_blockscale_256x32x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8_2x2_intrawave_v3<CTYPE>},                       \
       {{8, 59136, 8192},                                                                                                       \
        a4w4_blockscale_256x32x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8_2x2_intrawave_v3<CTYPE>},                       \
       {{8, 106496, 16384},                                                                                                       \
        a4w4_blockscale_256x64x512x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{16, 512, 7168},                                                                                                       \
        a4w4_blockscale_256x96x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{16, 4608, 7168},                                                                                                       \
        a4w4_blockscale_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{16, 7168, 2304},                                                                                                       \
        a4w4_blockscale_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{16, 51200, 5120},                                                                                                       \
        a4w4_blockscale_256x32x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8_2x2_intrawave_v3<CTYPE>},                       \
       {{16, 53248, 16384},                                                                                                       \
        a4w4_blockscale_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8_2x2_intrawave_v3<CTYPE>},                       \
       {{16, 57344, 8192},                                                                                                       \
        a4w4_blockscale_256x32x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8_2x2_intrawave_v3<CTYPE>},                       \
       {{16, 59136, 8192},                                                                                                       \
        a4w4_blockscale_256x32x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8_2x2_intrawave_v3<CTYPE>},                       \
       {{16, 106496, 16384},                                                                                                       \
        a4w4_blockscale_256x32x512x128_16x16_16x16_8x32x1_8x32x1_1x8x1x32_8_2x4_intrawave_v3<CTYPE>},                       \
       {{32, 512, 7168},                                                                                                       \
        a4w4_blockscale_256x96x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{32, 4096, 1024},                                                                                                       \
        a4w4_blockscale_256x32x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8_2x2_intrawave_v3<CTYPE>},                       \
       {{32, 4608, 7168},                                                                                                       \
        a4w4_blockscale_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{32, 7168, 2304},                                                                                                       \
        a4w4_blockscale_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{32, 51200, 5120},                                                                                                       \
        a4w4_blockscale_256x32x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8_2x2_intrawave_v3<CTYPE>},                       \
       {{32, 53248, 16384},                                                                                                       \
        a4w4_blockscale_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8_2x2_intrawave_v3<CTYPE>},                       \
       {{32, 57344, 8192},                                                                                                       \
        a4w4_blockscale_256x32x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8_2x2_intrawave_v3<CTYPE>},                       \
       {{32, 59136, 8192},                                                                                                       \
        a4w4_blockscale_256x32x256x128_16x16_16x16_8x32x1_8x32x1_1x8x1x32_8_2x4_intrawave_v3<CTYPE>},                       \
       {{32, 106496, 16384},                                                                                                       \
        a4w4_blockscale_256x32x512x128_16x16_16x16_8x32x1_8x32x1_1x8x1x32_8_2x4_intrawave_v3<CTYPE>},                       \
       {{64, 512, 7168},                                                                                                       \
        a4w4_blockscale_256x96x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{64, 4608, 7168},                                                                                                       \
        a4w4_blockscale_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{64, 7168, 2304},                                                                                                       \
        a4w4_blockscale_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{64, 51200, 5120},                                                                                                       \
        a4w4_blockscale_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{64, 53248, 16384},                                                                                                       \
        a4w4_blockscale_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{64, 57344, 8192},                                                                                                       \
        a4w4_blockscale_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8_2x2_intrawave_v3<CTYPE>},                       \
       {{64, 59136, 8192},                                                                                                       \
        a4w4_blockscale_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{64, 106496, 16384},                                                                                                       \
        a4w4_blockscale_256x64x512x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{65, 1280, 8192},                                                                                                       \
        a4w4_blockscale_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{112, 51200, 5120},                                                                                                       \
        a4w4_blockscale_256x128x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{127, 1280, 8192},                                                                                                       \
        a4w4_blockscale_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{128, 512, 7168},                                                                                                       \
        a4w4_blockscale_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{128, 4608, 7168},                                                                                                       \
        a4w4_blockscale_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{128, 7168, 2304},                                                                                                       \
        a4w4_blockscale_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{128, 12800, 5120},                                                                                                       \
        a4w4_blockscale_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{128, 24576, 1536},                                                                                                       \
        a4w4_blockscale_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{128, 28672, 4096},                                                                                                       \
        a4w4_blockscale_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{128, 51200, 5120},                                                                                                       \
        a4w4_blockscale_256x128x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{128, 53248, 16384},                                                                                                       \
        a4w4_blockscale_256x128x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{128, 57344, 8192},                                                                                                       \
        a4w4_blockscale_256x128x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{128, 59136, 8192},                                                                                                       \
        a4w4_blockscale_256x128x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{128, 106496, 16384},                                                                                                       \
        a4w4_blockscale_256x128x512x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{129, 1280, 8192},                                                                                                       \
        a4w4_blockscale_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{256, 512, 7168},                                                                                                       \
        a4w4_blockscale_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{256, 4608, 7168},                                                                                                       \
        a4w4_blockscale_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{256, 7168, 2304},                                                                                                       \
        a4w4_blockscale_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{256, 12288, 1536},                                                                                                       \
        a4w4_blockscale_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{256, 18432, 16384},                                                                                                       \
        a4w4_blockscale_256x64x384x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{256, 24576, 1536},                                                                                                       \
        a4w4_blockscale_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{256, 57344, 8192},                                                                                                       \
        a4w4_blockscale_256x256x256x128_16x16_16x16_8x32x1_8x32x1_1x8x1x32_8_2x4_intrawave_v3<CTYPE>},                       \
       {{256, 59136, 8192},                                                                                                       \
        a4w4_blockscale_256x256x256x128_16x16_16x16_8x32x1_8x32x1_1x8x1x32_8_2x4_intrawave_v3<CTYPE>},                       \
       {{256, 106496, 16384},                                                                                                       \
        a4w4_blockscale_256x256x256x128_16x16_16x16_8x32x1_8x32x1_1x8x1x32_8_2x4_intrawave_v3<CTYPE>},                       \
       {{384, 57344, 8192},                                                                                                       \
        a4w4_blockscale_256x128x384x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{512, 512, 7168},                                                                                                       \
        a4w4_blockscale_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{512, 3072, 1536},                                                                                                       \
        a4w4_blockscale_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{512, 4608, 7168},                                                                                                       \
        a4w4_blockscale_256x96x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{512, 7168, 2048},                                                                                                       \
        a4w4_blockscale_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{512, 7168, 2304},                                                                                                       \
        a4w4_blockscale_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{512, 12288, 1536},                                                                                                       \
        a4w4_blockscale_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{512, 24576, 1536},                                                                                                       \
        a4w4_blockscale_256x128x384x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{640, 8192, 1024},                                                                                                       \
        a4w4_blockscale_256x96x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8_2x2_intrawave_v3<CTYPE>},                       \
       {{768, 8192, 1024},                                                                                                       \
        a4w4_blockscale_256x96x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8_2x2_intrawave_v3<CTYPE>},                       \
       {{1024, 512, 7168},                                                                                                       \
        a4w4_blockscale_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{1024, 3072, 1536},                                                                                                       \
        a4w4_blockscale_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{1024, 4608, 7168},                                                                                                       \
        a4w4_blockscale_256x96x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{1024, 7168, 2048},                                                                                                       \
        a4w4_blockscale_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{1024, 7168, 2304},                                                                                                       \
        a4w4_blockscale_256x128x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{1024, 8192, 1024},                                                                                                       \
        a4w4_blockscale_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{1536, 512, 7168},                                                                                                       \
        a4w4_blockscale_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{1536, 1536, 7168},                                                                                                       \
        a4w4_blockscale_256x96x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{1536, 3072, 1536},                                                                                                       \
        a4w4_blockscale_256x96x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{1536, 4608, 7168},                                                                                                       \
        a4w4_blockscale_256x96x384x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{1536, 7168, 2048},                                                                                                       \
        a4w4_blockscale_256x128x384x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{1536, 7168, 2304},                                                                                                       \
        a4w4_blockscale_256x96x512x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{2048, 512, 7168},                                                                                                       \
        a4w4_blockscale_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{2048, 3072, 1536},                                                                                                       \
        a4w4_blockscale_256x128x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{2048, 4608, 7168},                                                                                                       \
        a4w4_blockscale_256x128x384x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{2048, 5120, 1280},                                                                                                       \
        a4w4_blockscale_256x128x384x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{2048, 7168, 2048},                                                                                                       \
        a4w4_blockscale_256x128x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{2048, 7168, 2304},                                                                                                       \
        a4w4_blockscale_256x128x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{4096, 512, 7168},                                                                                                       \
        a4w4_blockscale_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{4096, 3072, 1536},                                                                                                       \
        a4w4_blockscale_256x128x384x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{4096, 4608, 7168},                                                                                                       \
        a4w4_blockscale_256x128x384x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{4096, 7168, 2048},                                                                                                       \
        a4w4_blockscale_256x128x384x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{4096, 7168, 2304},                                                                                                       \
        a4w4_blockscale_256x96x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{8192, 512, 7168},                                                                                                       \
        a4w4_blockscale_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{8192, 3072, 1536},                                                                                                       \
        a4w4_blockscale_256x128x384x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{8192, 4608, 7168},                                                                                                       \
        a4w4_blockscale_256x128x384x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{8192, 7168, 2048},                                                                                                       \
        a4w4_blockscale_256x128x384x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{8192, 7168, 2304},                                                                                                       \
        a4w4_blockscale_256x128x384x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{16384, 512, 7168},                                                                                                       \
        a4w4_blockscale_256x128x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{16384, 3072, 1536},                                                                                                       \
        a4w4_blockscale_256x128x384x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{16384, 4608, 7168},                                                                                                       \
        a4w4_blockscale_256x128x384x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{16384, 7168, 2048},                                                                                                       \
        a4w4_blockscale_256x128x384x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{16384, 7168, 2304},                                                                                                       \
        a4w4_blockscale_256x128x384x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{16384, 18432, 16384},                                                                                                       \
        a4w4_blockscale_256x128x384x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{20480, 512, 7168},                                                                                                       \
        a4w4_blockscale_256x96x512x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{20480, 1536, 7168},                                                                                                       \
        a4w4_blockscale_256x128x384x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{20480, 3072, 1536},                                                                                                       \
        a4w4_blockscale_256x128x384x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{20480, 4608, 7168},                                                                                                       \
        a4w4_blockscale_256x128x384x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{20480, 7168, 2048},                                                                                                       \
        a4w4_blockscale_256x128x384x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
       {{20480, 7168, 2304},                                                                                                       \
        a4w4_blockscale_256x128x384x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x2_intrawave_v3<CTYPE>},                       \
   }

#endif // USE_ROCM
