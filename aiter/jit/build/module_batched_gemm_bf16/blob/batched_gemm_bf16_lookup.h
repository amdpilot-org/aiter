#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#ifdef USE_ROCM

#define GENERATE_LOOKUP_TABLE()                                                                                      \
   {                                                                                                                             \
       {{16, 1, 1280, 8192},                                                                                                       \
        bf16_batched_64x16x16x64_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4x4x1_1x1_interwave_v2},                       \
       {{16, 32, 1280, 8192},                                                                                                       \
        bf16_batched_256x32x96x128_16x16_1x3_16x16x1_16x16x1_1x32x1x8_4x4x1_1x1_intrawave_v3},                       \
       {{16, 64, 1280, 8192},                                                                                                       \
        bf16_batched_256x64x128x128_32x32_1x2_16x16x1_16x16x1_1x32x1x8_8x8x1_1x1_intrawave_v3},                       \
       {{16, 128, 1280, 8192},                                                                                                       \
        bf16_batched_256x128x128x64_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v5},                       \
       {{16, 192, 1280, 8192},                                                                                                       \
        bf16_batched_256x128x160x64_32x32_1x5_8x32x1_8x32x1_1x64x1x4_8x8x1_1x1_intrawave_v3},                       \
       {{16, 256, 1280, 8192},                                                                                                       \
        bf16_batched_256x128x160x64_32x32_1x5_8x32x1_8x32x1_1x64x1x4_8x8x1_1x1_intrawave_v3},                       \
       {{16, 320, 1280, 8192},                                                                                                       \
        bf16_batched_256x128x128x64_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3},                       \
       {{16, 512, 1280, 8192},                                                                                                       \
        bf16_batched_256x128x160x64_32x32_1x5_8x32x1_8x32x1_1x64x1x4_8x8x1_1x1_intrawave_v3},                       \
       {{16, 1024, 1280, 8192},                                                                                                       \
        bf16_batched_256x128x128x64_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v5},                       \
       {{16, 2048, 1280, 8192},                                                                                                       \
        bf16_batched_256x128x128x64_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v5},                       \
       {{16, 4096, 1280, 8192},                                                                                                       \
        bf16_batched_256x128x128x64_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v5},                       \
       {{16, 8192, 1280, 8192},                                                                                                       \
        bf16_batched_256x128x128x64_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v5},                       \
       {{16, 16384, 1280, 8192},                                                                                                       \
        bf16_batched_256x128x128x64_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v5},                       \
       {{16, 1, 8192, 1024},                                                                                                       \
        bf16_batched_128x16x32x64_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_interwave_v2},                       \
       {{16, 32, 8192, 1024},                                                                                                       \
        bf16_batched_256x32x64x256_16x16_1x2_32x8x1_32x8x1_1x32x1x8_8x8x1_1x2_intrawave_v3},                       \
       {{16, 64, 8192, 1024},                                                                                                       \
        bf16_batched_256x64x128x64_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3},                       \
       {{16, 128, 8192, 1024},                                                                                                       \
        bf16_batched_256x128x128x64_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3},                       \
       {{16, 192, 8192, 1024},                                                                                                       \
        bf16_batched_256x128x128x64_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3},                       \
       {{16, 256, 8192, 1024},                                                                                                       \
        bf16_batched_256x128x128x64_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3},                       \
       {{16, 320, 8192, 1024},                                                                                                       \
        bf16_batched_256x128x128x64_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3},                       \
       {{16, 512, 8192, 1024},                                                                                                       \
        bf16_batched_256x128x128x64_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3},                       \
       {{16, 1024, 8192, 1024},                                                                                                       \
        bf16_batched_256x128x128x64_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3},                       \
       {{16, 2048, 8192, 1024},                                                                                                       \
        bf16_batched_256x256x256x32_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4},                       \
       {{16, 4096, 8192, 1024},                                                                                                       \
        bf16_batched_256x256x256x32_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4},                       \
       {{16, 8192, 8192, 1024},                                                                                                       \
        bf16_batched_256x128x128x64_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3},                       \
       {{16, 16384, 8192, 1024},                                                                                                       \
        bf16_batched_256x256x256x32_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4},                       \
   }

#endif // USE_ROCM
