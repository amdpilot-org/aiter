#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#ifdef USE_ROCM

#define GENERATE_LOOKUP_TABLE(DTYPE, ETYPE)                                                                                      \
   {                                                                                                                             \
       {{192, 1536, 5120},                                                                                                       \
        a8w8_bpreshuffle_cktile_0x0x8x4x1x0x0x0x0x1_32x64x512_1x4x1_16x16x64_default<DTYPE, ETYPE>},                       \
   }

#endif // USE_ROCM
