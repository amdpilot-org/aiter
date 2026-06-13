#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#ifdef USE_ROCM

#define GENERATE_LOOKUP_TABLE(DTYPE, ETYPE)                                                                                      \
   {                                                                                                                             \
       {{16, 4096, 512},                                                                                                       \
        a8w8_blockscale_cktile_16x128x256_1x4x1_16x16x64_intrawave_0x1x0_2<DTYPE, ETYPE>},                       \
   }

#endif // USE_ROCM
