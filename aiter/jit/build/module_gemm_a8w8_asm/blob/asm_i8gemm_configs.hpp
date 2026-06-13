// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#include <unordered_map>


#define ADD_CFG(tile_m, tile_n, splitK, bpreshuffle, arch, path, knl_name, co_name)         \
    {                                         \
        arch knl_name, { knl_name, path co_name, arch, tile_m, tile_n, splitK, bpreshuffle }         \
    }

struct i8gemmConfig
{
    std::string knl_name;
    std::string co_name;
    std::string arch;
    int tile_m;
    int tile_n;
    int splitK;
    int bpreshuffle;
};

using CFG = std::unordered_map<std::string, i8gemmConfig>;

static CFG cfg_i8gemm_bf16_perTokenI8 = {
    ADD_CFG( 112,  256,    0,    1, "gfx942", "i8gemm/", "_ZN5aiter42I8gemm_bf16_perTokenI8_BpreShuffle_112x256E", "I8gemm_bf16_perTokenI8_BpreShuffle_112x256.co"),
    ADD_CFG(  32,  128,    1,    1, "gfx942", "i8gemm/", "_ZN5aiter41I8gemm_bf16_perTokenI8_BpreShuffle_32x128E", "I8gemm_bf16_perTokenI8_BpreShuffle_32x128.co"),
    ADD_CFG(  48,  128,    1,    1, "gfx942", "i8gemm/", "_ZN5aiter41I8gemm_bf16_perTokenI8_BpreShuffle_48x128E", "I8gemm_bf16_perTokenI8_BpreShuffle_48x128.co"),
    ADD_CFG( 128,  128,    1,    1, "gfx942", "i8gemm/", "_ZN5aiter42I8gemm_bf16_perTokenI8_BpreShuffle_128x128E", "I8gemm_bf16_perTokenI8_BpreShuffle_128x128.co"),
    ADD_CFG( 160,  128,    1,    1, "gfx942", "i8gemm/", "_ZN5aiter42I8gemm_bf16_perTokenI8_BpreShuffle_160x128E", "I8gemm_bf16_perTokenI8_BpreShuffle_160x128.co"),
    ADD_CFG(  64,  128,    1,    1, "gfx942", "i8gemm/", "_ZN5aiter41I8gemm_bf16_perTokenI8_BpreShuffle_64x128E", "I8gemm_bf16_perTokenI8_BpreShuffle_64x128.co"),
    ADD_CFG(  80,  128,    1,    1, "gfx942", "i8gemm/", "_ZN5aiter41I8gemm_bf16_perTokenI8_BpreShuffle_80x128E", "I8gemm_bf16_perTokenI8_BpreShuffle_80x128.co"),
    ADD_CFG(  96,  128,    1,    1, "gfx942", "i8gemm/", "_ZN5aiter41I8gemm_bf16_perTokenI8_BpreShuffle_96x128E", "I8gemm_bf16_perTokenI8_BpreShuffle_96x128.co"),
    ADD_CFG(  16,  128,    1,    1, "gfx942", "i8gemm/", "_ZN5aiter41I8gemm_bf16_perTokenI8_BpreShuffle_16x128E", "I8gemm_bf16_perTokenI8_BpreShuffle_16x128.co"),
};
