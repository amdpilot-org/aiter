// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#include <unordered_map>


#define ADD_CFG(tile_m, tile_n, splitK, bpreshuffle, arch, path, knl_name, co_name)         \
    {                                         \
        arch knl_name, { knl_name, path co_name, arch, tile_m, tile_n, splitK, bpreshuffle }         \
    }

struct fp8gemm_blockscaleConfig
{
    std::string knl_name;
    std::string co_name;
    std::string arch;
    int tile_m;
    int tile_n;
    int splitK;
    int bpreshuffle;
};

using CFG = std::unordered_map<std::string, fp8gemm_blockscaleConfig>;

static CFG cfg_fp8gemm_bf16_blockscale = {
    ADD_CFG( 128,  128,    1,    1, "gfx942", "fp8gemm_blockscale/", "_ZN5aiter43fp8gemm_bf16_blockscale_BpreShuffle_128x128E", "fp8gemm_bf16_blockscale_BpreShuffle_128x128.co"),
    ADD_CFG(  32,  128,    1,    1, "gfx942", "fp8gemm_blockscale/", "_ZN5aiter42fp8gemm_bf16_blockscale_BpreShuffle_32x128E", "fp8gemm_bf16_blockscale_BpreShuffle_32x128.co"),
    ADD_CFG(  48,  128,    1,    1, "gfx942", "fp8gemm_blockscale/", "_ZN5aiter42fp8gemm_bf16_blockscale_BpreShuffle_48x128E", "fp8gemm_bf16_blockscale_BpreShuffle_48x128.co"),
    ADD_CFG(  64,  128,    1,    1, "gfx942", "fp8gemm_blockscale/", "_ZN5aiter42fp8gemm_bf16_blockscale_BpreShuffle_64x128E", "fp8gemm_bf16_blockscale_BpreShuffle_64x128.co"),
    ADD_CFG(  80,  128,    1,    1, "gfx942", "fp8gemm_blockscale/", "_ZN5aiter42fp8gemm_bf16_blockscale_BpreShuffle_80x128E", "fp8gemm_bf16_blockscale_BpreShuffle_80x128.co"),
    ADD_CFG(  96,  128,    1,    1, "gfx942", "fp8gemm_blockscale/", "_ZN5aiter42fp8gemm_bf16_blockscale_BpreShuffle_96x128E", "fp8gemm_bf16_blockscale_BpreShuffle_96x128.co"),
};
