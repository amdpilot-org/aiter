// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#include <unordered_map>


#define ADD_CFG(tile_M, tile_N, splitK, bpreshuffle, arch, path, knl_name, co_name)         \
    {                                         \
        arch knl_name, { knl_name, path co_name, arch, tile_M, tile_N, splitK, bpreshuffle }         \
    }

struct f4gemmConfig
{
    std::string knl_name;
    std::string co_name;
    std::string arch;
    int tile_M;
    int tile_N;
    int splitK;
    int bpreshuffle;
};

using CFG = std::unordered_map<std::string, f4gemmConfig>;

static CFG cfg_f4gemm_bf16_per1x32Fp4 = {
    
};
