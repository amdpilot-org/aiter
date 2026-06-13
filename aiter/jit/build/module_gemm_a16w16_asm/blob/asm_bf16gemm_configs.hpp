// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#include <unordered_map>


#define ADD_CFG(tn, tileM, tileN, pf, bPreshuffle, splitK, subK, bias, arch, path, knl_name, co_name)         \
    {                                         \
        arch knl_name, { knl_name, path co_name, arch, tn, tileM, tileN, pf, bPreshuffle, splitK, subK, bias }         \
    }

struct bf16gemmConfig
{
    std::string knl_name;
    std::string co_name;
    std::string arch;
    int tn;
    int tileM;
    int tileN;
    int pf;
    int bPreshuffle;
    int splitK;
    int subK;
    int bias;
};

using CFG = std::unordered_map<std::string, bf16gemmConfig>;

static CFG cfg_bf16gemm_fp32bf16 = {
    ADD_CFG(   1,  128,   64,    0,    1,    0,   64,    1, "gfx942", "bf16gemm/", "_ZN5aiter43bf16gemm_fp32bf16_tn_128x64_bshuffle_splitkE", "bf16gemm_fp32bf16_tn_128x64_bshuffle_splitk.co"),
    ADD_CFG(   1,  160,   64,    0,    1,    0,   64,    1, "gfx942", "bf16gemm/", "_ZN5aiter43bf16gemm_fp32bf16_tn_160x64_bshuffle_splitkE", "bf16gemm_fp32bf16_tn_160x64_bshuffle_splitk.co"),
    ADD_CFG(   1,   32,   64,    0,    1,    0,   64,    1, "gfx942", "bf16gemm/", "_ZN5aiter42bf16gemm_fp32bf16_tn_32x64_bshuffle_splitkE", "bf16gemm_fp32bf16_tn_32x64_bshuffle_splitk.co"),
    ADD_CFG(   1,   32,   64,    3,    0,    0,   64,    1, "gfx942", "bf16gemm/", "_ZN5aiter37bf16gemm_fp32bf16_tn_32x64_pf3_splitkE", "bf16gemm_fp32bf16_tn_32x64_pf3_splitk.co"),
    ADD_CFG(   1,   48,   64,    0,    1,    0,   64,    1, "gfx942", "bf16gemm/", "_ZN5aiter42bf16gemm_fp32bf16_tn_48x64_bshuffle_splitkE", "bf16gemm_fp32bf16_tn_48x64_bshuffle_splitk.co"),
    ADD_CFG(   1,   48,   64,    3,    0,    0,   64,    1, "gfx942", "bf16gemm/", "_ZN5aiter37bf16gemm_fp32bf16_tn_48x64_pf3_splitkE", "bf16gemm_fp32bf16_tn_48x64_pf3_splitk.co"),
    ADD_CFG(   1,   64,   64,    0,    1,    0,   64,    1, "gfx942", "bf16gemm/", "_ZN5aiter42bf16gemm_fp32bf16_tn_64x64_bshuffle_splitkE", "bf16gemm_fp32bf16_tn_64x64_bshuffle_splitk.co"),
    ADD_CFG(   1,   64,   64,    3,    0,    0,   64,    1, "gfx942", "bf16gemm/", "_ZN5aiter37bf16gemm_fp32bf16_tn_64x64_pf3_splitkE", "bf16gemm_fp32bf16_tn_64x64_pf3_splitk.co"),
    ADD_CFG(   1,   96,   64,    0,    1,    0,   64,    1, "gfx942", "bf16gemm/", "_ZN5aiter42bf16gemm_fp32bf16_tn_96x64_bshuffle_splitkE", "bf16gemm_fp32bf16_tn_96x64_bshuffle_splitk.co"),
    ADD_CFG(   1,   96,   64,    3,    0,    0,   64,    1, "gfx942", "bf16gemm/", "_ZN5aiter37bf16gemm_fp32bf16_tn_96x64_pf3_splitkE", "bf16gemm_fp32bf16_tn_96x64_pf3_splitk.co"),
    ADD_CFG(   1,  128,   64,    0,    1,    1,   64,    1, "gfx942", "bf16gemm/", "_ZN5aiter49bf16gemm_fp32bf16_tn_128x64_bshuffle_splitk_cleanE", "bf16gemm_fp32bf16_tn_128x64_bshuffle_splitk_clean.co"),
    ADD_CFG(   1,   32,   64,    0,    1,    1,   64,    1, "gfx942", "bf16gemm/", "_ZN5aiter48bf16gemm_fp32bf16_tn_32x64_bshuffle_splitk_cleanE", "bf16gemm_fp32bf16_tn_32x64_bshuffle_splitk_clean.co"),
    ADD_CFG(   1,   48,   64,    0,    1,    1,   64,    1, "gfx942", "bf16gemm/", "_ZN5aiter48bf16gemm_fp32bf16_tn_48x64_bshuffle_splitk_cleanE", "bf16gemm_fp32bf16_tn_48x64_bshuffle_splitk_clean.co"),
    ADD_CFG(   1,   64,   64,    0,    1,    1,   64,    1, "gfx942", "bf16gemm/", "_ZN5aiter48bf16gemm_fp32bf16_tn_64x64_bshuffle_splitk_cleanE", "bf16gemm_fp32bf16_tn_64x64_bshuffle_splitk_clean.co"),
    ADD_CFG(   1,   80,   64,    0,    1,    1,   64,    1, "gfx942", "bf16gemm/", "_ZN5aiter48bf16gemm_fp32bf16_tn_80x64_bshuffle_splitk_cleanE", "bf16gemm_fp32bf16_tn_80x64_bshuffle_splitk_clean.co"),
    ADD_CFG(   1,   96,   64,    0,    1,    1,   64,    1, "gfx942", "bf16gemm/", "_ZN5aiter48bf16gemm_fp32bf16_tn_96x64_bshuffle_splitk_cleanE", "bf16gemm_fp32bf16_tn_96x64_bshuffle_splitk_clean.co"),
    ADD_CFG(   1,   96,   64,    0,    0,    1,   64,    1, "gfx942", "bf16gemm/", "_ZN5aiter39bf16gemm_fp32bf16_tn_96x64_splitk_cleanE", "bf16gemm_fp32bf16_tn_96x64_splitk_clean.co"),
    ADD_CFG(   1,  160,   64,    0,    1,    1,   64,    1, "gfx942", "bf16gemm/", "_ZN5aiter49bf16gemm_fp32bf16_tn_160x64_bshuffle_splitk_cleanE", "bf16gemm_fp32bf16_tn_160x64_bshuffle_splitk_clean.co"),
    ADD_CFG(   1,   32,   64,    0,    0,    1,   64,    1, "gfx942", "bf16gemm/", "_ZN5aiter39bf16gemm_fp32bf16_tn_32x64_splitk_cleanE", "bf16gemm_fp32bf16_tn_32x64_splitk_clean.co"),
    ADD_CFG(   1,   48,   64,    0,    0,    1,   64,    1, "gfx942", "bf16gemm/", "_ZN5aiter39bf16gemm_fp32bf16_tn_48x64_splitk_cleanE", "bf16gemm_fp32bf16_tn_48x64_splitk_clean.co"),
    ADD_CFG(   1,   64,   64,    0,    0,    1,   64,    1, "gfx942", "bf16gemm/", "_ZN5aiter39bf16gemm_fp32bf16_tn_64x64_splitk_cleanE", "bf16gemm_fp32bf16_tn_64x64_splitk_clean.co"),
    ADD_CFG(   1,   80,   64,    0,    0,    1,   64,    1, "gfx942", "bf16gemm/", "_ZN5aiter39bf16gemm_fp32bf16_tn_80x64_splitk_cleanE", "bf16gemm_fp32bf16_tn_80x64_splitk_clean.co"),
};
