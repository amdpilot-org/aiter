// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#include <unordered_map>


#define ADD_CFG(subm, num_experts, topk, dtype, arch, path, knl_name, co_name)         \
    {                                         \
        arch knl_name, { knl_name, path co_name, arch, subm, num_experts, topk, dtype }         \
    }

struct topksoftmaxConfig
{
    std::string knl_name;
    std::string co_name;
    std::string arch;
    int subm;
    int num_experts;
    int topk;
    std::string dtype;
};

using CFG = std::unordered_map<std::string, topksoftmaxConfig>;

static CFG cfg_topksoftmax = {
    ADD_CFG(   4,  128,    4, "fp32", "gfx942", "topksoftmax/", "_ZN5aiter19topksoftmax_4x128x4E", "topksoftmax_4x128x4.co"),
    ADD_CFG(   4,  128,    6, "fp32", "gfx942", "topksoftmax/", "_ZN5aiter19topksoftmax_4x128x6E", "topksoftmax_4x128x6.co"),
    ADD_CFG(   4,  128,    8, "fp32", "gfx942", "topksoftmax/", "_ZN5aiter19topksoftmax_4x128x8E", "topksoftmax_4x128x8.co"),
    ADD_CFG(   4,  256,    6, "fp32", "gfx942", "topksoftmax/", "_ZN5aiter19topksoftmax_4x256x6E", "topksoftmax_4x256x6.co"),
    ADD_CFG(   4,  256,    8, "fp32", "gfx942", "topksoftmax/", "_ZN5aiter19topksoftmax_4x256x8E", "topksoftmax_4x256x8.co"),
    ADD_CFG(   4,  384,    8, "fp32", "gfx942", "topksoftmax/", "_ZN5aiter19topksoftmax_4x384x8E", "topksoftmax_4x384x8.co"),
    ADD_CFG(  12,  128,    4, "fp32", "gfx942", "topksoftmax/", "_ZN5aiter20topksoftmax_12x128x4E", "topksoftmax_12x128x4.co"),
    ADD_CFG(  12,  128,    6, "fp32", "gfx942", "topksoftmax/", "_ZN5aiter20topksoftmax_12x128x6E", "topksoftmax_12x128x6.co"),
    ADD_CFG(  12,  128,    8, "fp32", "gfx942", "topksoftmax/", "_ZN5aiter20topksoftmax_12x128x8E", "topksoftmax_12x128x8.co"),
    ADD_CFG(  12,  256,    6, "fp32", "gfx942", "topksoftmax/", "_ZN5aiter20topksoftmax_12x256x6E", "topksoftmax_12x256x6.co"),
    ADD_CFG(  12,  256,    8, "fp32", "gfx942", "topksoftmax/", "_ZN5aiter20topksoftmax_12x256x8E", "topksoftmax_12x256x8.co"),
    ADD_CFG(   4,  128,    4, "bf16", "gfx942", "topksoftmax/", "_ZN5aiter24topksoftmax_4x128x4_bf16E", "topksoftmax_4x128x4_bf16.co"),
    ADD_CFG(   4,  128,    6, "bf16", "gfx942", "topksoftmax/", "_ZN5aiter24topksoftmax_4x128x6_bf16E", "topksoftmax_4x128x6_bf16.co"),
    ADD_CFG(   4,  128,    8, "bf16", "gfx942", "topksoftmax/", "_ZN5aiter24topksoftmax_4x128x8_bf16E", "topksoftmax_4x128x8_bf16.co"),
    ADD_CFG(   4,  256,    6, "bf16", "gfx942", "topksoftmax/", "_ZN5aiter24topksoftmax_4x256x6_bf16E", "topksoftmax_4x256x6_bf16.co"),
    ADD_CFG(   4,  256,    8, "bf16", "gfx942", "topksoftmax/", "_ZN5aiter24topksoftmax_4x256x8_bf16E", "topksoftmax_4x256x8_bf16.co"),
    ADD_CFG(   4,  384,    8, "bf16", "gfx942", "topksoftmax/", "_ZN5aiter24topksoftmax_4x384x8_bf16E", "topksoftmax_4x384x8_bf16.co"),
    ADD_CFG(  12,  128,    4, "bf16", "gfx942", "topksoftmax/", "_ZN5aiter25topksoftmax_12x128x4_bf16E", "topksoftmax_12x128x4_bf16.co"),
    ADD_CFG(  12,  128,    6, "bf16", "gfx942", "topksoftmax/", "_ZN5aiter25topksoftmax_12x128x6_bf16E", "topksoftmax_12x128x6_bf16.co"),
    ADD_CFG(  12,  128,    8, "bf16", "gfx942", "topksoftmax/", "_ZN5aiter25topksoftmax_12x128x8_bf16E", "topksoftmax_12x128x8_bf16.co"),
    ADD_CFG(  12,  256,    6, "bf16", "gfx942", "topksoftmax/", "_ZN5aiter25topksoftmax_12x256x6_bf16E", "topksoftmax_12x256x6_bf16.co"),
    ADD_CFG(  12,  256,    8, "bf16", "gfx942", "topksoftmax/", "_ZN5aiter25topksoftmax_12x256x8_bf16E", "topksoftmax_12x256x8_bf16.co"),
};
