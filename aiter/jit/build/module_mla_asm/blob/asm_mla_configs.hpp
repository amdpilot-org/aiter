// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#include <unordered_map>


#define ADD_CFG(qType, kvType, Gqa, ps, qSeqLen, prefill, causal, lse, arch, path, knl_name, co_name)         \
    {                                         \
        arch knl_name, { knl_name, path co_name, arch, qType, kvType, Gqa, ps, qSeqLen, prefill, causal, lse }         \
    }

struct mlaConfig
{
    std::string knl_name;
    std::string co_name;
    std::string arch;
    std::string qType;
    std::string kvType;
    int Gqa;
    int ps;
    int qSeqLen;
    int prefill;
    int causal;
    int lse;
};

using CFG = std::unordered_map<std::string, mlaConfig>;

static CFG cfg_mla_asm = {
    ADD_CFG("bf16", "bf16",  128,    0,    0,    0,    0,    0, "gfx942", "mla/", "_ZN5aiter41mla_dec_stage1_bf16_a16w16_subQ128_mqa128E", "mla_dec_stage1_bf16_a16w16_subQ128_mqa128.co"),
    ADD_CFG("bf16", "bf16",   16,    1,    4,    0,    0,    0, "gfx942", "mla/", "_ZN5aiter42mla_a16w16_qh16_m16x4_n16x1_coex0_mask1_psE", "mla_a16w16_qh16_m16x4_n16x1_coex0_mask1_ps.co"),
    ADD_CFG("bf16", "bf16",   16,    0,    1,    0,    0,    0, "gfx942", "mla/", "_ZN5aiter39mla_dec_stage1_bf16_a16w16_subQ16_mqa16E", "mla_dec_stage1_bf16_a16w16_subQ16_mqa16.co"),
    ADD_CFG("bf16", "bf16",   16,    0,    4,    0,    0,    0, "gfx942", "mla/", "_ZN5aiter39mla_a16w16_qh16_m16x4_n16x1_coex0_mask1E", "mla_a16w16_qh16_m16x4_n16x1_coex0_mask1.co"),
    ADD_CFG("bf16", "bf16",   16,    0,    8,    0,    0,    0, "gfx942", "mla/", "_ZN5aiter39mla_a16w16_qh16_m32x4_n16x1_coex0_mask1E", "mla_a16w16_qh16_m32x4_n16x1_coex0_mask1.co"),
    ADD_CFG("bf16", "fp8",   16,    1,    4,    0,    0,    0, "gfx942", "mla/", "_ZN5aiter41mla_a16w8_qh16_m16x4_n16x1_coex0_mask1_psE", "mla_a16w8_qh16_m16x4_n16x1_coex0_mask1_ps.co"),
    ADD_CFG("bf16", "byte",   16,    1,    4,    0,    0,    0, "gfx942", "mla/", "_ZN5aiter53mla_a16w8_qh16_m16x4_n16x1_coex0_mask1_ps_page64_ds32E", "mla_a16w8_qh16_m16x4_n16x1_coex0_mask1_ps_page64_ds32.co"),
    ADD_CFG("fp8", "fp8",   16,    1,    1,    0,    0,    0, "gfx942", "mla/", "_ZN5aiter36mla_a8w8_qh16_qseqlen1_gqaratio16_psE", "mla_a8w8_qh16_qseqlen1_gqaratio16_ps.co"),
    ADD_CFG("fp8", "fp8",   16,    1,    2,    0,    0,    0, "gfx942", "mla/", "_ZN5aiter36mla_a8w8_qh16_qseqlen2_gqaratio16_psE", "mla_a8w8_qh16_qseqlen2_gqaratio16_ps.co"),
    ADD_CFG("fp8", "fp8",   16,    1,    4,    0,    0,    0, "gfx942", "mla/", "_ZN5aiter36mla_a8w8_qh64_qseqlen4_gqaratio16_psE", "mla_a8w8_qh64_qseqlen4_gqaratio16_ps.co"),
    ADD_CFG("fp8", "fp8",   16,    0,    1,    0,    0,    0, "gfx942", "mla/", "_ZN5aiter33mla_a8w8_qh16_qseqlen1_gqaratio16E", "mla_a8w8_qh16_qseqlen1_gqaratio16.co"),
    ADD_CFG("fp8", "fp8",   16,    0,    2,    0,    0,    0, "gfx942", "mla/", "_ZN5aiter33mla_a8w8_qh16_qseqlen2_gqaratio16E", "mla_a8w8_qh16_qseqlen2_gqaratio16.co"),
    ADD_CFG("fp8", "fp8",   16,    0,    4,    0,    0,    0, "gfx942", "mla/", "_ZN5aiter33mla_a8w8_qh64_qseqlen4_gqaratio16E", "mla_a8w8_qh64_qseqlen4_gqaratio16.co"),
    ADD_CFG("fp8", "fp8",  128,    0,    0,    0,    0,    0, "gfx942", "mla/", "_ZN5aiter31mla_a8w8_qh128_m32x4_n16x2_msk1E", "mla_a8w8_qh128_m32x4_n16x2_msk1.co"),
    ADD_CFG("bf16", "bf16",   16,    0,    0,    1,    0,    0, "gfx942", "mla/", "_ZN5aiter39mla_pfl_bf16_a16w16_causal_subQ16_mqa16E", "mla_pfl_bf16_a16w16_causal_subQ16_mqa16.co"),
    ADD_CFG("bf16", "bf16",  128,    0,    0,    1,    0,    0, "gfx942", "mla/", "_ZN5aiter41mla_pfl_bf16_a16w16_causal_subQ128_mqa128E", "mla_pfl_bf16_a16w16_causal_subQ128_mqa128.co"),
    ADD_CFG("fp8", "fp8",  128,    1,    0,    0,    0,    0, "gfx942", "mla/", "_ZN5aiter34mla_a8w8_qh128_m32x4_n16x2_msk0_psE", "mla_a8w8_qh128_m32x4_n16x2_msk0_ps.co"),
    ADD_CFG("fp8", "fp8",    1,    1,    0,    1,    1,    0, "gfx942", "mla/", "_ZN5aiter40mla_pfl_qh192_vh128_m32x8_n128x1_causal1E", "mla_pfl_qh192_vh128_m32x8_n128x1_causal1.co"),
    ADD_CFG("fp8", "fp8",    1,    1,    0,    1,    0,    0, "gfx942", "mla/", "_ZN5aiter40mla_pfl_qh192_vh128_m32x8_n128x1_causal0E", "mla_pfl_qh192_vh128_m32x8_n128x1_causal0.co"),
    ADD_CFG("bf16", "bf16",   32,    0,    0,    0,    0,    0, "gfx942", "mla/", "_ZN5aiter39mla_a16w16_qh16_m32x1_n16x1_coex0_mask1E", "MLA_A16W16_1TG_4W_32mx1_16nx1_Coex0_Msk1_QH16.co"),
    ADD_CFG("bf16", "bf16",   64,    0,    0,    0,    0,    0, "gfx942", "mla/", "_ZN5aiter39mla_a16w16_qh16_m64x1_n16x1_coex0_mask1E", "MLA_A16W16_1TG_4W_64mx1_16nx1_Coex0_Msk1_QH16.co"),
};
