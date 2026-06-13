// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#include <unordered_map>


#define ADD_CFG(qType, kvType, Gqa, Mtp, Msk, Hp, blkSz, ps, qTile, quant_type, arch, path, knl_name, co_name)         \
    {                                         \
        arch knl_name, { knl_name, path co_name, arch, qType, kvType, Gqa, Mtp, Msk, Hp, blkSz, ps, qTile, quant_type }         \
    }

struct paConfig
{
    std::string knl_name;
    std::string co_name;
    std::string arch;
    std::string qType;
    std::string kvType;
    int Gqa;
    int Mtp;
    int Msk;
    int Hp;
    int blkSz;
    int ps;
    int qTile;
    int quant_type;
};

using CFG = std::unordered_map<std::string, paConfig>;

static CFG cfg_pa_asm = {
    ADD_CFG("bf16", "fp8",    0,    0,    1,    1,  256,    1,   16,    6, "gfx942", "pa/", "_ZN5aiter44PA_A16W8_BLK256_1TG_4W_16mx1_64nx4_MTP_PS_PBE", "pa_bf16_perblockFp8_blk256_1tg_4w_qlen16_msk1_ps.co"),
    ADD_CFG("bf16", "fp8",    0,    0,    1,    1,  256,    1,   32,    6, "gfx942", "pa/", "_ZN5aiter44PA_A16W8_BLK256_1TG_4W_32mx1_64nx4_MTP_PS_PBE", "pa_bf16_perblockFp8_blk256_1tg_4w_qlen32_msk1_ps.co"),
    ADD_CFG("bf16", "fp8",    0,    0,    1,    1,  256,    1,   40,    6, "gfx942", "pa/", "_ZN5aiter44PA_A16W8_BLK256_1TG_4W_40mx1_64nx4_MTP_PS_PBE", "pa_bf16_perblockFp8_blk256_1tg_4w_qlen40_msk1_ps.co"),
    ADD_CFG("bf16", "fp8",    0,    0,    1,    1,  256,    1,   48,    6, "gfx942", "pa/", "_ZN5aiter44PA_A16W8_BLK256_1TG_4W_48mx1_64nx4_MTP_PS_PBE", "pa_bf16_perblockFp8_blk256_1tg_4w_qlen48_msk1_ps.co"),
    ADD_CFG("bf16", "fp8",    0,    0,    1,    1,  256,    1,   64,    6, "gfx942", "pa/", "_ZN5aiter44PA_A16W8_BLK256_1TG_4W_64mx1_64nx4_MTP_PS_PBE", "pa_bf16_perblockFp8_blk256_1tg_4w_qlen64_msk1_ps.co"),
    ADD_CFG("bf16", "fp8",    0,    0,    1,    1, 1024,    1,   16,    7, "gfx942", "pa/", "_ZN5aiter45PA_A16W8_BLK1024_1TG_4W_16mx1_64nx4_MTP_PS_PBE", "pa_bf16_perblockFp8_blk1024_1tg_4w_qlen16_msk1_ps.co"),
    ADD_CFG("bf16", "fp8",    0,    0,    1,    1, 1024,    1,   32,    7, "gfx942", "pa/", "_ZN5aiter45PA_A16W8_BLK1024_1TG_4W_32mx1_64nx4_MTP_PS_PBE", "pa_bf16_perblockFp8_blk1024_1tg_4w_qlen32_msk1_ps.co"),
    ADD_CFG("bf16", "fp8",    0,    0,    1,    1, 1024,    1,   40,    7, "gfx942", "pa/", "_ZN5aiter45PA_A16W8_BLK1024_1TG_4W_40mx1_64nx4_MTP_PS_PBE", "pa_bf16_perblockFp8_blk1024_1tg_4w_qlen40_msk1_ps.co"),
    ADD_CFG("bf16", "fp8",    0,    0,    1,    1, 1024,    1,   48,    7, "gfx942", "pa/", "_ZN5aiter45PA_A16W8_BLK1024_1TG_4W_48mx1_64nx4_MTP_PS_PBE", "pa_bf16_perblockFp8_blk1024_1tg_4w_qlen48_msk1_ps.co"),
    ADD_CFG("bf16", "fp8",    0,    0,    1,    1, 1024,    1,   64,    7, "gfx942", "pa/", "_ZN5aiter45PA_A16W8_BLK1024_1TG_4W_64mx1_64nx4_MTP_PS_PBE", "pa_bf16_perblockFp8_blk1024_1tg_4w_qlen64_msk1_ps.co"),
    ADD_CFG("bf16", "fp8",    0,    0,    1,    1, 1024,    1,   16,    2, "gfx942", "pa/", "_ZN5aiter42PA_A16W8_BLK1024_1TG_4W_16mx1_64nx4_MTP_PSE", "pa_bf16_pertokenFp8_gqa16_1tg_4w_qlen16_msk1_ps.co"),
    ADD_CFG("bf16", "fp8",    0,    0,    1,    1, 1024,    1,   32,    2, "gfx942", "pa/", "_ZN5aiter42PA_A16W8_BLK1024_1TG_4W_32mx1_64nx4_MTP_PSE", "pa_bf16_pertokenFp8_gqa16_1tg_4w_qlen32_msk1_ps.co"),
    ADD_CFG("bf16", "fp8",    0,    0,    1,    1, 1024,    1,   40,    2, "gfx942", "pa/", "_ZN5aiter42PA_A16W8_BLK1024_1TG_4W_40mx1_64nx4_MTP_PSE", "pa_bf16_pertokenFp8_gqa16_1tg_4w_qlen40_msk1_ps.co"),
    ADD_CFG("bf16", "fp8",    0,    0,    1,    1, 1024,    1,   48,    2, "gfx942", "pa/", "_ZN5aiter42PA_A16W8_BLK1024_1TG_4W_48mx1_64nx4_MTP_PSE", "pa_bf16_pertokenFp8_gqa16_1tg_4w_qlen48_msk1_ps.co"),
    ADD_CFG("bf16", "fp8",    0,    0,    1,    1, 1024,    1,   64,    2, "gfx942", "pa/", "_ZN5aiter42PA_A16W8_BLK1024_1TG_4W_64mx1_64nx4_MTP_PSE", "pa_bf16_pertokenFp8_gqa16_1tg_4w_qlen64_msk1_ps.co"),
    ADD_CFG("bf16", "fp8",   16,    0,    0,    0,   16,    0,    0,    0, "gfx942", "pa/", "_ZN5aiter32pa_bf16_pertokenFp8_gqa16_2tg_4wE", "pa_bf16_pertokenFp8_gqa16_2tg_4w.co"),
    ADD_CFG("bf16", "int8",   16,    0,    0,    0,   16,    0,    0,    0, "gfx942", "pa/", "_ZN5aiter33pa_bf16_pertokenInt8_gqa16_2tg_4wE", "pa_bf16_pertokenInt8_gqa16_2tg_4w.co"),
    ADD_CFG("bf16", "bf16",   16,    0,    0,    0,   16,    0,    0,    0, "gfx942", "pa/", "_ZN5aiter28pa_bf16_noquant_gqa16_1tg_4wE", "pa_bf16_noquant_gqa16_1tg_4w.co"),
    ADD_CFG("bf16", "bf16",    8,    0,    0,    0,   16,    0,    0,    0, "gfx942", "pa/", "_ZN5aiter27pa_bf16_noquant_gqa8_1tg_4wE", "pa_bf16_noquant_gqa8_1tg_4w.co"),
    ADD_CFG("fp16", "fp8",    8,    1,    1,    0,   16,    0,    0,    0, "gfx942", "pa/", "_ZN5aiter40pa_fp16_pertokenFp8_gqa8_1tg_4w_mtp_msk1E", "pa_fp16_pertokenFp8_gqa8_1tg_4w_mtp_msk1.co"),
    ADD_CFG("fp16", "fp16",    8,    1,    0,    0,   16,    0,    0,    0, "gfx942", "pa/", "_ZN5aiter36pa_fp16_noquant_gqa8_1tg_4w_mtp_msk0E", "pa_fp16_noquant_gqa8_1tg_4w_mtp_msk0.co"),
    ADD_CFG("bf16", "int8",    8,    0,    0,    0,   16,    0,    0,    0, "gfx942", "pa/", "_ZN5aiter32pa_bf16_pertokenInt8_gqa8_2tg_4wE", "pa_bf16_pertokenInt8_gqa8_2tg_4w.co"),
    ADD_CFG("bf16", "fp8",    8,    0,    0,    0,   16,    0,    0,    0, "gfx942", "pa/", "_ZN5aiter31pa_bf16_pertokenFp8_gqa8_2tg_4wE", "pa_bf16_pertokenFp8_gqa8_2tg_4w.co"),
    ADD_CFG("fp16", "fp8",    8,    0,    0,    1,   16,    0,    0,    0, "gfx942", "pa/", "_ZN5aiter34pa_fp16_pertokenFp8_gqa8_2tg_4w_hpE", "pa_fp16_pertokenFp8_gqa8_2tg_4w_hp.co"),
    ADD_CFG("bf16", "fp8",    8,    1,    1,    0,   16,    0,    0,    0, "gfx942", "pa/", "_ZN5aiter40pa_bf16_pertokenFp8_gqa8_1tg_4w_mtp_msk1E", "pa_bf16_pertokenFp8_gqa8_1tg_4w_mtp_msk1.co"),
    ADD_CFG("bf16", "fp8",    8,    0,    0,    2,   16,    0,    0,    0, "gfx942", "pa/", "_ZN5aiter35pa_bf16_pertokenFp8_gqa8_2tg_4w_uhpE", "pa_bf16_pertokenFp8_gqa8_2tg_4w_uhp.co"),
    ADD_CFG("bf16", "bf16",    8,    1,    0,    0,   16,    0,    0,    0, "gfx942", "pa/", "_ZN5aiter36pa_bf16_noquant_gqa8_1tg_4w_mtp_msk0E", "pa_bf16_noquant_gqa8_1tg_4w_mtp_msk0.co"),
    ADD_CFG("fp16", "int8",    8,    0,    0,    0,   16,    0,    0,    0, "gfx942", "pa/", "_ZN5aiter32pa_fp16_pertokenInt8_gqa8_2tg_4wE", "pa_fp16_pertokenInt8_gqa8_2tg_4w.co"),
    ADD_CFG("bf16", "fp8",    8,    0,    0,    1,   16,    0,    0,    0, "gfx942", "pa/", "_ZN5aiter34pa_bf16_pertokenFp8_gqa8_2tg_4w_hpE", "pa_bf16_pertokenFp8_gqa8_2tg_4w_hp.co"),
    ADD_CFG("fp16", "fp16",    8,    1,    1,    0,   16,    0,    0,    0, "gfx942", "pa/", "_ZN5aiter36pa_fp16_noquant_gqa8_1tg_4w_mtp_msk1E", "pa_fp16_noquant_gqa8_1tg_4w_mtp_msk1.co"),
    ADD_CFG("fp16", "int8",    8,    1,    1,    0,   16,    0,    0,    0, "gfx942", "pa/", "_ZN5aiter41pa_fp16_pertokenInt8_gqa8_1tg_4w_mtp_msk1E", "pa_fp16_pertokenInt8_gqa8_1tg_4w_mtp_msk1.co"),
    ADD_CFG("bf16", "bf16",    8,    1,    1,    0,   16,    0,    0,    0, "gfx942", "pa/", "_ZN5aiter36pa_bf16_noquant_gqa8_1tg_4w_mtp_msk1E", "pa_bf16_noquant_gqa8_1tg_4w_mtp_msk1.co"),
    ADD_CFG("bf16", "fp8",   16,    1,    1,    0,   16,    0,    0,    0, "gfx942", "pa/", "_ZN5aiter41pa_bf16_pertokenFp8_gqa16_1tg_4w_mtp_msk1E", "pa_bf16_pertokenFp8_gqa16_1tg_4w_mtp_msk1.co"),
    ADD_CFG("bf16", "int8",    8,    1,    1,    0,   16,    0,    0,    0, "gfx942", "pa/", "_ZN5aiter41pa_bf16_pertokenInt8_gqa8_1tg_4w_mtp_msk1E", "pa_bf16_pertokenInt8_gqa8_1tg_4w_mtp_msk1.co"),
    ADD_CFG("bf16", "int8",   16,    1,    1,    0,   16,    0,    0,    0, "gfx942", "pa/", "_ZN5aiter42pa_bf16_pertokenInt8_gqa16_1tg_4w_mtp_msk1E", "pa_bf16_pertokenInt8_gqa16_1tg_4w_mtp_msk1.co"),
    ADD_CFG("fp16", "fp16",    8,    0,    0,    0,   16,    0,    0,    0, "gfx942", "pa/", "_ZN5aiter27pa_fp16_noquant_gqa8_1tg_4wE", "pa_fp16_noquant_gqa8_1tg_4w.co"),
    ADD_CFG("fp16", "fp8",   16,    0,    0,    0,   16,    0,    0,    0, "gfx942", "pa/", "_ZN5aiter32pa_fp16_pertokenFp8_gqa16_2tg_4wE", "pa_fp16_pertokenFp8_gqa16_2tg_4w.co"),
    ADD_CFG("fp16", "fp8",    8,    1,    0,    0,   16,    0,    0,    0, "gfx942", "pa/", "_ZN5aiter40pa_fp16_pertokenFp8_gqa8_1tg_4w_mtp_msk0E", "pa_fp16_pertokenFp8_gqa8_1tg_4w_mtp_msk0.co"),
    ADD_CFG("fp16", "int8",   16,    0,    0,    0,   16,    0,    0,    0, "gfx942", "pa/", "_ZN5aiter33pa_fp16_pertokenInt8_gqa16_2tg_4wE", "pa_fp16_pertokenInt8_gqa16_2tg_4w.co"),
    ADD_CFG("fp16", "int8",   16,    1,    0,    0,   16,    0,    0,    0, "gfx942", "pa/", "_ZN5aiter42pa_fp16_pertokenInt8_gqa16_1tg_4w_mtp_msk0E", "pa_fp16_pertokenInt8_gqa16_1tg_4w_mtp_msk0.co"),
    ADD_CFG("bf16", "fp8",    8,    1,    0,    0,   16,    0,    0,    0, "gfx942", "pa/", "_ZN5aiter40pa_bf16_pertokenFp8_gqa8_1tg_4w_mtp_msk0E", "pa_bf16_pertokenFp8_gqa8_1tg_4w_mtp_msk0.co"),
    ADD_CFG("fp16", "fp16",   16,    0,    0,    0,   16,    0,    0,    0, "gfx942", "pa/", "_ZN5aiter28pa_fp16_noquant_gqa16_1tg_4wE", "pa_fp16_noquant_gqa16_1tg_4w.co"),
    ADD_CFG("fp16", "fp8",    8,    0,    0,    2,   16,    0,    0,    0, "gfx942", "pa/", "_ZN5aiter35pa_fp16_pertokenFp8_gqa8_2tg_4w_uhpE", "pa_fp16_pertokenFp8_gqa8_2tg_4w_uhp.co"),
    ADD_CFG("bf16", "int8",    8,    1,    0,    0,   16,    0,    0,    0, "gfx942", "pa/", "_ZN5aiter41pa_bf16_pertokenInt8_gqa8_1tg_4w_mtp_msk0E", "pa_bf16_pertokenInt8_gqa8_1tg_4w_mtp_msk0.co"),
    ADD_CFG("bf16", "int8",   16,    1,    0,    0,   16,    0,    0,    0, "gfx942", "pa/", "_ZN5aiter42pa_bf16_pertokenInt8_gqa16_1tg_4w_mtp_msk0E", "pa_bf16_pertokenInt8_gqa16_1tg_4w_mtp_msk0.co"),
    ADD_CFG("fp16", "int8",   16,    1,    1,    0,   16,    0,    0,    0, "gfx942", "pa/", "_ZN5aiter42pa_fp16_pertokenInt8_gqa16_1tg_4w_mtp_msk1E", "pa_fp16_pertokenInt8_gqa16_1tg_4w_mtp_msk1.co"),
    ADD_CFG("fp16", "fp8",   16,    1,    0,    0,   16,    0,    0,    0, "gfx942", "pa/", "_ZN5aiter41pa_fp16_pertokenFp8_gqa16_1tg_4w_mtp_msk0E", "pa_fp16_pertokenFp8_gqa16_1tg_4w_mtp_msk0.co"),
    ADD_CFG("fp16", "fp8",   16,    1,    1,    0,   16,    0,    0,    0, "gfx942", "pa/", "_ZN5aiter41pa_fp16_pertokenFp8_gqa16_1tg_4w_mtp_msk1E", "pa_fp16_pertokenFp8_gqa16_1tg_4w_mtp_msk1.co"),
    ADD_CFG("bf16", "fp8",   16,    1,    0,    0,   16,    0,    0,    0, "gfx942", "pa/", "_ZN5aiter41pa_bf16_pertokenFp8_gqa16_1tg_4w_mtp_msk0E", "pa_bf16_pertokenFp8_gqa16_1tg_4w_mtp_msk0.co"),
    ADD_CFG("fp16", "fp8",    8,    0,    0,    0,   16,    0,    0,    0, "gfx942", "pa/", "_ZN5aiter31pa_fp16_pertokenFp8_gqa8_2tg_4wE", "pa_fp16_pertokenFp8_gqa8_2tg_4w.co"),
    ADD_CFG("fp16", "int8",    8,    1,    0,    0,   16,    0,    0,    0, "gfx942", "pa/", "_ZN5aiter41pa_fp16_pertokenInt8_gqa8_1tg_4w_mtp_msk0E", "pa_fp16_pertokenInt8_gqa8_1tg_4w_mtp_msk0.co"),
    ADD_CFG("bf16", "fp8",   10,    0,    0,    0, 1024,    0,    0,    0, "gfx942", "pa/", "_ZN5aiter43pa_bf16_pertokenFp8_gqa10_1tg_4w_qlen1_msk1E", "pa_bf16_pertokenFp8_gqa10_1tg_4w_qlen1_msk1.co"),
    ADD_CFG("bf16", "fp8",   10,   12,    1,    0, 1024,    0,    0,    0, "gfx942", "pa/", "_ZN5aiter43pa_bf16_pertokenFp8_gqa10_1tg_4w_qlen2_msk1E", "pa_bf16_pertokenFp8_gqa10_1tg_4w_qlen2_msk1.co"),
    ADD_CFG("bf16", "fp8",   10,   13,    1,    0, 1024,    0,    0,    0, "gfx942", "pa/", "_ZN5aiter43pa_bf16_pertokenFp8_gqa10_1tg_4w_qlen3_msk1E", "pa_bf16_pertokenFp8_gqa10_1tg_4w_qlen3_msk1.co"),
    ADD_CFG("bf16", "fp8",   10,   14,    1,    0, 1024,    0,    0,    0, "gfx942", "pa/", "_ZN5aiter43pa_bf16_pertokenFp8_gqa10_1tg_4w_qlen4_msk1E", "pa_bf16_pertokenFp8_gqa10_1tg_4w_qlen4_msk1.co"),
};
