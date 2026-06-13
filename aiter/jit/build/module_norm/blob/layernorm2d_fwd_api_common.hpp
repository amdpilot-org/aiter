
// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <ck_tile/core.hpp>
#include "layernorm2d_fwd.hpp"
#include <ck_tile/ops/epilogue.hpp>
#include <iostream>

#pragma once

using S = ck_tile::stream_config;
using A = layernorm2d_fwd_args;


// this is used to pattern-match internl kernel implementation, not to instantiate kernel
template <typename XDataType_,
          typename YDataType_,
          typename SmoothScaleDataType_,
          typename YScaleDataType_,
          ck_tile::index_t Repeat_M_,         // each thread repeat along M
          ck_tile::index_t Repeat_N_,         // each thread repeat along N
          ck_tile::index_t ThreadPerBlock_M_, // num threads along M
          ck_tile::index_t ThreadPerBlock_N_, // num threads along N
          ck_tile::index_t Vector_N_,         // vector size along N
          bool kPadN_,
          bool kSaveMeanInvStd_,
          bool kFastFDiv_,
          bool kWelford_,
          bool kTwoPass_,
          ck_tile::index_t kXbias_ = 0,
          ck_tile::index_t kFusedAdd_ = 0,
          ck_tile::index_t kFusedQuant_ = 0>
struct layernorm2d_fwd_traits_
{
    using XDataType = ck_tile::remove_cvref_t<XDataType_>;
    using YDataType = ck_tile::remove_cvref_t<YDataType_>;
    using SmoothScaleDataType = ck_tile::remove_cvref_t<SmoothScaleDataType_>;
    using YScaleDataType = ck_tile::remove_cvref_t<YScaleDataType_>;

    static constexpr bool is_warp_per_row = ThreadPerBlock_N_ <= ck_tile::get_warp_size();
    static_assert((ThreadPerBlock_M_ * ThreadPerBlock_N_) % ck_tile::get_warp_size() == 0);
    static constexpr ck_tile::index_t total_warps =
        (ThreadPerBlock_M_ * ThreadPerBlock_N_) / ck_tile::get_warp_size();

    // num of warps along m
    static constexpr ck_tile::index_t BlockWarps_M = []() {
        if constexpr(is_warp_per_row)
        {
            static_assert(ck_tile::get_warp_size() % ThreadPerBlock_N_ == 0);
            return total_warps;
        }
        else
        {
            // static_assert(ck_tile::get_warp_size() % ThreadPerBlock_M_ == 0);
            return total_warps / (ThreadPerBlock_N_ / ck_tile::get_warp_size());
        }
    }();

    // num of warps along n
    static constexpr ck_tile::index_t BlockWarps_N = []() {
        if constexpr(is_warp_per_row)
        {
            static_assert(ck_tile::get_warp_size() % ThreadPerBlock_N_ == 0);
            return 1;
        }
        else
        {
            static_assert(ThreadPerBlock_N_ % ck_tile::get_warp_size() == 0);
            return ThreadPerBlock_N_ / ck_tile::get_warp_size();
        }
    }();

    static constexpr ck_tile::index_t Repeat_M = Repeat_M_;
    static constexpr ck_tile::index_t Repeat_N = Repeat_N_;

    static constexpr ck_tile::index_t Block_M = Repeat_M_ * ThreadPerBlock_M_;
    static constexpr ck_tile::index_t Block_N = Repeat_N_ * ThreadPerBlock_N_ * Vector_N_;

    using BlockTile  = ck_tile::sequence<Block_M, Block_N>;
    using Vector     = ck_tile::sequence<1, Vector_N_>;
    using ThreadPerBlock = ck_tile::sequence<ThreadPerBlock_M_, ThreadPerBlock_N_>;

    using Shape = ck_tile::Generic2dBlockShape<BlockTile, ThreadPerBlock, Vector>;

    static constexpr bool kPadN           = kPadN_;
    static constexpr bool kSaveMeanInvStd = kSaveMeanInvStd_;
    static constexpr bool kFastFDiv       = kFastFDiv_;
    static constexpr bool kWelford        = kWelford_;
    static constexpr bool kTwoPass        = kTwoPass_;
    static constexpr ck_tile::index_t kXbias = kXbias_;
    static constexpr ck_tile::index_t kFusedAdd = kFusedAdd_;
    static constexpr ck_tile::index_t kFusedQuant = kFusedQuant_;
};

template <typename XDataType_,
          typename YDataType_,
          typename SmoothScaleDataType_,
          typename YScaleDataType_,
          ck_tile::index_t Repeat_M_,         // each thread repeat along M
          ck_tile::index_t Repeat_N_,         // each thread repeat along N
          ck_tile::index_t ThreadPerBlock_M_, // num threads along M
          ck_tile::index_t ThreadPerBlock_N_, // num threads along N
          ck_tile::index_t Vector_N_,         // vector size along N
          bool kPadN_,
          bool kSaveMeanInvStd_,
          bool kFastFDiv_,
          bool kWelford_,
          bool kTwoPass_,
          int  kXbias_,
          int  kFusedAdd_,
          int  kFusedQuant_>
using traits_ = layernorm2d_fwd_traits_<XDataType_,
                                       YDataType_,
                                       SmoothScaleDataType_,
                                       YScaleDataType_,
                                       Repeat_M_,
                                       Repeat_N_,
                                       ThreadPerBlock_M_,
                                       ThreadPerBlock_N_,
                                       Vector_N_,
                                       kPadN_,
                                       kSaveMeanInvStd_,
                                       kFastFDiv_,
                                       kWelford_,
                                       kTwoPass_,
                                       kXbias_,
                                       kFusedAdd_,
                                       kFusedQuant_>;


template <typename Traits_>
float layernorm2d_fwd_(const S& s, A a)
{
    using XDataType = typename Traits_::XDataType;
    using YDataType = typename Traits_::YDataType;
    using SmoothScaleDataType = typename Traits_::SmoothScaleDataType;
    using YScaleDataType = typename Traits_::YScaleDataType;
    using ComputeDataType = typename LayerNormTypeConfig<XDataType, YDataType, SmoothScaleDataType, YScaleDataType>::ComputeDataType;

    using PipelineTraits = ck_tile::Layernorm2dFwdTraits<Traits_::kPadN,
        Traits_::kSaveMeanInvStd,
        Traits_::kFastFDiv,
        Traits_::kWelford,
        Traits_::kTwoPass,
        static_cast<ck_tile::Layernorm2dXBiasEnum>(Traits_::kXbias),
        static_cast<ck_tile::Layernorm2dFusedAddEnum>(Traits_::kFusedAdd),
        static_cast<ck_tile::Layernorm2dFusedQuantEnum>(Traits_::kFusedQuant)>;
    using PipelineProblem = ck_tile::Layernorm2dFwdPipelineProblem<
        typename LayerNormTypeConfig<XDataType, YDataType, SmoothScaleDataType, YScaleDataType>::XDataType,
        typename LayerNormTypeConfig<XDataType, YDataType, SmoothScaleDataType, YScaleDataType>::XBiasDataType,
        typename LayerNormTypeConfig<XDataType, YDataType, SmoothScaleDataType, YScaleDataType>::GammaDataType,
        typename LayerNormTypeConfig<XDataType, YDataType, SmoothScaleDataType, YScaleDataType>::BetaDataType,
        typename LayerNormTypeConfig<XDataType, YDataType, SmoothScaleDataType, YScaleDataType>::ComputeDataType,
        typename LayerNormTypeConfig<XDataType, YDataType, SmoothScaleDataType, YScaleDataType>::YDataType,
        typename LayerNormTypeConfig<XDataType, YDataType, SmoothScaleDataType, YScaleDataType>::MeanDataType,
        typename LayerNormTypeConfig<XDataType, YDataType, SmoothScaleDataType, YScaleDataType>::InvStdDataType,
        typename LayerNormTypeConfig<XDataType, YDataType, SmoothScaleDataType, YScaleDataType>::SmoothScaleDataType,
        typename LayerNormTypeConfig<XDataType, YDataType, SmoothScaleDataType, YScaleDataType>::YScaleDataType,
        typename Traits_::Shape,
        PipelineTraits>;

    using OnePassPipeline = ck_tile::Layernorm2dFwdPipelineOnePass<PipelineProblem>;
    using TwoPassPipeline = ck_tile::Layernorm2dFwdPipelineTwoPass<PipelineProblem>;
    using Pipeline        = std::conditional_t<Traits_::kTwoPass, TwoPassPipeline, OnePassPipeline>;

    using Default2DEpilogueProblem = ck_tile::Default2DEpilogueProblem<ComputeDataType, YDataType, false, Traits_::kPadN, true>;
    using Default2DEpilogue = ck_tile::Default2DEpilogue<Default2DEpilogueProblem>;

    static constexpr bool UseSmoothInputScale = Traits_::kFusedQuant == 1;
    static constexpr bool UseRawStore = sizeof(YDataType) == 4;
    using DynamicQuantEpilogueProblem = ck_tile::DynamicQuantEpilogueProblem<ComputeDataType, SmoothScaleDataType, YScaleDataType, YDataType, typename Traits_::Shape,
            ck_tile::DynamicQuantEpilogueTraits<false, Traits_::kPadN, UseSmoothInputScale, UseRawStore,  true/*max3*/>>;

    using DynamicQuantEpilogue = ck_tile::DynamicQuantEpilogue<DynamicQuantEpilogueProblem>;

    using Epilogue = std::conditional_t<Traits_::kFusedQuant == 1, DynamicQuantEpilogue,  Default2DEpilogue>;

    using Kernel = ck_tile::Layernorm2dFwd<Pipeline, Epilogue>;

    const dim3 grids                       = Kernel::GridSize(a);
    const dim3 blocks                      = Kernel::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = 1;

    auto kargs = Kernel::MakeKargs(a);
    if(s.log_level_ > 0)
        std::cout << ", " << Kernel::GetName() << std::flush;

    return ck_tile::launch_kernel(
        s, ck_tile::make_kernel<kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
}

