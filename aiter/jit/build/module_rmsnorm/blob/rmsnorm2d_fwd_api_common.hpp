
// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#include <ck_tile/core.hpp>
#include "rmsnorm2d_fwd.hpp"
#include <ck_tile/ops/epilogue.hpp>
#include <iostream>

#pragma once

using S = ck_tile::stream_config;
using A = rmsnorm2d_fwd_args;


// this is used to pattern-match internl kernel implementation, not to instantiate kernel
template <typename XDataType_,
          typename YDataType_,
          typename SmoothScaleDataType_,
          typename YScaleDataType_,
          typename UnquantYDataType_,
          ck_tile::index_t Repeat_M_,         // each thread repeat along M
          ck_tile::index_t Repeat_N_,         // each thread repeat along N
          ck_tile::index_t ThreadPerBlock_M_, // num threads along M
          ck_tile::index_t ThreadPerBlock_N_, // num threads along N
          ck_tile::index_t Vector_N_,         // vector size along N
          bool kPadN_,
          bool kSaveInvRms_,
          bool kSaveUnquant_,
          bool kTwoPass_,
          ck_tile::index_t kFusedAdd_ = 0,
          ck_tile::index_t kFusedQuant_ = 0,
          ck_tile::index_t kUseModelSensitiveRMSNorm_ = 0>
struct rmsnorm2d_fwd_traits_
{
    using XDataType           = ck_tile::remove_cvref_t<XDataType_>;
    using YDataType           = ck_tile::remove_cvref_t<YDataType_>;
    using SmoothScaleDataType = ck_tile::remove_cvref_t<SmoothScaleDataType_>;
    using YScaleDataType      = ck_tile::remove_cvref_t<YScaleDataType_>;
    using UnquantYDataType    = ck_tile::remove_cvref_t<UnquantYDataType_>;

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

    static constexpr bool kPadN        = kPadN_;
    static constexpr bool kSaveInvRms  = kSaveInvRms_;
    static constexpr bool kSaveUnquant = kSaveUnquant_;
    static constexpr bool kTwoPass     = kTwoPass_;
    static constexpr ck_tile::index_t kFusedAdd                 = kFusedAdd_;
    static constexpr ck_tile::index_t kFusedQuant               = kFusedQuant_;
    static constexpr ck_tile::index_t kUseModelSensitiveRMSNorm = kUseModelSensitiveRMSNorm_;
};

template <typename XDataType_,
          typename YDataType_,
          typename SmoothScaleDataType_,
          typename YScaleDataType_,
          typename UnquantYDataType_,
          ck_tile::index_t Repeat_M_,         // each thread repeat along M
          ck_tile::index_t Repeat_N_,         // each thread repeat along N
          ck_tile::index_t ThreadPerBlock_M_, // num threads along M
          ck_tile::index_t ThreadPerBlock_N_, // num threads along N
          ck_tile::index_t Vector_N_,         // vector size along N
          bool kPadN_,
          bool kSaveInvRms_,
          bool kSaveUnquant_,
          bool kTwoPass_,
          int  kFusedAdd_,
          int  kFusedQuant_,
          int  kUseModelSensitiveRMSNorm_>
using traits_ = rmsnorm2d_fwd_traits_<XDataType_,
                                      YDataType_,
                                      SmoothScaleDataType_,
                                      YScaleDataType_,
                                      UnquantYDataType_,
                                      Repeat_M_,
                                      Repeat_N_,
                                      ThreadPerBlock_M_,
                                      ThreadPerBlock_N_,
                                      Vector_N_,
                                      kPadN_,
                                      kSaveInvRms_,
                                      kSaveUnquant_,
                                      kTwoPass_,
                                      kFusedAdd_,
                                      kFusedQuant_,
                                      kUseModelSensitiveRMSNorm_>;


template <typename Traits_>
float rmsnorm2d_fwd_(const S& s, A a)
{
    using XDataType           = typename Traits_::XDataType;
    using YDataType           = typename Traits_::YDataType;
    using SmoothScaleDataType = typename Traits_::SmoothScaleDataType;
    using YScaleDataType      = typename Traits_::YScaleDataType;
    using UnquantYDataType    = typename Traits_::UnquantYDataType;
    using ComputeDataType     = typename RmsnormTypeConfig<XDataType, YDataType, SmoothScaleDataType, YScaleDataType>::ComputeDataType;

    using PipelineTraits =
        ck_tile::Rmsnorm2dFwdTraits<Traits_::kPadN,
                                    Traits_::kSaveInvRms,
                                    Traits_::kSaveUnquant,
                                    Traits_::kTwoPass,
                                    static_cast<ck_tile::Rmsnorm2dFusedAddEnum>(Traits_::kFusedAdd),
                                    static_cast<ck_tile::Rmsnorm2dFusedQuantEnum>(Traits_::kFusedQuant),
                                    static_cast<ck_tile::Rmsnorm2dSensitiveEnum>(Traits_::kUseModelSensitiveRMSNorm)>;

    using PipelineProblem =
        ck_tile::Rmsnorm2dFwdPipelineProblem<typename RmsnormTypeConfig<XDataType, YDataType, SmoothScaleDataType, YScaleDataType>::XDataType,
                                             typename RmsnormTypeConfig<XDataType, YDataType, SmoothScaleDataType, YScaleDataType>::GammaDataType,
                                             typename RmsnormTypeConfig<XDataType, YDataType, SmoothScaleDataType, YScaleDataType>::ComputeDataType,
                                             typename RmsnormTypeConfig<XDataType, YDataType, SmoothScaleDataType, YScaleDataType>::YDataType,
                                             typename RmsnormTypeConfig<XDataType, YDataType, SmoothScaleDataType, YScaleDataType>::InvRmsDataType,
                                             typename RmsnormTypeConfig<XDataType, YDataType, SmoothScaleDataType, YScaleDataType>::UnquantYDataType,
                                             typename RmsnormTypeConfig<XDataType, YDataType, SmoothScaleDataType, YScaleDataType>::SmoothScaleDataType,
                                             typename RmsnormTypeConfig<XDataType, YDataType, SmoothScaleDataType, YScaleDataType>::YScaleDataType,
                                             typename Traits_::Shape,
                                             PipelineTraits>;

    using OnePassPipeline = ck_tile::Rmsnorm2dFwdPipelineOnePass<PipelineProblem>;
    using TwoPassPipeline = ck_tile::Rmsnorm2dFwdPipelineTwoPass<PipelineProblem>;
    using T5PassPipeline  = ck_tile::Rmsnorm2dFwdPipelineModelSensitiveT5Pass<PipelineProblem>;

    using Pipeline = std::conditional_t<
        (Traits_::kUseModelSensitiveRMSNorm == 0 || Traits_::kTwoPass), // TODO: consider TwoPass for T5PassPipeline
        std::conditional_t<Traits_::kTwoPass, TwoPassPipeline, OnePassPipeline>,  // kUseModelSensitiveRMSNorm == 0
        T5PassPipeline
    >;

    using Default2DEpilogueProblem = ck_tile::Default2DEpilogueProblem<ComputeDataType, YDataType, false, Traits_::kPadN, false>;
    using Default2DEpilogue = ck_tile::Default2DEpilogue<Default2DEpilogueProblem>;

    static constexpr bool UseSmoothInputScale = Traits_::kFusedQuant == 1;
    using DynamicQuantEpilogueProblem = ck_tile::DynamicQuantEpilogueProblem<ComputeDataType, SmoothScaleDataType, YScaleDataType, YDataType, typename Traits_::Shape,
            ck_tile::DynamicQuantEpilogueTraits<false, Traits_::kPadN, UseSmoothInputScale, false,  true/*max3*/>>;

    using DynamicQuantEpilogue = ck_tile::DynamicQuantEpilogue<DynamicQuantEpilogueProblem>;

    using Default2DAndDynamicQuantEpilogueProblem = ck_tile::Default2DAndDynamicQuantEpilogueProblem<
        ComputeDataType, SmoothScaleDataType, YScaleDataType, YDataType, UnquantYDataType, typename Traits_::Shape,
        ck_tile::Default2DAndDynamicQuantEpilogueTraits<false, Traits_::kPadN, UseSmoothInputScale, false,  true/*max3*/>>;
    using Default2DAndDynamicQuantEpilogue = ck_tile::Default2DAndDynamicQuantEpilogue<Default2DAndDynamicQuantEpilogueProblem>;

    using Epilogue = std::conditional_t<Traits_::kFusedQuant != 0,
                                        std::conditional_t<Traits_::kSaveUnquant,
                                                           Default2DAndDynamicQuantEpilogue,
                                                           DynamicQuantEpilogue>,
                                        Default2DEpilogue>;

    using Kernel = ck_tile::Rmsnorm2dFwd<Pipeline, Epilogue>;

    const dim3 grids                       = Kernel::GridSize(a);
    const dim3 blocks                      = Kernel::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = 1;

    auto kargs = Kernel::MakeKargs(a);
    if(s.log_level_ > 0)
        std::cout << ", " << Kernel::GetName() << std::flush;

    return ck_tile::launch_kernel(
        s, ck_tile::make_kernel<kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
}

