# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared state for norm+rope CUDAGraph fusion.

When ``rmsnorm2d_fwd`` is called twice followed by ``rope_2c_fwd`` (the
standard GQA decode pattern), the three HIP kernel launches can be captured
into a single CUDAGraph and replayed as one graph launch, eliminating
per-launch CPU overhead on the decode hot path.

Flow:
  iter 1 (capture): rmsnorm2d_fwd launches eagerly & records pending args;
                     rope_2c_fwd captures a CUDAGraph of all 3 kernels.
  iter 2+ (replay): rmsnorm2d_fwd skips eager launch (returns fused buffer);
                     rope_2c_fwd replays the cached graph (1 launch).
"""

# Pending rmsnorm calls: list of (input, weight, eps, out_buffer)
_pending_norms = []

# Input data_ptrs handled by a fused graph (skip eager kernel launch)
_fused_norm_ptrs = set()

# Maps input data_ptr -> reused output buffer (the one the graph captured)
_fused_norm_outs = {}

# Graph cache: gkey -> (graph, out_x, out_y) or None (capture failed)
_fused_graph_cache = {}
