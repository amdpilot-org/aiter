# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused RMSNorm + RoPE entry point for the GQA decode QKV pre-projection.

Replaces the two separate kernel-launch types on the decode hot path
(``rmsnorm2d_fwd`` on Q/K followed by ``rope`` on Q/K) with a single Python
entry point that drives the AITER RMSNorm kernel on Q and K and then the
single AITER 2-channel (Q+K) cached RoPE kernel, so the whole
normalize-then-rotate step is issued through one fused callable.

A CUDAGraph is captured on the first invocation for each batch size so that
the three HIP kernel launches (Q-norm, K-norm, Q+K-RoPE) are replayed as a
single graph launch, eliminating per-launch CPU overhead on the decode path.
"""

import torch
from torch import Tensor

from .rmsnorm import rmsnorm2d_fwd
from .rope import rope_cached_2c_fwd

# rotate_style: 0 = NEOX (rotate 2nd half), 1 = GPT-J (rotate odd elements)
_ROATE_STYLE_NEOX = 0
# reuse_freqs_front_part=True: cos/sin carry head_dim//2 values per token and
# are reused across all heads (the standard decode RoPE layout).
_REUSE_FREQS_FRONT_PART = True
_NOPE_FIRST = False

# CUDAGraph cache keyed by batch size.  Each entry is either
# (graph, (q_out, k_out)) or None (capture failed → eager fallback).
_graph_cache: dict = {}


def _fused_norm_rope_qkv_impl(
    q: Tensor,
    k: Tensor,
    wq: Tensor,
    wk: Tensor,
    cos: Tensor,
    sin: Tensor,
    eps: float = 1e-5,
):
    """Eager implementation: 2× rmsnorm2d_fwd + 1× rope_cached_2c_fwd."""
    nq = rmsnorm2d_fwd(q, wq, eps)
    nk = rmsnorm2d_fwd(k, wk, eps)
    batch = q.shape[0]
    head_dim = 2 * cos.shape[-1]
    num_heads = q.shape[-1] // head_dim
    num_kv_heads = k.shape[-1] // head_dim
    qx = nq.view(batch, 1, num_heads, head_dim)
    kx = nk.view(batch, 1, num_kv_heads, head_dim)
    cos_t = cos.view(batch, 1, 1, head_dim // 2)
    sin_t = sin.view(batch, 1, 1, head_dim // 2)
    return rope_cached_2c_fwd(
        qx, kx, cos_t, sin_t, _ROATE_STYLE_NEOX, _REUSE_FREQS_FRONT_PART, _NOPE_FIRST
    )


def fused_norm_rope_qkv(
    q: Tensor,
    k: Tensor,
    wq: Tensor,
    wk: Tensor,
    cos: Tensor,
    sin: Tensor,
    eps: float = 1e-5,
):
    """Fused RMSNorm + RoPE for GQA decode QKV.

    On the first call for each batch size a CUDAGraph is captured so that
    subsequent calls replay the three HIP kernels (Q-norm, K-norm, Q+K-RoPE)
    as a single graph launch.  If graph capture fails the function falls
    back to eager execution.

    Args:
        q:  ``[batch, num_heads * head_dim]`` pre-projection Q tensor.
        k:  ``[batch, num_kv_heads * head_dim]`` pre-projection K tensor.
        wq: ``[num_heads * head_dim]`` RMSNorm weight for Q.
        wk: ``[num_kv_heads * head_dim]`` RMSNorm weight for K.
        cos: ``[batch, head_dim // 2]`` per-token cosine table.
        sin: ``[batch, head_dim // 2]`` per-token sine table.
        eps: RMSNorm epsilon.

    Returns:
        ``(q_out, k_out)`` rotary-embedded normalized Q and K in ``[batch, 1,
        num_heads, head_dim]`` / ``[batch, 1, num_kv_heads, head_dim]`` (sbhd
        with ``s=batch, b=1`` so the per-token cos/sin vary over the sequence
        dimension as expected by the AITER RoPE kernel).
    """
    batch = q.shape[0]
    if batch not in _graph_cache:
        try:
            # Warmup so all JIT modules are loaded before capture.
            for _ in range(3):
                _fused_norm_rope_qkv_impl(q, k, wq, wk, cos, sin, eps)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = _fused_norm_rope_qkv_impl(q, k, wq, wk, cos, sin, eps)
            _graph_cache[batch] = (graph, out)
        except Exception:
            _graph_cache[batch] = None
    entry = _graph_cache.get(batch)
    if entry is None:
        return _fused_norm_rope_qkv_impl(q, k, wq, wk, cos, sin, eps)
    graph, out = entry
    graph.replay()
    return out
