# gfx942 ASM MLA decode investigation

## Result

The original issue #4350 failure did **not** reproduce on the assigned MI300X
(gfx942). With unpatched main, 20 cold-process split-16 trials (10 eager and 10
CUDA-graph with cache churn) produced no NaNs and passed the same strict output
and LSE gates used after the fix.

A separate, confirmed defect did reproduce on gfx942: `get_meta_param` cached a
device tensor as its only owner. A CUDA graph captured a cache hit as a bare
pointer, LRU eviction freed the tensor, allocator reuse scribbled over it, and
replay read a corrupted split-offset table. Upstream PR #5194's focused test
failed on unpatched main at batch 480 and passed after applying its fix. This
matches the earlier upstream analysis in PR #4587, but uses the smaller #5194
implementation.

The fix caches only the pure integer split-count heuristic and rebuilds
`num_kv_splits_indptr` on every call. During graph capture, `torch.arange` is
recorded as a fill kernel, so replay reinitializes the buffer rather than relying
on allocator lifetime.

## Environment

- Qualified image: `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`
- Qualified image ID: `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`
- Tested mirror base: `amdpilot-org/aiter` `main` at `f0321c0e8927d1d90a29385433f71e592b1c51f5`
- Preinstalled image source: `/sgl-workspace/aiter` at `c16d44b93a528b2a4bfd6d8d3409116d465872a9`
- Python: `/opt/venv/bin/python` (Python 3.10.12)
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- HIP: `7.2.26015-fc0010cf6a`
- ROCm: `/opt/rocm`; compiler: `/opt/rocm/bin/hipcc`
- GPU: AMD Instinct MI300X, capability `(9, 4)`, `gfx942`
- Tested JIT module: `/job/aiter/aiter/jit/module_mla_asm.so`
- Tested HSA object: `/job/aiter/hsa/gfx942/mla/mla_dec_stage1_bf16_a16w16_subQ16_mqa16.co`
- AITER JIT build cache: `/job/aiter/aiter/jit/build/module_mla_asm`
- Baseline worktree JIT cache: `/job/aiter-baseline/aiter/jit/build/module_mla_asm`
- Triton cache: `/job/.triton/cache`

## Test coverage

The reproducer uses finite deterministic BF16 tensors with seed 0:

- Batch 1, 16 heads, 576 QK dimensions, 512 latent/value dimensions
- 2,048 selected KV positions from a 65,536-token pool
- Page size 1, one query token, causal decode
- Explicit split counts: 8, 12, 16, 24, and 32
- Fresh process for every trial
- Eager execution and CUDA-graph capture/replay with 160 cache-churn calls
- Independent PyTorch reference for both output and LSE
- ASM verification by wrapping `mla_decode_stage1_asm_fwd`
- Unchanged gates: output max absolute error <= 5e-4, LSE max absolute error <= 1e-4, no NaNs

Patched results:

- Eager: 50/50 passed; 10 trials per split count
- CUDA graph: 50/50 passed; 10 trials per split count
- Total: 100/100 passed
- Worst output error: `1.9183754920959473e-4`
- Worst LSE error: `9.5367431640625e-7`
- Output NaNs: 0
- LSE NaNs: 0
- ASM stage calls: 1 per eager run, 2 per graph run (warmup plus capture)

Unpatched split-16 issue-shape control:

- Eager: 10/10 passed
- CUDA graph with cache churn: 10/10 passed
- Total: 20/20 passed
- No output or LSE NaNs

Upstream candidate regression:

- Unpatched `main`: fails at batch 480; replayed indptr begins
  `[4608, 4620, 4632, ...]` instead of `[0, 12, 24, ...]`
- Patched branch: passes batch 8 and batch 480
- Focused pytest: 1 passed
- Python compilation checks: passed; Ruff was not installed in the image

## Reproduction

From the repository root:

```bash
ROCM_PATH=/opt/rocm GPU_ARCHS=gfx942 /opt/venv/bin/python \
  reports/j-943d535054d6/reproduce_mla_nan.py \
  --mode graph --splits 16 --trial 0 --cache-churn

ROCM_PATH=/opt/rocm GPU_ARCHS=gfx942 /opt/venv/bin/python -m pytest -q \
  op_tests/test_mla_split_indptr_cudagraph.py
``+

Set `AITER_SOURCE_DIR` to an alternate checkout to test that tree while reusing
this reproducer.

## Limitations

- The original #4350 report is on gfx950; no gfx950 device was available here.
- The B=1 issue shape did not reproduce on gfx942, so this work does not prove
  that the captured-indptr lifetime bug is the exact root cause of #4350.
- The confirmed gfx942 defect is nevertheless real and can turn finite inputs
  into non-finite graph-replay results after allocator reuse.
- No full-model or performance benchmark was run, as requested.
