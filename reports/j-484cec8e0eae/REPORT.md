# `mhc_pre` MI300X launch investigation

## Result

The reported `CUDA error: invalid configuration argument` did not reproduce in a
standalone `mhc_pre` launch on the assigned MI300X. The exact DeepSeek-V4
`profile_run` shape launches successfully on both current mirror `main` and the
upstream issue commit. The current-shape GPU result also matches a PyTorch eager
reference without relaxing the accuracy gates.

This is a validated negative result, not a claim that the original eight-worker
`vllm serve` failure is invalid. The standalone result rules out an invalid
configuration intrinsic to `mhc_pre` at the production shape on this qualified
Torch/ROCm stack. The original failure may depend on the reporter's ROCm 7.1.1 /
PyTorch 2.14 stack, eight-way tensor-parallel memory pressure, or an earlier
async kernel error surfaced by the allocation at `aiter/ops/mhc.py:350`.

## Environment

- Image: `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`
- Image ID: `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`
- OS: Ubuntu 22.04.5 LTS
- Python: `/opt/venv/bin/python` (CPython 3.10.12)
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- ROCm/HIP: `7.2.26015-fc0010cf6a`
- GPU: one AMD Instinct MI300X, gfx942, capability `(9, 4)`, 304 CUs

## Source and native paths

- Tested mirror checkout: `/job/aiter`, branch `main`, commit
  `f0321c0e8927d1d90a29385433f71e592b1c51f5`
- Current-main JIT cache: `/job/.cache/aiter-jit-current`
- Current-main native modules:
  `/job/.cache/aiter-jit-current/module_aiter_core.so` and
  `/job/.cache/aiter-jit-current/module_mhc.so`
- Historical issue worktree: `/job/aiter-issue-17f24ec`, commit
  `17f24ec6e93f48722a6b4ec8e54738434194f3d6`
- Historical JIT cache: `/job/.cache/aiter-jit-issue-17f24ec-src`
- Preinstalled context source: `/sgl-workspace/aiter`, detached commit
  `c16d44b93a528b2a4bfd6d8d3409116d465872a9`; it was not the tested source.

## Production shape derivation

The shape was derived from vLLM commit `4719a9b8f5cbfef78508369ba1a51abe2f5a79ce`,
not invented for a generic kernel test:

1. `profile_run()` calls `_dummy_run(self.max_num_tokens, skip_attn=True,
   is_profile=True)`.
2. `max_num_tokens` is `scheduler_config.max_num_batched_tokens`.
3. `vllm serve` uses the OpenAI API server usage context. On a device with at
   least 70 GiB and not A100, that context defaults to 8192 tokens.
4. DeepSeek-V4-Flash-0731 metadata gives `hidden_size=4096` and `hc_mult=4`.

Therefore the production inputs are:

- `residual`: `(8192, 4, 4096)`, `torch.bfloat16`
- `fn`: `(24, 16384)`, `torch.float32`
- `hc_scale`: `(3,)`, `torch.float32`
- `hc_base`: `(24,)`, `torch.float32`

Only the model's `config.json` metadata was fetched; no model weights were
downloaded and no full-model benchmark was run.

## Reference validation

`reproduce_mhc_pre.py` compares the current-main production result with a PyTorch
eager equivalent:

| Output | Shape | Tolerance | Max abs error | Mismatches |
|---|---:|---:|---:|---:|
| `post_mix` | `(8192, 4, 1)` fp32 | `rtol=atol=1e-3` | `1.239776611328125e-05` | 0 |
| `comb_mix` | `(8192, 4, 4)` fp32 | `rtol=atol=1e-3` | `6.407499313354492e-06` | 0 |
| `layer_input` | `(8192, 4096)` bf16 | `rtol=atol=1e-2` | `0.0625` | 0 |

The historical issue-commit launch also returned the expected output shapes and
dtypes, but it was not revalidated numerically because the purpose of that run
was only to test launch configuration.

## Boundary probe

| Case | Shape | Result |
|---|---:|---|
| Production | `8192x4x4096` | Validated against eager reference |
| Token off-by-one | `8191x4x4096`, `8193x4x4096` | Launched |
| Token tile boundary | `8128x4x4096`, `8256x4x4096` | Launched |
| Big-fuse non-temporal boundary | `2431x4x4096` through `2433x4x4096` | Launched |
| Big-fuse dispatch boundary | `3647x4x4096` through `3649x4x4096` | Launched |
| `hc_mult` off-by-one | `8192x3x4096`, `8192x5x4096` | Unsupported by explicit `hc_mult=4` guard |
| Hidden off-by-one | `8192x4x4095`, `8192x4x4097` | Unsupported by explicit hidden-size guard |

The unsupported cases are explicit limits, not numerical passes. The native
`mhc_pre_big_fuse` path supports `hc_mult=4` and hidden sizes
`{7168, 4096, 2560, 1280}`.

## Upstream context

ROCm/aiter issue 4633 was open with no comments when read through the public
GitHub API. It reports the allocation line rather than a kernel-launch traceback
and includes blocking-launch evidence from the original eight-GPU run.

No commit between issue commit `17f24ec6e` and current `main` explicitly fixes
this launch failure. The relevant MHC changes are the August 13 rocprim/hipcub
cleanup and the August 14 detorch/OPUS refactor; neither changes the gfx942
production launch geometry. Since both the historical and current exact-shape
launches succeed here, no candidate fix was written.

## Reproduction

```bash
cd /job/aiter
mkdir -p /job/.cache/aiter-jit-current
HIP_LAUNCH_BLOCKING=1 \
AITER_JIT_DIR=/job/.cache/aiter-jit-current \
/opt/venv/bin/python reports/j-484cec8e0eae/reproduce_mhc_pre.py \
  --json-output reports/j-484cec8e0eae/results.json
```

The script requires exactly one gfx942 device, uses deterministic CUDA inputs,
runs the production and boundary cases, and exits nonzero only for launch or
accuracy failures. Unsupported architecture guards are reported as unsupported.

## Limits and unfinished work

- Only one assigned MI300X was available; the original eight-way TP worker
  context was not reproduced.
- No full vLLM `profile_run`, model load, or eight-worker serve run was performed.
- No candidate fix is included because neither current `main` nor the issue commit
  failed at the exact production shape on this stack.
- The historical launch used the qualified Torch 2.9.1 / ROCm 7.2 stack, not the
  reporter's Torch 2.14 / ROCm 7.1.1 stack.
