# `mhc_post` noncontiguous layout validation

## Scope and result

Validated the public `aiter.mhc_post` API on one AMD Instinct MI300X (gfx942).
Before the change, contiguous controls were correct, while noncontiguous
`post_layer_mix`, `comb_res_mix`, and output views were accepted but silently
corrupted. The output view also overwrote sentinel storage outside the view.

The public wrapper now dispatches any noncontiguous `post_layer_mix`,
`comb_res_mix`, or output to the existing stride-aware Triton `mhc_post`
implementation. The contiguous HIP path and its numerical behavior are
unchanged. After the change, every tested layout matches an independent Torch
reference within the existing `atol=rtol=1e-2` gate, has a bad-element ratio of
zero, contains no NaN, and preserves all sentinel elements outside output views.

Reference issue ROCm/aiter 5292 was read through the unauthenticated GitHub API.
It had no comments or timeline events at the time. No upstream issue, PR, or
comment was modified.

A bounded read-only search for `mhc_post`, `mhc_post stride`, and
`mhc_post noncontiguous` found no merged fix for this defect. Open ROCm/aiter
PR 4432 concerns residual HC-slice indexing and explicitly leaves `mhc_post`
output offsets packed, so it is not a candidate fix for mix or output strides.

## Environment

- Campaign: `repo-e2e-20260909`
- Coordination tracker: amdpilot-org/amdpilotv2 issue 402
- Image: `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`
- Local image ID: `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`
- GPU: one AMD Instinct MI300X, gfx942, serial `692440004306`, GUID `9845`
- Python: `/opt/venv/bin/python` (Python 3.10.12)
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- Working clone: `/job/j-8dc7afb2b11c/aiter`
- Mirror base commit: `f0321c0e8927d1d90a29385433f71e592b1c51f5`
- Public API: `aiter/ops/mhc.py`
- Stride-aware Triton implementation: `aiter/ops/triton/fusions/mhc.py`
- Native kernel: `csrc/kernels/mhc_kernels.cu`
- Job-private built module: `/job/j-8dc7afb2b11c/aiter/aiter/jit/module_mhc.so`
- Installed source context: `/sgl-workspace/aiter` at
  `c16d44b93a528a2a4bfd6d8d3409116d465872a9`
- Installed native module context:
  `/sgl-workspace/aiter/aiter/jit/module_mhc.so`

## Synthetic case

`M=37`, `n=4`, `C=768`, `torch.bfloat16`, seed `1234`. Inputs were finite and
nonzero. The independent reference was:

```python
combined = torch.bmm(comb_mix.mT, residual.float())
expected = x.float().unsqueeze(-2) * post_mix.unsqueeze(-1) + combined
```

The bad-element ratio used the unchanged numerical gate:
`relative_error > atol + rtol * expected`, with `atol=rtol=1e-2`.

## Raw pre-fix results

The native public API returns `None` and mutates `out`. All layouts were accepted.

| Case | Post strides | Comb strides | Output strides | Max abs | Max rel | Bad ratio | NaN | Sentinels |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Contiguous control | `(4,1)` | `(16,4,1)` | `(3072,768,1)` | `0.0625` | `0.0077464217` | `0` | no | n/a |
| Noncontiguous post | `(8,2)` | `(16,4,1)` | `(3072,768,1)` | `6.5451126099` | `4875.5151367` | `0.8706362844` | no | n/a |
| Noncontiguous comb | `(4,1)` | `(32,8,2)` | `(3072,768,1)` | `15.3219642639` | `5562.7475586` | `0.9847093225` | no | n/a |
| Noncontiguous output | `(4,1)` | `(16,4,1)` | `(6144,1536,1)` | `12363.37890625` | `12352000` | `0.9887739419` | no | failed |
| All noncontiguous | `(8,2)` | `(32,8,2)` | `(6144,1536,1)` | `12363.58496094` | `12352000` | `0.9944573641` | no | failed |

## Raw post-fix results

All layouts were accepted. The noncontiguous output view had nontrivial strides
in all three dimensions: `(12288,3072,2)`.

| Case | Post strides | Comb strides | Output strides | Max abs | Max rel | Bad ratio | NaN | Sentinels |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Contiguous control | `(4,1)` | `(16,4,1)` | `(3072,768,1)` | `0.0625` | `0.0077464217` | `0` | no | n/a |
| Noncontiguous post | `(8,2)` | `(16,4,1)` | `(3072,768,1)` | `0.0311994553` | `0.0038893817` | `0` | no | passed |
| Noncontiguous comb | `(4,1)` | `(32,8,2)` | `(3072,768,1)` | `0.0308876038` | `0.0038901044` | `0` | no | passed |
| Noncontiguous output | `(4,1)` | `(16,4,1)` | `(12288,3072,2)` | `0.0307245255` | `0.0038893947` | `0` | no | passed |
| All noncontiguous | `(8,2)` | `(32,8,2)` | `(12288,3072,2)` | `0.0307455063` | `0.0038886664` | `0` | no | passed |

## Commands

```bash
git clone --filter=blob:none https://github.com/amdpilot-org/aiter.git aiter
gh issue view 5292 --repo ROCm/aiter --json number,title,state,body,comments,createdAt,updatedAt,url
curl -L --fail --silent --show-error --retry 3 --retry-delay 5 \
  -H 'Accept: application/vnd.github+json' \
  https://api.github.com/repos/ROCm/aiter/issues/5292
curl -L --fail --silent --show-error --retry 3 --retry-delay 5 \
  -H 'Accept: application/vnd.github+json' \
  https://api.github.com/repos/ROCm/aiter/issues/5292/comments
curl -L --fail --silent --show-error --retry 3 --retry-delay 5 \
  -H 'Accept: application/vnd.github+json' \
  https://api.github.com/repos/ROCm/aiter/issues/5292/timeline
curl -L --silent --show-error --retry 3 --retry-delay 5 \
  --get --data-urlencode 'q=repo:ROCm/aiter mhc_post' \
  --data-urlencode 'per_page=20' \
  https://api.github.com/search/issues
curl -L --silent --show-error --retry 3 --retry-delay 5 \
  -H 'Accept: application/vnd.github+json' \
  https://api.github.com/repos/ROCm/aiter/pulls/4432
rocm-smi --showproduct --showserial
PYTHONPATH=/job/j-8dc7afb2b11c/aiter \
AITER_CACHE_DIR=/job/j-8dc7afb2b11c/cache \
/opt/venv/bin/python /job/j-8dc7afb2b11c/validate_mhc_post.py
PYTHONPATH=/job/j-8dc7afb2b11c/aiter \
AITER_CACHE_DIR=/job/j-8dc7afb2b11c/cache \
/opt/venv/bin/python -m pytest -q op_tests/test_mhc_post_layouts.py
git diff --check
```

The authenticated `gh issue view` command failed because the current fine-grained
GitHub token is forbidden by the `ROCm` organization. The unauthenticated API
requests succeeded and were read-only.

## Limits

- No full model weights or alternate framework stacks were downloaded.
- Build artifacts and Triton/HIP caches stayed under `/job/j-8dc7afb2b11c/cache`
  and the working clone.
- The fix does not alter the native HIP kernel or its contiguous performance.
- `store_nt` remains a native-path performance hint and is not applicable to the
  Triton fallback.
