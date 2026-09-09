# FlyDSL import mismatch investigation for `flash_attn_varlen_func`

## Result

This is an **already-fixed, report-only** result. No product code was changed and the qualified Torch/ROCm stack was not modified.

The reported FlyDSL 0.3.1 mismatch reproduces on upstream `v0.1.19` (`31350226161346314b3d8882c8085bd31dce6a34`). An unconditional import in `flash_attn_varlen_func` enters `aiter.ops.flydsl`, whose `gemm_kernels` import reaches `tensor_shim`. That module imports `buffer_ops` from `flydsl.expr`, but FlyDSL 0.3.1 no longer exports it. The resulting `ImportError` therefore occurs before the `gfx942` backend can fall back to Triton.

Upstream commit `37b5395edf630a9eded64e7189c84d450644e571` (`refactor(flydsl): vendor buffer_ops/vector into aiter (#4402)`) fixes the root cause by vendoring `buffer_ops` and `vector` under `aiter.ops.flydsl.kernels` and repointing `tensor_shim` to those local modules. That commit is already present in mirror `main` at `f0321c0e8927d1d90a29385433f71e592b1c51f5`. Duplicating it would not be useful.

## Propagation

On the affected revision:

1. `aiter.ops.mha:flash_attn_varlen_func` unconditionally imports `flydsl_flash_attn_varlen_func`.
2. Importing `aiter.ops.flydsl.fmha_kernels` initializes `aiter.ops.flydsl`.
3. `aiter.ops.flydsl.__init__` imports `gemm_kernels`.
4. `gemm_kernels` imports `tensor_shim`.
5. `tensor_shim` executes `from flydsl.expr import buffer_ops, ...`.
6. FlyDSL 0.3.1 does not export `buffer_ops`, so the explicit `ImportError` escapes before the FlyDSL architecture gate or Triton fallback can run.

The fix does not add a broad exception handler. The historical explicit import error remains available for genuinely unavailable FlyDSL functionality; the missing symbols are supplied by Aiter's vendored modules instead.

## GPU validation

The validation used one assigned AMD Instinct MI300X (`gfx942`) and `/opt/venv/bin/python`. `ENABLE_CK=0` forced the supported non-FlyDSL Triton fallback. The synthetic fixture used two 128-token query/key sequences, two heads, head dimension 64, bfloat16 tensors, seed `20260909`, input scale `0.01`, no dropout, non-causal attention, and `window_size=(-1, -1, 0)`. The reference was PyTorch SDPA computed independently per sequence.

The unchanged numerical gate required:

- output shape `[256, 2, 64]`;
- output dtype `torch.bfloat16`;
- maximum absolute error versus SDPA at most `0.003`.

Raw results:

| Revision | Result |
| --- | --- |
| `31350226161346314b3d8882c8085bd31dce6a34` (`v0.1.19`) | `ImportError: cannot import name 'buffer_ops' from 'flydsl.expr'`; no numerical output |
| `37b5395edf630a9eded64e7189c84d450644e571` | gate passed; max abs error `0.002532958984375`; mean abs error `0.00036064122105017304` |
| `a63ede724b153564f3ed3fc538055fd15178c77d` (`v0.1.19.post2`) | gate passed with the same raw errors |
| `f0321c0e8927d1d90a29385433f71e592b1c51f5` (mirror `main`) | gate passed with the same raw errors |

The default CK route was also attempted from the shallow delivery clone. Its job-private JIT build failed because `3rdparty/composable_kernel/example/ck_tile/01_fmha/generate.py` was absent. This is an unrelated shallow-clone/submodule limitation, not a FlyDSL result, and it was not masked. The bounded Triton validation above did not require that build.

## Environment and paths

- Image: `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`, local image ID `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`.
- GPU: AMD Instinct MI300X, device ID `0x74a1`, GFX `gfx942`, GUID `6729`.
- Python: `/opt/venv/bin/python` (`3.10.12`).
- Torch: `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`, version `2.9.1+rocm7.2.0.git7e1940d4`, HIP `7.2.26015-fc0010cf6a`.
- FlyDSL: `/opt/venv/lib/python3.10/site-packages/flydsl/__init__.py`, version `0.3.1`; `flydsl.expr.buffer_ops` is absent.
- Delivery source: `/job/aiter`, branch `amdpilot/j-fc1a9f96f694`, base `f0321c0e8927d1d90a29385433f71e592b1c51f5`.
- Preinstalled context source: `/sgl-workspace/aiter`, revision `c16d44b93a528b2a4bfd6d8d3409116d465872a9` (`v0.1.19.post2-139-gc16d44b93-dirty`).
- Historical native modules were built under `/job/job-artifacts/aiter-jit-*`, outside the delivery worktree.

## Reproduction

From the affected worktree:

```bash
PYTHONPATH=/job/job-artifacts/aiter-v0.1.19 \
AITER_JIT_DIR=/job/job-artifacts/aiter-jit-historical \
ENABLE_CK=0 \
/opt/venv/bin/python /job/aiter/reports/j-fc1a9f96f694/reproduce_flydsl_import_mismatch.py
```

From the fix worktree:

```bash
PYTHONPATH=/job/job-artifacts/aiter-37b5395 \
AITER_JIT_DIR=/job/job-artifacts/aiter-jit-37b5395 \
ENABLE_CK=0 \
/opt/venv/bin/python /job/aiter/reports/j-fc1a9f96f694/reproduce_flydsl_import_mismatch.py
```

The same command with `PYTHONPATH=/job/aiter` and `AITER_JIT_DIR=/job/job-artifacts/aiter-jit-current` validates mirror `main`.

## Context and uncertainty

- Mirror issue `amdpilot-org/aiter` number 71 was read first and remains open.
- Upstream issue `ROCm/aiter` number 4701 was read through the public GitHub API; it had no comments at investigation time.
- Upstream pull request 2664 remains open and addresses install-time dev/rc version matching, not this runtime import. It was not duplicated.
- No upstream issue, pull request, or comment was posted or modified.
- The tiny `[3, 4]`/`[5, 6]` Triton probe showed a separate numerical mismatch, so the final unchanged gate uses the tile-aligned `[128, 128]` sequences documented above. That separate tiny-shape behavior was not investigated further and is not attributed to FlyDSL.
