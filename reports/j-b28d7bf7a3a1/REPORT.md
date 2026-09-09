# FMHA v3 FP16 admission and dispatch on gfx942

## Result

`fmha_v3_fwd` admitted FP16 at the C++ dtype check but had no FP16 rows in the gfx942 forward CSV, so dispatch failed with the opaque error `invalid argument for fmha_fwd`. The smallest truthful correction is to reject FP16 at admission. BF16 and FP8 remain admitted and dispatch real gfx942 kernels.

## Environment and provenance

- Source: `/job/aiter`, base revision `f0321c0e8927d1d90a29385433f71e592b1c51f5`
- Native JIT build: `/job/.cache/aiter-jit/build/module_fmha_v3_fwd`
- Image: `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`, image ID `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`
- GPU: one AMD Instinct MI300X, gfx942, CUDA capability `(9, 4)`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`; HIP runtime `7.2.26015-fc0010cf6a`

## Investigation

Upstream ROCm/aiter issue #3762 reports the same inconsistency. The issue had no comments when read through the GitHub API. Current mirror source still admitted FP16 in `csrc/py_itfs_cu/asm_mha_fwd.cu`, while `hsa/gfx942/fmha_v3_fwd/fmha_fwd.csv` contains BF16 and FP8 rows but no FP16 rows. The Python-level `can_impl_fmha_v3_fwd` gate already excludes FP16, confirming that the low-level API should not claim FP16 support.

## Baseline behavior

All baseline GPU cases used batch 2, sequence length 2, and no dropout. The sequence length does not affect dtype admission or CSV lookup. The baseline logs are retained at `/job/fmha_v3_baseline.log` and `/job/fmha_v3_fixed.log`; the final validation log is `/job/fmha_v3_final.log`.

| Case | Admission | Dispatch | Numerical result |
|---|---|---|---|
| BF16 `128x128`, non-causal | admitted | `_ZN5aiter24fmha_fwd_hd128_bf16_rtneE` | max abs error `0.0075268745`, mean `0.0009681422`, max LSE error `3.58e-7` |
| BF16 `128x128`, causal | admitted | `_ZN5aiter31fmha_fwd_hd128_bf16_causal_rtneE` | max abs error `0.0070221424`, mean `0.0005068232`, max LSE error `1.49e-7` |
| BF16 `192x128`, non-causal GQA | admitted | `_ZN5aiter28fmha_fwd_hd192x128_bf16_rtneE` | max abs error `0.0069243908`, mean `0.0008779658`, max LSE error `4.77e-7` |
| BF16 `192x128`, causal GQA | admitted | `_ZN5aiter35fmha_fwd_hd192x128_bf16_causal_rtneE` | max abs error `0.0139422417`, mean `0.0005216831`, max LSE error `2.53e-7` |
| BF16 `64x64` | admitted by dtype/head checks | no kernel | unsupported; `invalid argument for fmha_fwd` |
| FP16 `128x128` | admitted by dtype check | no kernel | failed dispatch; `invalid argument for fmha_fwd` |
| FP16 `192x128` | admitted by dtype check | no kernel | failed dispatch; `invalid argument for fmha_fwd` |

The reference is explicit FP32 `einsum` attention with softmax and logsumexp. BF16 gates are max absolute output error `<= 0.02` and max LSE error `<= 1e-4`. These gates were not relaxed.

## Corrected behavior

The C++ admission check now accepts only BF16 and FP8 E4M3. FP16 fails immediately with `fmha_v3_fwd only supports bf16 and fp8_e4m3 data type`. BF16 `64x64` remains unsupported and still fails at kernel lookup; it is not counted as a pass.

| Case | Admission | Dispatch | Numerical result |
|---|---|---|---|
| BF16 `128x128`, non-causal | admitted | `fwd_hd128_bf16_rtne.co` | max abs error `0.0017413497`, mean `0.0001657863`, max LSE error `9.54e-7`; pass |
| BF16 `128x128`, causal | admitted | `fwd_hd128_bf16_causal_rtne.co` | max abs error `0.0077030659`, mean `0.0002724839`, max LSE error `9.54e-7`; pass |
| BF16 `192x128`, non-causal GQA | admitted | `fwd_hd192x128_bf16_rtne.co` | max abs error `0.0016360283`, mean `0.0001663328`, max LSE error `9.54e-7`; pass |
| BF16 `192x128`, causal GQA | admitted | `fwd_hd192x128_bf16_causal_rtne.co` | max abs error `0.0075218678`, mean `0.0002723562`, max LSE error `9.54e-7`; pass |
| BF16 `64x64` | rejected at capability lookup | none | unsupported, as expected |
| FP16 `128x128` | rejected at admission | none | unsupported, as expected |
| FP16 `192x128` | rejected at admission | none | unsupported, as expected |
| FP8 E4M3FNuz `128x128` | admitted | `fwd_hd128_fp8.co` | max abs error `0.0123902857`, mean `0.0017915224`, max LSE error `9.54e-7`; pass at `0.06` |

The FP8 check uses unit descales and the existing repository FP8 tolerance of `0.06` (the repository's FP8 test uses `0.055`; this focused check uses a slightly rounder gate). The FP8 case is a regression check for the corrected admission expression, not part of the requested BF16/FP16 matrix.

## Reproduction

```bash
AITER_JIT_DIR=/job/.cache/aiter-jit \
  /opt/venv/bin/python reports/j-b28d7bf7a3a1/validate.py \
  --aiter-root /job/aiter --jit-dir /job/.cache/aiter-jit
```

The script exits nonzero if a supported case misses its numerical gate or an unsupported case is unexpectedly admitted. AITER's `LoadKernel` lines identify the real dispatched `.co` kernels.

## Limits

- This is a focused direct `fmha_v3_fwd` investigation, not a full-model or end-to-end benchmark.
- FP16 is intentionally left unsupported by this low-level API. No FP16 numerical output is claimed.
- The high-level `flash_attn_func` Python dispatch already routes FP16 away from FMHA v3; that fallback path was not expanded or benchmarked here.
