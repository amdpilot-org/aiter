# BF16 residual rounding validation

## Environment

- Image: `amdpilotv2/miles-job:central-d957-20260909`
- GPU: AMD Instinct MI355X, gfx950, serial `692517002127`, GUID `42583`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- Torch source: `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`
- ROCm/HIP: `7.2.26015-fc0010cf6a`
- Compiler: `/opt/rocm-7.2.0/bin/hipcc`, AMD clang `22.0.0git`
- Source checkout: `/job/aiter`
- JIT cache and native modules: `/tmp/aiter-jit-cache-fixed15`
- Native Opus module: `/tmp/aiter-jit-cache-fixed15/module_rmsnorm.so`
- Common RMSNorm module: `/tmp/aiter-jit-cache-fixed15/module_rmsnorm_quant.so`

## Baseline

The investigation began at PR28 head
`f410a963874c26c3f6836b8e3d5e539d734dab16`. Hidden `16384` routes through
`add_rms_norm_opus`.

On this gfx950 GPU, ordinary normal BF16 ties already used hardware RNE, so a
random hidden-`16384` tensor produced `0/131072` mismatches against a CPU
double-precision reference. The source-level truncation default still affects
portable pre-gfx950 builds, but it did not reproduce as normal-value bias on
this assigned GPU.

The baseline did fail the explicit subnormal residual cases:

| Case | Baseline bits | Independent bits |
| --- | ---: | ---: |
| positive subnormal tie | `0x0000` | `0x0002` |
| negative subnormal tie | `0x8000` | `0x8002` |

The module is compiled with `-fgpu-flush-denormals-to-zero`, so changing only
the conversion rounding mode was insufficient; subnormal operands also needed a
double-precision fallback.

## Final bit-level results

The updated test compares returned BF16 residual bits against a CPU
double-precision RNE reference for random finite data and explicit independent
bit expectations for edge cases. At hidden `8192` and `16384`, all of the
following match:

- positive and negative normal ties that round up and down
- large finite overflow to infinity
- positive and negative subnormal ties
- mixed subnormal/normal values
- positive and negative cancellation to zero
- positive and negative infinity plus finite
- opposite infinities
- positive and negative NaN plus finite

The repo's existing common-path NaN convention is preserved: opposite
infinities produce sign-bearing quiet NaN `0xffc0`, while explicit NaN operands
canonicalize to `0x7fc0`. IEEE does not mandate a NaN payload; this PR pins the
existing Aiter convention so both dispatch paths agree.

## Tests

Commands run on the assigned GPU:

```bash
for hidden in 8192 12288 16384 20480 32768; do
    AITER_JIT_DIR=/tmp/aiter-jit-cache-fixed15 \
    python op_tests/test_rmsnorm2d.py --dtype bf16 --m 1 --n "$hidden"
done

AITER_JIT_DIR=/tmp/aiter-jit-cache-fixed15 \
python op_tests/test_rmsnorm2d.py --dtype fp16 --m 1 --n 16384

AITER_JIT_DIR=/tmp/aiter-jit-cache-fixed15 \
python op_tests/test_rmsnorm2d.py --dtype fp32 --m 1 --n 16384
```

All commands passed.

## Synchronized latency

CUDA-event timing used 20 warmups, 100 timed calls, three trials, `rows=128`,
and synchronized before and after each trial. Baseline and fixed modules were
run as separate processes from the same source checkout and seed.

| Hidden | Baseline mean | Fixed mean | Change |
| ---: | ---: | ---: | ---: |
| 8192 | 14.272217 us | 14.883320 us | +4.28% |
| 12288 | 15.668770 us | 15.610923 us | -0.37% |
| 16384 | 15.417834 us | 15.656257 us | +1.55% |
| 20480 | 15.476764 us | 15.499193 us | +0.14% |
| 32768 | 20.751247 us | 22.643380 us | +9.12% |

The first software-RNE implementation regressed hidden `16384` by about 49% and
hidden `32768` by about 110%. Hoisting the subnormal check to an eight-element
vector and using gfx950 hardware RNE for normal vectors reduced the final
tradeoff to the table above.

## Limitations

- Only the assigned MI355X/gfx950 was available; gfx942 software-RNE latency was
  not measured.
- The normal path uses hardware RNE only on gfx950/gfx1250/gfx1200/gfx1201 with
  clang 20 or newer. Other AMD GPUs use the software RNE fallback in this
  residual store.
- Quantized RMSNorm casts are intentionally unchanged; only the returned BF16
  residual store in the Opus norm kernel is corrected.
