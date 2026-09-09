# gfx942/cu304 FP8 MoE coverage evidence

## Runtime

- Image Python: `/opt/venv/bin/python`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- `torch.version.hip`: `7.2.26015-fc0010cf6a`
- Device: `AMD Instinct MI300X`, `gfx942:sramecc+:xnack-`, 304 CUs, one visible GPU
- Driver: `6.19.14.31400000`; host: `Linux banff-cyxtera-cx57-4 5.18.2-mi300-build-140423-ubuntu-22.04+`
- Image userspace: Ubuntu 22.04.5; this is the actual local job image, not a claim of byte identity with another image
- Preinstalled source revision: `c16d44b93a528b2a4bfd6d8d3409116d465872a9`
- Checkout base: `f0321c0e8927d1d90a29385433f71e592b1c51f5`
- Validation explicitly used `PYTHONPATH=/job/aiter`, so it imported the modified checkout rather than `/sgl-workspace/aiter`.

## Reproduction

Command:

```bash
PYTHONPATH=/job/aiter AITER_MOE_EXPERT_BALANCE=true AITER_ONLINE_TUNE=0 \
/opt/venv/bin/python op_tests/test_moe_2stage.py \
  -t 2048 -dim 7168,256 -e 256 -k 8 -q 5 -a silu -s f -p t -hip 0,0 \
  --no-flydsl-csv
```

Observed dispatch:

```text
[aiter] [fused_moe] using 1stage default for ('gfx942', 304, 2048, 7168, 256, 256, 8, <ActivationType.Silu: 0>, 'torch.bfloat16', 'torch.float8_e4m3fnuz', 'torch.float8_e4m3fnuz', 'QuantType.per_1x128', True, False)
```

The default loaded `_ZN5aiter50fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_ps_32x256E` with both serving weights marked `is_shuffled=True`. The existing test's relative similarity error was `8.13832e-05`; its elementwise `allclose` check used an unsuitable absolute tolerance for this large-output workload.

## Bounded one-stage measurements

The scan was limited to four already shipped gfx942 SiLU FP8 block-scale one-stage kernels. Each candidate ran in a fresh process with seeds 0 and 1, serving `shuffle_weight(..., layout=(16, 16))` tags on both weights, and an independently dequantized torch MoE reference. Timings are warm, synchronized CUDA-event medians in microseconds.

| Token | Kernel | Seed 0 / seed 1 median | Relative-L2 range | Finite |
|---:|---|---:|---:|---|
| 2048 | `vs_silu_1tg_ps_32x256` (default) | 1048.66 / 1052.02 | 0.007396–0.007409 | yes |
| 2048 | `vs_silu_1tg_32x256` | 1058.20 / 1047.83 | 0.007397–0.007406 | yes |
| 2048 | `vs_silu_1tg_ps_32x128` | 1142.11 / 1153.02 | 0.008145–0.008159 | yes |
| 2048 | `vs_ps_silu_64x256` | 1115.11 / 1102.56 | 0.007393–0.007408 | yes |
| 32768 | `vs_silu_1tg_ps_32x256` (default) | 8583.42 / 8549.12 | 0.007455–0.007480 | yes |
| 32768 | `vs_silu_1tg_32x256` | 6762.96 / 6671.98 | 0.007455–0.007479 | yes |
| 32768 | `vs_silu_1tg_ps_32x128` | 9006.22 / 9017.48 | 0.008199–0.008222 | yes |
| 32768 | `vs_ps_silu_64x256` | 5534.48 / 5558.81 | 0.007455–0.007480 | yes |

The M2048 default and non-`ps` 32x256 kernel tie within noise, so the change keeps the measured default. At M32768, 64x256 is about 35% faster than the default with the same reference error, so the change adopts it only for that exact tuple.

## Fixed rows and final validation

`aiter/configs/tuned_fmoe.csv` adds exactly two `cu_num=304` rows:

- M2048: block 32, `_ZN5aiter50fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_ps_32x256E`
- M32768: block 64, `_ZN5aiter46fmoe_bf16_blockscaleFp8_g1u1_vs_ps_silu_64x256E`

Final checkout-CSV runtime lookup selected those exact kernels and blocks. Fresh-process operator validation with the emitted CSV produced finite outputs and relative-L2 of 0.007394–0.007406 at M2048 and 0.007455–0.007480 at M32768.

The existing exact-tuple regression command was:

```bash
PYTHONPATH=/job/aiter AITER_MOE_EXPERT_BALANCE=true AITER_ONLINE_TUNE=0 \
/opt/venv/bin/python op_tests/test_moe_2stage.py \
  -t 2048 -dim 7168,256 -e 256 -k 8 -q 5 -a silu -s f -p t -hip 0,0 \
  --no-flydsl-csv
```

The same command with `-t 32768` selected the new 64x256 row. Their similarity errors were `8.14037e-05` and `8.14058e-05`; the test's fixed `atol=0.01, rtol=0.01` elementwise check still reports failure because outputs are large, which is why relative-L2 and finiteness were checked independently.

## Scope and limits

- No other FMoE rows or default heuristics were changed.
- The known gfx942 CK two-stage OOB boundary was not exercised or changed.
- These are operator-only measurements; no end-to-end model or TP8 speedup is claimed.
- No model weights or distributed run were used.
