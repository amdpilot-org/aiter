# gfx942/cu304 FP8 MoE coverage evidence

## Runtime

- Image Python: `/opt/venv/bin/python`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- `torch.version.hip`: `7.2.26015-fc0010cf6a`
- Device: `AMD Instinct MI300X`, `gfx942:sramecc+:xnack-`, 304 CUs, one visible GPU
- Checkout base: `f0321c0e8927d1d90a29385433f71e592b1c51f5`

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

The default loaded `_ZN5aiter50fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_ps_32x256E` with both serving weights marked `is_shuffled=True`. The existing test's relative similarity error was `8.13832e-05`; its elementwise `allclose` check used an unsuitable absolute tolerance for this large-output workload. Independent dequantized-reference validation and bounded candidate measurements are in progress and will be recorded below.
