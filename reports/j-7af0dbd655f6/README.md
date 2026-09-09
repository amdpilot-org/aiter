# Per-tensor MoE quantization with local token counts

## Result

- Baseline at mirror `main` commit `f0321c0e8927d1d90a29385433f71e592b1c51f5` reproduced `AssertionError: num_rows is not supported for per_tensor_quant_hip`.
- Upstream issue `ROCm/aiter#5256` had no candidate fix. The cross-referenced `ROCm/aiter#5255` only validates topk index dtypes and does not address this contract.
- This change threads `num_rows` and `num_rows_factor` through `static_per_tensor_quant` and `dynamic_per_tensor_quant`. Device kernels skip rows beyond `num_rows * num_rows_factor`, so padding cannot affect the dynamic scale or quantized output.
- The focused synthetic test passes on one AMD Instinct MI300X (`gfx942`): dynamic/static quantized values and scales are exact; the fused stage-2 quantized tensor and scale also matched the independent reference exactly. Final MoE mean relative error is `0.019416` and cosine error is `9.81388e-06`.

## Paths

- Python quantization wrapper: `aiter/ops/quant.py`
- Native declarations: `csrc/include/quant.h`
- Pybind declarations: `csrc/include/rocm_ops.hpp`
- HIP kernels: `csrc/kernels/quant_kernels.cu`
- Synthetic GPU regression: `op_tests/test_moe_2stage_per_tensor_num_local_tokens.py`
- Job-local native modules: `aiter/jit/module_aiter_core.so`, `aiter/jit/module_quant.so`, `aiter/jit/module_moe_ck2stages_f8_f8_preshuffle_on_b16_silu_per_tensor_mulWeightStage2.so`

## Environment

- Image: `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`, local image ID `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`
- GPU: one AMD Instinct MI300X, `gfx942`, serial `692440004395`
- Python: `/opt/venv/bin/python` (Python 3.10.12)
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`, HIP `7.2.26015-fc0010cf6a`

## Commands

```bash
git clone --depth=50 https://github.com/amdpilot-org/aiter.git /job/aiter
git switch -c amdpilot/j-7af0dbd655f6

AITER_JIT_DIR=/job/aiter/aiter/jit AITER_REBUILD=1 PYTHONPATH=/job/aiter \
  /opt/venv/bin/python -c 'import aiter; from aiter.ops.quant import per_tensor_quant_hip'

AITER_JIT_DIR=/job/aiter/aiter/jit PYTHONPATH=/job/aiter \
  /opt/venv/bin/python op_tests/test_moe_2stage_per_tensor_num_local_tokens.py

AITER_JIT_DIR=/job/aiter/aiter/jit PYTHONPATH=/job/aiter \
  /opt/venv/bin/python op_tests/test_moe_2stage.py \
  --no-flydsl-csv -q 1 -t 2 -dim 4096,384 -e 128 -k 8
```

## Unchanged gate

The existing `op_tests/test_moe_2stage.py` per-tensor case was run unchanged at the supported tuned shape. It exits successfully under its existing non-strict policy, with `logits_diff=8.81655e-06`. Its legacy `checkAllclose(atol=0.01, rtol=0.01)` diagnostic reports 15.2% mismatched elements and max absolute delta `512`; no existing gate or threshold was changed.

The tiny untuned shape `(token=5, model_dim=128, inter_dim=64, E=2, topk=2)` also has a pre-existing final-output mismatch on `main` (`logits_diff=0.9744878527187893`). It is unrelated to this fix and was not altered.
