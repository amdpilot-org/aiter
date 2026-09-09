# AITER issue 3852 evidence report

## Conclusion

No A8W8 BLOCKSCALE config rows are missing on reviewed `main`
`f0321c0e8927d1d90a29385433f71e592b1c51f5`.

The real `get_CKGEMM_config` lookup does not fall through for the three
`M=32768` prefill tuples. It tries the exact key, then `get_padded_m(..., gl=0)`,
then `get_padded_m(..., gl=1)`. On this MI300X, `gl=1` maps each `M=32768`
request to `M=8192`, and the DSv3 family table already contains the matching
`gfx942,cu_num=304` rows. This is the intentional reachable-row scheme described
by tuning commit `84493f0e5b083b02c9ff3d60961f58bae547045a`.

No tuner sweep was run because the condition for adding rows was not met. No
GEMM family table or runtime source was changed. The three existing `M=2048`
decode rows remain byte-for-byte unchanged.

## Selected rows

| Request `(M,N,K)` | Lookup `(M,N,K)` | `kernelId` | `splitK` | Kernel |
|---|---:|---:|---:|---|
| `(32768,4608,7168)` | `(8192,4608,7168)` | 0 | 0 | `a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3` |
| `(32768,7168,2304)` | `(8192,7168,2304)` | 0 | 0 | `a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3` |
| `(32768,7168,256)` | `(8192,7168,256)` | 16 | 0 | `a8w8_blockscale_1x128x128_256x64x64x128_16x16_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v1` |

The exact runtime lookup log is in `runtime_lookup.log`.

## Hardware and revisions

- Device: AMD Instinct MI300X, `gfx942`, 304 CUs, one visible device.
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`.
- HIP: `7.2.26015-fc0010cf6a`.
- Checkout: `f0321c0e8927d1d90a29385433f71e592b1c51f5`.
- Preinstalled distribution: `amd-aiter 0.1.19.post3.dev139+gc16d44b93`.
- Validation imports the checkout at `/job/aiter/aiter/__init__.py`, not the
  preinstalled distribution.

## Reproduction

From the checkout root:

```bash
AITER_LOG_TUNED_CONFIG=1 \
AITER_CONFIG_GEMM_A8W8_BLOCKSCALE=$PWD/aiter/configs/model_configs/a8w8_blockscale_tuned_gemm_ds_v3.csv \
/opt/venv/bin/python - <<'PY'
from aiter.jit.core import AITER_CONFIGS
from aiter.ops.gemm_op_a8w8 import get_CKGEMM_config

path = AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_FILE
for shape in [
    (2048, 4608, 7168),
    (2048, 7168, 2304),
    (2048, 7168, 256),
    (32768, 4608, 7168),
    (32768, 7168, 2304),
    (32768, 7168, 256),
]:
    print(shape, get_CKGEMM_config(*shape, path))
PY
```

The six-shape correctness and timing reproduction is:

```bash
AITER_CONFIG_GEMM_A8W8_BLOCKSCALE=$PWD/aiter/configs/model_configs/a8w8_blockscale_tuned_gemm_ds_v3.csv \
/opt/venv/bin/python reports/j-f58ff018ba03/validate_issue_3852.py \
  --output reports/j-f58ff018ba03/validation.json
```

## Validation results

The selected row and the no-row default path were both compared with an
independent FP32 dequantized `torch.matmul` reference. Inputs used AITER's
ROCm-native `dtypes.fp8` (`torch.float8_e4m3fnuz`). Across two seeds, all six
shapes and both paths produced finite outputs and zero mismatches at
`rtol=1e-2, atol=1e-2`.

Median synchronized kernel timings over four 20-iteration samples:

| Shape | Selected | No-row default | Default / selected |
|---|---:|---:|---:|
| `(2048,4608,7168)` | 167.976 us | 519.533 us | 3.093x |
| `(2048,7168,2304)` | 85.844 us | 269.923 us | 3.144x |
| `(2048,7168,256)` | 26.362 us | 56.197 us | 2.132x |
| `(32768,4608,7168)` | 2661.269 us | 7944.111 us | 2.985x |
| `(32768,7168,2304)` | 1252.321 us | 4066.351 us | 3.247x |
| `(32768,7168,256)` | 363.192 us | 808.876 us | 2.227x |

Raw timings and numeric summaries are in `validation.json`; the console log is
`validation.log`.

An in-memory equivalent of the existing config merge collision check found
39,130 rows across the canonical table and all matching model tables, with
zero duplicate `(gfx,cu_num,M,N,K)` keys. The full unittest could not write its
lock under the container-owned root directory `/tmp/aiter_configs`; this is an
environment permission limitation, not a config result. The equivalent check is
recorded in `config_collision.log`.

## Unverified and out of scope

- No B-preshuffle, A4W4, BF16, or other GEMM family was tuned or changed.
- No end-to-end vLLM/model benchmark was run; performance claims here are
  kernel-level only.
- The full `test_config_shape_collision.py` suite was not completed because of
  the `/tmp/aiter_configs` permission blocker described above.
