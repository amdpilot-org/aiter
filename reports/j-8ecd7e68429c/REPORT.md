# gfx942 FP8 fused_moe boundary investigation

## Result

The reported crash did not reproduce on the tested mirror commit. All five reported
`(M, N, K)` cases and all ten neighboring controls completed in separate bounded
processes with exit status 0, finite outputs, and no GPU memory-permission fault.

The current dispatcher pads `M=288..352` to tuning token `512`, misses an exact tuned
row, and then selects the default one-stage path rather than the issue's fault-prone
two-stage fallback:

- metadata: `run_1stage=True`, `block_m=32`, `ksplit=0`, `stage1=fused_moe_1stage`
- loaded kernel: `_ZN5aiter50fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_ps_32x256E`
- HSACO: `/job/aiter/hsa/gfx942/fmoe/silu/fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_ps_32x256.co`

This is an already-fixed/negative result for the crash boundary on this commit. No
product-code fix was made. The report-only harness and raw logs are retained instead.

## Environment

- Campaign: `repo-e2e-20260909`
- Job: `j-8ecd7e68429c`
- Working clone: `/job/aiter`
- Base commit: `f0321c0e8927d1d90a29385433f71e592b1c51f5` (`main`)
- GPU: one AMD Instinct MI300X, `gfx942`, 304 CUs, GUID `39656`, node ID `3`
- Driver: `6.19.14.31400000`; VRAM total `206141652992` bytes
- Qualified image (operator-specified): `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`
- Local image ID (operator-specified): `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`
- Python: `/opt/venv/bin/python` (Python 3.10.12)
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`, `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`
- Aiter Python source: `/job/aiter/aiter/__init__.py`
- Native modules used: `/job/aiter/aiter/jit/module_aiter_core.so`, `module_moe_sorting_opus.so`, `module_quant.so`, and `module_moe_fmoe_asm.so`
- Preinstalled context (not used for the sweep): `/sgl-workspace/aiter`, upstream merge commit `c16d44b93a528b2a4bfd6d8d3409116d465872a9`

The sweep explicitly set `PYTHONPATH=/job/aiter`. Without that, script-style imports can
resolve the preinstalled `/sgl-workspace/aiter` package instead of the delivery clone.

## Upstream context

- ROCm/aiter issue 4783 was read through the public GitHub API. It was open, had no
  comments, and its timeline cross-referenced issue 4788.
- ROCm/aiter issue 4788 was also read. It concerns silent garbage from GLM decode
  tuned rows, not the 304/336 crash boundary.
- Related ROCm/aiter PR 4811 (head `cff8d43085ab4683e05f54f13d2a9fac0651cc7c`,
  merge `c16d44b93a528b2a4bfd6d8d3409116d465872a9`) retunes only token buckets
  32, 64, 128, and 256 for `model_dim=6144`, `inter_dim=256`. It does not address
  the reported prefill boundary and was not duplicated.
- No upstream issue, PR, review, or comment was created or modified.

## Synthetic input

The harness uses bounded per-rank synthetic tensors, not model weights:

- `E=32` local experts, `topk=8`
- BF16 hidden state, FP8 `e4m3fnuz` weights and activation
- `QuantType.per_128x128`, remapped by `fused_moe` to `per_1x128`
- 1x128 activation blocks and 128x128 weight blocks
- separated G1U1 weights, SiLU, `doweight_stage1=False`
- deterministic seed `12345`
- FP32 route weights and int32 expert IDs

The independent reference dequantizes the FP8 activation and both FP8 weight tensors,
then evaluates the equivalent routed dense G1U1/SiLU/down-projection MoE in Torch FP32.
It does not use AITER's fused MoE reference implementation. No numerical gate was
changed: raw default `checkAllclose(rtol=0.01, atol=0.01)` results and the existing
non-strict `logits_diff <= 0.01` threshold are both recorded.

## Raw results

`checkAllclose` is elementwise and failed in every case, as it also does in the
repository's existing FP8 MoE test. The existing non-strict logit-difference threshold
passed 11/15 cases. It exceeded `0.01` for all four `(N=6144, K=2048)` cases, including
neighboring controls, so that numerical drift is not specific to `M=304` or `M=336`.

| M | N | K | Exit | Dispatch | Loaded kernel | logits_diff | Existing threshold | Cosine | Relative L2 | Max abs | Mean abs |
|---:|---:|---:|---:|---|---|---:|---|---:|---:|---:|---:|
| 288 | 3072 | 1536 | 0 | 1stage, block_m=32, ksplit=0 | `..._vs_silu_1tg_ps_32x256` | 0.007450 | pass | 0.992775 | 0.123383 | 0.256605 | 0.006645 |
| 288 | 3072 | 6144 | 0 | 1stage, block_m=32, ksplit=0 | `..._vs_silu_1tg_ps_32x256` | 0.008356 | pass | 0.991925 | 0.130843 | 1.342111 | 0.029724 |
| 288 | 6144 | 2048 | 0 | 1stage, block_m=32, ksplit=0 | `..._vs_silu_1tg_ps_32x256` | 0.013962 | exceed | 0.986845 | 0.170585 | 0.702585 | 0.018045 |
| 304 | 3072 | 1536 | 0 | 1stage, block_m=32, ksplit=0 | `..._vs_silu_1tg_ps_32x256` | 0.005633 | pass | 0.994491 | 0.106992 | 0.264417 | 0.005605 |
| 304 | 3072 | 6144 | 0 | 1stage, block_m=32, ksplit=0 | `..._vs_silu_1tg_ps_32x256` | 0.007365 | pass | 0.992854 | 0.122658 | 1.373361 | 0.026246 |
| 304 | 6144 | 2048 | 0 | 1stage, block_m=32, ksplit=0 | `..._vs_silu_1tg_ps_32x256` | 0.024876 | exceed | 0.977710 | 0.231625 | 0.604901 | 0.025754 |
| 320 | 3072 | 1536 | 0 | 1stage, block_m=32, ksplit=0 | `..._vs_silu_1tg_ps_32x256` | 0.005619 | pass | 0.994504 | 0.106855 | 0.256605 | 0.005659 |
| 320 | 3072 | 6144 | 0 | 1stage, block_m=32, ksplit=0 | `..._vs_silu_1tg_ps_32x256` | 0.005893 | pass | 0.994242 | 0.109463 | 1.388986 | 0.024086 |
| 320 | 6144 | 2048 | 0 | 1stage, block_m=32, ksplit=0 | `..._vs_silu_1tg_ps_32x256` | 0.026578 | exceed | 0.976400 | 0.240117 | 0.687206 | 0.026122 |
| 336 | 3072 | 1536 | 0 | 1stage, block_m=32, ksplit=0 | `..._vs_silu_1tg_ps_32x256` | 0.003836 | pass | 0.996216 | 0.088036 | 0.163360 | 0.005065 |
| 336 | 3072 | 6144 | 0 | 1stage, block_m=32, ksplit=0 | `..._vs_silu_1tg_ps_32x256` | 0.006425 | pass | 0.993737 | 0.114393 | 1.404611 | 0.025291 |
| 336 | 6144 | 2048 | 0 | 1stage, block_m=32, ksplit=0 | `..._vs_silu_1tg_ps_32x256` | 0.024762 | exceed | 0.977827 | 0.231103 | 0.677918 | 0.027194 |
| 352 | 3072 | 1536 | 0 | 1stage, block_m=32, ksplit=0 | `..._vs_silu_1tg_ps_32x256` | 0.006435 | pass | 0.993726 | 0.114481 | 0.278689 | 0.005763 |
| 352 | 3072 | 6144 | 0 | 1stage, block_m=32, ksplit=0 | `..._vs_silu_1tg_ps_32x256` | 0.007793 | pass | 0.992453 | 0.126254 | 1.729555 | 0.028392 |
| 352 | 6144 | 2048 | 0 | 1stage, block_m=32, ksplit=0 | `..._vs_silu_1tg_ps_32x256` | 0.019756 | exceed | 0.981892 | 0.204792 | 0.705566 | 0.023818 |

The complete JSON result line, dispatch metadata, loaded HSACO path, and process log
for each case are retained under `logs/`. The existing repository test was also run for
`M=304, N=3072, K=1536`; it selected the same one-stage kernel and reported
`logits_diff=0.0033379059581262194`.

## Reproduction

Single case:

```bash
cd /job/aiter
PYTHONPATH=/job/aiter AITER_LOG_LEVEL=INFO CUDA_VISIBLE_DEVICES=0 \
  /opt/venv/bin/python reports/j-8ecd7e68429c/boundary_harness.py \
  --M 304 --N 3072 --K 1536 --E 32 --topk 8 --seed 12345
```

Bounded sweep (one `timeout 420` process per case):

```bash
cd /job/aiter
for M in 288 304 320 336 352; do
  for NK in '3072 1536' '3072 6144' '6144 2048'; do
    set -- $NK
    timeout 420 env PYTHONPATH=/job/aiter AITER_LOG_LEVEL=INFO CUDA_VISIBLE_DEVICES=0 \
      /opt/venv/bin/python reports/j-8ecd7e68429c/boundary_harness.py \
      --M "$M" --N "$1" --K "$2" --E 32 --topk 8 --seed 12345 \
      >"reports/j-8ecd7e68429c/logs/M${M}_N${1}_K${2}.log" 2>&1
  done
done
```

Existing-test cross-check:

```bash
cd /job/aiter
PYTHONPATH=/job/aiter AITER_LOG_LEVEL=INFO CUDA_VISIBLE_DEVICES=0 \
  /opt/venv/bin/python op_tests/test_moe_2stage.py \
  -q 5 -t 304 -dim 1536,3072 -e 32 -k 8 --no-flydsl-csv
```

## Interpretation and limitations

- The crash reported in ROCm/aiter issue 4783 is negative/already-fixed on mirror
  commit `f0321c0e8927d1d90a29385433f71e592b1c51f5`: the dispatcher no longer selects
  the reported two-stage fallback for these shapes.
- No minimal boundary fix is warranted from this evidence.
- The `N=6144, K=2048` logit-difference exceedance appears in controls as well as the
  reported M values, so it is not an M-boundary regression. It was not investigated
  further because changing numerical behavior or gates was out of scope.
- The synthetic topology uses `E=32/topk=8`, matching a bounded per-rank shape, but it
  is not a full GLM or vLLM serving reproduction.
- The image ID is the operator-provided local image identity; the container cannot
  independently resolve a Docker image digest from inside itself.
