# BF16 accumulation investigation: RMSNorm and unquantized fused MoE

## Executive summary

- **RMSNorm:** production variance accumulation is already FP32. The measured
  differences against an independent FP32 reference come from the final
  FP32-to-BF16 store, which uses truncation on the tested gfx942 path. Comparing
  against a truncating FP32-to-BF16 reference leaves only 49–169,607 nonexact
  elements out of 16,777,216 depending on the case, and produces **0/4096**
  downstream argmax flips. No RMSNorm accumulation fix is warranted by this
  evidence.
- **Unquantized fused MoE:** both CK GEMMs use FP32 K-dimension accumulation.
  The remaining production route accumulation is BF16: stage 2 atomically adds
  route contributions into a BF16 output. On a near-uniform adversarial case,
  this changes **2/512** downstream argmax decisions relative to an FP32 route
  accumulator.
- **Draft change:** this branch adds an opt-in
  `fused_moe(..., fp32_route_accumulation=True)` mode for the supported
  unquantized BF16 Silu G1U1 CK two-stage path. It allocates an FP32 route
  accumulator, generates/loads an FP32 CK stage-2 instance, and casts once to
  the caller's BF16 output dtype. It is not enabled by default.
- **Measured effect:** the patched output has **0/512** argmax disagreements
  with the FP32-route emulator, while default output has **2/512**. The patch
  still has **2/512** disagreements with a fully independent FP32 MoE reference
  because the BF16 stage-1 intermediate and final BF16 rounding remain.
- **Performance:** on the small 512-token case, default averaged 0.1724 ms and
  the FP32-route option averaged 0.1711 ms over 20 iterations. This is not a
  measurable regression on this input, but the route accumulator doubles output
  scratch bytes (BF16 to FP32) and the first use JIT-compiles a new CK module.

## Environment and revisions

- Image: `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`
  (`sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`)
- GPU: one AMD Instinct MI300X, gfx942, Torch capability `(9, 4)`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`
- Tested source: `/job/aiter`, base `f0321c0e8927d1d90a29385433f71e592b1c51f5`
- Preinstalled context snapshot (not the tested revision):
  `/sgl-workspace/aiter` at `c16d44b93a528b2a4bfd6d8d3409116d465872a9`
- Private JIT cache: `/job/aiter-jit`
- Native modules used:
  - `/job/aiter-jit/module_rmsnorm_quant.so`
  - `/job/aiter-jit/module_moe_sorting_opus.so`
  - `/job/aiter-jit/module_moe_ck2stages_b16_b16_preshuffle_on_b16_silu_no_mulWeightStage2.so`
  - `/job/aiter-jit/module_moe_ck2stages_b16_b16_preshuffle_on_f32_silu_no_mulWeightStage2.so`

The first FP32 CK stage-2 build took approximately 111 seconds. Subsequent runs
loaded the cached module.

## Issue review

Upstream ROCm/aiter issue 3806 has no comments and no linked AITER fix. The
linked vLLM PR 45509 is a CI workaround/model-routing change; it does not
change AITER accumulation. At the tested base revision there was no
`fp32_route_accumulation` option, so this branch does not duplicate an existing
HEAD fix.

## Source findings

### RMSNorm

- The common 2-D BF16/FP16, hidden-size <= 8192 path dispatches to
  `module_rmsnorm_quant` (`aiter/ops/rmsnorm.py:404`).
- `csrc/kernels/rmsnorm_quant_kernels.cu:141` uses `float square_sum`, FP32
  multiply-adds, and `block_reduce<float, ...>` at line 149.
- The OPus fallback also uses `float acc` at
  `csrc/include/opus/rmsnorm_opus_kernel.hpp:155`.
- The tested BF16 store uses truncation:
  `OPUS_FP32_to_BF16_DEFAULT` is `2` (truncate) at
  `csrc/include/opus/opus.hpp:40`.

### Unquantized fused MoE

- The tested shape dispatches to CK two-stage
  `module_moe_ck2stages_b16_b16_preshuffle_on_b16_silu_no_mulWeightStage2`.
- CK instance generation sets `AccDataType` to `F32` for BF16 inputs
  (`csrc/ck_gemm_moe_2stages_codegen/gen_instances.py:928` and line 959).
  Thus the K-dimension GEMM accumulation is FP32 in both stages.
- Stage 1's intermediate is BF16 (`EDataType=B16`), and default stage 2 writes
  and atomically accumulates route contributions into a BF16 `EDataType`.
  The generated CK instance therefore performs route accumulation in BF16.
- `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages.cu:200` selects the CK
  instance from the actual output dtype. With the new option, this loads the
  `..._f32...` module and performs route accumulation in FP32.

## Reproduction

```bash
cd /job/aiter
export PYTHONPATH=/job/aiter \
       AITER_JIT_DIR=/job/aiter-jit \
       GPU_ARCHS=gfx942 \
       HIP_VISIBLE_DEVICES=0 \
       AITER_LOG_MORE=0

/opt/venv/bin/python reports/j-153e366e6337/investigate.py \
  > reports/j-153e366e6337/results.json
```

The machine-readable results are retained in
`reports/j-153e366e6337/results.json`. The harness uses seed 0, disables TF32,
and constructs:

- RMSNorm: `[4096, 4096]` random, large-magnitude `1e17`, alternating
  cancellation at `1e17`, and dynamic range `1e17` versus `1e-20`.
- MoE: 512 tokens, hidden 1024, intermediate 2048, 8 experts, top-2, BF16
  inputs/weights, Silu, `QuantType.No`, and a near-uniform router.

## RMSNorm measurements

The downstream projection uses a 32,000-column FP32 random language head and
row normalization to create near-uniform logits.

| Case | Max abs vs FP32 | Mean abs vs FP32 | Argmax vs FP32 | Argmax vs truncating BF16 |
|---|---:|---:|---:|---:|
| random BF16 | 6.2448e-2 | 1.7627e-3 | 15/4096 | 0/4096 |
| large `1e17` | 6.2445e-2 | 1.7625e-3 | 15/4096 | 0/4096 |
| alternating cancellation `1e17` | 6.2498e-2 | 1.7633e-3 | 14/4096 | 0/4096 |
| dynamic range | 3.7405e-37 | 1.1209e-38 | 0/4096 | 0/4096 |

The truncating reference is not exact for every element because the independent
Torch FP32 reduction and the kernel's FP32 block reduction can use different
orders. It nevertheless isolates the dominant effect: the kernel agrees with a
truncated final BF16 result closely enough that no downstream argmax flip
appears in these cases. Comparing against RNE-rounded BF16 is the wrong reference
for this kernel and produces 24–28 flips.

## Fused MoE measurements

The independent reference performs all GEMMs, Silu, route weighting, and route
accumulation in FP32. The two-stage emulator preserves the production BF16
stage-1 intermediate and then selects either BF16 or FP32 route accumulation.

| Comparison | Max abs | Mean abs | Exact elements | Argmax flips |
|---|---:|---:|---:|---:|
| default vs independent FP32 | 7.7315e-5 | 7.9988e-6 | 5/524288 | 2/512 |
| default vs BF16-route emulator | 6.1035e-5 | 4.4884e-8 | 522104/524288 | 0/512 |
| default vs FP32-route emulator | 1.2207e-4 | 4.7122e-6 | 327824/524288 | 2/512 |
| patched vs default | 1.2207e-4 | 4.7093e-6 | 327951/524288 | 2/512 |
| patched vs FP32-route emulator | 6.1035e-5 | 4.3813e-8 | 519581/524288 | 0/512 |
| patched vs independent FP32 | 4.9580e-5 | 6.4734e-6 | 6/524288 | 2/512 |

These measurements isolate BF16 route accumulation as a real argmax-changing
effect. They also show what the option does **not** fix: the BF16 stage-1
intermediate and final BF16 rounding still cause disagreements with a fully
independent FP32 reference.

## Change and tradeoffs

`fp32_route_accumulation=True`:

- allocates the route accumulator as FP32;
- keeps stage-1 metadata and the stage-1 intermediate BF16;
- generates/loads an FP32 CK stage-2 instance;
- casts the FP32 route result once to the requested BF16 output;
- supports caller-provided output buffers;
- raises `NotImplementedError` for quantized paths, EP, bias, stage-1 route
  weighting, scatter, one-stage configs, and non-CK stage-2 functions.

Numerical tradeoff:

- removes the isolated BF16 route-accumulation effect;
- does not remove BF16 stage-1 intermediate rounding or final BF16 rounding;
- can change 2/512 near-uniform argmax decisions relative to default.

Resource tradeoff:

- route output scratch grows from 2 to 4 bytes per output element;
- adds one final FP32-to-BF16 cast;
- first use JIT-compiles a new CK module;
- measured small-input time was 0.1711 ms versus 0.1724 ms default, so no
  regression was measurable in this 20-iteration run.

## Validation

- `PYTHONPATH=/job/aiter AITER_JIT_DIR=/job/aiter-jit GPU_ARCHS=gfx942`
  harness run completed on MI300X and produced
  `reports/j-153e366e6337/results.json`.
- `/opt/venv/bin/python -m py_compile` passed for both changed Python modules,
  the CK generator, and the investigation harness.
- `git diff --check` passed.
- A caller-provided BF16 output buffer returned the same tensor and matched a
  second patched call exactly.
- A quantized call with `fp32_route_accumulation=True` raised the intended
  `NotImplementedError`.
- No accuracy gate was relaxed.

## Limits and honest negative results

- This is a bounded kernel-level numerical investigation, not a full-model
  benchmark or an end-to-end quality claim.
- Empirical results are for gfx942/MI300X only. The source findings are not
  gfx-specific, but no gfx950/MI355X device was available in this assignment.
- RMSNorm accumulation was not changed: source and measurements show FP32
  accumulation already exists. Changing final BF16 rounding semantics would be a
  separate compatibility decision.
- The FP32 route option does not make patched MoE equal an independent FP32
  reference; the remaining 2/512 argmax disagreements are reported as failures
  of that stronger equality goal.
- Unsupported fused-MoE configurations intentionally fail loudly rather than
  silently falling back. This is a limitation of the draft fix, not a passing
  numerical check for those paths.
