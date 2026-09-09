# MoE `atol=100` calibration report

Job: `j-53652bc76ddd`

Date: 2026-09-09

Mirror issue: [amdpilot-org/aiter#55](https://github.com/amdpilot-org/aiter/issues/55)

Upstream issue: [ROCm/aiter#5031](https://github.com/ROCm/aiter/issues/5031)

## Executive summary

- The stock predicate `rtol=0.01, atol=100` accepted uniform `+99` and `+100`
  corruptions in all eight supported real-output comparisons.
- The same predicate accepted six real reference/AITER comparisons outright and
  reported mismatches in two others.  Those two test processes still exited zero
  because `checkAllclose` logs and returns a ratio instead of raising.
- A per-row experimental criterion accepted every real output in the retained
  run, while rejecting `+99`/`+100` in three of eight cases.  It was looser than
  stock for some high-magnitude int8 rows, so this is **not** a qualified global
  tightening.
- Recommendation: keep the defaults unchanged, land the opt-in
  `op_tests/calibrate_moe_tolerance.py` harness, and gather more shapes before
  considering any default change.

## Source and issue context

- Tested mirror revision: `f0321c0e8927d1d90a29385433f71e592b1c51f5` (`main`).
- Upstream #5031 remains open.  Its current description and 2026-08-27 comment
  report that a one-rounding derived-tolerance prototype flips int8 from pass
  to fail on MI300X and explicitly should not land.
- The current source still has six active `atol=100` call sites:
  `op_tests/test_moe.py:199`, `:314`, `:317` and
  `op_tests/test_moe_tkw1.py:194`, `:299`, `:302`.
- No derived-tolerance helper or replacement for those sites exists in the tested
  revision.  This work therefore does not duplicate an already-landed fix.
- Upstream #5030 (make `checkAllclose` capable of failing) and #5032 (per-row
  tuner scoring) are related.  Both are relevant before a default criterion is
  changed.

## Environment

- Qualified image (operator-provided identity):
  `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`,
  local image ID `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`.
- GPU: one AMD Instinct MI300X, `gfx942`, capability `(9, 4)`.
- Python: `/opt/venv/bin/python` (3.10.12).
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`; HIP runtime
  `7.2.26015-fc0010cf6a`.
- Tested source checkout: `/job/aiter`.
- Preinstalled native modules/HSACOs loaded from `/sgl-workspace/aiter`, for
  example `/sgl-workspace/aiter/aiter/jit/module_moe_asm.so` and
  `/sgl-workspace/aiter/hsa//gfx942/fmoe/silu/`.
- Job-private native cache: `TORCH_EXTENSIONS_DIR=/tmp/aiter-jit-cache`;
  tuned MoE config cache: `/tmp/aiter_configs`.
- The image warned that NUMA balancing was enabled.  No system setting was
  changed.

## Method

The calibration used one bounded shape:

- tokens: 32
- model dimension: 2048
- intermediate dimension: 1792
- experts: 32
- top-k: 5 (or 1 for `test_moe_tkw1.py`)
- output dtype: bf16

Supported comparisons:

1. `test_moe.py:g1u1_int8quant:a8w8`
2. `test_moe.py:g1u1_fp8quant:a8w8`
3. `test_moe.py:g1u1_int8smoothquant:a8w8`
4. `test_moe.py:g1u1_int8smoothquant:a16w8`
5. `test_moe.py:g1u1_fp8smoothquant:a8w8`
6. `test_moe.py:g1u1_wint4afp8smoothquant:a8w4`
7. `test_moe.py:g1u0_int8smoothquant:a8w8`
8. `test_moe_tkw1.py:g1u1_fp8quant:a8w8`

The harness generates seeded inputs and captures real Torch references and
AITER outputs.  It then applies labeled corruptions to each reference:
uniform offsets of 1, 10, 50, 99, 100, 101, and 500; a 1% multiplicative
scale; masked zeroing of 1%, 10%, and 50%; and masked 1% sign flips.

“Stock accepted” below means the elementwise `torch.isclose` predicate accepted
every element.  It does **not** mean the current test can fail: `checkAllclose`
returns zero only for an all-close result and returns the mismatch ratio
otherwise, while the existing MoE scripts ignore that return value.

The experimental criterion is:

```text
abs(reference - candidate)
  <= max(1, safety_factor * bf16_ulp(max_abs(reference_row)))
     + 0.01 * abs(candidate)
```

It uses the bf16 output ULP and a per-row reference magnitude.  The safety
factors are empirical powers of two from this bounded run:

- int8: 32
- fp8: 16
- int4: 32

This is deliberately not the rejected upstream one-rounding formula and does not
claim a reduction-depth law.  Upstream #5031 showed that pure `sqrt(K)`/`K`
scaling is not justified by the observed int8 error.

## Real output results

The retained artifact is one real run.  A same-seed repeat showed native
run-to-run variation, so the harness should not be described as bit-reproducible.

| Case | Stock all-close | Stock mismatches | Max error | Max abs reference | Required ULP safety | Experimental accepted |
|---|---:|---:|---:|---:|---:|---:|
| g1u1 int8 a8w8 | yes | 0 | 68 | 1352 | 14 | yes |
| g1u1 fp8 a8w8 | yes | 0 | 60.125 | 1328 | 14 | yes |
| g1u1 int8 smooth a8w8 | yes | 0 | 92 | 1512 | 23 | yes |
| g1u1 int8 smooth a16w8 | yes | 0 | 80 | 1512 | 16 | yes |
| g1u1 fp8 smooth a8w8 | yes | 0 | 72 | 1536 | 14 | yes |
| g1u1 int4/fp8 a8w4 | yes | 0 | 55 | 788 | 17 | yes |
| g1u0 int8 smooth a8w8 | no | 1 | 112 | 3024 | 13 | yes |
| tkw1 fp8 a8w8 | no | 65 | 160 | 2560 | 15.5 | yes |

Observed same-seed max-error ranges across three runs included:

- fp8 plain: 58.125–60.125
- int8 smooth a8w8: 92–96
- int8 smooth a16w8: 72–80
- fp8 smooth: 72–78
- int4/fp8: 53–55
- g1u0 int8 smooth: 112–136, with 1–4 stock mismatches

The top-k 1 fp8 case consistently had 65 stock mismatches and max error 160 in
the runs observed.  Its existing test log says `warning!` but exits zero.

## Corruption verdict matrix

Each row counts how many of the eight real-output cases accepted the corruption.

| Corruption | Stock accepted | Experimental accepted |
|---|---:|---:|
| `add_1_all` | 8/8 | 8/8 |
| `add_10_all` | 8/8 | 8/8 |
| `add_50_all` | 8/8 | 8/8 |
| `add_99_all` | 8/8 | 5/8 |
| `add_100_all` | 8/8 | 5/8 |
| `add_101_all` | 0/8 | 5/8 |
| `add_500_all` | 0/8 | 0/8 |
| `scale_1pct_all` | 8/8 | 8/8 |
| `zero_1pct` | 0/8 | 0/8 |
| `zero_10pct` | 0/8 | 0/8 |
| `zero_50pct` | 0/8 | 0/8 |
| `sign_flip_1pct` | 0/8 | 0/8 |

The experimental criterion rejected `+99`/`+100` in the lower-row-scale fp8 and
int4 cases but accepted those offsets in higher-row-scale int8 cases.  It also
accepted `+101` in five cases where stock rejected it.  This confirms that a
row-scaled criterion changes the error geometry; it does not uniformly tighten
the gate.

## Unsupported and skipped paths

- `test_moe.py:test_fmoe_16_bit` (g1u0 bf16) aborted with HIP error 700,
  `an illegal memory access was encountered`, while loading/running the bf16
  MoE path.  This matches the limitation reported in upstream #5031.  It is not
  treated as a numerical pass.  Log: `logs/stock_bf16_g1u0.log`.
- `g1u1_no_quant` is not a real kernel comparison in the current test: it sets
  `out_b = ref2` and only prints that g1u1 supports quant/smoothquant.
- The optional a16 comparison is exercised for int8 smooth quant at this shape.
  The fp8 smooth a16 branch is skipped by the existing divisibility condition at
  1792.
- No accuracy gate was relaxed to make any case pass.

## Recommendation

1. Do not replace the six default `atol=100` call sites from this one-shape
   sample.
2. Keep `op_tests/calibrate_moe_tolerance.py` as an opt-in experiment.  It
   records real tensors, corruption verdicts, and the row-scaled criterion
   without changing existing test behavior.
3. Resolve the “logging-only” behavior from upstream #5030 before arming any
   new default gate.  A wrong verdict that cannot redden CI is still invisible
   to the build.
4. Extend calibration across more magnitudes, token counts, expert counts,
   activations, and real-model distributions.  A per-row or p99-keyed criterion
   is more coherent than one scalar, as upstream #5032 argues for the tuner.
5. Treat the bf16 g1u0 illegal-memory-access failure as a separate kernel bug.

## Artifacts

- JSON summary: `data/moe_tolerance_calibration.json`
  (`sha256:08bba67203392651ba16f4fda52bd2b51977e499f9197053c771ec95997d21eb`)
- Real reference/AITER tensors: `data/moe_tolerance_calibration.npz`
  (`sha256:f9bbc431745198433ed38f818765511733e74fe1196b930e97afcf5cdc70dd50`)
- Calibration log: `logs/calibration_harness.log`
- Stock-test logs: `logs/stock_*.log`

## Reproduction

```bash
cd /job/aiter
export TORCH_EXTENSIONS_DIR=/tmp/aiter-jit-cache
export AITER_REBUILD=0
/opt/venv/bin/python op_tests/calibrate_moe_tolerance.py \
  --output reports/j-53652bc76ddd/data/moe_tolerance_calibration.json \
  --tensor-output reports/j-53652bc76ddd/data/moe_tolerance_calibration.npz
```

Representative stock-test invocations are recorded in `logs/stock_*.log`.  The
calibration script refuses non-`gfx942` devices rather than reporting an
unsupported architecture as a numerical pass.
