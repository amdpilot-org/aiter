# BF16 FMHA extreme-negative-logit investigation

## Scope

This investigation covers mirror issue amdpilot-org/aiter#43, upstream ROCm/aiter
issue 5086, and the existing upstream candidate ROCm/aiter PR 5087 (commit
`6b05791ec140d44e4bd98ea7f34ff43aed98193f`). The upstream issue had no user
comments when checked. PR 5087 had only the repository CI-guide bot comment. The
related FlashInfer PR 4401 fix identified the same finite-score clipping class of
error.

All meaningful GPU results below were collected on one AMD Instinct MI355X
(`gfx950`, node 3) with PyTorch `2.9.1+rocm7.2.0.git7e1940d4` and HIP
`7.2.26015-fc0010cf6a`, using `/opt/venv/bin/python`. The tested checkout was
imported from `/job/aiter`; JIT artifacts were kept in `/job/aiter-jit-cache`.

## Findings

### Current main

Current mirror `main` is `f0321c0e8927d1d90a29385433f71e592b1c51f5`.

- BF16 D128 OPUS passed all 24 cases at raw logits from `-2^99` through `-2^102`
  (seeds 0–2, causal and noncausal). It then returned all-zero output and
  non-finite LSE for all 12 cases from `-2^103` through `-2^127`.
- BF16 D192/V128 OPUS failed all 24 cases at raw logits from `-2^99` through
  `-2^102`. Focused threshold checks found that raw `-2^36` passed, NaN began at
  raw `-2^37`, and output became zero below approximately `-2^100`.
- BF16 D128 v3 passed the same `-2^99` through `-2^102` matrix. BF16 D192/V128
  v3 failed all 24 cases.
- The initial CK matrix was not a meaningful kernel result on this host: BF16 D64
  could not build because the CK submodule was absent, and FP16 D128 at exponents
  99–102 overflowed while constructing inputs. FP16 cannot represent those
  BF16-scale logits.

### Upstream candidate

PR 5087 replaces the `-1e30f` softmax maximum seeds with the lowest finite FP32
value in native D64, OPUS D128, and OPUS D192/V128. It was verified before making
a duplicate patch and cherry-picked as local commit `df682a9`.

On MI355X, the candidate fixed the D128 seed-clipping behavior: D128 OPUS passed
all 24 cases at `-2^99` through `-2^102`, and the added regression covers
`-2^127`. It did not fix D192/V128: all 24 candidate-matrix cases failed with
NaN, including the candidate's own focused D192 test at `-2^101`.

### D192/V128 root cause

OPUS D192/V128 deferred softmax scaling until after the QK GEMM. It computed the
row maximum from huge raw FP32 scores, then fused `score * scale - row_max` before
`exp2`. Tiny MFMA accumulation differences at huge raw magnitudes become large
after scaling and can overflow or underflow `exp2`.

The fix pre-scales Q by `softmax_scale * log2(e)` before the GEMM, matching the
existing D128 OPUS design, and then uses plain row-max subtraction on the already
scaled scores. The upstream lowest-finite seed changes are retained. This is a
required D192 runtime correction, not a duplicate of PR 5087.

## Regression coverage

`op_tests/test_mha_extreme_logits.py` uses the actual public entrypoints
`aiter.flash_attn_func` and `aiter.flash_attn_varlen_func`, with direct-wrapper
identity checks to prove routing. Values are deliberately nonzero random tensors.
The reference computes stable float32 softmax from the actual input tensors and
checks that all probabilities are unambiguously equal.

Coverage includes:

- OPUS D128 and D192/V128 BF16 dense GQA `(B=2, S=256, Hq=8, Hkv=2)`.
- Raw exponents 36, 101, and 127; seeds 0, 1, and 2; causal and noncausal.
- OPUS D192/V128 varlen groups `(64, 192)` with `Hq=4`, `Hkv=1`, causal and
  noncausal, at exponents 36 and 127.
- Native D64 through the public entrypoint with split-KV identity, skipped on
  MI355X because that native path is gfx942-only.
- Unchanged tolerance `rtol=1e-2, atol=1e-2` for random V; exact zero tolerance
  only where constant V makes the expected output exactly one.

## GPU validation

- New regression: `40 passed, 1 skipped`.
- Existing dense OPUS suite (`op_tests/test_mha.py -k opus`): `64 passed`.
- Existing varlen OPUS suite (`op_tests/test_mha_varlen.py -k opus`):
  `32 passed`.
- Neighboring CK fallbacks through `aiter.flash_attn_func`: `36 passed, 0 failed`.
  - BF16 D64 CK: exponents 30, 100, and 120; seeds 0–2; causal and noncausal.
  - FP16 D128 CK: representable exponents 8, 12, and 15; seeds 0–2; causal and
    noncausal.
  - All public outputs matched the direct `mha_fwd` wrapper exactly, were finite
    and nonzero, and had finite LSE. Maximum absolute difference was
    `0.00390625`, within the unchanged tolerance.
- `git diff --check`: passed.

The exact reported v3 cases were also preserved. On this MI355X build, D128 v3 at
`-1.914088313930279e38` produced NaN output and infinite LSE. D192/V128 v3 passed
at `-9.50737950171172e29` and `-3.802951800684688e30` with output exactly one.
FMHA v3 remains unfixed because it uses precompiled code objects.

## Reproduction

From `/job/aiter`:

```bash
AITER_JIT_DIR=/job/aiter-jit-cache GPU_ARCHS=gfx950 \
  /opt/venv/bin/python -m pytest op_tests/test_mha_extreme_logits.py -v

AITER_JIT_DIR=/job/aiter-jit-cache GPU_ARCHS=gfx950 \
  /opt/venv/bin/python -m pytest op_tests/test_mha.py -k opus -q

AITER_JIT_DIR=/job/aiter-jit-cache GPU_ARCHS=gfx950 \
  /opt/venv/bin/python -m pytest op_tests/test_mha_varlen.py -k opus -q

AITER_JIT_DIR=/job/aiter-jit-cache GPU_ARCHS=gfx950 \
  /opt/venv/bin/python /job/repro_ck_extreme_logits.py
```

The broad current-main, upstream-candidate, D128 large-scale, exact-v3, and CK
result artifacts are retained outside the checkout as:

- `/job/extreme-logits-current-main.json`
- `/job/extreme-logits-pr5087.json`
- `/job/extreme-logits-current-main-d128-large.json`
- `/job/extreme-logits-reported-v3.json`
- `/job/extreme-logits-ck-neighbors.json`

They include input/output SHA-256 hashes, minima/maxima, nonzero counts, finite
flags, direct-routing identity, tolerances, and failure messages.

## Limitations

- FMHA v3 is precompiled and is intentionally not changed by this patch.
- Native D64 was not run on MI355X because the public path is gfx942-only; the
  upstream candidate reports MI300X validation.
- FP16 was tested only at representable nearby scales, not BF16-scale exponents.
- CK coverage is limited to the supported BF16 D64 and FP16 D128 neighbors used
  here; it is not an exhaustive dtype/shape sweep.
- No performance benchmark was collected for the D192 pre-scale change.
- `black`, `ruff`, and `clang-format` binaries were unavailable in this image;
  style was checked manually and with `git diff --check`.
