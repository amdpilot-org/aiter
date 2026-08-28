# Investigation: gfx950 `gemm_a8w8_blockscale_bpreshuffle` v1 accumulator nondeterminism

## TL;DR

The defect described in the task is **real and reproducible at the pinned
source revision** (`ROCm/AITER` `3c5f0ba9` / Composable Kernel `10cb6916`):
repeated launches of the untuned (heuristic) A8W8 block-scale B-preshuffle GEMM
on gfx950/MI35x with fixed inputs produce a different output every launch.

The proposed stabilizer (`rocm-libraries` `408a5bbe`, based on `ce3e67b2`) **does
fix it** at that pinned revision: `unique_hashes=1/1000`, with correctness
preserved.

**However, `amdpilot-org/aiter` `main` (`2f9ee5e`) pins Composable Kernel
`15e12dd`, which already contains a working accumulator anchor that resolves the
defect.** Verified on `main` as-is with the issue's exact reproduction:
`unique_hashes=1/1000` (the task's passing criterion), plus the focused K rows
and an 8-shape correctness sweep. The stabilized pinned-revision output is
byte-for-byte identical to `main`'s output.

**No code change is warranted on `main`.** Applying the `408a5bbe` stabilizer to
`main`'s CK `15e12dd` would be inappropriate: the diff does not apply to
`15e12dd` (the files differ), and `15e12dd` already ships a different, working
anchor. Replacing a working fix with a ported alternative would risk regression
for no benefit. This PR therefore documents the investigation rather than
patching `main`.

## The problem

On MI35x/gfx950, `aiter.gemm_a8w8_blockscale_bpreshuffle` (the CK B-preshuffle
path) for WKV-shaped, **untuned** A8W8 block-scale GEMMs is nondeterministic:
with fixed inputs and a fixed launch path, each launch yields a different
output. At `M=8192, N=512, K=4096` this manifests as one unique output hash per
launch.

The untuned path is the heuristic v1 kernel
`a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1`
(the `BlockGemmPipelineVersion::v1` fallback in
`csrc/ck_gemm_a8w8_blockscale_bpreshuffle/gemm_a8w8_blockscale_bpreshuffle.cu`).
For `M=8192,N=512,K=4096` there is no tuned CK entry in
`aiter/configs/a8w8_blockscale_bpreshuffle_tuned_gemm.csv` (gfx950/256CU rows
are K=7168 `asm`, or other M/N/K), so the dispatch falls through to this v1
heuristic.

Root cause: in the CK blockwise B-preshuffle v1 pipeline
(`blockwise_gemm_pipeline_xdlops_blockscale_b_preshuffle_v1.hpp`), the packed
FMA accumulations into the post-scale accumulator are not forced to be
materialized at a stable program point. The compiler is free to schedule/reuse
MFMA accumulator state across loop iterations in a way whose result depends on
launch-time scheduling, producing a different (always-correct-tolerance) bit
pattern each launch.

## What I did

### 1. Reproduced the defect at the pinned revision

- AITER `3c5f0ba9` (from the `amdpilot-org/aiter` mirror) with CK submodule at
  `10cb6916` (note: CK `10cb6916`'s own message states it corresponds to
  `rocm-libraries` commit `ce3e67b2` — i.e. exactly the stabilizer's base).
- Built the JIT module `module_gemm_a8w8_blockscale_bpreshuffle` (heuristic v1
  instance only, via an empty tune file so only `default_kernels_dict[-1]` is
  compiled — the heuristic is always built and is what the untuned shape uses).
- Ran the issue's reproduction: fixed seed 1, FP8 `x=[8192,4096]`,
  `weight=[512,4096]`, fp32 block scales (`x_scale=[8192,32]`,
  `w_scale=[4,32]`, block `(128,128)`), `x_scale` transposed,
  `shuffle_weight(weight, layout=(16,16))`, BF16 output, `HIP_VISIBLE_DEVICES=0`,
  fresh `AITER_JIT_DIR`.

Result (unpatched `10cb6916`):

```
RESULT M=8192 N=512 K=4096 seed=1 unique_hashes=20/20
```

i.e. one unique hash per launch — the reported behavior.

### 2. Inspected and applied the proposed stabilizer

- Inspected `rocm-libraries` `408a5bbe` ("Stabilize CK bpreshuffle blockscale
  GEMM accumulator"). It modifies exactly the two files named in the task. The
  approach: thread a runtime-null `p_accum_observer` (and tile/mrepeat/thread
  selectors) from the gridwise `Problem` struct into the blockwise v1 `Run()`,
  and add `anchor_accumulator_value(...)` calls (a null-gated
  `asm volatile("" : : "v"(value))` plus an unconditional
  `type_convert<float>(accumulator)` materialization) after the packed FMA sites
  in the main, even-tail, and odd-tail paths. The pointer is null in normal
  execution, so the `asm` never fires at runtime and the numerical result is
  unchanged; only the (unconditional) accumulator materialization keeps the
  compiler from reordering the accumulations nondeterministically.
- The diff applies cleanly to CK `10cb6916` after remapping the path prefix
  (`projects/composablekernel/include/ck` -> `include/ck`), confirming
  `10cb6916` == stabilizer base `ce3e67b2`.

Result (`10cb6916` + stabilizer `408a5bbe`):

```
RESULT M=8192 N=512 K=4096 seed=1 unique_hashes=1/1000
```

Passing criterion met. Correctness preserved (CK vs torch reference,
`checkAllclose atol=0.01 rtol=0.01` passed, `err=0`) across 8 shapes, and the
focused K rows are all stable:

```
DET M=8192 N=512 K=384:  unique_hashes=1/200
DET M=8192 N=512 K=640:  unique_hashes=1/200
DET M=8192 N=512 K=3968: unique_hashes=1/200
DET M=8192 N=512 K=4096: unique_hashes=1/1000
DET M=8192 N=512 K=4224: unique_hashes=1/200
```

### 3. Checked `main`

`amdpilot-org/aiter` `main` (`2f9ee5e`) pins CK `15e12dd` (newer than
`10cb6916`). `15e12dd` **already contains an accumulator anchor** in the
blockwise v1 header — a different but equivalent implementation:

```cpp
// It keeps each post-scale accumulator definition visible to the
// optimizer, which enables determinism for bitwise-stable repeated launches.
auto anchor_accumulator_value = [&](auto& value) { asm volatile("" : "+v"(value)); };
```

called immediately after each packed
`c_acc_vec(t) = __builtin_elementwise_fma(...)` at all four FMA sites (main and
tail paths). It is unconditional (no runtime observer), uses an empty
read-write `asm` (no numerical change), and needs no gridwise/`Problem` changes
— satisfying all of the task's constraints (appropriate for the pinned CK
revision; does not alter numerical semantics; does not activate runtime
observer work).

Verified on `main` **as-is** (no modifications):

```
RESULT M=8192 N=512 K=4096 seed=1 unique_hashes=1/1000
DET M=8192 N=512 K=384:  unique_hashes=1/200
DET M=8192 N=512 K=640:  unique_hashes=1/200
DET M=8192 N=512 K=3968: unique_hashes=1/200
DET M=8192 N=512 K=4096: unique_hashes=1/1000
DET M=8192 N=512 K=4224: unique_hashes=1/200
```

Correctness sweep (8 shapes, CK vs torch reference, all `err=0`, `passed`):
`8192x512x4096`, `8192x512x384`, `8192x512x4224`, `256x512x4096`,
`4096x4096x4096`, `8192x1536x4096`, `128x512x1280`, `8192x512x7168`.

Two *tuned* v1 instances (different kernel configs, built from a 2-row tune
file) are also stable on `main`:

```
TUNED-DET M=4096 N=1536 K=7168: unique_hashes=1/100   (256x64x128x128 ... v1)
TUNED-DET M=128  N=4096 K=1280: unique_hashes=1/100   (256x16x128x256 ... v1)
```

### 4. Equivalence check

The SHA-256 of the stabilized pinned-revision (`10cb6916` + `408a5bbe`) output
is **identical** to `main`'s (`15e12dd`) output for the primary shape:

```
4cee14cfe82d325ba739f357780c17484f474fc66610b4673e17526dbbaaf2e0
```

i.e. both fixes produce the same, correct, deterministic result.

## Conclusion

The defect is already resolved on `main` by CK `15e12dd`'s accumulator anchor.
The task's passing criterion (`unique_hashes=1/1000` for `M=8192,N=512,K=4096`
after ten warmups) is met on `main` without any change. The `408a5bbe`
stabilizer is a valid alternative fix (verified on the pinned `10cb6916`), but
porting it onto `main`'s `15e12dd` is neither necessary nor appropriate. Hence
this PR carries the investigation + reproduction artifacts rather than a code
patch.

If a backport of the fix to the pinned revision (`3c5f0ba9`/CK`10cb6916`) is
ever needed, `stabilizer_ck.diff` in this directory is the path-remapped
`408a5bbe` diff that applies cleanly to `10cb6916` and was used to verify the
fix.

## How to reproduce

Environment used: MI355X (gfx950, 256 CUs), ROCm 7.2.0, torch 2.9.1+rocm7.2,
triton 3.5.1 (note: `main`'s gluon kernels want triton>=3.6; the gemm path does
not use gluon, so set `AITER_USE_SYSTEM_TRITON=1` to import `aiter`).

```sh
# clone main (CK submodule at 15e12dd, already fixed)
git clone https://github.com/amdpilot-org/aiter.git
cd aiter && git submodule update --init --depth 1 3rdparty/composable_kernel

# minimal build: only the heuristic v1 instance (empty tune file)
printf 'gfx,cu_num,M,N,K,libtype,kernelId,splitK,us,kernelName,tflops,bw,errRatio\n' > /tmp/empty.csv

HIP_VISIBLE_DEVICES=0 \
AITER_JIT_DIR=/tmp/jit-main \
AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE=/tmp/empty.csv \
AITER_REBUILD=1 AITER_USE_SYSTEM_TRITON=1 \
python repro.py --warmups 10 --iters 1000
# -> RESULT M=8192 N=512 K=4096 seed=1 unique_hashes=1/1000
```

`repro.py` (in this directory) constructs the exact tensors from
`op_tests/test_gemm_a8w8_blockscale.py` (FP8 x/weight, fp32 block scales with
block (128,128), transposed x_scale, `shuffle_weight(..., layout=(16,16))`,
BF16 output) and calls `gemm_a8w8_blockscale_bpreshuffle_ck` (the CK JIT module
-> C++ heuristic v1 dispatch) directly, hashing each synchronized output
byte-for-byte. To reproduce the *buggy* behavior, check out AITER `3c5f0ba9`
with CK `10cb6916` (unpatched) and run the same script — expect
`unique_hashes=N/N`.

## What I did not do / where I was unsure

- **No code change on `main`**: by design — `main` already meets the passing
  criterion. I did not port `408a5bbe` onto `15e12dd` because it does not apply
  and would replace a working fix.
- **Triton version**: `main` requires triton>=3.6 for gluon kernels; the
  environment has 3.5.1. I bypassed the import guard with
  `AITER_USE_SYSTEM_TRITON=1`. The B-preshuffle GEMM path does not touch gluon,
  so this does not affect the reproduction. I did not upgrade triton.
- **Minimal tune files for build speed**: I built with an empty (or 2-row) tune
  file so only the heuristic (plus, for the tuned check, two selected) v1
  instances compile (~35s each). I did **not** run the full 13-instance tuned
  build or the full `op_tests/test_gemm_a8w8_blockscale.py` suite (which also
  builds/exercises the ASM path and needs the full tune table). The anchor lives
  in the shared blockwise v1 header used by every v1 instance, and the two tuned
  v1 instances I did build were stable.
- **AOT/wheel build**: not exercised. Only the JIT path (`fresh AITER_JIT_DIR`)
  was used, as the task specified.
- **Upstream provenance of the `15e12dd` anchor**: I did not trace the exact CK
  commit that introduced the anchor (the submodule was checked out shallow). I
  confirmed its presence and effect directly.
- **Single device/GPU**: tested on one MI355X (`HIP_VISIBLE_DEVICES=0`).
