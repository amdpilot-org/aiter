# Follow-up: PR19 FP8 MHA causal LSE failures on MI300X

## Scope

- Follow-up to [PR29](https://github.com/amdpilot-org/aiter/pull/29), which validated [PR19](https://github.com/amdpilot-org/aiter/pull/19) for [ROCm/aiter issue 5000](https://github.com/ROCm/aiter/issues/5000).
- Draft delivery PR: [PR32](https://github.com/amdpilot-org/aiter/pull/32).
- This run targets only the two actual PR29 LSE failures. It does not repeat the full 56-case matrix and does not claim the other 26 runtime rejections were checked or resolved.
- Validation control: `/job/aiter-validation`, detached at exact PR19 head `94996d72b232ac3e640c692e10d1d908710520a5`, clean.
- Candidate control: `/job/aiter-candidate`, same exact head with only the unmerged candidate patch in `candidate-fix.patch`.
- Delivery checkout: `/job/aiter-main`, branch `amdpilot/j-d6e546987377`, cut from current `main` at `f0321c0e8927d1d90a29385433f71e592b1c51f5`.
- The validation checkout was not merged with refreshed main. Native caches were kept under `/job/aiter-build-cache` and `/job/aiter-candidate-build-cache`, outside all checkouts.

## Environment

- GPU: exactly 1 visible AMD Instinct MI300X, `gfx942`, capability `(9, 4)`.
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`.
- HIP: `7.2.26015-fc0010cf6a`; AMD clang `22.0.0git` (`roc-7.2.0 26014`).
- Imported Aiter source: `/job/aiter-validation/aiter/__init__.py` and `/job/aiter-validation/aiter/ops/mha.py` for the control; `/job/aiter-candidate/aiter/ops/mha.py` for the candidate.
- Native modules were imported from the external cache paths recorded in the raw logs.
- Validation submodule: `3rdparty/composable_kernel` at `af9e1d1f1ae347c22feeb08fd2d42645075e0c5d`.

## Exact failures reproduced

PR29's exact invocations were rerun in bounded subprocesses against the clean validation head:

```bash
python op_tests/test_mha_fp8.py \
  -b 1 -n 8 -nk 8 -q 512 -k 512 -d 128 -dv 128 --lse -c
python op_tests/test_mha_varlen_fp8.py \
  -b 1 -n 8 -nk 8 -q 512 -k 512 -d 128 -dv 128 --lse -c
```

| Path | Sequence | Causal | Output max error | Production LSE error | Result |
|---|---:|---|---:|---:|---|
| fixed | 512 | yes | 0.0390625 | 3.2928357124328613 | failed unchanged 0.01 LSE gate |
| varlen | 512 | yes | 0.04296875 | 3.2909841537475586 | failed unchanged 0.01 LSE gate |

Both output errors remain below the unchanged 0.055 output gate. The deterministic input dtype is `torch.float8_e4m3fnuz` (FNUZ). All per-tensor descales are `0.004150390625`, and the softmax scale is `128 ** -0.5 = 0.08838834764831845`.

`repro_lse.py` independently dequantizes the FNUZ tensors to float32, computes QK logits and causal masks directly, and computes float32 `logsumexp`. It also compares against the existing `attention_ref` convention and diagnostic scale/mask variants.

## Isolation

- The direct float32 reference and the existing test reference agree within `1.9073486328125e-06` for both paths.
- The production LSE and direct reference have identical `-inf` patterns. The fixed path has no fully masked rows; the varlen path has 128 fully masked rows, as expected from its packed lengths (`q=508`, `k=492`).
- Therefore the mismatch is in the returned production LSE, not the reference convention, scaling, FNUZ dequantization, or test plumbing.
- For both paths, query rows 0 through 255 agree within about `9.5e-7`. Every tested head fails beginning exactly at query row 256. This isolates the numerical defect to the gfx942 FP8 D128 causal `fmha_v3` ASM LSE path beyond its 256-row tile boundary.
- A direct CK FP8 LSE fallback probe was rejected with `RuntimeError: invalid argument for fmha_fwd`, so rerouting FP8 to CK is not a valid fix.
- A BF16 MHA LSE probe on the dequantized FNUZ tensors agrees with the direct float32 reference within `1.9073486328125e-06` and preserves the `-inf` pattern.

## Candidate fix

`candidate-fix.patch` is an unmerged production patch with exact base `94996d72b232ac3e640c692e10d1d908710520a5`. It is not part of current `main` and must not be represented as merged.

The narrowly gated wrapper workaround:

- Applies only on `gfx942`, for FNUZ FP8 D128, causal, per-tensor descale, with LSE requested and more than 256 query rows.
- Runs the existing FP8 ASM forward without LSE so the production FP8 output remains unchanged.
- Dequantizes Q/K/V to BF16 and uses the existing BF16 MHA forward solely to return the LSE.
- Does not alter either the 0.055 output gate or the 0.01 LSE gate.

This is a safe wrapper workaround, not a repair of the underlying ASM LSE instruction sequence. A proper ASM kernel fix remains open.

## Candidate results

| Path | Sequence | Causal | Output max error | LSE error | `-inf` agreement | Result |
|---|---:|---|---:|---:|---|---|
| fixed | 512 | yes | 0.0390625 | 1.9073486328125e-06 | yes | passed original invocation |
| varlen | 512 | yes | 0.04296875 | 1.9073486328125e-06 | yes | passed original invocation |
| fixed | 512 | no | 0.0078125 | 1.9073486328125e-06 | yes | passed original invocation |
| varlen | 128 | yes | 0.04296875 | 9.5367431640625e-07 | yes | passed original invocation |

Additional direct-reference controls:

| Path | Sequence | Causal | Output max error | LSE error | Result |
|---|---:|---|---:|---:|---|
| fixed | 512 | no | 0.004843294620513916 | 1.9073486328125e-06 | passed |
| fixed | 256 | yes | 0.017711162567138672 | 9.5367431640625e-07 | passed |
| varlen | 512 | no | 0.004723191261291504 | 1.9073486328125e-06 | passed |
| varlen | 256 | yes | 0.02078425884246826 | 9.5367431640625e-07 | passed |
| varlen | 128 | yes | 0.020784318447113037 | 9.5367431640625e-07 | passed |
| varlen | 128 | no | 0.0071582794189453125 | 9.5367431640625e-07 | passed |

Fixed LSE at sequence 128 remains unsupported and is explicitly rejected by the public call with `RuntimeError: invalid argument for fmha_fwd`, matching PR29. It was not silently skipped.

## Reproduction

From this delivery checkout, with the validation checkout at the exact head above:

```bash
export AITER_VALIDATION_ROOT=/job/aiter-validation
export AITER_BUILD_CACHE=/job/aiter-build-cache
export AITER_JIT_DIR=$AITER_BUILD_CACHE/jit
export TRITON_CACHE_DIR=$AITER_BUILD_CACHE/triton
export TORCH_EXTENSIONS_DIR=$AITER_BUILD_CACHE/torch-extensions
export TMPDIR=$AITER_BUILD_CACHE/tmp
export PYTHONPATH=$AITER_VALIDATION_ROOT

cd "$AITER_VALIDATION_ROOT"
python "$OLDPWD/reports/j-d6e546987377/repro_lse.py" \
  --path fixed --seq 512 --causal
python "$OLDPWD/reports/j-d6e546987377/repro_lse.py" \
  --path varlen --seq 512 --causal
```

Use `--tensor-out FILE` to save the deterministic FNUZ tensors, production/direct LSE, outputs, causal masks, cumulative lengths, and descales. The two baseline tensor bundles are included under `raw/tensors/`.

## Raw artifacts

- `raw/baseline/`: exact original failures and direct-reference measurements.
- `raw/candidate/`: patched candidate results and original invocation logs.
- `raw/controls/`: noncausal and shorter-sequence controls, including explicit fixed-128 rejections.
- `raw/probes/`: CK FP8 rejection and BF16 LSE fallback probes.
- `raw/tensors/`: deterministic baseline tensor bundles for both exact failures.
- `candidate-fix.patch`: exact-base unmerged production candidate.

## Limitations

- Only one MI300X was available; no other architecture was tested.
- The candidate workaround covers the per-tensor descale shape used by the exact public failures. Per-head descale above row 256 remains an open case and was not silently claimed fixed.
- The underlying ASM LSE defect was isolated but not repaired at instruction level.
- The 26 PR29 runtime rejections were not rechecked and remain unresolved in this report.
