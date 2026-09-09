# MI300 validation report for PR19 FP8 MHA dimension coverage

## Scope

- Validated the MI300 half of ROCm/aiter issue 5000 against PR19 at exact head `94996d72b232ac3e640c692e10d1d908710520a5`.
- Matched source base: `c6677e075198a78dfee0d3490c97b37533b5610f`.
- Delivery branch is cut from current `main` at `f0321c0e8927d1d90a29385433f71e592b1c51f5`.
- No source or kernel changes were made.

## Environment

- GPU: one AMD Instinct MI300X, `gfx942`, capability `(9, 4)`.
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`.
- Validation checkout: `/job/aiter-validation`, detached at `94996d72b232ac3e640c692e10d1d908710520a5`, clean.
- Imported Aiter path: `/job/aiter-validation/aiter/__init__.py`.
- Native build and cache root: `/job/aiter-build-cache`, outside the validation checkout.
- Required Composable Kernel submodule was initialized at its recorded commit `af9e1d1f1ae347c22feeb08fd2d42645075e0c5d`.

## Method

- Used the existing public harnesses:
  - `op_tests/test_mha_fp8.py`
  - `op_tests/test_mha_varlen_fp8.py`
- Read the supported CLI options before testing and used only those options.
- Exercised fixed and variable-length public FP8 MHA paths with their BF16 baselines.
- Requested dimensions:
  - `(D_QK,D_V) = (128,128), (192,128), (128,64), (256,256)`
  - causal and noncausal
  - batch `1`, heads `8`, sequence lengths `128` and `512`
- Exercised optional LSE where supported.
- Preserved the harness output tolerance `<0.055` and LSE tolerance `<0.01`.
- Preserved the harness assertion that `-inf` LSE masks are identical.
- Did not widen tolerances, replace the Torch stack, or alter kernels.

## Results

### Non-LSE public paths

- Fixed: 12 of 16 cases passed.
- Variable-length: 12 of 16 cases passed.
- All eight `(256,256)` cases were explicitly unavailable on `gfx942` and failed with:
  - `RuntimeError: invalid argument for fmha_fwd`
- Maximum passing output max-absolute-error: `0.05078125`, below `0.055`.

### Optional LSE

- Fixed LSE: 1 of 8 exercised cases passed.
- Variable-length LSE: 3 of 16 exercised cases passed.
- Maximum passing LSE max-absolute-error: `1.9073486328125e-06`, below `0.01`.
- All passing LSE cases also passed the harness’s identical `-inf` mask assertion.
- Fixed LSE at sequence `128` was probed and rejected with `RuntimeError: invalid argument for fmha_fwd`; it is recorded as an explicit skip rather than silently omitted.
- Fixed LSE at sequence `512`:
  - Noncausal `(128,128)` passed.
  - Causal `(128,128)` produced LSE error `3.2928357124328613`, above `0.01`.
  - `(192,128)`, `(128,64)`, and `(256,256)` were rejected with `RuntimeError: invalid argument for fmha_fwd`.
- Variable-length LSE:
  - `(128,128)` noncausal and causal at sequence `128` passed.
  - `(128,128)` noncausal at sequence `512` passed.
  - `(128,128)` causal at sequence `512` produced LSE error `3.2909841537475586`, above `0.01`.
  - All other dimensions were rejected with `RuntimeError: invalid argument for fmha_fwd`.

### Passing measurements

Actual `quant_fwd_us` and `fwd_us` values are in `raw/results.csv`. The passing cases are:

| Path | Seq | D_QK | D_V | Causal | LSE | Max output error | LSE error | quant_fwd_us | fwd_us |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|
| fixed | 128 | 128 | 128 | no | no | 0.009765625 |  | 14.999273 | 16.703714 |
| fixed | 128 | 128 | 128 | yes | no | 0.05078125 |  | 15.562566 | 13.061547 |
| fixed | 128 | 128 | 64 | no | no | 0.0078125 |  | 11.084190 | 9.152511 |
| fixed | 128 | 128 | 64 | yes | no | 0.046875 |  | 12.656367 | 10.154830 |
| fixed | 128 | 192 | 128 | no | no | 0.01171875 |  | 15.428200 | 17.863720 |
| fixed | 128 | 192 | 128 | yes | no | 0.046875 |  | 16.728400 | 19.386980 |
| fixed | 512 | 128 | 128 | no | no | 0.0078125 |  | 20.033650 | 27.407440 |
| fixed | 512 | 128 | 128 | yes | no | 0.0390625 |  | 32.526423 | 45.270213 |
| fixed | 512 | 128 | 64 | no | no | 0.0078125 |  | 21.049708 | 21.490359 |
| fixed | 512 | 128 | 64 | yes | no | 0.046875 |  | 22.326880 | 22.936606 |
| fixed | 512 | 192 | 128 | no | no | 0.0078125 |  | 26.691737 | 19.840790 |
| fixed | 512 | 192 | 128 | yes | no | 0.04296875 |  | 28.514870 | 20.180840 |
| fixed | 512 | 128 | 128 | no | yes | 0.0078125 | 1.9073486328125e-06 | 19.485960 | 27.59986 |
| varlen | 128 | 128 | 128 | no | no | 0.01171875 |  | 11.707640 | 13.171750 |
| varlen | 128 | 128 | 128 | yes | no | 0.04296875 |  | 12.157020 | 14.034980 |
| varlen | 128 | 128 | 64 | no | no | 0.01171875 |  | 11.467900 | 9.774351 |
| varlen | 128 | 128 | 64 | yes | no | 0.0390625 |  | 12.428100 | 10.974855 |
| varlen | 128 | 192 | 128 | no | no | 0.01171875 |  | 15.278230 | 10.699115 |
| varlen | 128 | 192 | 128 | yes | no | 0.0390625 |  | 16.410776 | 10.938260 |
| varlen | 512 | 128 | 128 | no | no | 0.0078125 |  | 20.913210 | 27.998800 |
| varlen | 512 | 128 | 128 | yes | no | 0.04296875 |  | 34.024280 | 46.395290 |
| varlen | 512 | 128 | 64 | no | no | 0.0078125 |  | 22.697470 | 22.875450 |
| varlen | 512 | 128 | 64 | yes | no | 0.0390625 |  | 25.591860 | 25.286286 |
| varlen | 512 | 192 | 128 | no | no | 0.0078125 |  | 28.465938 | 20.224440 |
| varlen | 512 | 192 | 128 | yes | no | 0.0390625 |  | 30.822930 | 20.467410 |
| varlen | 128 | 128 | 128 | no | yes | 0.01171875 | 9.5367431640625e-07 | 10.718480 | 13.10880 |
| varlen | 128 | 128 | 128 | yes | yes | 0.04296875 | 9.5367431640625e-07 | 11.689673 | 14.03780 |
| varlen | 512 | 128 | 128 | no | yes | 0.0078125 | 1.9073486328125e-06 | 19.976730 | 28.20650 |

## Limitations

- This is a bounded MI300 architecture check, not a full CI matrix.
- Only one MI300X GPU was used.
- Only sequence lengths `128` and `512` were tested.
- Only batch `1` and heads `8` were tested.
- No larger shapes or additional architectures were run.
- Fixed LSE at sequence `128` was not part of the passing matrix because the public call rejected it.
- The report does not claim untested kernels are correct.

## Reproduction

Run:

```bash
./reports/j-b234e3858812/run_validation.sh
```

The script uses the exact validation checkout and keeps native builds and caches under `/job/aiter-build-cache`.

## Raw artifacts

- `raw/environment.log`
- `raw/results.csv`
- `raw/fixed_no_lse_results.tsv`
- `raw/fixed_lse_512_results.tsv`
- `raw/varlen_no_lse_results.tsv`
- `raw/varlen_lse_results.tsv`
- Individual raw harness logs under `raw/`
