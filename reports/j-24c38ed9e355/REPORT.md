# PR27 gfx950 two-GPU follow-up validation

## Result

The unresolved gfx950 portion of [PR27](https://github.com/amdpilot-org/aiter/pull/27) is complete for the requested bounded scope. No production change was needed, and this report-only mirror branch does not duplicate or merge PR27's runtime change.

The validation checkout was a separate ordinary clone checked out at exactly `29790464f03a9c626355a7c963aefee25a85242a`, detached and clean. The delivery branch is cut from current `main` at `f0321c0e8927d1d90a29385433f71e592b1c51f5`.

## Production consumers

The two actual production GEMM call sites using `get_num_xcds` are:

- `aiter/ops/triton/gemm/basic/gemm_a8w8.py:128`: `gemm_a8w8(..., backend="gluon")`
- `aiter/ops/triton/gemm/basic/gemm_a8w8.py:288`: `gemm_a8w8_preshuffle(...)`

Both were run directly on each visible ordinal. No unrelated GEMM was substituted.

## Getter and device contract

Both visible devices are AMD Instinct MI355X, architecture `gfx950`, with 256 compute units. `rocm-smi` reported exactly two cards at PCI `05:00.0` and `15:00.0`.

For each ordinal, direct `hipDeviceGetAttribute(..., hipDeviceAttributeNumberOfXccs=10018, ...)` returned status `0` and value `8`. The current-device and explicit-device getter forms both returned `8`. Repeating current-device switches in the order `0, 1, 0, 1` preserved equality; the cache ended with 2 misses, 78 hits, and current size 2.

## Consumer correctness

Each consumer ran on ordinals 0 and 1 with `float8_e4m3fn` input, `bfloat16` output, `gfx950` Gluon kernels, and the default `GEMM-A8W8` config. Results were compared against independent `torch.mm`/scale/bias references with `torch.testing.assert_close(atol=0.02, rtol=1e-2)`.

Representative shape `(M=1024, N=1024, K=1024)` passed for both consumers on both devices. Boundary shapes `(1, 16, 256)` for the normal consumer and `(16, 16, 256)` for preshuffle also passed. Consumer and reference outputs were finite in every case.

Graph capture and replay passed for both consumers on both devices at the representative shape. Boundary cases did not repeat graph capture; replay was checked once per consumer/device. An earlier in-process invocation of the harness (before graph probes were isolated) terminated with `c10::AcceleratorError: HIP error: an illegal memory access was encountered` in `CUDAGraph::~CUDAGraph`. Isolated normal and preshuffle teardown commands then completed successfully; the final harness isolates each graph probe in a subprocess.

Synchronized CUDA-event pair timings were recorded only as a bounded sanity check. Representative consumer/reference timings ranged from about 105–136 µs and 54–55 µs, respectively; these are not cross-architecture performance claims.

## Environment

- Python: `3.10.12`, `/opt/venv/bin/python`
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`, `/opt/venv/lib/python3.10/site-packages/torch/__init__.py`
- Torch HIP: `7.2.26015-fc0010cf6a`
- Triton: `3.6.0`, `/sgl-workspace/triton-custom/python/triton/__init__.py`
- hipcc: HIP `7.2.26015-fc0010cf6a`, AMD clang `22.0.0git`
- HIP library: `libamdhip64.so.7`
- Native AITER module: `/job/aiter-validation/aiter/jit/module_aiter_core.so`

No private compatible Triton/Python dependency was needed. Native caches were kept under `/job/.triton-cache` and `/job/.triton-home`, outside both checkouts.

## Reproduction

From the exact validation checkout:

```sh
cd /job/aiter-validation
git checkout --detach 29790464f03a9c626355a7c963aefee25a85242a
export PYTHONPATH=/job/aiter-validation
export TRITON_CACHE_DIR=/job/.triton-cache
export TRITON_HOME=/job/.triton-home
/opt/venv/bin/python -m pytest -q --disable-warnings \
  'op_tests/triton_tests/gemm/basic/test_gemm_a8w8.py::test_gemm_fp8[gluon-fp8e4m3-1024-1024-1024]' \
  'op_tests/triton_tests/gemm/basic/test_gemm_a8w8.py::test_gemm_fp8[gluon_shuffle-fp8e4m3-1024-1024-1024]'
/opt/venv/bin/python /job/validate_pr27_gfx950.py
```

The focused harness is preserved at `reports/j-24c38ed9e355/validate_pr27_gfx950.py`. Raw JSON and logs are under `reports/j-24c38ed9e355/raw/`.

## Limitations and scope

- This is a follow-up report only; PR27 remains unmerged and no production patch is claimed.
- The initial in-process graph teardown failure is recorded above, but did not affect the final isolated replay results.
- Boundary graph replay was intentionally not repeated after the supported representative replay check.
- No fix was made merely because the original PR report lacked gfx950 architecture coverage.
