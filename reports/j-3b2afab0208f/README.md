# MoE correctness verdict investigation

## Result

`checkAllclose` printed an over-threshold `failed!` verdict but returned only
the mismatch ratio, which MoE op-tests discarded. The process therefore exited
zero even when half of a real gfx942 MoE output was deliberately corrupted.

The correction adds an opt-in `strict` mode to `checkAllclose` and arms it for
the validated `op_tests/test_moe.py` CI invocation with
`AITER_STRICT_ALLCLOSE=1`. The default remains non-strict, quiet tuner calls
still return their ratio, and no numerical tolerance is changed.

## Environment

- Image: `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`, image ID
  `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`.
- Source: `/job/aiter`, mirror `amdpilot-org/aiter`, base `main` at
  `f0321c0e8927d1d90a29385433f71e592b1c51f5`.
- Python: `/opt/venv/bin/python` 3.10.12.
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`; HIP: `7.2.26015-fc0010cf6a`.
- GPU: one AMD Instinct MI300X, `gfx942`, capability `(9, 4)`.
- Job-private JIT cache: `/tmp/aiter-j-3b2afab0208f-jit`.
- Native modules loaded from that cache, including
  `module_aiter_core.so`, `module_moe_asm.so`, `module_moe_sorting.so`, and
  `module_moe_fmoe_asm.so`.
- ASM kernels loaded from `/job/aiter/hsa/gfx942`, including
  `fmoe_b16.co` and the quantized MoE kernels used by `test_moe.py`.

## Investigation

ROCm/aiter issue 5030 describes the same defect and proposes an opt-in strict
mode. It had no comments or linked pull requests when read, and current source
did not contain `AITER_STRICT_ALLCLOSE` or a `strict` argument. The mirror's
issue 54 is the local task record.

The MoE op-test files using `checkAllclose` are:

- `op_tests/test_flydsl_moe_aux.py` (23 calls)
- `op_tests/test_moe.py` (3 calls)
- `op_tests/test_moeTopkSoftmax.py` (11 calls)
- `op_tests/test_moe_2stage.py` (1 call)
- `op_tests/test_moe_blockscale.py` (2 calls)
- `op_tests/test_moe_dp_share_expert.py` (1 call)
- `op_tests/test_moe_ep.py` (6 calls)
- `op_tests/test_moe_local_expert_ids.py` (3 calls)
- `op_tests/test_moe_sorting.py` (4 calls)
- `op_tests/test_moe_sorting_mxfp4.py` (6 calls)
- `op_tests/test_moe_tkw1.py` (3 calls)
- `op_tests/test_moe_topk_gating.py` (2 calls)

Most call sites discard the returned ratio. The bounded correction does not
arm every historical MoE file at once; it provides the reusable strict contract
and enables it for the real path validated here.

## GPU evidence

The selected supported path was the no-quant g1u0 case in
`op_tests/test_moe.py`, which explicitly runs the ASM kernel on gfx942:

```text
token=128 model_dim=4096 inter_dim=1024 E=32 topk=5 dtype=bf16
```

### Before correction

Clean invocation:

```text
exit_code=0
[checkAllclose atol=100 rtol=0.01 passed~]
```

Deliberate labeled corruption perturbed the first half of the real ASM output
rows by `+1000`:

```text
exit_code=0
[CORRUPTION: perturb half output rows +1000]
[checkAllclose atol=100 rtol=0.01 failed!]
max abs delta: 1040.0
delta details: 50.0% (262144 of 524288) elements
```

The failed verdict was therefore printed while the invocation still passed.

### After correction

With `AITER_STRICT_ALLCLOSE=1`, the clean invocation remained:

```text
exit_code=0
[checkAllclose atol=100 rtol=0.01 passed~]
```

The same labeled corruption produced:

```text
exit_code=1
[CORRUPTION: perturb half output rows +1000]
[checkAllclose atol=100 rtol=0.01 failed!]
max abs delta: 1032.0
delta details: 50.0% (262144 of 524288) elements
AssertionError: checkAllclose failed: 50.0% (262144 of 524288) elements
```

The raw tensor samples, delta samples, maximum delta, and mismatch count are
logged before the assertion raises.

## Validation

`op_tests/test_checkallclose_strict.py` passes six assertions:

- default behavior still returns the ratio;
- explicit `strict=True` raises;
- the environment variable arms the logging path;
- `printLog=False` remains non-raising under the environment variable;
- a close tensor passes in strict mode;
- a mismatch below `tol_err_ratio` remains a warning.

## Limits

- This was a one-GPU, bounded investigation, not a full MoE suite baseline.
- Other MoE files remain default-off and can be armed incrementally after their
  existing numerical baselines are reviewed.
- Warnings below `tol_err_ratio` intentionally remain warnings.
- No accuracy tolerance, kernel numeric, or supported-case gate was changed.
