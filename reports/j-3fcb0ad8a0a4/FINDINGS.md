# Findings: eight-GPU cold sampler startup

## Conclusion

The reported failure did not reproduce on this qualified MI300X image. Across 59 bounded child
processes, every process returned zero, no child received a signal or timed out, and every sampler
contract check passed. The evidence does not support changing the sampler or adding the previously
suggested lock.

This is a non-reproduction with useful negative evidence, not proof that the upstream reports are
invalid. The original race attribution was already withdrawn, and this run does not resurrect it.

## Environment

- Host: eight AMD Instinct MI300X GPUs, all `gfx942:sramecc+:xnack-`.
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`.
- HIP: `7.2.26015-fc0010cf6a`.
- Runtime: `/opt/venv/bin/python` from the required image.
- Mirror base: `f0321c0e`.
- Upstream `main` at `337f15b5` has no diff in the sampler wrapper, sampler template, sampling
  kernel, or `csrc/cpp_itfs/utils.py`; therefore there was no upstream candidate to validate.
- The required `3rdparty/composable_kernel` submodule was initialized at
  `af9e1d1f1ae347c22feeb08fd2d42645075e0c5d`. Torch and ROCm were not replaced.

## Harness

The child calls the registered production operator:

```python
torch.ops.aiter.top_k_top_p_sampling_from_probs(
    probabilities,
    None,
    top_k_tensor,
    0,
    top_p_tensor,
    0.0,
    deterministic=True,
)
```

It uses a valid 16 x 248320 float32 probability tensor, tensor-valued `top_k=50`, and
tensor-valued `top_p=0.95`. Each process is assigned one physical GPU with `HIP_VISIBLE_DEVICES`.
Both `AITER_ROOT_DIR` and `AITER_JIT_DIR` point inside the job-private cache root.

The harness records:

- readiness and synchronized start;
- existence checks, lock entry/exit, actual build entry/exit, compile calls, and loads;
- library path, size, and SHA-256;
- return code, signal, timeout status, stdout, and stderr;
- exact first and second sampled IDs and probabilities;
- same-seed deterministic replay;
- probability validity, ID range, and joint top-k/top-p support;
- a 4,096-trial empirical distribution comparison.

The parent bounds each child at 900 seconds and the full matrix was run under a 7,000-second
outer timeout.

## Matrix

| Scenario | Processes | Result |
| --- | ---: | --- |
| Fresh single-worker cold | 1 | Pass |
| Synchronized eight-worker cold, six fresh caches | 48 | Pass |
| Deliberate prebuild | 1 | Pass |
| Warm eight-worker start using prebuild | 8 | Pass |
| Warm loader trace | 1 | Pass |

Totals:

- 59/59 processes returned zero.
- 0 signals.
- 0 timeouts.
- 59/59 passed the deterministic contract.
- Maximum total-variation distance in applicable 4,096-trial checks: `0.05741338059306145`.
- The one-trial prebuild and loader-trace cases intentionally skip the distribution threshold;
  they still pass deterministic replay, support, range, and probability checks.

The first full matrix used `compile_start` to mean a call to `compile_lib`. A follow-up fresh
eight-worker validation added explicit `build_start` instrumentation. It showed 8 compile calls but
exactly 1 actual sampler build: the lock winner runs `main_func`, while the other seven wait and
skip it. This confirms that the check outside the lock causes redundant `compile_lib` calls, not
redundant compilation.

## Artifact analysis

The ten final library paths have the same size, 80,712 bytes, but eight distinct whole-library
SHA-256 values. Section analysis shows:

- one host `.text` hash across all artifacts;
- one embedded device `.text` hash across all artifacts;
- one embedded device `.note` hash across all artifacts;
- varying `.dynstr`, `.gnu.hash`, and `.hip_fatbin` metadata.

The varying strings contain different random `__hip_cuid_<16 hex>` values. The embedded AMDGPU
ELF device code and metadata notes remain identical. Thus the whole-library hash variation is
confined to CUID-bearing loader metadata in this run; it is not evidence of varying device code.

The `set(cxxflags)` behavior does vary Makefile flag order. The matrix assigned different
`PYTHONHASHSEED` values to each process and produced eight normalized flag orders. Despite those
different orders and different final-builder seeds, host `.text` and embedded device `.text` were
byte-identical. This run therefore does not support flag-order instability as a source of device
code variation for this sampler.

## Classification

- **Loader/build publication:** No partial library, loader failure, or unresolved sampler symbol
  was observed. `ldd` resolves all dependencies. The `LD_DEBUG=libs` trace shows the sampler
  library's init and fini calls and exit zero. It also contains a non-fatal Torch `libc10.so`
  lookup diagnostic unrelated to the sampler library.
- **Missing headers/ABI:** The fresh checkout initially lacked the uninitialized CK submodule's
  include directory. After initializing that required submodule, all builds succeeded. There was
  no missing header or ABI failure in the production sampler build.
- **Device numerical defect:** No defect was observed. Same-seed calls produced identical IDs,
  sampled IDs stayed in the joint top-k/top-p support, sampled probabilities were positive, and
  empirical distributions passed the total-variation check.

## Code observations

The demoted upstream observations remain visible in the code:

- `not_built()` is checked before `compile_lib()` acquires its lock.
- `run_lib()` loads outside that lock.
- `not_built()` only checks for the existence of `lib.so`.

In this run, all eight workers could observe a missing library and call `compile_lib`, but the lock
allowed only one actual build. No worker loaded a partially published artifact, and no failure
occurred. Because the original race attribution was withdrawn and this evidence does not support a
crash mechanism, no production lock or atomic-publication change is included.

## Honest limits

- Only eight distinct cold caches were exercised, versus the 22 cold builds described upstream.
- The original host-side report used a different Torch/ROCm build; this task intentionally
  preserved the supplied image stack.
- The harness exercises the real AITER sampler directly, not a full vLLM/NCCL startup.
- No crashing artifact was obtained, so no crashing-versus-healthy device-code diff is possible.
- The distribution check is statistical; it does not prove every floating-point input is correct.
- `LD_DEBUG=libs` can alter loader timing, so it was used only in a separate warm trace.
- NUMA auto-balancing remained enabled, as reported in the preserved logs.

## Reproduce

```bash
/opt/venv/bin/python reports/j-3fcb0ad8a0a4/run_matrix.py \
  --cache-root /job/aiter-cache-j-3fcb0ad8a0a4 \
  --output-root /job/aiter-matrix-j-3fcb0ad8a0a4 \
  --cold-cycles 6 \
  --trials 4096 \
  --child-timeout 900
```

Analyze the resulting matrix with:

```bash
/opt/venv/bin/python reports/j-3fcb0ad8a0a4/analyze_matrix.py \
  --matrix-root /job/aiter-matrix-j-3fcb0ad8a0a4 \
  --output reports/j-3fcb0ad8a0a4/evidence/artifact-analysis.json
```

Compact per-process outcomes, artifact metadata, build stderr, loader trace, Makefiles, and event
excerpts are committed under `evidence/`. The complete 418 MB raw matrix remains at
`/job/aiter-matrix-j-3fcb0ad8a0a4`, and the private cache remains at
`/job/aiter-cache-j-3fcb0ad8a0a4`.
