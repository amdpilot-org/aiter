# AITER issue 4888 cold-replay investigation

## Result

The reported failure did not reproduce in the standalone eight-GPU harness. This is a
negative result for the harness and does not establish that the vLLM engine symptom is
fixed.

- 12 fresh one-stage processes and 12 matched two-stage processes each used 8 MI300X GPUs.
- Each process captured one fused all-reduce + RMSNorm HIP graph without an eager fused warmup.
- Each rank recorded the first replay separately, followed by two warm replays with new deterministic data.
- Immediate probes and probes after a controlled 75-second idle interval both passed.
- Every all-reduce, residual, and normalization comparison passed the existing `atol=rtol=5e-2` tolerance.
- Across 96 cold rank-replays per stage, there were zero tolerance mismatches. The 192 warm rank-replays per stage also had zero mismatches.
- Maximum observed absolute errors were bounded by the reference arithmetic: all-reduce `0.0463867`, residual `0.046875`, and normalization `0.0625`. The normalization maximum exceeds `atol` alone but passes the existing combined `atol + rtol * |reference|` criterion.

## Probe design

- GPUs: 8x AMD Instinct MI300X (`gfx942`).
- Image: `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`.
- Runtime: `/opt/venv/bin/python`, Torch `2.9.1+rocm7.2.0.git7e1940d4`, HIP `7.2.26015-fc0010cf6a`, RCCL `2.27.7`.
- Source base: `f0321c0e8927d1d90a29385433f71e592b1c51f5`.
- Hidden size: 6144, dtype `bfloat16`, world size 8.
- Token counts: 8, 10, 11, and 16. At 6144 columns in bf16 these are 96 KiB, 120 KiB, 132 KiB, and 192 KiB, straddling the 128-KiB dispatch boundary.
- Stage selection: `AITER_AR_1STAGE=1` or `AITER_AR_1STAGE=0`; all other settings were fixed.
- Inputs and residual inputs were rank-distinct and deterministic. Each replay used a new seed.
- The fused operator's all-reduce contribution was recovered as `residual_output - local_residual`, then compared with an independent RCCL `dist.all_reduce` result.
- The residual output was compared with `RCCL_all_reduce(input) + local_residual`.
- The normalized output was compared with independent `torch.nn.functional.rms_norm`.
- No eager fused operator call occurred before the first replay. Graph capture records the work without executing the captured kernels.
- The controlled idle probes slept for 75 seconds after capture and before the first replay.

## Reproduce

Run from the repository root with the eight assigned GPUs:

```bash
export AITER_JIT_DIR=/job/.aiter-jit
export TORCH_EXTENSIONS_DIR=/job/.torch-extensions
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

/opt/venv/bin/torchrun --standalone --nproc_per_node=8 \
  op_tests/multigpu_tests/test_fused_ar_rms_cold_replay.py \
  --stage 1stage --tokens 8 --hidden 6144 --replays 3 \
  --idle-seconds 75 --output /tmp/aiter-4888-1stage.json

/opt/venv/bin/torchrun --standalone --nproc_per_node=8 \
  op_tests/multigpu_tests/test_fused_ar_rms_cold_replay.py \
  --stage 2stage --tokens 8 --hidden 6144 --replays 3 \
  --idle-seconds 75 --output /tmp/aiter-4888-2stage.json
```

The `results/` directory contains the raw JSON receipts. Each file records the stage,
token count, idle interval, seed, tolerance, Torch/HIP identity, and per-rank checksums,
error statistics, and pass/fail state for every replay.

## Scope and limitations

- No synchronization or initialization defect was observed, so no correctness fix was made.
- The standalone graph is much smaller than the vLLM piecewise-graph engine and does not include FP8 block-scale GEMMs, MoE, scheduler state, cache eviction, or model-specific buffer reuse.
- The harness uses the actual AITER fused operator and real interprocess IPC, but the all-reduce contribution is derived from the fused residual output because the fused binding does not return the intermediate all-reduce tensor directly.
- These results narrow the report to the standalone operator path: first cold replay, 75-second idle, the 128-KiB boundary, and the one-stage/two-stage contrast were all clean on this image and source.
