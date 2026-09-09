# `top_k_renorm_probs` gfx942 reproducer

## Conclusion

The reported memory fault is reproducible on the assigned MI300X (`gfx942`).

- Mirror `main` at `f0321c0e8927d1d90a29385433f71e592b1c51f5`: **23/27 cases pass; 4/27 fail with `hipErrorIllegalAddress`**.
- All four failures use a per-row `top_k` tensor and vocabulary size `1,048,576`; scalar `k`, smaller mixed batches, both tested layouts, and all three tested dtypes otherwise pass.
- The exact mixed batch `[50, 1048576]` fails **5/5 fresh-process repeats**.
- Upstream candidate `04906a670e187c6a6091c77e03ff3781cfd2867f` (ROCm/aiter pull request 5191) passes **27/27 matrix cases** and **5/5 targeted repeats**.

This is therefore a reproduced fault for which an already-identified, still-open upstream candidate fix works. This PR deliberately does not duplicate that native change.

## Coverage

- Vocabulary sizes: `111`, `500`, `32000`, `128256`, and `1048576`.
- Scalar `k`: `1`, `50`, `100`, `vocab - 1`, `vocab`, and `vocab + 1`.
- Per-row `k`: mixtures of `10`, `50`, `100`, `vocab - 1`, `vocab`, and `vocab + 1`, including disabled top-k encoded as `k = vocab`.
- Input dtypes: `float32`, `float16`, and `bfloat16`.
- Input layouts: row-major contiguous and row-major contiguous with a nonzero storage offset.
- Probabilities are positive and normalized. A `0.05` lower bound is applied before normalization so `float16` conversion cannot underflow a synthetic value to zero.
- Every case runs in a fresh child process because the native workspace is process-local static storage.
- Non-crashing cases are compared with an independent PyTorch reference using `torch.allclose(..., rtol=1e-3, atol=1e-3)`.

The maximum absolute difference among passing main cases was `1.4901161193847656e-08`; the candidate's maximum was identical. No numerical gate was relaxed.

## Environment

- Image: `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`, local ID `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`.
- GPU: one AMD Instinct MI300X, `gfx942`, GUID `39656`.
- Python: `/opt/venv/bin/python`, version `3.10.12`.
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`; HIP `7.2.26015-fc0010cf6a`.
- Python wrapper: `aiter/ops/sampling.py`.
- Python JIT interface: `csrc/cpp_itfs/sampling/top_k_renorm_probs.py`.
- Native source: `csrc/cpp_itfs/sampling/sampling.cuh`.
- Main native module: `/job/aiter-build-main/build/top_k_renorm_probs_d41d8cd98f00b204e9800998ecf8427e/lib.so`.
- Candidate native module: `/job/aiter-build-candidate/build/top_k_renorm_probs_d41d8cd98f00b204e9800998ecf8427e/lib.so`.

Structured details are in `environment.json`. The result JSON files embed the `rocm-smi` output used to verify `gfx942`.

## Commands

The working clone was created with:

```bash
git clone --depth 50 https://github.com/amdpilot-org/aiter.git /job/aiter
cd /job/aiter
git checkout -b amdpilot/j-8e1daed2f804 origin/main
git submodule update --init --depth 1 3rdparty/composable_kernel
```

Read-only upstream context was fetched from the public GitHub API for ROCm/aiter issue 5192, its comments/timeline, and pull request 5191. No upstream issue, pull request, or comment was created or modified.

Main-branch campaign:

```bash
export AITER_ROOT_DIR=/job/aiter-build-main
export PYTORCH_TUNABLEOP_ENABLED=0
export OMP_NUM_THREADS=1
/opt/venv/bin/python reports/j-8e1daed2f804/reproduce_top_k_renorm_probs.py \
  --label main \
  --output reports/j-8e1daed2f804/results-main.json \
  --image-id sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1 \
  --timeout 180
```

Targeted main-branch fault control:

```bash
/opt/venv/bin/python reports/j-8e1daed2f804/reproduce_top_k_renorm_probs.py \
  --label main-fault-repeat \
  --output reports/j-8e1daed2f804/results-main-fault-repeat.json \
  --filter per-row-v1048576-k50-disabled \
  --repeat 5 \
  --image-id sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1 \
  --timeout 180
```

The candidate was tested in a separate detached worktree at `04906a670e187c6a6091c77e03ff3781cfd2867f`:

```bash
export AITER_ROOT_DIR=/job/aiter-build-candidate
/opt/venv/bin/python reports/j-8e1daed2f804/reproduce_top_k_renorm_probs.py \
  --label candidate-04906a6 \
  --repo-root /job/aiter-candidate \
  --output reports/j-8e1daed2f804/results-candidate-04906a6.json \
  --image-id sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1 \
  --timeout 180
```

The same targeted five-repeat command with `--label candidate-04906a6-fault-repeat`, `--repo-root /job/aiter-candidate`, and output `results-candidate-04906a6-fault-repeat.json` was used for the candidate control.

## Raw Results

- `results-main.json`: 27 runs, 23 pass, 4 fail. Each failing stderr contains `HIP error: an illegal memory access was encountered` / `hipErrorIllegalAddress`.
- `results-main-fault-repeat.json`: 5 runs, 0 pass, 5 fail.
- `results-candidate-04906a6.json`: 27 runs, 27 pass, 0 fail.
- `results-candidate-04906a6-fault-repeat.json`: 5 runs, 5 pass, 0 fail.

The four main-branch failures are:

- `per-row-v1048576-k50-disabled`
- `per-row-v1048576-mixed`
- `per-row-v1048576-boundary`
- `per-row-v1048576-offset-disabled`

## Interpretation

The observed failures match the mechanism described by ROCm/aiter pull request 5191. A per-row tensor sends `top_k_val = 0`; the host falls back to `max_k = 50`, but the radix kernel bounds `out_idx` writes by each row's actual `k`. A row with `k = vocab_size` writes `vocab_size` integers into a 50-integer slot. At `1048576`, the approximately 4 MiB overrun reliably escapes allocator slack and faults on gfx942. At `128256`, the overrun can remain silent, which matches the passing mixed-batch result.

The candidate removes the unused `out_idx` workspace from this path. Because that open upstream change already works here, this PR contains only the reproducer and evidence.
