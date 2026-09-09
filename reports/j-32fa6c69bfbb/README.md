# fused_qk_norm_rope_cache_pts_quant_shuffle packed-stride validation

## Result

The stride/addressing defect described by ROCm/aiter issue 4356 is already fixed
in this mirror by upstream commit `97f5bf71acb0002bd08b97310e6e147390ba8344`
(PR 4312). No operator source change is needed. This report adds an independent
synthetic validation harness and records gfx942 results both at that exact fix
commit and at current mirror `main`.

The fix replaces the old contiguous-within-block cache guard with an innermost
`head_size` contiguity guard and passes block, token, and head strides into the
native kernel. The packed layout used here is
`[num_blocks, num_heads_kv, block_size, 2 * head_size]`, with K and V taken as
innermost-dim slices and permuted to the operator's
`[num_blocks, block_size, num_heads_kv, head_size]` logical shape.

## Environment

- GPU: AMD Instinct MI300X, Device ID `0x74a1`, GUID `9845`, gfx `gfx942`.
- Image: `amdpilotv2/open-job-mi300:jit-config-readable-35122-260909`, local
  image ID `sha256:dfc9419089c338b5712da4841768b38b1ab79f3da41f8c58c3cd4dfcc1147ff1`.
- Python: `/opt/venv/bin/python` 3.10.12.
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`.
- Working clone: `/job/aiter` at `f0321c0e8927d1d90a29385433f71e592b1c51f5`.
- Python wrapper: `aiter/ops/fused_qk_norm_rope_cache_quant.py`.
- Public caller: `aiter/rotary_embedding.py`.
- Native dispatch: `csrc/kernels/fused_qk_norm_rope_cache_quant.cu`.
- Native kernel: `csrc/kernels/rope/rope_common.h`.
- Native declaration: `csrc/include/fused_qk_norm_rope_cache_quant.h`.
- Built native module: `/job/aiter/aiter/jit/module_fused_qk_norm_rope_cache_quant_shuffle.so`.

## Validation

The harness uses synthetic Q/K/V tensors only. It independently implements
RMSNorm in float32, partial NeoX/GPT-J RoPE in float32, and scaled fp8
quantization separately from the operator. It runs the same logical data through
a contiguous cache and a packed strided cache, compares both views bit-for-bit,
and checks that unused slots, an extra contiguous block, and packed padding
lanes retain sentinel value `57`.

Realistic shapes use 32 query heads, 8 KV heads, 17–19 tokens, 4 blocks, and
block size 16. Cases cover bf16 head sizes 128 and 256, NeoX and GPT-J RoPE,
and gfx942-qualified `float8_e4m3fnuz` head size 128.

### Numerical gates

- Contiguous versus packed-strided parity: exact `torch.equal`.
- Sentinel regions: exact equality to `57`.
- Independent bf16 reference: `rtol=1e-2`, `atol=0.05`.
- Independent fp8 reference: `rtol=0`, `atol=0.0625`.

### Raw results

| Case | View | Max abs error | Max rel error | Pass |
|---|---|---:|---:|---|
| bf16, D=128, NeoX | contiguous | 0 | 0 | PASS |
| bf16, D=128, NeoX | packed strided | 0 | 0 | PASS |
| bf16, D=256, GPT-J | contiguous | 0.00048828125 | 0.005102040711790323 | PASS |
| bf16, D=256, GPT-J | packed strided | 0.00048828125 | 0.005102040711790323 | PASS |
| fp8_e4m3fnuz, D=128, NeoX | contiguous | 0.0009765625 | 0.004739336669445038 | PASS |
| fp8_e4m3fnuz, D=128, NeoX | packed strided | 0.0009765625 | 0.004739336669445038 | PASS |

All quantized K/V outputs and cache writes matched the independent quantization
reference exactly in these runs. All view-parity and sentinel checks passed.
Complete per-output metrics for current `main` are in `results.json`; the exact
fix-commit run is preserved in `fix_commit_results.json`. Both runs produced
the same maxima shown above and passed every view, reference, and sentinel gate.

## Reproduction

```bash
cd /job/aiter
/opt/venv/bin/python reports/j-32fa6c69bfbb/validate_packed_strides.py
```

The first invocation JIT-builds the operator module under the job-private
`aiter/jit/build` directory. No full model weights or framework stack are
downloaded.

## Commands recorded

- `git clone --depth 50 https://github.com/amdpilot-org/aiter.git /job/aiter`
- `git fetch --deepen=500 origin main`
- `rocm-smi --showproduct --showid`
- `/opt/venv/bin/python reports/j-32fa6c69bfbb/validate_packed_strides.py`
- `git worktree add --detach /job/aiter-fix-97f5 97f5bf71acb0002bd08b97310e6e147390ba8344`
- `/opt/venv/bin/python /job/aiter-fix-97f5/reports/j-32fa6c69bfbb/validate_packed_strides.py`

## Context

- ROCm/aiter issue 4356 requested stride-aware packed KV-cache support and had
  no comments when read through the public GitHub API.
- amdpilot-org/aiter issue 68 tracks this validation.
- Upstream PR 4312 / commit `97f5bf71acb0002bd08b97310e6e147390ba8344` is the
  already-merged fix tested here.
- No upstream issue, PR, or comment was posted or modified.
