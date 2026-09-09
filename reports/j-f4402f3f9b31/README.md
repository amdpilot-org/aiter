# DSV4 A8W8 blockscale B-preshuffle investigation

## Result

The DSV4 gfx942 tuned rows were keyed for 80 CUs even though the assigned
MI300X reports 304 CUs. All 105 gfx942 rows in
`aiter/configs/model_configs/dsv4_a8w8_blockscale_bpreshuffle_tuned_gemm.csv`
now use `cu_num=304`. Only that field changed: kernel IDs, kernel names,
split-K settings, timings, and every gfx1250/gfx950 row are preserved.

The reported Opus binding failure did not reproduce on the tested revision and
qualified software stack. The real Opus path built kid 11000 and passed the
same numerical gate, so no Opus code change was made.

## Environment

- Qualified image: `amdpilotv2/open-job-mi300x:jit-config-readable-260909-banff5`,
  image ID `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`.
- Tested AITER revision: `f0321c0e8927d1d90a29385433f71e592b1c51f5` from
  `amdpilot-org/aiter`, checkout `/job/aiter`.
- No preinstalled AITER checkout was found; `/opt/tilelang` is unrelated
  source context.
- Python: `/opt/venv/bin/python` 3.10.12.
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`.
- ROCm/HIP runtime: `7.2.26015-fc0010cf6a`.
- GPU: one AMD Instinct MI300X, `gfx942`, 304 CUs.
- Job-private JIT cache: `/job/.cache/aiter-jit`.
- Native modules used: `/job/.cache/aiter-jit/module_gemm_a8w8_blockscale_bpreshuffle.so`,
  `/job/.cache/aiter-jit/module_gemm_a8w8_blockscale_bpreshuffle_asm.so`, and
  `/job/.cache/aiter-jit/module_deepgemm_opus.so`.
- ASM code objects came from `/job/aiter/hsa/gfx942/fp8gemm_blockscale/`.

## Row verification

Before the fix, `get_CKGEMM_config()` returned no configuration for every
tested DSV4 shape because its runtime key was `(gfx942, 304, M, N, K)` while
the CSV contained `(gfx942, 80, M, N, K)`.

After the key correction:

- All 105 gfx942 rows are selected by the real `get_CKGEMM_config()` path.
- The selected `libtype`, `kernelId`, and `kernelName` match each CSV row.
- 192 gfx1250/256-CU rows and 745 gfx950/256-CU rows remain unchanged.
- A field-by-field diff audit confirms only `cu_num` changed, and only on the
  105 gfx942 rows.

## Bounded numerical validation

Each case used the real selector and `gemm_a8w8_blockscale_bpreshuffle()` path,
a `(16,16)` B-preshuffle, and an independent dequantized float32 Torch
`F.linear` reference. The accuracy gate remained `rtol=1e-2, atol=1e-2`.

| M | N | K | Backend | Kernel ID | Max abs error | Mean abs error | Result |
|---:|---:|---:|---|---:|---:|---:|---|
| 16 | 1536 | 4096 | CK | 7 | 0.015625 | 6.35783e-07 | PASS |
| 128 | 1536 | 4096 | ASM | 0 | 0.03125 | 0.00773108 | PASS |
| 1408 | 1536 | 4096 | Opus | 11000 | 0.03125 | 0.00756926 | PASS |
| 16 | 8192 | 1024 | CK | 7 | 0.00390625 | 7.45058e-08 | PASS |
| 128 | 8192 | 1024 | ASM | 5 | 0.0078125 | 0.00180619 | PASS |
| 256 | 8192 | 1024 | Opus | 11000 | 0.0078125 | 0.0017444 | PASS |

## Opus diagnosis

The generator prints `baked 0 tuned entries` for
`opus_gemm_lookup.h`, but that runtime shape table is for the A16W16 family.
A8W8 blockscale B-preshuffle uses the separate ID-based
`opus_gemm_a8w8_tune_lookup.h`. On this revision that table contains kid
11000, and the real dispatch path resolves and launches it successfully.

The issue-era revision `17f24ec6e` and tested revision both contain the same
A8W8 ID-table dispatch mechanism. The reported
`TypeError: opus_gemm_a8w8_blockscale_bpreshuffle_tune(): incompatible
function arguments` was therefore not reproduced. A deliberate use of
`torch.float8_e4m3fn` instead of the gfx942-supported
`torch.float8_e4m3fnuz` produced a different unsupported-dtype error; using
the supported subtype succeeded. No accuracy gate was relaxed.

Upstream issue 4634 has no comments or linked fix at investigation time. A
GitHub search found only the open issue and the separate gfx950 issue 4735;
no already-landed DSV4 gfx942 correction was found in current source.

## Reproduction

```bash
export PYTHONPATH=/job/aiter
export AITER_JIT_DIR=/job/.cache/aiter-jit
export GPU_ARCHS=gfx942
/opt/venv/bin/python op_tests/test_dsv4_bpreshuffle_config.py
```

The bounded numerical cases were run with the same environment by calling
`get_CKGEMM_config()`, `shuffle_weight(layout=(16,16))`,
`gemm_a8w8_blockscale_bpreshuffle()`, and an independent dequantized Torch
reference. No full-model benchmark was run.
