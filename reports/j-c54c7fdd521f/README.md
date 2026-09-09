# Triton DeepSeek MoE `expert_map` validation

## Result

The global-to-local mapping contract is already fixed in current `main` by upstream PR 4500, commit `10b192f5b5bda90f2af33ceae7a6c2f416bfc674`. No production change was needed and that fix was not duplicated.

The DeepSeek `routing()` entry still emits global expert IDs. Under expert parallelism, the post-all-to-all path uses `ep_sort_routing()` with a global-to-local `expert_map`. The validation confirms that:

- selected global experts owned by this rank map to the expected local IDs;
- nonlocal experts produce invalid gates, zero scatter destinations, and no local histogram entries;
- identity mapping preserves expert IDs;
- a bounded local MoE reference with local weights and hidden size 8 matches the mapped routing output.

## Regression

`op_tests/triton_tests/moe/test_moe_routing_expert_map.py` covers three synthetic cases:

- `nonlocal`: 8 global experts, 2 local experts, and a map containing `-1` for other ranks;
- `identity`: 4 global experts and an identity map;
- `bounded_local_reference`: 6 global experts, 3 local experts, local weights, and hidden size 8.

Each case builds deterministic grouped-top-k logits, runs `routing()`, runs `ep_sort_routing()`, and checks the mapping, histogram, gather/scatter inverse, gate validity, weights, and local-weight reference output.

## GPU evidence

Raw output is preserved in `reports/j-c54c7fdd521f/gpu-evidence.txt`.

- GPU: one AMD Instinct MI300X, `gfx942`, serial `692440003`, unique ID `0x439e01ac3221d888`.
- Image: `amdpilotv2/open-job-mi300:jit-config-readable-260909-banff5`, local ID `sha256:39fe745feda79ecf4c17f4d806d8ef12150bef720f2f07f5c63a20b3ccfd63f1`.
- Python: `/opt/venv/bin/python` 3.10.12.
- Torch: `2.9.1+rocm7.2.0.git7e1940d4`.
- Triton: `3.7.0`.
- Regression command: `HIP_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/job/.triton-cache /opt/venv/bin/python -m pytest -q -s op_tests/triton_tests/moe/test_moe_routing_expert_map.py`.
- Regression result: `3 passed, 2 warnings`.
- Existing routing numerical gate: `8 skipped, 244 deselected`; the upstream test skips non-CDNA4 architectures such as `gfx942`. No numerical threshold or production gate was changed.

## Paths

- Read source: `aiter/ops/triton/moe/moe_routing/routing.py`.
- Read Triton kernel: `aiter/ops/triton/_triton_kernels/moe/moe_routing/routing.py`.
- Added regression: `op_tests/triton_tests/moe/test_moe_routing_expert_map.py`.
- Generated native module: `/job/aiter/aiter/jit/module_aiter_core.so`.
- Generated native build directory: `/job/aiter/aiter/jit/build/module_aiter_core`.
- Job-private Triton cache: `/job/.triton-cache`.

## Upstream context

- ROCm/aiter issue 3995 reports that the Triton DeepSeek fused routing path dropped `expert_map` under expert parallelism.
- ROCm/aiter PR 4500 added the expert-parallel routing sort and was merged on 2026-09-03.
- Current mirror `main` is `f0321c0e8927d1d90a29385433f71e592b1c51f5` and contains the fix.

## Reproduction

```bash
cd /job/aiter
HIP_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/job/.triton-cache \
  /opt/venv/bin/python -m pytest -q -s \
  op_tests/triton_tests/moe/test_moe_routing_expert_map.py
```

## Notes

The runtime has no Docker or Podman CLI, so the image ID above is the qualified local image ID supplied by the task environment. The container marker `/.dockerenv` and the recorded Torch/ROCm/GPU stack are preserved in the raw evidence.
