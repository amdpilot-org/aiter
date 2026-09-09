# Eight-GPU cold sampler startup investigation

This directory contains a bounded reproduction harness for
`top_k_top_p_sampling_from_probs(deterministic=True)` on MI300X.

The child exercises the registered production operator with a valid
16 x 248320 probability tensor, tensor-valued `top_k=50` and `top_p=0.95`,
and records:

- existence-check, lock, compile, and load ordering;
- library path, size, and SHA-256;
- return code, signal, timeout status, stdout, and stderr;
- exact sampled IDs and probabilities;
- same-seed deterministic replay;
- joint top-k/top-p support validation; and
- a 4,096-trial total-variation distribution check.

Run the matrix with:

```bash
/opt/venv/bin/python reports/j-3fcb0ad8a0a4/run_matrix.py \
  --cache-root /job/aiter-cache-j-3fcb0ad8a0a4 \
  --output-root /job/aiter-matrix-j-3fcb0ad8a0a4 \
  --cold-cycles 6 \
  --trials 4096 \
  --child-timeout 900
```

The cache root must initially be empty. The harness keeps both the
`cpp_itfs` cache (`AITER_ROOT_DIR`) and the Python JIT cache
(`AITER_JIT_DIR`) under that private root.

`compile_start` counts calls to `compile_lib`; `build_start` counts actual
builds by the lock winner. This distinction matters for synchronized cold
starts. After a matrix run, `analyze_matrix.py` compares whole-library,
host-code, embedded-device-code, note, CUID, and flag-order hashes.

See `FINDINGS.md` for the completed eight-GPU result and its limits.
