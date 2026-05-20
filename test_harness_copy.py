#!/usr/bin/env python3
import json
import sys
import os
import hashlib
from pathlib import Path

checks_passed = 0
checks_total = 0
check_results = []

def check(name, condition, detail=""):
    global checks_passed, checks_total
    checks_total += 1
    detail = str(detail).strip()
    check_results.append({
        "name": name,
        "passed": bool(condition),
        "detail": detail,
    })
    if condition:
        checks_passed += 1
    status = "PASS" if condition else "FAIL"
    msg = f"  [{status}] {name}"
    if detail and not condition:
        msg += f": {detail}"
    print(msg)
    return condition


def run_repro(iters=1000, warmup=10):
    """Run the WKV shape GEMM many times and hash outputs."""
    import torch
    import aiter
    from aiter.ops.shuffle import shuffle_weight

    M, N, K = 8192, 512, 4096
    block_shape = (128, 128)

    dtype_out = torch.bfloat16
    device = "cuda"

    torch.manual_seed(0)

    # Inputs
    a_fp32 = (torch.randn((M, K), device=device, dtype=torch.float32) * 0.1)
    b_fp32 = (torch.randn((N, K), device=device, dtype=torch.float32) * 0.1)

    # Quantize to fp8 (e4m3)
    fp8_dtype = torch.float8_e4m3fnuz if hasattr(torch, "float8_e4m3fnuz") else torch.float8_e4m3fn
    a_max = a_fp32.abs().max().clamp(min=1e-4)
    b_max = b_fp32.abs().max().clamp(min=1e-4)
    fp8_max = torch.finfo(fp8_dtype).max
    a_scale_global = fp8_max / a_max
    b_scale_global = fp8_max / b_max
    A = (a_fp32 * a_scale_global).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    B = (b_fp32 * b_scale_global).clamp(-fp8_max, fp8_max).to(fp8_dtype)

    # Block scales
    block_n, block_k = block_shape
    scale_n = (N + block_n - 1) // block_n
    scale_k = (K + block_k - 1) // block_k
    scale_m = M  # per-token along M for A, per (block_n, block_k) for B
    # AITER blockscale layout:
    # A scales shape: (scale_k, M) or (M, scale_k); check by trying
    a_scales = torch.full((M, scale_k), 1.0 / a_scale_global.item(), device=device, dtype=torch.float32)
    b_scales = torch.full((scale_n, scale_k), 1.0 / b_scale_global.item(), device=device, dtype=torch.float32)

    # Shuffle B with (16,16) layout
    B_shuffled = shuffle_weight(B, layout=(16, 16))

    out = torch.empty((M, N), device=device, dtype=dtype_out)

    fn = aiter.gemm_a8w8_blockscale_bpreshuffle

    # Try a couple of scale-orderings to find what works
    last_err = None
    call = None
    candidates = [
        ("a_scales,b_scales", (A, B_shuffled, a_scales, b_scales, out)),
        ("a_scales_T,b_scales", (A, B_shuffled, a_scales.t().contiguous(), b_scales, out)),
        ("a_scales,b_scales_T", (A, B_shuffled, a_scales, b_scales.t().contiguous(), out)),
    ]
    for label, args in candidates:
        try:
            fn(*args)
            torch.cuda.synchronize()
            call = args
            print(f"  using arg layout: {label}")
            break
        except Exception as e:
            last_err = e
            continue
    if call is None:
        raise RuntimeError(f"could not call gemm_a8w8_blockscale_bpreshuffle: {last_err}")

    # Warmup
    for _ in range(warmup):
        fn(*call)
    torch.cuda.synchronize()

    hashes = []
    first_hash = None
    sample_unique = []
    for i in range(iters):
        out.zero_()
        fn(*call)
        torch.cuda.synchronize()
        h = hashlib.sha256(out.detach().cpu().numpy().tobytes()).hexdigest()[:16]
        hashes.append(h)
        if first_hash is None:
            first_hash = h
        if h not in sample_unique and len(sample_unique) < 5:
            sample_unique.append(h)

    unique = len(set(hashes))
    return {
        "iters": iters,
        "unique_hashes": unique,
        "first_hash": first_hash,
        "sample_unique": sample_unique,
    }


def main():
    print("=" * 60)
    print("issue-aiter-2-r25 Stage0 baseline reproduction harness")
    print("=" * 60)

    # Static / env
    aiter_path = Path("/sgl-workspace/aiter")
    check("aiter_checkout_present", aiter_path.exists(), f"{aiter_path} missing")

    # Import + GPU check
    have_gpu = False
    try:
        import torch
        have_gpu = torch.cuda.is_available()
        check("torch_imports", True, "")
        check("cuda_available", have_gpu, "no GPU")
        if have_gpu:
            name = torch.cuda.get_device_name(0)
            print(f"  device: {name}")
    except Exception as e:
        check("torch_imports", False, str(e))

    aiter_ok = False
    try:
        import aiter
        aiter_ok = hasattr(aiter, "gemm_a8w8_blockscale_bpreshuffle")
        check("aiter_has_gemm_a8w8_blockscale_bpreshuffle", aiter_ok, "")
    except Exception as e:
        check("aiter_import", False, str(e))

    if not (have_gpu and aiter_ok):
        # Cannot run runtime metric. Save and exit with partial pass.
        Path("/workspace/check_results.json").write_text(json.dumps(check_results, indent=2))
        score = (checks_passed / checks_total * 100.0) if checks_total else 0.0
        print(f"Results: {checks_passed}/{checks_total} checks passed")
        print(f"SCORE: {score:.1f}")
        sys.exit(0 if checks_passed == checks_total else 1)

    # Runtime metric
    iters = int(os.environ.get("ITERS", "1000"))
    warmup = int(os.environ.get("WARMUP", "10"))
    expect_min_unique = int(os.environ.get("EXPECT_MIN_UNIQUE", "2"))

    result = None
    err = None
    try:
        result = run_repro(iters=iters, warmup=warmup)
    except Exception as e:
        import traceback
        err = f"{e}\n{traceback.format_exc()}"

    if check("repro_ran", result is not None, err or ""):
        u = result["unique_hashes"]
        print(f"  unique_hashes={u}/{result['iters']}")
        print(f"  first_hash={result['first_hash']}")
        print(f"  sample_unique={result['sample_unique']}")
        print(f"  metric_unique_hashes={u}")

        # Write metric artifact
        Path("/workspace/metric_unique_hashes.json").write_text(
            json.dumps({
                "metric": "unique_hashes",
                "value": u,
                "iters": result["iters"],
                "first_hash": result["first_hash"],
                "sample_unique": result["sample_unique"],
            }, indent=2)
        )

        check(
            "baseline_nondeterminism_reproduced",
            u >= expect_min_unique,
            f"expected unique_hashes >= {expect_min_unique}, got {u}",
        )

    Path("/workspace/check_results.json").write_text(json.dumps(check_results, indent=2))
    score = (checks_passed / checks_total * 100.0) if checks_total else 0.0
    print(f"Results: {checks_passed}/{checks_total} checks passed")
    print(f"SCORE: {score:.1f}")
    sys.exit(0 if checks_passed == checks_total else 1)


if __name__ == "__main__":
    main()