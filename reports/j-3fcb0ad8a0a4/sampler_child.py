#!/usr/bin/env python3
import argparse
import faulthandler
import functools
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback


def emit(prefix, payload):
    record = {"time_ns": time.time_ns(), "pid": os.getpid()}
    record.update(payload)
    print(f"{prefix} {json.dumps(record, sort_keys=True)}", flush=True)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as artifact:
        for block in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def instrument_utils(rank):
    import csrc.cpp_itfs.utils as utils

    original_not_built = utils.not_built
    original_compile_lib = utils.compile_lib
    original_mp_lock = utils.mp_lock
    original_run_lib = utils.run_lib

    def not_built(folder):
        result = original_not_built(folder)
        emit(
            "EVENT_JSON",
            {
                "rank": rank,
                "event": "exists_check",
                "folder": folder,
                "missing": result,
            },
        )
        return result

    def compile_lib(*args, **kwargs):
        folder = args[1] if len(args) > 1 else kwargs.get("folder")
        emit("EVENT_JSON", {"rank": rank, "event": "compile_start", "folder": folder})
        result = original_compile_lib(*args, **kwargs)
        emit("EVENT_JSON", {"rank": rank, "event": "compile_end", "folder": folder})
        return result

    def mp_lock(*args, **kwargs):
        arguments = list(args)
        lock_path = arguments[0] if arguments else kwargs.get("lock_path")
        if len(arguments) > 1:
            main_func = arguments[1]
        else:
            main_func = kwargs["main_func"]

        def instrumented_main():
            emit(
                "EVENT_JSON",
                {"rank": rank, "event": "build_start", "lock_path": lock_path},
            )
            result = main_func()
            emit(
                "EVENT_JSON",
                {"rank": rank, "event": "build_end", "lock_path": lock_path},
            )
            return result

        if len(arguments) > 1:
            arguments[1] = instrumented_main
        else:
            kwargs["main_func"] = instrumented_main
        emit("EVENT_JSON", {"rank": rank, "event": "lock_start", "lock_path": lock_path})
        result = original_mp_lock(*arguments, **kwargs)
        emit("EVENT_JSON", {"rank": rank, "event": "lock_end", "lock_path": lock_path})
        return result

    @functools.wraps(original_run_lib)
    def run_lib(*args, **kwargs):
        func_name = args[0] if args else kwargs.get("func_name")
        folder = kwargs.get("folder")
        if folder is None and len(args) > 1:
            folder = args[1]
        if folder is None:
            folder = func_name
        library_path = Path(utils.BUILD_DIR) / folder / "lib.so"
        emit(
            "EVENT_JSON",
            {
                "rank": rank,
                "event": "load_start",
                "func_name": func_name,
                "library_path": str(library_path),
            },
        )
        result = original_run_lib(*args, **kwargs)
        emit(
            "EVENT_JSON",
            {
                "rank": rank,
                "event": "load_end",
                "func_name": func_name,
                "library_path": str(library_path),
                "sha256": file_sha256(library_path),
                "size": library_path.stat().st_size,
            },
        )
        return result

    utils.not_built = not_built
    utils.compile_lib = compile_lib
    utils.mp_lock = mp_lock
    utils.run_lib = run_lib
    return utils


def make_probability_tensor(batch_size, vocab_size, seed):
    import torch

    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    values = torch.rand(
        (batch_size, vocab_size), device="cuda", generator=generator, dtype=torch.float32
    )
    probabilities = values / values.sum(dim=-1, keepdim=True)
    return probabilities


def joint_valid_mask(probabilities, top_k, top_p, epsilon=1e-4):
    import torch

    batch_size, vocab_size = probabilities.shape
    ascending_order = torch.argsort(probabilities, dim=-1)
    ascending_probabilities = torch.gather(probabilities, 1, ascending_order)
    cumulative = torch.cumsum(ascending_probabilities, dim=-1)
    top_p_mask = torch.zeros_like(probabilities, dtype=torch.int32)
    top_p_mask.scatter_(
        1, ascending_order, (cumulative > (1.0 - top_p) - epsilon).to(torch.int32)
    )
    descending_probabilities, _ = torch.sort(probabilities, dim=-1, descending=True)
    top_k_pivot = descending_probabilities[:, top_k - 1]
    top_k_mask = (
        probabilities >= top_k_pivot.unsqueeze(-1)
    ).to(torch.int32)
    return torch.minimum(top_p_mask, top_k_mask)


def call_sampler(probabilities, top_k_tensor, top_p_tensor):
    import torch

    return torch.ops.aiter.top_k_top_p_sampling_from_probs(
        probabilities,
        None,
        top_k_tensor,
        0,
        top_p_tensor,
        0.0,
        deterministic=True,
    )


def validate(probabilities, top_k_tensor, top_p_tensor, seed, trials):
    import torch

    batch_size, vocab_size = probabilities.shape
    top_k = int(top_k_tensor[0].item())
    top_p = float(top_p_tensor[0].item())
    valid_mask = joint_valid_mask(probabilities, top_k, top_p)

    torch.cuda.manual_seed(seed)
    first_output = call_sampler(
        probabilities, top_k_tensor, top_p_tensor
    )
    torch.cuda.synchronize()

    torch.cuda.manual_seed(seed)
    second_output = call_sampler(
        probabilities, top_k_tensor, top_p_tensor
    )
    torch.cuda.synchronize()

    row_indices = torch.arange(batch_size, device="cuda")
    first_sampled_probabilities = probabilities[row_indices, first_output.long()]
    second_sampled_probabilities = probabilities[row_indices, second_output.long()]
    deterministic_equal = torch.equal(first_output, second_output)
    ids_in_range = torch.all((first_output >= 0) & (first_output < vocab_size))
    sampled_in_mask = torch.all(valid_mask[row_indices, first_output.long()] == 1)
    sampled_probabilities_positive = torch.all(first_sampled_probabilities > 0)
    probabilities_valid = torch.all(
        torch.isfinite(probabilities) & (probabilities >= 0)
    ) and torch.allclose(
        probabilities.sum(dim=-1),
        torch.ones(batch_size, device="cuda"),
        rtol=1e-5,
        atol=1e-5,
    )

    torch.cuda.manual_seed(seed + 1)
    samples = torch.empty(
        (trials, batch_size), device="cuda", dtype=torch.int32
    )
    for trial in range(trials):
        samples[trial] = call_sampler(probabilities, top_k_tensor, top_p_tensor)
    torch.cuda.synchronize()

    observed = torch.zeros(
        (batch_size, vocab_size), device="cuda", dtype=torch.float32
    )
    for row in range(batch_size):
        observed[row] = torch.bincount(
            samples[:, row], minlength=vocab_size
        ).to(torch.float32)
    observed /= observed.sum(dim=-1, keepdim=True)
    expected = probabilities * valid_mask
    expected /= expected.sum(dim=-1, keepdim=True)
    total_variation = 0.5 * (observed - expected).abs().sum(dim=-1)
    max_probability_error = (observed - expected).abs().max(dim=-1).values

    return {
        "batch_size": batch_size,
        "vocab_size": vocab_size,
        "top_k": top_k,
        "top_p": top_p,
        "trials": trials,
        "first_ids": first_output.tolist(),
        "second_ids": second_output.tolist(),
        "first_sampled_probabilities": first_sampled_probabilities.tolist(),
        "second_sampled_probabilities": second_sampled_probabilities.tolist(),
        "deterministic_equal": bool(deterministic_equal),
        "ids_in_range": bool(ids_in_range),
        "sampled_in_mask": bool(sampled_in_mask),
        "sampled_probabilities_positive": bool(sampled_probabilities_positive),
        "probabilities_valid": bool(probabilities_valid),
        "total_variation": total_variation.tolist(),
        "max_total_variation": float(total_variation.max()),
        "max_probability_error": max_probability_error.tolist(),
        "max_max_probability_error": float(max_probability_error.max()),
        "distribution_pass": bool(total_variation.max() < 0.2),
        "distribution_applicable": trials >= 256,
        "valid_mask_counts": valid_mask.sum(dim=-1).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--go-file", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=4096)
    parser.add_argument("--go-timeout", type=float, default=120.0)
    args = parser.parse_args()

    faulthandler.enable()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.ready_file.parent.mkdir(parents=True, exist_ok=True)
    args.go_file.parent.mkdir(parents=True, exist_ok=True)

    import torch

    torch.cuda.set_device(0)
    device = torch.cuda.get_device_properties(0)
    utils = instrument_utils(args.rank)
    import aiter.ops.sampling  # noqa: F401

    emit(
        "EVENT_JSON",
        {
            "rank": args.rank,
            "event": "ready",
            "device_name": device.name,
            "device_uuid": str(device.uuid),
            "gcn_arch_name": device.gcnArchName,
            "build_dir": utils.BUILD_DIR,
            "torch_version": torch.__version__,
            "hip_version": torch.version.hip,
        },
    )
    args.ready_file.write_text(str(os.getpid()))

    deadline = time.monotonic() + args.go_timeout
    while not args.go_file.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"go file not created: {args.go_file}")
        time.sleep(0.01)

    started = time.perf_counter()
    probabilities = make_probability_tensor(16, 248320, 0x5EED0000 + args.rank)
    top_k_tensor = torch.full((16,), 50, device="cuda", dtype=torch.int32)
    top_p_tensor = torch.full((16,), 0.95, device="cuda", dtype=torch.float32)
    result = validate(
        probabilities,
        top_k_tensor,
        top_p_tensor,
        0x5EED1000 + args.rank,
        args.trials,
    )
    result.update(
        {
            "rank": args.rank,
            "physical_gpu": args.rank,
            "device_uuid": str(device.uuid),
            "elapsed_seconds": time.perf_counter() - started,
            "library_path": str(
                Path(utils.BUILD_DIR)
                / "top_k_top_p_sampling_from_probs_51f5779b2644c0a61347d39ea3295db7"
                / "lib.so"
            ),
        }
    )
    if Path(result["library_path"]).exists():
        result["library_sha256"] = file_sha256(result["library_path"])
        result["library_size"] = Path(result["library_path"]).stat().st_size
    required_checks = [
        "deterministic_equal",
        "ids_in_range",
        "sampled_in_mask",
        "sampled_probabilities_positive",
        "probabilities_valid",
    ]
    if result["distribution_applicable"]:
        required_checks.append("distribution_pass")
    result["contract_pass"] = all(result[key] for key in required_checks)
    emit("RESULT_JSON", result)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        emit(
            "ERROR_JSON",
            {
                "rank": int(sys.argv[sys.argv.index("--rank") + 1]),
                "traceback": traceback.format_exc(),
            },
        )
        raise
