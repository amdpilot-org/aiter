import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F


ATOL = 5e-2
RTOL = 5e-2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Probe cold and warm HIP-graph replays of fused AR+RMSNorm."
    )
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=6144)
    parser.add_argument("--stage", choices=("1stage", "2stage"), required=True)
    parser.add_argument("--replays", type=int, default=3)
    parser.add_argument("--idle-seconds", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=4888)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def deterministic_tensor(shape, seed, rank, dtype, scale):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed * 100003 + rank)
    values = torch.randn(shape, generator=generator, dtype=torch.float32)
    return (values * scale).to(dtype)


def tensor_checksum(tensor):
    if tensor.element_size() == 2:
        raw = tensor.detach().contiguous().view(torch.uint16)
    elif tensor.element_size() == 4:
        raw = tensor.detach().contiguous().view(torch.uint32)
    else:
        raw = tensor.detach().contiguous().view(torch.uint8)
    return hashlib.sha256(raw.cpu().numpy().tobytes()).hexdigest()[:16]


def error_stats(actual, reference):
    difference = (actual.float() - reference.float()).abs()
    close = torch.isclose(actual.float(), reference.float(), atol=ATOL, rtol=RTOL)
    return {
        "max_abs_error": difference.max().item(),
        "mean_abs_error": difference.mean().item(),
        "mismatches": close.logical_not().sum().item(),
        "elements": close.numel(),
        "passed": bool(close.all().item()),
    }


def probe_record(
    replay_index,
    seed,
    input_buffer,
    residual_buffer,
    residual_output,
    norm_output,
    group,
):
    reference_allreduce = input_buffer.clone()
    dist.all_reduce(reference_allreduce, group=group)
    reference_residual = reference_allreduce + residual_buffer
    reference_norm = F.rms_norm(
        reference_residual,
        (input_buffer.shape[-1],),
        weight=torch.ones_like(input_buffer[0]),
        eps=1e-6,
    )
    actual_allreduce = residual_output.float() - residual_buffer.float()
    return {
        "replay_index": replay_index,
        "phase": "cold" if replay_index == 0 else f"warm_{replay_index}",
        "seed": seed,
        "input_checksum": tensor_checksum(input_buffer),
        "residual_input_checksum": tensor_checksum(residual_buffer),
        "allreduce": error_stats(actual_allreduce, reference_allreduce),
        "residual": error_stats(residual_output, reference_residual),
        "norm": error_stats(norm_output, reference_norm),
        "allreduce_checksum": tensor_checksum(actual_allreduce.to(input_buffer.dtype)),
        "residual_output_checksum": tensor_checksum(residual_output),
        "norm_output_checksum": tensor_checksum(norm_output),
    }


def main():
    args = parse_args()
    os.environ["AITER_AR_1STAGE"] = "1" if args.stage == "1stage" else "0"

    from aiter.dist.communication_op import (
        tensor_model_parallel_fused_allreduce_rmsnorm,
    )
    from aiter.dist.device_communicators.communicator_cuda import CudaCommunicator
    from aiter.dist.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        ensure_model_parallel_initialized,
        graph_capture,
        get_tp_group,
        init_distributed_environment,
        set_custom_all_reduce,
    )

    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    assert world_size == 8, f"this probe requires 8 GPUs, got {world_size}"
    assert torch.cuda.device_count() == 8, (
        f"this probe requires 8 visible GPUs, got {torch.cuda.device_count()}"
    )
    assert CudaCommunicator._ar_1stage_override == (args.stage == "1stage")

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dtype = torch.bfloat16
    shape = (args.tokens, args.hidden)
    input_buffer = torch.empty(shape, dtype=dtype, device=device)
    residual_buffer = torch.empty(shape, dtype=dtype, device=device)
    weight = torch.ones(args.hidden, dtype=dtype, device=device)

    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        distributed_init_method="env://",
        local_rank=local_rank,
    )
    ensure_model_parallel_initialized(world_size, 1)
    group = get_tp_group().device_group

    initial_seed = args.seed + args.tokens
    input_buffer.copy_(
        deterministic_tensor(shape, initial_seed, rank, dtype, 0.2).to(device)
    )
    residual_buffer.copy_(
        deterministic_tensor(shape, initial_seed + 1, rank, dtype, 0.1).to(device)
    )

    graph = torch.cuda.CUDAGraph()
    with graph_capture() as capture_context, torch.cuda.graph(
        graph, stream=capture_context.stream
    ):
        norm_output, residual_output = tensor_model_parallel_fused_allreduce_rmsnorm(
            input_buffer,
            residual_buffer,
            weight,
            1e-6,
        )

    torch.cuda.synchronize()
    if args.idle_seconds:
        time.sleep(args.idle_seconds)

    records = []
    for replay_index in range(args.replays):
        probe_seed = initial_seed + replay_index * 2
        if replay_index:
            input_buffer.copy_(
                deterministic_tensor(shape, probe_seed, rank, dtype, 0.2).to(device)
            )
            residual_buffer.copy_(
                deterministic_tensor(shape, probe_seed + 1, rank, dtype, 0.1).to(
                    device
                )
            )
        norm_output.fill_(0)
        residual_output.fill_(0)
        graph.replay()
        torch.cuda.synchronize()
        records.append(
            probe_record(
                replay_index,
                probe_seed,
                input_buffer,
                residual_buffer,
                residual_output,
                norm_output,
                group,
            )
        )

    gathered_records = [None] * world_size
    dist.all_gather_object(gathered_records, records)
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result = {
            "issue": "ROCm/aiter#4888",
            "world_size": world_size,
            "tokens": args.tokens,
            "hidden": args.hidden,
            "dtype": str(dtype),
            "stage": args.stage,
            "requested_use_1stage": args.stage == "1stage",
            "seed": args.seed,
            "input_bytes": input_buffer.numel() * input_buffer.element_size(),
            "dispatch_boundary_bytes": 128 * 1024,
            "idle_seconds": args.idle_seconds,
            "replays": args.replays,
            "tolerance": {"atol": ATOL, "rtol": RTOL},
            "torch_version": torch.__version__,
            "hip_version": torch.version.hip,
            "device_name": torch.cuda.get_device_name(device),
            "ranks": gathered_records,
        }
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    destroy_model_parallel()
    destroy_distributed_environment()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
