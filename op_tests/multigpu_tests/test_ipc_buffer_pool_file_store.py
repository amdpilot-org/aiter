import os
import tempfile
from multiprocessing import Pool, freeze_support, set_start_method

import torch
import torch.distributed as dist

from aiter.dist.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    ensure_model_parallel_initialized,
    get_tp_group,
    graph_capture,
    init_distributed_environment,
    set_custom_all_reduce,
)


set_start_method("spawn", force=True)


def ipc_buffer_pool_file_store_worker(world_size, worker_rank, store_path):
    device = torch.device(f"cuda:{worker_rank}")
    torch.cuda.set_device(device)

    store = dist.FileStore(store_path, world_size)
    dist.init_process_group(
        backend="nccl",
        store=store,
        rank=worker_rank,
        world_size=world_size,
    )
    init_distributed_environment(
        world_size=world_size,
        rank=worker_rank,
        local_rank=worker_rank,
        backend="nccl",
    )
    set_custom_all_reduce(True)
    ensure_model_parallel_initialized(
        tensor_model_parallel_size=world_size,
        pipeline_model_parallel_size=1,
    )

    group = get_tp_group()
    custom_allreduce = group.device_communicator.ca_comm
    assert custom_allreduce is not None and not custom_allreduce.disabled

    numel = 1024
    indices = torch.arange(numel, dtype=torch.float32)
    values = (indices * (worker_rank + 1) * 0.125) % 17 - 8
    input_tensor = values.to(device=device, dtype=torch.bfloat16)
    reference = input_tensor.to(torch.float32).contiguous()
    dist.all_reduce(reference, group=group.device_group)

    output = group.all_reduce(input_tensor)
    torch.cuda.synchronize()
    assert torch.allclose(
        output.float(), reference, rtol=2e-2, atol=2e-2
    ), (worker_rank, "eager")

    graph = torch.cuda.CUDAGraph()
    with graph_capture() as capture_context, torch.cuda.graph(
        graph, stream=capture_context.stream
    ):
        graph_output = group.all_reduce(input_tensor)
    graph_output.fill_(0)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.allclose(
        graph_output.float(), reference, rtol=2e-2, atol=2e-2
    ), (worker_rank, "graph")

    destroy_model_parallel()
    destroy_distributed_environment()
    return worker_rank


def test_ipc_buffer_pool_file_store(world_size=2):
    store_path = os.path.join(
        tempfile.mkdtemp(prefix="aiter-ipc-file-store-"), "store"
    )
    pool = Pool(processes=world_size)
    results = [
        pool.apply_async(
            ipc_buffer_pool_file_store_worker,
            args=(world_size, worker_rank, store_path),
        )
        for worker_rank in range(world_size)
    ]
    pool.close()
    pool.join()
    assert [result.get() for result in results] == list(range(world_size))


if __name__ == "__main__":
    freeze_support()
    test_ipc_buffer_pool_file_store()
    print("ipc_buffer_pool_file_store_test_passed")
