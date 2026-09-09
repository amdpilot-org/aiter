import gc

import pytest
import torch

import aiter.mla


def _capture(pool, function):
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=pool):
        output = function()
    return graph, output


def _check_replayed_indptr(batch_size):
    pool = torch.cuda.graph_pool_handle()
    arguments = (None, batch_size, batch_size * 4096, 16, 1, torch.bfloat16, 1, 1)

    splits, _ = aiter.mla.get_meta_param(*arguments)
    expected = torch.arange(
        0,
        (batch_size + 1) * splits,
        splits,
        dtype=torch.int,
        device="cuda",
    )
    destination = torch.zeros(batch_size + 1, dtype=torch.int32, device="cuda")
    graph, _ = _capture(
        pool,
        lambda: destination.copy_(aiter.mla.get_meta_param(*arguments)[1]),
    )

    for offset in range(1024):
        aiter.mla.get_meta_param(
            None,
            batch_size,
            batch_size * 4096 + offset + 1,
            16,
            1,
            torch.bfloat16,
            1,
            1,
        )

    scribble = torch.zeros(1 << 16, dtype=torch.int32, device="cuda")
    scribble_graph, _ = _capture(
        pool,
        lambda: scribble.copy_(
            torch.full((1 << 16,), -777, dtype=torch.int32, device="cuda")
        ),
    )
    scribble_graph.replay()
    torch.cuda.synchronize()

    destination.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(destination, expected)



@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_get_meta_param_indptr_survives_replay():
    for batch_size in (8, 480):
        _check_replayed_indptr(batch_size)
        gc.collect()
