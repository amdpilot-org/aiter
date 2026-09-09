import pytest
import torch

from aiter.ops.triton.moe.moe_routing.routing import (
    _compute_expt_data_internal,
    ep_sort_routing,
    routing,
)
from aiter.ops.triton.moe.moe_routing.topk import grouped_topk


def _make_logits(selected_experts, num_tokens, num_global_experts, device):
    logits = torch.full(
        (num_tokens, num_global_experts),
        -16.0,
        dtype=torch.float32,
        device=device,
    )
    for expert_id, logit in zip(selected_experts, [8.0, 7.0, 6.0, 5.0, 4.0]):
        logits[:, expert_id] = logit
    return logits


def _run_exp_routing_case(
    selected_experts,
    expert_map,
    num_expert_group,
    topk_group,
    device,
):
    num_tokens = 8
    num_global_experts = expert_map.numel()
    num_local_experts = int((expert_map >= 0).sum().item())
    topk = len(selected_experts)
    logits = _make_logits(selected_experts, num_tokens, num_global_experts, device)
    expert_group = (
        torch.arange(num_global_experts, device=device, dtype=torch.int32)
        // (num_global_experts // num_expert_group)
    )

    global_weights, global_ids, _ = grouped_topk(
        logits,
        topk,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        score_mode="sigmoid",
        renorm=True,
    )
    routing_data, gather_indx, scatter_indx = routing(
        logits,
        topk,
        score_mode="sigmoid",
        renorm=True,
        use_grouped_topk=True,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        expert_group=expert_group,
    )

    expected_global_hist = torch.bincount(
        global_ids.reshape(-1).to(torch.int64), minlength=num_global_experts
    ).to(torch.int32)
    assert torch.equal(routing_data.expt_hist, expected_global_hist)
    assert torch.equal(
        gather_indx.to(torch.int64)[scatter_indx.to(torch.int64)],
        torch.arange(num_tokens * topk, device=device),
    )

    dispatch_ids = global_ids.to(torch.int32).contiguous()
    dispatch_weights = global_weights.to(torch.float32).contiguous()
    num_gates = num_tokens * topk
    block_m = 16
    expt_data_bufs = _compute_expt_data_internal(
        num_local_experts, num_gates, block_m, device
    )
    local_hist, local_gather, local_scatter, local_gate_scal, gate_valid, _ = (
        ep_sort_routing(
            dispatch_weights,
            dispatch_ids,
            expert_map,
            num_local_experts,
            None,
            num_tokens,
            topk,
            num_gates,
            expt_data_bufs,
        )
    )

    flat_global_ids = dispatch_ids.reshape(-1)
    flat_global_weights = dispatch_weights.reshape(-1)
    flat_gate_valid = gate_valid.reshape(-1).to(torch.bool)
    expected_local_ids = expert_map[flat_global_ids]
    assert torch.equal(flat_gate_valid, expected_local_ids >= 0)
    assert torch.all(expected_local_ids[flat_gate_valid] < num_local_experts)

    expected_local_hist = torch.bincount(
        expected_local_ids[flat_gate_valid].to(torch.int64),
        minlength=num_local_experts,
    ).to(torch.int32)
    assert torch.equal(local_hist[:num_local_experts], expected_local_hist)

    flat_local_scatter = local_scatter.reshape(-1)
    flat_local_gather = local_gather.reshape(-1)
    live_positions = flat_local_scatter[flat_gate_valid]
    assert torch.equal(
        torch.sort(live_positions).values,
        torch.arange(live_positions.numel(), device=device, dtype=torch.int32),
    )
    assert torch.equal(
        torch.sort(flat_local_gather[live_positions.to(torch.int64)]).values,
        torch.nonzero(flat_gate_valid, as_tuple=True)[0].to(torch.int32),
    )
    assert torch.all(flat_local_scatter[~flat_gate_valid] == 0)
    assert torch.equal(
        local_gate_scal[live_positions.to(torch.int64)],
        flat_global_weights[flat_gate_valid],
    )

    hidden_size = 8
    activations = torch.randn(
        (num_tokens, hidden_size), dtype=torch.float32, device=device
    )
    local_weights = torch.randn(
        (num_local_experts, hidden_size, hidden_size),
        dtype=torch.float32,
        device=device,
    )
    reference_output = torch.zeros_like(activations)
    for token_index in range(num_tokens):
        for slot_index in range(topk):
            gate_index = token_index * topk + slot_index
            if not flat_gate_valid[gate_index]:
                continue
            local_expert_id = int(expected_local_ids[gate_index])
            reference_output[token_index] += (
                flat_global_weights[gate_index]
                * (activations[token_index] @ local_weights[local_expert_id])
            )

    mapped_output = torch.zeros_like(activations)
    for token_index in range(num_tokens):
        for slot_index in range(topk):
            gate_index = token_index * topk + slot_index
            if not flat_gate_valid[gate_index]:
                continue
            sorted_position = int(flat_local_scatter[gate_index])
            source_gate_index = int(flat_local_gather[sorted_position])
            assert source_gate_index == gate_index
            global_expert_id = int(flat_global_ids[source_gate_index])
            local_expert_id = int(expert_map[global_expert_id])
            mapped_output[token_index] += (
                local_gate_scal[sorted_position]
                * (activations[token_index] @ local_weights[local_expert_id])
            )

    assert torch.allclose(reference_output, mapped_output)
    return {
        "global_ids": flat_global_ids.reshape(num_tokens, topk),
        "local_ids": expected_local_ids.reshape(num_tokens, topk),
        "gate_valid": flat_gate_valid.reshape(num_tokens, topk),
        "local_hist": local_hist[:num_local_experts],
    }


@pytest.mark.parametrize(
    "case_name,selected_experts,expert_map,num_expert_group,topk_group",
    [
        (
            "nonlocal",
            [0, 1, 4, 5],
            torch.tensor([-1, -1, -1, -1, 0, 1, -1, -1], dtype=torch.int32),
            2,
            2,
        ),
        (
            "identity",
            [0, 2],
            torch.tensor([0, 1, 2, 3], dtype=torch.int32),
            2,
            2,
        ),
        (
            "bounded_local_reference",
            [0, 1, 3],
            torch.tensor([0, 1, 2, -1, -1, -1], dtype=torch.int32),
            2,
            2,
        ),
    ],
)
def test_routing_expert_map(
    case_name, selected_experts, expert_map, num_expert_group, topk_group
):
    if not torch.cuda.is_available():
        pytest.skip("A CUDA/HIP GPU is required")
    device = torch.device("cuda")
    expert_map = expert_map.to(device)
    result = _run_exp_routing_case(
        selected_experts, expert_map, num_expert_group, topk_group, device
    )
    assert case_name in {
        "nonlocal",
        "identity",
        "bounded_local_reference",
    }
    assert torch.all(result["global_ids"] >= 0)
    assert torch.all(result["global_ids"] < expert_map.numel())
    assert torch.all(result["local_ids"][result["gate_valid"]] >= 0)
    assert torch.all(
        result["local_ids"][result["gate_valid"]] < result["local_hist"].numel()
    )
