import json
import math
import os
import sys
import traceback

import torch
from torch.nn.functional import scaled_dot_product_attention


def main():
    torch.manual_seed(20260909)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    query_lengths = [128, 128]
    key_lengths = [128, 128]
    heads = 2
    head_dim = 64
    scale = 1.0 / math.sqrt(head_dim)

    cu_seqlens_q = torch.tensor([0, *query_lengths], dtype=torch.int32, device=device)
    cu_seqlens_k = torch.tensor([0, *key_lengths], dtype=torch.int32, device=device)
    input_scale = 0.01
    q = input_scale * torch.randn(
        sum(query_lengths), heads, head_dim, device=device, dtype=dtype
    )
    k = input_scale * torch.randn(
        sum(key_lengths), heads, head_dim, device=device, dtype=dtype
    )
    v = input_scale * torch.randn(
        sum(key_lengths), heads, head_dim, device=device, dtype=dtype
    )

    import aiter
    from aiter.jit.utils.chip_info import get_gfx

    result = {
        "python": sys.executable,
        "aiter_module": aiter.__file__,
        "gfx": get_gfx(),
        "enable_ck": os.environ.get("ENABLE_CK", "1"),
        "query_lengths": query_lengths,
        "key_lengths": key_lengths,
        "heads": heads,
        "head_dim": head_dim,
        "input_scale": input_scale,
        "dtype": str(dtype),
    }

    try:
        import importlib.metadata

        result["flydsl_version"] = importlib.metadata.version("flydsl")
    except Exception as error:
        result["flydsl_version_error"] = repr(error)

    with torch.no_grad():
        output = aiter.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max(query_lengths),
            max(key_lengths),
            dropout_p=0.0,
            softmax_scale=scale,
            causal=False,
            window_size=(-1, -1, 0),
        )

        reference_chunks = []
        for sequence_index, (query_length, key_length) in enumerate(
            zip(query_lengths, key_lengths)
        ):
            query_start = int(cu_seqlens_q[sequence_index])
            key_start = int(cu_seqlens_k[sequence_index])
            query = q[query_start : query_start + query_length].transpose(0, 1).unsqueeze(0)
            key = k[key_start : key_start + key_length].transpose(0, 1).unsqueeze(0)
            value = v[key_start : key_start + key_length].transpose(0, 1).unsqueeze(0)
            reference_chunks.append(
                scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    dropout_p=0.0,
                    is_causal=False,
                    scale=scale,
                )
                .squeeze(0)
                .transpose(0, 1)
            )
        reference = torch.cat(reference_chunks, dim=0)

    absolute_error = (output.float() - reference.float()).abs()
    result.update(
        {
            "output_shape": list(output.shape),
            "output_dtype": str(output.dtype),
            "reference_shape": list(reference.shape),
            "max_abs_error": float(absolute_error.max()),
            "mean_abs_error": float(absolute_error.mean()),
            "numerical_gate": bool(
                output.shape == reference.shape
                and output.dtype == dtype
                and float(absolute_error.max()) <= 0.003
            ),
        }
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["numerical_gate"]:
        raise SystemExit("numerical gate failed")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
