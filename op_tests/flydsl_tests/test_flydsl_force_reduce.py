import pytest

from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import _force_flydsl_stage2_reduce, get_2stage_cfgs


@pytest.mark.parametrize("force", [False, True])
def test_force_flydsl_stage2_reduce(force):
    v2_atomic = "flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_persist_nt_sbm16"
    native_atomic = "flydsl_mxmoe_g2_a4w4_16x256x256_atomic"
    native_atomic_nt = "flydsl_mxmoe_g2_a4w4_16x256x256_atomic_nt"
    reduce_name = (
        "flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_reduce_persist_nt_sbm16"
    )

    if not force:
        expected_v2 = v2_atomic
        expected_native = native_atomic
        expected_native_nt = native_atomic_nt
    else:
        expected_v2 = reduce_name
        expected_native = "flydsl_mxmoe_g2_a4w4_16x256x256"
        expected_native_nt = "flydsl_mxmoe_g2_a4w4_16x256x256_nt"

    assert _force_flydsl_stage2_reduce(v2_atomic, force=force) == expected_v2
    assert _force_flydsl_stage2_reduce(native_atomic, force=force) == expected_native
    assert (
        _force_flydsl_stage2_reduce(native_atomic_nt, force=force)
        == expected_native_nt
    )
    assert _force_flydsl_stage2_reduce(reduce_name, force=force) == reduce_name


def test_get_2stage_cfgs_force_reduce_cache_key():
    get_2stage_cfgs.cache_clear()
    args = (
        1024,
        6144,
        512,
        257,
        9,
        dtypes.bf16,
        dtypes.fp4x2,
        dtypes.fp4x2,
        QuantType.per_1x32,
        True,
        ActivationType.Silu,
        False,
        0,
        0,
    )
    kwargs = {
        "is_shuffled": True,
        "gate_mode": "separated",
        "is_ep": False,
        "has_stage2_bias": False,
        "opus_weights_shuffled": True,
        "config_file": None,
    }

    atomic = get_2stage_cfgs(*args, force_reduce=False, **kwargs)
    reduce = get_2stage_cfgs(*args, force_reduce=True, **kwargs)
    atomic_again = get_2stage_cfgs(*args, force_reduce=False, **kwargs)

    assert atomic is not reduce
    assert atomic_again is atomic
    assert "_atomic_" in atomic.stage2.keywords["kernelName"]
    assert "_reduce_" in reduce.stage2.keywords["kernelName"]
    cache_info = get_2stage_cfgs.cache_info()
    assert cache_info.hits == 1
    assert cache_info.misses == 2
    assert cache_info.currsize == 2
