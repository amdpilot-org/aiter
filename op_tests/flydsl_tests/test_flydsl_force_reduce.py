import pytest

from aiter.fused_moe import _force_flydsl_stage2_reduce


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
