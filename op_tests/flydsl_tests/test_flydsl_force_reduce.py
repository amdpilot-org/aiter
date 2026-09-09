import pytest

from aiter.fused_moe import _force_flydsl_stage2_reduce


@pytest.mark.parametrize("enabled", [None, "0", "1"])
def test_force_flydsl_stage2_reduce(enabled):
    v2_atomic = "flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_atomic_persist_nt_sbm16"
    native_atomic = "flydsl_mxmoe_g2_a4w4_16x256x256_atomic"
    native_atomic_nt = "flydsl_mxmoe_g2_a4w4_16x256x256_atomic_nt"
    reduce_name = (
        "flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x256_reduce_persist_nt_sbm16"
    )

    if enabled is None:
        expected_v2 = v2_atomic
        expected_native = native_atomic
        expected_native_nt = native_atomic_nt
    else:
        expected_v2 = reduce_name if enabled == "1" else v2_atomic
        expected_native = (
            "flydsl_mxmoe_g2_a4w4_16x256x256" if enabled == "1" else native_atomic
        )
        expected_native_nt = (
            "flydsl_mxmoe_g2_a4w4_16x256x256_nt" if enabled == "1" else native_atomic_nt
        )

    with pytest.MonkeyPatch.context() as monkeypatch:
        if enabled is None:
            monkeypatch.delenv("AITER_FLYDSL_FORCE_REDUCE", raising=False)
        else:
            monkeypatch.setenv("AITER_FLYDSL_FORCE_REDUCE", enabled)
        assert _force_flydsl_stage2_reduce(v2_atomic) == expected_v2
        assert _force_flydsl_stage2_reduce(native_atomic) == expected_native
        assert _force_flydsl_stage2_reduce(native_atomic_nt) == expected_native_nt
        assert _force_flydsl_stage2_reduce(reduce_name) == reduce_name
