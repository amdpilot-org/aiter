"""Row-aware scoring for the MoE tuner's cosine-diff acceptance metric."""

import os

import torch

__all__ = [
    "COS_DIFF_THRESHOLD",
    "COS_DIFF_ROW_ENERGY_FLOOR",
    "rowwise_enabled",
    "worst_row_cos_diff",
]


COS_DIFF_THRESHOLD = 1e-1
COS_DIFF_ROW_ENERGY_FLOOR = 1e-3

_ROWWISE_ENV = "AITER_MOE_COS_DIFF_ROWWISE"


def rowwise_enabled():
    """Return whether opt-in per-row cosine-diff scoring is enabled."""

    return os.environ.get(_ROWWISE_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def worst_row_cos_diff(x, y, energy_floor=COS_DIFF_ROW_ENERGY_FLOOR):
    """Return the largest per-row normalized squared error.

    Rows whose reference energy is below ``energy_floor`` times the peak row
    energy are skipped. This prevents an all-but-empty row from rejecting an
    otherwise correct candidate.
    """

    if x.shape != y.shape or x.dim() < 2 or x.shape[-1] < 2:
        return 0.0

    x = x.double().reshape(-1, x.shape[-1])
    y = y.double().reshape(-1, y.shape[-1])
    finite = torch.isfinite(x).all(dim=-1)
    if not finite.any():
        return 0.0
    if not finite.all():
        x = x[finite]
        y = y[finite]
    if not torch.isfinite(y).all():
        return float("inf")
    ref_energy = (x * x).sum(dim=-1)
    result_energy = (y * y).sum(dim=-1)
    peak_energy = ref_energy.max()
    if peak_energy <= 0:
        return 0.0

    keep = ref_energy > peak_energy * energy_floor
    if not keep.any():
        return 0.0

    error = ((x - y) ** 2).sum(dim=-1)[keep]
    energy = (ref_energy + result_energy)[keep]
    return (error / energy.clamp_min(1e-12)).max().item()
