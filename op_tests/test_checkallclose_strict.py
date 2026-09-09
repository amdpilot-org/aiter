# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Behavioural contract for checkAllclose strict mode."""

import os

import torch

from aiter.test_common import checkAllclose

STRICT_ENV = "AITER_STRICT_ALLCLOSE"


def _pair(bad_rows=32, rows=64, cols=128):
    torch.manual_seed(0)
    ref = torch.randn(rows, cols)
    out = ref.clone()
    out[:bad_rows] += 10.0
    return out, ref


def _raises(**kwargs):
    out, ref = _pair()
    try:
        checkAllclose(out, ref, **kwargs)
    except AssertionError:
        return True
    return False


def test_default_logs_and_returns_ratio():
    os.environ.pop(STRICT_ENV, None)
    out, ref = _pair()
    ratio = checkAllclose(out, ref, msg="default: ")
    assert ratio == 0.5, f"expected 0.5 mismatch ratio, got {ratio}"


def test_explicit_strict_raises():
    os.environ.pop(STRICT_ENV, None)
    assert _raises(msg="explicit: ", strict=True)


def test_env_var_enables_strict():
    os.environ[STRICT_ENV] = "1"
    try:
        assert _raises(msg="env: ")
    finally:
        os.environ.pop(STRICT_ENV, None)


def test_env_var_does_not_affect_tuner_path():
    os.environ[STRICT_ENV] = "1"
    try:
        out, ref = _pair()
        ratio = checkAllclose(out, ref, msg="tuner: ", printLog=False)
        assert ratio == 0.5, f"expected 0.5 mismatch ratio, got {ratio}"
    finally:
        os.environ.pop(STRICT_ENV, None)


def test_strict_passes_when_close():
    os.environ[STRICT_ENV] = "1"
    try:
        torch.manual_seed(0)
        ref = torch.randn(64, 128)
        assert checkAllclose(ref.clone(), ref, msg="clean: ") == 0
    finally:
        os.environ.pop(STRICT_ENV, None)


def test_strict_respects_tol_err_ratio():
    os.environ[STRICT_ENV] = "1"
    try:
        torch.manual_seed(0)
        ref = torch.randn(64, 128)
        out = ref.clone()
        out[0, 0] += 10.0
        checkAllclose(out, ref, msg="below-ratio: ")
    finally:
        os.environ.pop(STRICT_ENV, None)


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
            print(f"ok  {name}")
    print("all checkAllclose strict-mode tests passed")
