# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Validate DSV4 B-preshuffle tuned-config hardware keys."""

import os

import pandas as pd


REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.path.join(
    REPO_DIR,
    "aiter",
    "configs",
    "model_configs",
    "dsv4_a8w8_blockscale_bpreshuffle_tuned_gemm.csv",
)


def test_gfx942_rows_use_mi300x_cu_count():
    config = pd.read_csv(CONFIG_PATH)
    gfx942 = config[config["gfx"] == "gfx942"]
    assert not gfx942.empty
    assert (gfx942["cu_num"] == 304).all()


if __name__ == "__main__":
    test_gfx942_rows_use_mi300x_cu_count()
    print("DSV4 gfx942 B-preshuffle config keys: PASS")
