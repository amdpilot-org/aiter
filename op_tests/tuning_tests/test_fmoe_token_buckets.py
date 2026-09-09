# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import csv
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


AITER_ROOT = Path(__file__).resolve().parents[2]
TUNER_PATH = (
    AITER_ROOT
    / "csrc"
    / "ck_gemm_moe_2stages_codegen"
    / "gemm_moe_tune.py"
)


def _load_tuner():
    spec = importlib.util.spec_from_file_location("fmoe_tuner_under_test", TUNER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestFmoeTokenBuckets(unittest.TestCase):
    KEYS = [
        "gfx",
        "cu_num",
        "token",
        "model_dim",
        "inter_dim",
        "expert",
        "topk",
        "act_type",
        "dtype",
        "q_dtype_a",
        "q_dtype_w",
        "q_type",
        "use_g1u1",
        "doweight_stage1",
    ]

    def _write_csv(self, path, tokens):
        columns = self.KEYS[2:]
        with open(path, "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for token in tokens:
                writer.writerow(
                    {
                        "token": token,
                        "model_dim": 512,
                        "inter_dim": 256,
                        "expert": 4,
                        "topk": 2,
                        "act_type": "ActivationType.Silu",
                        "dtype": "torch.bfloat16",
                        "q_dtype_a": "torch.bfloat16",
                        "q_dtype_w": "torch.bfloat16",
                        "q_type": "QuantType.No",
                        "use_g1u1": 1,
                        "doweight_stage1": 0,
                    }
                )

    def _preprocess(self, tokens, last=False):
        module = _load_tuner()
        tuner = module.FmoeTuner("test", self.KEYS, [], "test")
        tuner.get_cu_num = lambda: 256
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(module, "get_gfx_runtime", return_value="gfx950"),
        ):
            untuned = os.path.join(directory, "untuned.csv")
            tuned = os.path.join(directory, "tuned.csv")
            self._write_csv(untuned, tokens)
            with open(tuned, "w", newline="") as stream:
                csv.DictWriter(stream, fieldnames=self.KEYS).writeheader()
            args = argparse.Namespace(
                all=False,
                last=last,
                untune_file=untuned,
                tune_file=tuned,
            )
            with patch.dict(os.environ, {"AITER_CONFIG_FMOE": tuned}):
                tuner.pre_process(args)
            return tuner.untunedf

    def test_equivalent_token_requests_use_runtime_bucket_once(self):
        untuned = self._preprocess(
            [96, 128, 192, 256, 384, 512, 768, 1024, 32767, 32768, 40000, 131072]
        )
        self.assertEqual(
            untuned["token"].tolist(), [128, 256, 512, 1024, 32768, 131072]
        )
        self.assertFalse(untuned.duplicated(subset=self.KEYS).any())

    def test_last_request_is_normalized_before_tuning(self):
        untuned = self._preprocess([96], last=True)
        self.assertEqual(untuned["token"].tolist(), [128])


if __name__ == "__main__":
    unittest.main()
