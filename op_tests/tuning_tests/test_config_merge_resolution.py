import os
import shutil
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

import pandas as pd

try:
    from aiter.jit import core

    _IMPORT_ERR = None
except Exception as e:  # noqa: BLE001
    core = None
    _IMPORT_ERR = e


ENV_NAME = "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE"
MERGE_NAME = "a8w8_blockscale_tuned_gemm"
HEADER = [
    "gfx",
    "cu_num",
    "M",
    "N",
    "K",
    "libtype",
    "kernelId",
    "splitK",
    "us",
    "kernelName",
    "tflops",
    "bw",
    "errRatio",
]


def _ck_row(us):
    return [
        "gfx950",
        256,
        16,
        2048,
        3072,
        "ck",
        8,
        2,
        us,
        "a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1_1_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1",
        20.0,
        700.0,
        0.0,
    ]


def _cktile_row(us):
    return [
        "gfx950",
        256,
        16,
        2048,
        3072,
        "cktile",
        20,
        3,
        us,
        "a8w8_blockscale_cktile_32x128x128_1x4x1_16x16x64_intrawave_0x1x0_4",
        23.39,
        744.13,
        0.0,
    ]


@unittest.skipUnless(core is not None, f"aiter.jit.core not importable: {_IMPORT_ERR}")
class TestConfigMergeResolution(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="aiter_config_merge_")
        self.root = os.path.join(self.tmpdir, "root")
        configs_dir = os.path.join(self.root, "aiter", "configs")
        os.makedirs(configs_dir)
        with open(
            os.path.join(configs_dir, "a8w8_blockscale_untuned_gemm.csv"), "w"
        ) as untuned_file:
            untuned_file.write("M,N,K\n")

        self.source_a = os.path.join(self.tmpdir, "source_a.csv")
        self.source_b = os.path.join(self.tmpdir, "source_b.csv")
        self._write_source(self.source_a, [_ck_row(8.0)])
        self._write_source(self.source_b, [_cktile_row(8.6089)])
        os.chmod(self.source_a, 0o444)
        os.chmod(self.source_b, 0o444)
        self.source_bytes = (
            Path(self.source_a).read_bytes(),
            Path(self.source_b).read_bytes(),
        )

        self.old_root = core.AITER_ROOT_DIR
        self.old_env = os.environ.pop(ENV_NAME, None)
        core.AITER_ROOT_DIR = self.root
        type(core.AITER_CONFIGS).get_config_file.cache_clear()

    def tearDown(self):
        core.AITER_ROOT_DIR = self.old_root
        if self.old_env is None:
            os.environ.pop(ENV_NAME, None)
        else:
            os.environ[ENV_NAME] = self.old_env
        type(core.AITER_CONFIGS).get_config_file.cache_clear()
        os.chmod(self.source_a, 0o644)
        os.chmod(self.source_b, 0o644)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_source(self, path, rows, header=HEADER):
        pd.DataFrame(rows, columns=header).to_csv(path, index=False)

    def _merged_path(self):
        os.environ[ENV_NAME] = f"{self.source_a}{os.pathsep}{self.source_b}"
        with mock.patch.object(core.tempfile, "gettempdir", return_value=self.tmpdir):
            return core.AITER_CONFIGS.get_config_file(
                ENV_NAME, "/unused/default.csv", MERGE_NAME
            )

    def _assert_sources_unchanged(self):
        self.assertEqual(Path(self.source_a).read_bytes(), self.source_bytes[0])
        self.assertEqual(Path(self.source_b).read_bytes(), self.source_bytes[1])

    def test_valid_duplicate_resolves_and_repeats(self):
        merged_path = self._merged_path()
        merged = pd.read_csv(merged_path)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged.iloc[0]["libtype"], "ck")
        self.assertEqual(float(merged.iloc[0]["us"]), 8.0)
        self.assertEqual(self._merged_path(), merged_path)
        pd.testing.assert_frame_equal(merged, pd.read_csv(merged_path))
        self._assert_sources_unchanged()

    def test_equal_us_selects_first_source_row(self):
        os.chmod(self.source_b, 0o644)
        self._write_source(self.source_b, [_cktile_row(8.0)])
        os.chmod(self.source_b, 0o444)
        self.source_bytes = (
            Path(self.source_a).read_bytes(),
            Path(self.source_b).read_bytes(),
        )
        merged = pd.read_csv(self._merged_path())
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged.iloc[0]["libtype"], "ck")
        self._assert_sources_unchanged()

    def test_missing_us_column_refuses(self):
        os.chmod(self.source_a, 0o644)
        os.chmod(self.source_b, 0o644)
        no_us_header = [column for column in HEADER if column != "us"]
        self._write_source(
            self.source_a, [_ck_row(8.0)[:8] + _ck_row(8.0)[9:]], no_us_header
        )
        self._write_source(
            self.source_b,
            [_cktile_row(8.6089)[:8] + _cktile_row(8.6089)[9:]],
            no_us_header,
        )
        os.chmod(self.source_a, 0o444)
        os.chmod(self.source_b, 0o444)
        self.source_bytes = (
            Path(self.source_a).read_bytes(),
            Path(self.source_b).read_bytes(),
        )
        with self.assertRaisesRegex(RuntimeError, "No 'us' column"):
            self._merged_path()
        self._assert_sources_unchanged()

    def test_invalid_us_refuses(self):
        os.chmod(self.source_b, 0o644)
        self._write_source(self.source_b, [_cktile_row("not-a-number")])
        os.chmod(self.source_b, 0o444)
        self.source_bytes = (
            Path(self.source_a).read_bytes(),
            Path(self.source_b).read_bytes(),
        )
        with self.assertRaisesRegex(RuntimeError, "missing or non-numeric"):
            self._merged_path()
        self._assert_sources_unchanged()

    def test_concurrent_readers_get_same_resolved_table(self):
        os.environ[ENV_NAME] = f"{self.source_a}{os.pathsep}{self.source_b}"
        source_paths = os.environ[ENV_NAME]
        merge_name = MERGE_NAME
        with (
            mock.patch.object(core.tempfile, "gettempdir", return_value=self.tmpdir),
            ThreadPoolExecutor(max_workers=4) as executor,
        ):
            paths = list(
                executor.map(
                    lambda _: core.AITER_CONFIGS.update_config_files(
                        source_paths, merge_name
                    ),
                    range(4),
                )
            )
        self.assertEqual(len(set(paths)), 1)
        merged = pd.read_csv(paths[0])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged.iloc[0]["libtype"], "ck")
        self._assert_sources_unchanged()


if __name__ == "__main__":
    unittest.main()
