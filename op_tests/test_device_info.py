import ctypes
import unittest
from unittest import mock

import torch

from aiter.ops.triton.utils import device_info


class FakeHipRuntime:
    def __init__(self, values=None, status=0):
        self.values = values or {}
        self.status = status
        self.device_ids = []

    def hipDeviceGetAttribute(self, value, attribute, device_id):
        self.device_ids.append(device_id)
        result = ctypes.cast(value, ctypes.POINTER(ctypes.c_int)).contents
        result.value = self.values.get(device_id, 8)
        return self.status


class TestGetNumXcds(unittest.TestCase):
    def tearDown(self):
        device_info._get_num_xcds.cache_clear()

    def test_current_visible_device_is_queried(self):
        fake_hip = FakeHipRuntime({0: 8, 1: 6})
        with (
            mock.patch.object(torch.cuda, "current_device", return_value=1),
            mock.patch.object(torch.cuda, "device_count", return_value=2),
            mock.patch.object(device_info.ctypes, "CDLL", return_value=fake_hip),
        ):
            self.assertEqual(device_info.get_num_xcds(), 6)
        self.assertEqual(fake_hip.device_ids, [1])

    def test_cache_is_keyed_by_visible_device(self):
        fake_hip = FakeHipRuntime({0: 8, 1: 6})
        with (
            mock.patch.object(torch.cuda, "device_count", return_value=2),
            mock.patch.object(device_info.ctypes, "CDLL", return_value=fake_hip),
        ):
            self.assertEqual(device_info.get_num_xcds(0), 8)
            self.assertEqual(device_info.get_num_xcds(1), 6)
            self.assertEqual(device_info.get_num_xcds(0), 8)
        self.assertEqual(fake_hip.device_ids, [0, 1])

    def test_invalid_visible_device_is_rejected(self):
        with (
            mock.patch.object(torch.cuda, "device_count", return_value=1),
            mock.patch.object(device_info.ctypes, "CDLL") as load_hip,
        ):
            with self.assertRaises(ValueError):
                device_info.get_num_xcds(1)
        load_hip.assert_not_called()

    def test_unsupported_attribute_falls_back_only_for_valid_device(self):
        for status in (
            device_info._HIP_ERROR_INVALID_VALUE,
            device_info._HIP_ERROR_NOT_SUPPORTED,
        ):
            with self.subTest(status=status):
                fake_hip = FakeHipRuntime(status=status)
                self.assertEqual(device_info._query_num_xcds(0, fake_hip, 6), 8)

    def test_modern_runtime_does_not_fallback_for_unsupported_attribute(self):
        for status in (
            device_info._HIP_ERROR_INVALID_VALUE,
            device_info._HIP_ERROR_NOT_SUPPORTED,
        ):
            with self.subTest(status=status):
                fake_hip = FakeHipRuntime(status=status)
                with self.assertRaises(RuntimeError):
                    device_info._query_num_xcds(0, fake_hip, 7)

    def test_older_runtime_uses_real_measurement_when_available(self):
        fake_hip = FakeHipRuntime({0: 6})
        self.assertEqual(device_info._query_num_xcds(0, fake_hip, 6), 6)

    def test_zero_and_failed_queries_are_errors(self):
        for status, value in ((0, 0), (999, 8)):
            with self.subTest(status=status, value=value):
                fake_hip = FakeHipRuntime({0: value}, status=status)
                with self.assertRaises(RuntimeError):
                    device_info._query_num_xcds(0, fake_hip, 7)

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.version.hip,
        "Requires an available HIP device",
    )
    def test_matches_hip_attribute_for_current_device(self):
        device_info._get_num_xcds.cache_clear()
        getter_value = device_info.get_num_xcds()
        hip_value = ctypes.c_int(0)
        libhip = ctypes.CDLL("libamdhip64.so")
        status = libhip.hipDeviceGetAttribute(
            ctypes.byref(hip_value),
            device_info._HIP_DEVICE_ATTRIBUTE_NUMBER_OF_XCCS,
            torch.cuda.current_device(),
        )
        if status == device_info._HIP_ERROR_INVALID_VALUE:
            self.assertEqual(getter_value, device_info._NUM_XCDS_COMPATIBILITY_FALLBACK)
        else:
            self.assertEqual(status, 0)
            self.assertEqual(getter_value, hip_value.value)
            self.assertGreater(getter_value, 0)


if __name__ == "__main__":
    unittest.main()
