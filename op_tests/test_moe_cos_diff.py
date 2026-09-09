import os
import unittest

import torch

from aiter.utility.cos_diff import (
    COS_DIFF_ROW_ENERGY_FLOOR,
    COS_DIFF_THRESHOLD,
    rowwise_enabled,
    worst_row_cos_diff,
)


ROWS, COLS = 512, 256


def _reference():
    torch.manual_seed(0)
    rows = torch.randn(ROWS, COLS, dtype=torch.float64)
    return rows * (8.0 / rows.norm(dim=1, keepdim=True))


def _whole_tensor_diff(x, y):
    x = x.flatten()
    y = y.flatten()
    return 1 - 2 * (x * y).sum().item() / max((x * x + y * y).sum().item(), 1e-12)


class TestMoeCosDiff(unittest.TestCase):
    def test_zeroed_rows_follow_closed_form_and_pass_whole_tensor_gate(self):
        reference = _reference()
        for count in (26, 51, 92):
            result = reference.clone()
            result[:count] = 0.0
            fraction = count / ROWS
            diff = _whole_tensor_diff(reference, result)
            with self.subTest(count=count):
                self.assertAlmostEqual(diff, fraction / (2 - fraction), places=3)
                self.assertLess(diff, COS_DIFF_THRESHOLD)

    def test_row_scoring_rejects_zeroed_rows(self):
        reference = _reference()
        for count in (1, 8, 64):
            result = reference.clone()
            result[:count] = 0.0
            with self.subTest(count=count):
                self.assertEqual(worst_row_cos_diff(reference, result), 1.0)

    def test_row_scoring_rejects_garbage_the_aggregate_accepts(self):
        reference = _reference()
        torch.manual_seed(1)
        result = reference.clone()
        garbage = torch.randn(16, COLS, dtype=torch.float64)
        result[:16] = garbage * (8.0 / garbage.norm(dim=1, keepdim=True))
        self.assertLess(_whole_tensor_diff(reference, result), COS_DIFF_THRESHOLD)
        self.assertGreater(
            worst_row_cos_diff(reference, result), COS_DIFF_THRESHOLD
        )

    def test_row_scoring_rejects_sign_flipped_rows(self):
        reference = _reference()
        result = reference.clone()
        result[: int(ROWS * 0.04)] *= -1.0
        self.assertLess(_whole_tensor_diff(reference, result), COS_DIFF_THRESHOLD)
        self.assertEqual(worst_row_cos_diff(reference, result), 2.0)

    def test_correct_low_energy_row_is_not_scored(self):
        reference = _reference()
        reference[0] = 0.0
        result = reference.clone()
        result[0] = 1e-9
        self.assertLess(worst_row_cos_diff(reference, result), COS_DIFF_THRESHOLD)

    def test_non_finite_reference_padding_is_not_scored(self):
        reference = _reference()
        result = reference.clone()
        reference[0] = float("nan")
        result[0] = float("nan")
        self.assertLess(worst_row_cos_diff(reference, result), COS_DIFF_THRESHOLD)

    def test_non_finite_candidate_row_is_rejected(self):
        reference = _reference()
        result = reference.clone()
        result[0] = float("nan")
        self.assertEqual(worst_row_cos_diff(reference, result), float("inf"))

    def test_correct_rounding_noise_passes_row_scoring(self):
        reference = _reference()
        torch.manual_seed(2)
        result = reference * (
            1.0 + (torch.rand_like(reference) - 0.5) * 2 * 2**-8
        )
        self.assertLess(
            worst_row_cos_diff(reference, result), COS_DIFF_THRESHOLD
        )

    def test_default_and_opt_in_behavior(self):
        os.environ.pop("AITER_MOE_COS_DIFF_ROWWISE", None)
        self.assertFalse(rowwise_enabled())

    def test_row_scoring_never_loosens_the_tuner_result(self):
        reference = _reference()
        torch.manual_seed(3)
        result = reference + torch.randn_like(reference) * 0.1
        whole = _whole_tensor_diff(reference, result)
        self.assertGreaterEqual(max(whole, worst_row_cos_diff(reference, result)), whole)
        self.assertEqual(COS_DIFF_ROW_ENERGY_FLOOR, 1e-3)


if __name__ == "__main__":
    unittest.main()
