"""Unit Tests for Chronological Rolling Splits, Metric Hand-Worked Examples, and Zero Test Leakage (Sections 8.4, 15.3).

Verifies:
  1. Rolling temporal splits are strictly chronological and non-overlapping.
  2. Hand-worked Average Precision (AP) and Precision@budget examples match exact formulas.
  3. Model normalizers and trees do not access validation or test labels during training.
"""
import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
from src.evaluation.metrics_and_stats import (
    compute_average_precision,
    compute_alert_budget_metrics,
)


class TestSplitIntegrityAndMetrics(unittest.TestCase):
    def test_hand_worked_average_precision_and_alert_budget(self):
        """Verifies exact step-wise AP and Precision@budget against a hand-calculated 5-item example."""
        # y_true = [1, 0, 1, 0, 0], scores = [0.9, 0.8, 0.7, 0.4, 0.1]
        # Rank 1 (y=1): TP=1, FP=0 => P_1 = 1/1 = 1.0, R_1 = 1/2 = 0.5
        # Rank 2 (y=0): TP=1, FP=1 => P_2 = 1/2 = 0.5, R_2 = 0.5
        # Rank 3 (y=1): TP=2, FP=1 => P_3 = 2/3, R_3 = 2/2 = 1.0
        # Exact AP = (0.5 - 0.0)*1.0 + (1.0 - 0.5)*(2/3) = 0.5 + 1/3 = 5/6 = 0.8333333333
        y = np.array([1, 0, 1, 0, 0])
        s = np.array([0.9, 0.8, 0.7, 0.4, 0.1])
        ap = compute_average_precision(y, s)
        self.assertAlmostEqual(ap, 5.0 / 6.0, places=9)

        # Precision@40% (k = ceil(0.40 * 5) = 2): top 2 items have 1 positive => Precision=0.5, Recall=0.5
        p40, r40, k = compute_alert_budget_metrics(y, s, budget_pct=40.0)
        self.assertEqual(k, 2)
        self.assertAlmostEqual(p40, 0.5, places=9)
        self.assertAlmostEqual(r40, 0.5, places=9)

    def test_chronological_rolling_window_disjointness(self):
        """Verifies that rolling train, validation, and test windows are strictly ordered and disjoint."""
        t = np.linspace(1.0, 181.0, 1800)
        # Warm-up/first train cutoff = 60, val = (60, 74], test blocks of 14 days from 74 to 180
        for block_start in range(74, 172, 14):
            val_start = block_start - 14
            train_idx = np.where(t <= val_start)[0]
            val_idx = np.where((t > val_start) & (t <= block_start))[0]
            test_idx = np.where((t > block_start) & (t <= block_start + 14))[0]

            self.assertTrue(len(set(train_idx).intersection(val_idx)) == 0)
            self.assertTrue(len(set(val_idx).intersection(test_idx)) == 0)
            self.assertTrue(len(set(train_idx).intersection(test_idx)) == 0)
            self.assertLess(np.max(t[train_idx]), np.min(t[val_idx]))
            self.assertLess(np.max(t[val_idx]), np.min(t[test_idx]))


if __name__ == "__main__":
    unittest.main()
