"""Unit Tests for Delay Distribution CDFs, Medians, Bounds, and Posterior Calibration (Sections 9.3, 14.2, 15.3).

Verifies:
  1. D1 truncated exponential on [0, 14] days has exact theoretical median 1.0 day,
     zero survival at a >= 14.0 days, and Monte Carlo median within tolerance.
  2. D2 piecewise-linear CDF reproduces exact knots [(1h, 1%), (1d, 30%), (7d, 60%), (10d, 80%), (30d, 100%)].
  3. D3 mixture (80% D2 + 20% Uniform[30, 90]) has non-zero survival at 30 days (S(30)=0.20)
     and zero survival at M=90 days.
  4. Numerical posterior q(a) matches empirical hidden fraud fraction among unreported rows in synthetic draws.
"""
import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
from src.delay_simulation.delay_models import (
    delay_cdf,
    delay_survival,
    sample_fraud_delays,
    compute_posterior_q,
)


class TestReproducibleDelayDraws(unittest.TestCase):
    def test_d1_truncated_exponential_bounds_and_median(self):
        self.assertAlmostEqual(float(delay_cdf(0.0, "D1")), 0.0, places=10)
        self.assertAlmostEqual(float(delay_cdf(1.0, "D1")), 0.50, places=9)
        self.assertAlmostEqual(float(delay_cdf(14.0, "D1")), 1.00, places=10)
        self.assertEqual(float(delay_survival(14.0, "D1")), 0.0)

        draws = sample_fraud_delays(50000, "D1", seed=12345)
        self.assertTrue(np.all((draws >= 0.0) & (draws <= 14.0)))
        self.assertAlmostEqual(float(np.median(draws)), 1.0, delta=0.025)

    def test_d2_piecewise_linear_knots(self):
        self.assertAlmostEqual(float(delay_cdf(1.0 / 24.0, "D2")), 0.01, places=9)
        self.assertAlmostEqual(float(delay_cdf(1.0, "D2")), 0.30, places=9)
        self.assertAlmostEqual(float(delay_cdf(7.0, "D2")), 0.60, places=9)
        self.assertAlmostEqual(float(delay_cdf(10.0, "D2")), 0.80, places=9)
        self.assertAlmostEqual(float(delay_cdf(30.0, "D2")), 1.00, places=9)

    def test_d3_long_tail_survival(self):
        # At 30 days, D2 component is 100% complete (0.80), tail component is at 0% -> CDF(30)=0.80, S(30)=0.20
        self.assertAlmostEqual(float(delay_cdf(30.0, "D3")), 0.80, places=9)
        self.assertAlmostEqual(float(delay_survival(30.0, "D3")), 0.20, places=9)
        self.assertAlmostEqual(float(delay_survival(90.0, "D3")), 0.00, places=9)

    def test_posterior_q_empirical_calibration(self):
        """Checks that q(a) matches empirical hidden fraud fraction among unreported rows."""
        rng = np.random.default_rng(2026)
        n = 200000
        pi = 0.04
        y = (rng.uniform(0.0, 1.0, size=n) < pi).astype(np.int32)
        ages = rng.uniform(0.0, 30.0, size=n)
        delays = np.full(n, np.inf)
        n_f = int(np.sum(y == 1))
        delays[y == 1] = sample_fraud_delays(n_f, "D2", seed=99)

        unreported = (delays > ages)
        q_pred = compute_posterior_q(ages[unreported], pi, "D2", maturity_horizon_days=30.0)
        y_unrep = y[unreported]
        # Mean predicted posterior across unreported pool must match empirical hidden fraud rate
        self.assertAlmostEqual(float(np.mean(q_pred)), float(np.mean(y_unrep)), delta=0.002)


if __name__ == "__main__":
    unittest.main()
