"""Unit Tests for Label Visibility States, Target Isolation, and Toy Examples (Sections 8.4, 9.4, 15.3).

Verifies:
  1. Future-label invariance: changing an unobserved/future label Y_i when its report_time > tau
     cannot alter any model-facing label state, V0/V1/V2 target, or neighbor feature.
  2. Target-isolation check: a transaction's own final label Y_i never enters its own neighbor features.
  3. Hand-calculated toy example for Visible Positive (state=1), Mature Negative (state=0),
     and Immature/Unreported (state=2) plus exact posterior q(a) edge cases (q=0 at a>=M, q=pi at a=0).
"""
import sys
import os
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
from src.delay_simulation.delay_models import (
    compute_posterior_q,
    reconstruct_label_state_at_cutoff,
)
from src.temporal_graph.causal_graph import extract_relational_features_and_graph_state


class TestLabelVisibilityAndIsolation(unittest.TestCase):
    def test_posterior_edge_cases_and_monotonicity(self):
        """Checks q(0)=pi, q(a>=M)=0, and strict monotonicity on [0, M]."""
        pi = 0.04
        q0 = float(compute_posterior_q(0.0, pi, "D2", maturity_horizon_days=30.0))
        q_mid = float(compute_posterior_q(7.0, pi, "D2", maturity_horizon_days=30.0))
        q_mat = float(compute_posterior_q(30.0, pi, "D2", maturity_horizon_days=30.0))
        q_late = float(compute_posterior_q(45.0, pi, "D2", maturity_horizon_days=30.0))

        self.assertAlmostEqual(q0, pi, places=10)
        self.assertEqual(q_mat, 0.0)
        self.assertEqual(q_late, 0.0)
        # At 7 days under D2, CDF=0.60 => S(7)=0.40 => q(7) = (0.04*0.40)/(0.04*0.40 + 0.96)
        expected_q7 = (0.04 * 0.40) / (0.04 * 0.40 + 0.96)
        self.assertAlmostEqual(q_mid, expected_q7, places=10)
        self.assertTrue(0.0 < q_mid < q0)

    def test_hand_worked_label_states_and_future_label_invariance(self):
        """Verifies state reconstruction at cutoff tau=40.0 and proves hidden final labels cannot leak."""
        # 4 transactions at days [5.0, 20.0, 32.0, 45.0], M=30.0 days, cutoff tau=40.0
        # Row 0 (t=5, age=35 >= 30): no report -> Mature Negative (state 0, y_v2=0.0)
        # Row 1 (t=20, age=20 < 30): report arrived at 25.0 <= 40.0 -> Visible Positive (state 1, y_v2=1.0)
        # Row 2 (t=32, age=8 < 30): report arrives at 44.0 > 40.0 -> Immature (state 2, y_v2=q(8))
        # Row 3 (t=45 > 40): Future transaction (state -1, mask=False)
        event_t = np.array([5.0, 20.0, 32.0, 45.0])
        rep_t = np.array([np.inf, 25.0, 44.0, 46.0])
        true_y_1 = np.array([0, 1, 1, 1])
        true_y_2 = np.array([1, 1, 0, 0])  # Flipping hidden final labels without changing permitted event log

        st1 = reconstruct_label_state_at_cutoff(event_t, rep_t, 40.0, 30.0, "D2", 0.05, true_y_1)
        st2 = reconstruct_label_state_at_cutoff(event_t, rep_t, 40.0, 30.0, "D2", 0.05, true_y_2)

        np.testing.assert_array_equal(st1["state_code"], np.array([0, 1, 2, -1], dtype=np.int8))
        np.testing.assert_array_equal(st1["mask_v1"], np.array([True, True, False, False]))
        np.testing.assert_array_equal(st1["mask_v2"], np.array([True, True, True, False]))
        # Changing true_labels_for_oracle_only must NOT alter y_v0, y_v1, y_v2, or state_code
        np.testing.assert_allclose(st1["y_v0"], st2["y_v0"], atol=1e-12)
        np.testing.assert_allclose(st1["y_v1"], st2["y_v1"], atol=1e-12)
        np.testing.assert_allclose(st1["y_v2"], st2["y_v2"], atol=1e-12)

    def test_target_isolation_in_neighbor_features(self):
        """Verifies that a transaction's own label never enters its own neighbor features."""
        df = pd.DataFrame({
            "TransactionID": [1, 2],
            "TransactionDT": [86400, 172800],
            "event_time_days": [1.0, 2.0],
            "TransactionAmt": [100.0, 200.0],
            "card1": [500, 500],
            "isFraud": [0, 1],
        })
        rep_1 = np.array([np.inf, 2.0])  # Row 1 is immediately reported at t=2.0
        rep_2 = np.array([np.inf, np.inf])  # Row 1 is never reported

        out1 = extract_relational_features_and_graph_state(
            df, ["card1"], ["TransactionAmt"], df["event_time_days"].to_numpy(),
            rep_1, np.array([0, 1]), "P3", "D1", 14.0, 0.05
        )
        out2 = extract_relational_features_and_graph_state(
            df, ["card1"], ["TransactionAmt"], df["event_time_days"].to_numpy(),
            rep_2, np.array([0, 0]), "P3", "D1", 14.0, 0.05
        )
        # Row 1's own features must be identical regardless of Row 1's own label/report time
        np.testing.assert_allclose(out1["rel_feats"][1], out2["rel_feats"][1], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
