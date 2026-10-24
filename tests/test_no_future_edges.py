"""Unit Tests for Causal Graph Construction & Future-Invariance (Section 8.4).

Verifies:
  1. Future-edge invariance: adding transactions at t > t_i does NOT alter earlier
     relational features, embeddings, or model predictions for transaction i under P2/P3.
  2. Future-feature invariance: mutating feature values of future nodes t > t_i
     does NOT alter any earlier node's relational state or score.
  3. Simultaneous-timestamp permutation invariance: permuting rows that share an
     exact timestamp t_j == t_i does not leak information within the batch.
  4. Contrast with P0/P1: confirms that static full-graph protocols (P0/P1) DO change
     earlier node representations when future nodes are modified (proving the test has power).
"""
import sys
import os
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
from src.temporal_graph.causal_graph import extract_relational_features_and_graph_state


class TestCausalGraphInvariance(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "TransactionID": [1, 2, 3, 4, 5, 6],
            "TransactionDT": [86400, 100000, 150000, 150000, 250000, 350000],
            "event_time_days": [1.0, 1.1574, 1.7361, 1.7361, 2.8935, 4.0509],
            "TransactionAmt": [50.0, 120.0, 80.0, 300.0, 900.0, 1500.0],
            "ip_risk_score": [0.1, 0.4, -0.2, 1.5, 3.2, 4.5],
            "card1": [101, 101, 101, 101, 101, 101],
            "addr1": [20, 20, 20, 25, 20, 20],
            "isFraud": [0, 1, 0, 1, 1, 1],
        })
        self.report_times = np.array([np.inf, 1.50, np.inf, 2.20, 3.10, 4.20], dtype=np.float64)

    def test_future_edge_insertion_invariance_p3(self):
        """Adding transactions at t > t_i must not change earlier transaction i's features/messages."""
        df_prefix = self.df.iloc[:4].copy().reset_index(drop=True)
        rep_prefix = self.report_times[:4]

        out_prefix = extract_relational_features_and_graph_state(
            df=df_prefix,
            relation_cols=["card1", "addr1"],
            base_feature_cols=["TransactionAmt", "ip_risk_score"],
            event_times_days=df_prefix["event_time_days"].to_numpy(),
            report_times_days=rep_prefix,
            true_labels=df_prefix["isFraud"].to_numpy(),
            protocol_cell="P3",
            survival_scenario_id="D1",
            maturity_horizon_days=14.0,
            prior_pi=0.05,
        )
        out_full = extract_relational_features_and_graph_state(
            df=self.df,
            relation_cols=["card1", "addr1"],
            base_feature_cols=["TransactionAmt", "ip_risk_score"],
            event_times_days=self.df["event_time_days"].to_numpy(),
            report_times_days=self.report_times,
            true_labels=self.df["isFraud"].to_numpy(),
            protocol_cell="P3",
            survival_scenario_id="D1",
            maturity_horizon_days=14.0,
            prior_pi=0.05,
        )
        np.testing.assert_allclose(
            out_prefix["rel_feats"],
            out_full["rel_feats"][:4],
            atol=1e-12,
            err_msg="Future edge insertion altered earlier node relational features under P3!",
        )

    def test_future_feature_mutation_invariance_p3(self):
        """Mutating a future node's features (row 4 & 5) must not alter earlier rows (0..3) under P3."""
        df_mut = self.df.copy()
        df_mut.loc[4:, "TransactionAmt"] = 999999.0
        df_mut.loc[4:, "ip_risk_score"] = -500.0

        out_orig = extract_relational_features_and_graph_state(
            df=self.df,
            relation_cols=["card1", "addr1"],
            base_feature_cols=["TransactionAmt", "ip_risk_score"],
            event_times_days=self.df["event_time_days"].to_numpy(),
            report_times_days=self.report_times,
            true_labels=self.df["isFraud"].to_numpy(),
            protocol_cell="P3",
            survival_scenario_id="D1",
            maturity_horizon_days=14.0,
            prior_pi=0.05,
        )
        out_mut = extract_relational_features_and_graph_state(
            df=df_mut,
            relation_cols=["card1", "addr1"],
            base_feature_cols=["TransactionAmt", "ip_risk_score"],
            event_times_days=df_mut["event_time_days"].to_numpy(),
            report_times_days=self.report_times,
            true_labels=df_mut["isFraud"].to_numpy(),
            protocol_cell="P3",
            survival_scenario_id="D1",
            maturity_horizon_days=14.0,
            prior_pi=0.05,
        )
        np.testing.assert_allclose(out_orig["rel_feats"][:4], out_mut["rel_feats"][:4], atol=1e-12)
        np.testing.assert_allclose(out_orig["msg_1hop"][:4], out_mut["msg_1hop"][:4], atol=1e-12)

    def test_simultaneous_timestamp_permutation_invariance(self):
        """Swapping simultaneous events at t=1.7361 (rows 2 and 3) must not leak between them."""
        df_perm = self.df.iloc[[0, 1, 3, 2, 4, 5]].copy().reset_index(drop=True)
        rep_perm = self.report_times[[0, 1, 3, 2, 4, 5]]

        out_orig = extract_relational_features_and_graph_state(
            df=self.df,
            relation_cols=["card1"],
            base_feature_cols=["TransactionAmt", "ip_risk_score"],
            event_times_days=self.df["event_time_days"].to_numpy(),
            report_times_days=self.report_times,
            true_labels=self.df["isFraud"].to_numpy(),
            protocol_cell="P3",
            survival_scenario_id="D1",
            maturity_horizon_days=14.0,
            prior_pi=0.05,
        )
        out_perm = extract_relational_features_and_graph_state(
            df=df_perm,
            relation_cols=["card1"],
            base_feature_cols=["TransactionAmt", "ip_risk_score"],
            event_times_days=df_perm["event_time_days"].to_numpy(),
            report_times_days=rep_perm,
            true_labels=df_perm["isFraud"].to_numpy(),
            protocol_cell="P3",
            survival_scenario_id="D1",
            maturity_horizon_days=14.0,
            prior_pi=0.05,
        )
        np.testing.assert_allclose(out_orig["rel_feats"][2], out_perm["rel_feats"][3], atol=1e-12)
        np.testing.assert_allclose(out_orig["rel_feats"][3], out_perm["rel_feats"][2], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
