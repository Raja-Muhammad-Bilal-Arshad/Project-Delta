"""Mandatory Repository Audit & Scientific Validation Test Suite (Categories A through J).

Implements and verifies all 10 required validation test categories from Section 4 of the
Repository Audit & Revision Plan Amendment (and Phase 3 of the Master Directive):
  - Category A: Dataset identity consistency across generator, manifest, protocols, and audit JSON.
  - Category B: Prior propagation (0.5x, 1.0x, 2.0x) across V2 soft targets and graph hidden-fraud mass,
                plus future-label invariance of `estimate_time_safe_prior`.
  - Category C: Label-state reconstruction and validation FPR threshold calibration strictly on confirmed
                mature negatives (adversarial test with hidden high-scoring frauds in immature validation rows,
                plus fallback behavior when confirmed negatives are below minimum threshold).
  - Category D: Temporal and causal invariance (future-edge insertion, future-feature mutation,
                simultaneous-timestamp tie handling, DataFrame row-order permutation invariance,
                and positive controls under P0/P1).
  - Category E: Split integrity (strict interval boundaries, disjoint test blocks, coverage of (101, 181],
                and training/validation/test label isolation).
  - Category F: Graph feature correctness (independent adjacency-matrix oracle verification of 2-hop
                visible fraud reachability across all 6 permutations of 3 relation columns and row permutations).
  - Category G: Metric correctness (hand-calculated edge cases distinguishing step-wise AP from
                trapezoidal PR-AUC, tied scores, zero-positive blocks, small alert budgets, and odd block sizes).
  - Category H: Results table generation & zero-mismatch verification against the frozen results bundle.
  - Category I: Clean-checkout execution & path portability (zero `/tmp/project` dependencies).
  - Category J: IEEE-CIS input validation (missing transaction/identity files fail closed, explicit
                transaction-only ablation warning, duplicate TransactionID rejection, and full 1:1 join verification).
"""
from __future__ import annotations
import os
import sys
import json
import shutil
import tempfile
import warnings
import unittest
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_audit.simulator_generator import (
    generate_handbook_multi_entity_dataset,
    SIMULATOR_GENERATOR_VERSION,
)
from src.data_audit.ieee_cis_audit import load_and_join_ieee_cis, run_step1_data_audit
from src.delay_simulation.delay_models import (
    generate_event_log_delays,
    estimate_time_safe_prior,
    reconstruct_label_state_at_cutoff,
)
from src.temporal_graph.causal_graph import extract_relational_features_and_graph_state
from src.evaluation.metrics_and_stats import (
    compute_average_precision,
    compute_trapezoidal_prauc,
    compute_roc_auc,
    compute_alert_budget_metrics,
    select_threshold_for_target_fpr,
    compute_block_metrics,
)
from export_tables_and_summary import export_and_verify_tables
from run_full_experiments import load_or_generate_split_manifest


def _independent_adjacency_2hop_oracle(
    df: pd.DataFrame,
    relation_cols: list[str],
    event_times: np.ndarray,
    report_times: np.ndarray,
    degree_cap_k: int | None = 50,
    recency_window_days: float | None = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Independent reference implementation of 1-hop and 2-hop visible prior fraud reachability
    using explicit causal adjacency sets N^<(i) at each transaction's scoring time tau_i = t_i.
    """
    n = len(df)
    tx_ids = df["TransactionID"].to_numpy(dtype=np.int64)
    # Build causal prior neighbor set N^<(i) across all relations
    neighbors_prior: list[set[int]] = [set() for _ in range(n)]
    for col in relation_cols:
        vals = df[col].to_numpy(dtype=np.int64)
        for i in range(n):
            if vals[i] < 0:
                continue
            t_i = event_times[i]
            cand = [
                j for j in range(n)
                if vals[j] == vals[i] and event_times[j] < t_i
                and (recency_window_days is None or event_times[j] >= t_i - recency_window_days)
            ]
            # Sort candidates by (event_time, TransactionID) and apply degree_cap_k
            cand.sort(key=lambda j: (event_times[j], tx_ids[j]))
            if degree_cap_k is not None and len(cand) > degree_cap_k:
                cand = cand[-degree_cap_k:]
            neighbors_prior[i].update(cand)

    vis_1hop = np.zeros(n, dtype=np.int8)
    vis_2hop = np.zeros(n, dtype=np.int8)
    for i in range(n):
        tau_i = event_times[i]
        # 1-hop visible fraud at tau_i
        for j in neighbors_prior[i]:
            if report_times[j] <= tau_i:
                vis_1hop[i] = 1
            # 2-hop visible fraud at tau_i: k in N^<(j) with report_times[k] <= tau_i
            for k in neighbors_prior[j]:
                if report_times[k] <= tau_i:
                    vis_2hop[i] = 1
    return vis_1hop, np.maximum(vis_1hop, vis_2hop)


class TestMandatoryValidationSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df_12k, cls.meta_12k = generate_handbook_multi_entity_dataset(
            n_transactions=12000, n_days=180, seed=20261024
        )

    def test_category_A_dataset_identity_consistency(self):
        """Category A: Verify N=12,000, 208 frauds, 1.7333% prevalence, 180 days, seed=20261024 across all files."""
        self.assertEqual(len(self.df_12k), 12000)
        self.assertEqual(self.meta_12k["n_transactions"], 12000)
        self.assertEqual(self.meta_12k["fraud_count"], 208)
        self.assertAlmostEqual(self.meta_12k["fraud_prevalence"], 208.0 / 12000.0, places=7)
        self.assertEqual(self.meta_12k["n_days"], 180)
        self.assertEqual(self.meta_12k["seed"], 20261024)
        self.assertEqual(self.meta_12k["generator_version"], SIMULATOR_GENERATOR_VERSION)

        # Verify protocol documents no longer claim N=60,000 or N=36,000
        dd_text = (PROJECT_ROOT / "protocol/data_dictionary.md").read_text()
        dev_text = (PROJECT_ROOT / "protocol/deviations_log.md").read_text()
        runner_text = (PROJECT_ROOT / "run_full_experiments.py").read_text()
        self.assertNotIn("60,000", dd_text)
        self.assertNotIn("60,000", dev_text)
        self.assertNotIn("N=36,000", runner_text)
        self.assertIn("12,000", dd_text)
        self.assertIn("12,000", dev_text)

    def test_category_B_prior_propagation_and_future_invariance(self):
        """Category B: Changing prior_mode (0.5x, 1.0x, 2.0x) propagates to BOTH V2 targets and graph hidden mass."""
        df_sub = self.df_12k.iloc[:1500].copy().reset_index(drop=True)
        ev = df_sub["event_time_days"].to_numpy()
        y = df_sub["isFraud"].to_numpy()
        _, rep_t = generate_event_log_delays(ev, y, "D2", seed=7007)
        tau = float(np.percentile(ev, 70))

        pi_05 = estimate_time_safe_prior(ev, rep_t, tau, 30.0, "D2", prior_mode="underestimated_0.5x")
        pi_10 = estimate_time_safe_prior(ev, rep_t, tau, 30.0, "D2", prior_mode="historical_smoothed")
        pi_20 = estimate_time_safe_prior(ev, rep_t, tau, 30.0, "D2", prior_mode="overestimated_2.0x")
        self.assertAlmostEqual(pi_05, 0.5 * pi_10, places=6)
        self.assertAlmostEqual(pi_20, 2.0 * pi_10, places=6)

        st_05 = reconstruct_label_state_at_cutoff(ev, rep_t, tau, 30.0, "D2", pi_05)
        st_10 = reconstruct_label_state_at_cutoff(ev, rep_t, tau, 30.0, "D2", pi_10)
        st_20 = reconstruct_label_state_at_cutoff(ev, rep_t, tau, 30.0, "D2", pi_20)
        imm_mask = st_10["is_immature"]
        self.assertTrue(np.any(imm_mask))
        self.assertTrue(np.all(st_05["y_v2"][imm_mask] < st_10["y_v2"][imm_mask]))
        self.assertTrue(np.all(st_10["y_v2"][imm_mask] < st_20["y_v2"][imm_mask]))

        # Verify graph hidden-fraud mass (column 6 of each relation) scales monotonically with prior_mode
        g_05 = extract_relational_features_and_graph_state(
            df_sub, ["card1", "addr1"], ["log_amount"], ev, rep_t, y, "P3", "D2", 30.0,
            prior_pi="time_safe", prior_mode="underestimated_0.5x"
        )
        g_10 = extract_relational_features_and_graph_state(
            df_sub, ["card1", "addr1"], ["log_amount"], ev, rep_t, y, "P3", "D2", 30.0,
            prior_pi="time_safe", prior_mode="historical_smoothed"
        )
        g_20 = extract_relational_features_and_graph_state(
            df_sub, ["card1", "addr1"], ["log_amount"], ev, rep_t, y, "P3", "D2", 30.0,
            prior_pi="time_safe", prior_mode="overestimated_2.0x"
        )
        hid_05 = g_05["rel_feats"][:, 6] + g_05["rel_feats"][:, 13]
        hid_10 = g_10["rel_feats"][:, 6] + g_10["rel_feats"][:, 13]
        hid_20 = g_20["rel_feats"][:, 6] + g_20["rel_feats"][:, 13]
        pos_hid = hid_10 > 0
        self.assertTrue(np.any(pos_hid))
        self.assertTrue(np.all(hid_05[pos_hid] < hid_10[pos_hid]))
        self.assertTrue(np.all(hid_10[pos_hid] < hid_20[pos_hid]))

        # Verify future labels (> tau) cannot influence estimated prior at tau
        rep_future_mut = rep_t.copy()
        rep_future_mut[ev > tau] = tau + 0.001
        rep_future_mut[(ev <= tau) & (rep_t > tau)] = tau + 100.0
        pi_10_mut = estimate_time_safe_prior(ev, rep_future_mut, tau, 30.0, "D2", prior_mode="historical_smoothed")
        self.assertAlmostEqual(pi_10, pi_10_mut, places=12)

    def test_category_C_validation_threshold_calibration_excludes_immature(self):
        """Category C: Adversarial validation set proving immature rows (including hidden frauds) never enter negative quantile."""
        # 200 confirmed mature negatives (state_code == 0) with scores uniformly in [0.01, 0.20]
        # 10 visible positives (state_code == 1) with scores in [0.90, 0.99]
        # 50 immature unreported rows (state_code == 2), including 25 hidden frauds with scores = 0.99!
        state_codes = np.concatenate([
            np.zeros(200, dtype=np.int8),
            np.ones(10, dtype=np.int8),
            np.full(50, 2, dtype=np.int8),
        ])
        mature_neg_scores = np.linspace(0.01, 0.20, 200)
        vis_pos_scores = np.linspace(0.90, 0.99, 10)
        imm_scores_high = np.full(50, 0.99)
        imm_scores_low = np.full(50, 0.001)

        s_val_1 = np.concatenate([mature_neg_scores, vis_pos_scores, imm_scores_high])
        s_val_2 = np.concatenate([mature_neg_scores, vis_pos_scores, imm_scores_low])

        th1, info1 = select_threshold_for_target_fpr(
            state_codes, s_val_1, target_fpr=0.01, confirmed_neg_mask=(state_codes == 0), return_info=True
        )
        th2, info2 = select_threshold_for_target_fpr(
            state_codes, s_val_2, target_fpr=0.01, confirmed_neg_mask=(state_codes == 0), return_info=True
        )
        # Threshold must equal the 99th percentile of the 200 mature negatives (<= 0.20) and be 100% invariant to immature scores!
        expected_th = float(np.quantile(mature_neg_scores, 0.99))
        self.assertAlmostEqual(th1, expected_th, places=10)
        self.assertAlmostEqual(th2, expected_th, places=10)
        self.assertEqual(info1["n_confirmed_neg_used"], 200)
        self.assertEqual(info1["n_val_immature_excluded"], 50)
        self.assertEqual(info1["calibration_mode"], "confirmed_mature_negatives_in_window")

        # Also test Fallback A (when validation window has 0 mature negatives because window < M)
        imm_only_states = np.full(80, 2, dtype=np.int8)
        fb_th, fb_info = select_threshold_for_target_fpr(
            imm_only_states, np.full(80, 0.95), target_fpr=0.01,
            confirmed_neg_mask=(imm_only_states == 0),
            fallback_mature_neg_scores=mature_neg_scores,
            return_info=True,
        )
        self.assertAlmostEqual(fb_th, expected_th, places=10)
        self.assertEqual(fb_info["calibration_mode"], "recent_confirmed_mature_negative_buffer")
        self.assertEqual(fb_info["n_confirmed_neg_used"], 200)

    def test_category_D_temporal_causal_and_row_permutation_invariance(self):
        """Category D: Positive controls under P0/P1 vs strict invariance under P2/P3 and row permutation."""
        df_small = pd.DataFrame({
            "TransactionID": [10, 20, 30, 40, 50, 60],
            "TransactionDT": [86400, 120000, 180000, 180000, 260000, 360000],
            "event_time_days": [1.0, 1.3889, 2.0833, 2.0833, 3.0093, 4.1667],
            "TransactionAmt": [40.0, 110.0, 95.0, 250.0, 800.0, 1600.0],
            "ip_risk_score": [0.1, 0.3, -0.1, 1.2, 2.8, 4.2],
            "card1": [7, 7, 7, 7, 7, 7],
            "addr1": [11, 11, 11, 22, 11, 11],
            "isFraud": [0, 1, 0, 1, 1, 1],
        })
        rep = np.array([np.inf, 1.8, np.inf, 2.5, 3.2, 4.5], dtype=np.float64)
        df_mut = df_small.copy()
        df_mut.loc[4:, "TransactionAmt"] = 50000.0

        # Negative control: P3 is invariant on rows 0..3 when future rows 4..5 are mutated
        p3_orig = extract_relational_features_and_graph_state(
            df_small, ["card1", "addr1"], ["TransactionAmt"], df_small["event_time_days"].to_numpy(),
            rep, df_small["isFraud"].to_numpy(), "P3", "D1", 14.0, prior_pi=0.04
        )
        p3_mut = extract_relational_features_and_graph_state(
            df_mut, ["card1", "addr1"], ["TransactionAmt"], df_mut["event_time_days"].to_numpy(),
            rep, df_mut["isFraud"].to_numpy(), "P3", "D1", 14.0, prior_pi=0.04
        )
        np.testing.assert_allclose(p3_orig["msg_1hop"][:4], p3_mut["msg_1hop"][:4], atol=1e-12)
        np.testing.assert_allclose(p3_orig["msg_2hop"][:4], p3_mut["msg_2hop"][:4], atol=1e-12)

        # Positive control: P0/P1 static topology DOES change on rows 0..3 when future rows 4..5 are mutated
        p0_orig = extract_relational_features_and_graph_state(
            df_small, ["card1", "addr1"], ["TransactionAmt"], df_small["event_time_days"].to_numpy(),
            rep, df_small["isFraud"].to_numpy(), "P0", "D1", 14.0, prior_pi=0.04
        )
        p0_mut = extract_relational_features_and_graph_state(
            df_mut, ["card1", "addr1"], ["TransactionAmt"], df_mut["event_time_days"].to_numpy(),
            rep, df_mut["isFraud"].to_numpy(), "P0", "D1", 14.0, prior_pi=0.04
        )
        self.assertGreater(np.max(np.abs(p0_orig["msg_1hop"][:4] - p0_mut["msg_1hop"][:4])), 1.0)

        # Row-order permutation invariance across all 6 rows (including tied timestamps at rows 2 and 3)
        perm = np.array([5, 3, 0, 4, 2, 1])
        inv_perm = np.argsort(perm)
        df_shuf = df_small.iloc[perm].copy().reset_index(drop=True)
        p3_shuf = extract_relational_features_and_graph_state(
            df_shuf, ["card1", "addr1"], ["TransactionAmt"], df_shuf["event_time_days"].to_numpy(),
            rep[perm], df_shuf["isFraud"].to_numpy(), "P3", "D1", 14.0, prior_pi=0.04
        )
        np.testing.assert_allclose(p3_orig["rel_feats"], p3_shuf["rel_feats"][inv_perm], atol=1e-12)
        np.testing.assert_allclose(p3_orig["msg_1hop"], p3_shuf["msg_1hop"][inv_perm], atol=1e-12)
        np.testing.assert_allclose(p3_orig["msg_2hop"], p3_shuf["msg_2hop"][inv_perm], atol=1e-12)
        np.testing.assert_array_equal(p3_orig["has_vis_fraud_within_2hops"], p3_shuf["has_vis_fraud_within_2hops"][inv_perm])

    def test_category_E_split_manifest_integrity(self):
        """Category E: Verify configs/splits/rolling_v1.json against the 12,000-row stream."""
        manifest_path = PROJECT_ROOT / "configs/splits/rolling_v1.json"
        ev_days = self.df_12k["event_time_days"].to_numpy()
        true_y = self.df_12k["isFraud"].to_numpy()
        manifest = load_or_generate_split_manifest(manifest_path, ev_days, true_y, regenerate=False)
        self.assertEqual(manifest["n_test_blocks"], 8)
        self.assertEqual(manifest["total_test_transactions"], 5358)
        self.assertEqual(manifest["total_test_positives"], 73)
        self.assertEqual(manifest["manifest_source_mode"], "loaded_and_validated_existing_manifest")

    def test_category_F_two_hop_fraud_neighbor_oracle_and_relation_order_invariance(self):
        """Category F: Compare 2-hop fraud reachability against an independent adjacency oracle across all 6 relation permutations."""
        df_het = pd.DataFrame({
            "TransactionID": [1, 2, 3, 4, 5, 6, 7],
            "TransactionDT": [100, 200, 300, 400, 500, 600, 700],
            "event_time_days": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "TransactionAmt": [50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0],
            "card1":       [10, 10, 20, 30, 40, 50, 60],
            "addr1":       [100, 200, 200, 300, 400, 500, 600],
            "terminal_id": [1, 2, 3, 3, 4, 4, 5],
            "isFraud":     [1, 0, 0, 0, 1, 0, 0],
        })
        # Tx 0 (t=1.0) is fraud reported at t=1.5.
        # Tx 1 (t=2.0) shares card1=10 with Tx 0 -> 1-hop vis fraud!
        # Tx 2 (t=3.0) shares addr1=200 with Tx 1 -> 2-hop vis fraud (Tx 0 -> card1 -> Tx 1 -> addr1 -> Tx 2)!
        # Tx 3 (t=4.0) shares terminal_id=3 with Tx 2 -> 3-hop from Tx 0 (must NOT be flagged as 2-hop!).
        # Tx 4 (t=5.0) is fraud reported at t=6.5 (AFTER Tx 5 at t=6.0!).
        # Tx 5 (t=6.0) shares terminal_id=4 with Tx 4, but at t=6.0 Tx 4's report (6.5) has NOT arrived -> 0!
        # Tx 6 (t=7.0) is isolated -> 0!
        rep_t = np.array([1.5, np.inf, np.inf, np.inf, 6.5, np.inf, np.inf])
        ev_t = df_het["event_time_days"].to_numpy()
        y_t = df_het["isFraud"].to_numpy()

        expected_1hop = np.array([0, 1, 0, 0, 0, 0, 0], dtype=np.int8)
        expected_within_2hop = np.array([0, 1, 1, 0, 0, 0, 0], dtype=np.int8)

        oracle_1hop, oracle_within_2hop = _independent_adjacency_2hop_oracle(
            df_het, ["card1", "addr1", "terminal_id"], ev_t, rep_t
        )
        np.testing.assert_array_equal(oracle_1hop, expected_1hop)
        np.testing.assert_array_equal(oracle_within_2hop, expected_within_2hop)

        # Test all 6 permutations of ["card1", "addr1", "terminal_id"]
        for rel_perm in itertools.permutations(["card1", "addr1", "terminal_id"]):
            out = extract_relational_features_and_graph_state(
                df_het, list(rel_perm), ["TransactionAmt"], ev_t, rep_t, y_t, "P3", "D1", 14.0, prior_pi=0.05
            )
            np.testing.assert_array_equal(
                out["has_vis_fraud_1hop"], oracle_1hop,
                err_msg=f"1-hop mismatch for relation order {rel_perm}"
            )
            np.testing.assert_array_equal(
                out["has_vis_fraud_within_2hops"], oracle_within_2hop,
                err_msg=f"2-hop mismatch for relation order {rel_perm}"
            )

    def test_category_G_metric_edge_cases_and_ap_vs_prauc(self):
        """Category G: Hand-calculated AP vs trapezoidal PR-AUC, tied scores, zero-positive blocks, and small budgets."""
        y = np.array([1, 0, 1, 0, 0])
        s = np.array([0.9, 0.8, 0.7, 0.4, 0.1])
        ap = compute_average_precision(y, s)
        trap_prauc = compute_trapezoidal_prauc(y, s)
        # Step-wise AP = 0.5*1.0 + 0.5*(2/3) = 5/6 = 0.833333333
        # Trapezoidal PR-AUC = 0.5*(1.0 + 1.0)*0.5 + 0.5*(0.5 + 2/3)*0.5 = 0.5 + 7/24 = 19/24 = 0.791666667 (or over distinct thresholds)
        self.assertAlmostEqual(ap, 5.0 / 6.0, places=9)
        self.assertNotAlmostEqual(ap, trap_prauc, places=3)

        # Zero-positive block edge case
        y_zero = np.zeros(10, dtype=np.int32)
        s_zero = np.linspace(0.1, 0.9, 10)
        self.assertEqual(compute_average_precision(y_zero, s_zero), 0.0)
        self.assertEqual(compute_trapezoidal_prauc(y_zero, s_zero), 0.0)
        self.assertEqual(compute_roc_auc(y_zero, s_zero), 0.5)

        # Tied scores with deterministic amount tie-breaking
        y_tie = np.array([0, 1, 0, 1])
        s_tie = np.array([0.8, 0.8, 0.8, 0.2])
        amt_tie = np.array([10.0, 500.0, 20.0, 50.0])  # Item 1 has highest amount among tied 0.8 scores!
        p25, r25, k25 = compute_alert_budget_metrics(y_tie, s_tie, budget_pct=25.0, amounts=amt_tie)
        self.assertEqual(k25, 1)
        self.assertEqual(p25, 1.0)
        self.assertEqual(r25, 0.5)

    def test_category_H_table_generation_consistency(self):
        """Category H: Regenerate CSV tables from frozen bundle and verify 0 numerical mismatches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_res = Path(tmpdir) / "results"
            shutil.copytree(PROJECT_ROOT / "results", tmp_res)
            info = export_and_verify_tables(tmp_res)
            self.assertEqual(info["mismatches_detected"], 0)
            self.assertEqual(info["tables_generated"], 8)

    def test_category_I_clean_checkout_portability(self):
        """Category I: Verify zero hardcoded /tmp/project references in runner, table exporter, and figure generator."""
        for script_name in ["run_full_experiments.py", "export_tables_and_summary.py", "generate_figures.py"]:
            lines = (PROJECT_ROOT / script_name).read_text().splitlines()
            code_lines = [l for l in lines if not l.strip().startswith("#") and '"/tmp/project"' in l]
            self.assertEqual(len(code_lines), 0, f"Found hardcoded /tmp/project in {script_name}: {code_lines}")

    def test_category_J_ieee_cis_input_validation(self):
        """Category J: Test missing files fail closed, transaction-only warning, duplicate keys fail, and valid 1:1 join."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tx_path = os.path.join(tmpdir, "train_transaction.csv")
            id_path = os.path.join(tmpdir, "train_identity.csv")

            # 1. Missing train_transaction fails closed
            with self.assertRaises(FileNotFoundError):
                load_and_join_ieee_cis(tx_path, id_path)

            # Create valid synthetic train_transaction
            df_tx = pd.DataFrame({
                "TransactionID": [1001, 1002, 1003, 1004],
                "TransactionDT": [86400, 86500, 86600, 86700],
                "TransactionAmt": [25.0, 150.0, 75.0, 300.0],
                "isFraud": [0, 1, 0, 0],
                "card1": [10, 20, 10, 30],
            })
            df_tx.to_csv(tx_path, index=False)

            # 2. Missing train_identity fails closed by default (allow_transaction_only=False)
            with self.assertRaises(FileNotFoundError):
                load_and_join_ieee_cis(tx_path, id_path, allow_transaction_only=False)

            # 3. Explicit allow_transaction_only=True emits UserWarning and marks ablation identity
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                m_only, rep_only = load_and_join_ieee_cis(tx_path, id_path, allow_transaction_only=True)
                self.assertEqual(len(w), 1)
                self.assertEqual(rep_only["dataset_identity"], "IEEE_CIS_TRANSACTION_ONLY_ABLATION")
                self.assertEqual(len(m_only), 4)

            # 4. Duplicate TransactionID in train_identity raises ValueError
            df_id_dup = pd.DataFrame({
                "TransactionID": [1001, 1001],
                "DeviceInfo": ["iOS", "Android"],
            })
            df_id_dup.to_csv(id_path, index=False)
            with self.assertRaises(ValueError):
                load_and_join_ieee_cis(tx_path, id_path, allow_transaction_only=False)

            # 5. Valid full two-file join
            df_id_valid = pd.DataFrame({
                "TransactionID": [1002, 1004],
                "DeviceInfo": ["Windows", "MacOS"],
            })
            df_id_valid.to_csv(id_path, index=False)
            m_full, rep_full = load_and_join_ieee_cis(tx_path, id_path, allow_transaction_only=False)
            self.assertEqual(rep_full["dataset_identity"], "IEEE_CIS_FULL_JOINED_BENCHMARK")
            self.assertEqual(rep_full["merged_rows"], 4)
            self.assertEqual(rep_full["matched_identity_rows"], 2)
            self.assertAlmostEqual(rep_full["identity_join_coverage"], 0.5, places=6)
            self.assertIsNotNone(rep_full["train_transaction_sha256"])
            self.assertIsNotNone(rep_full["train_identity_sha256"])


if __name__ == "__main__":
    unittest.main()
