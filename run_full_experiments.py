"""Master Experimental Execution Script for Project Delta (Corrected & Validated v2.0).

Executes and logs all registered experiments on the Extended Fraud Detection Handbook
Multi-Entity Transaction Stream (N=12,000 transactions, 208 frauds [1.7333% prevalence]
over 180 temporal days across 8 rolling 10-day test blocks consuming configs/splits/rolling_v1.json),
and verifies the IEEE-CIS ingestion/audit harness.
"""
from __future__ import annotations
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
import time
import json
import hashlib
import resource
import argparse
import platform
import functools
import tracemalloc
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

print = functools.partial(print, flush=True)

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_audit.simulator_generator import (
    generate_handbook_multi_entity_dataset,
    SIMULATOR_GENERATOR_VERSION,
)
from src.data_audit.ieee_cis_audit import run_step1_data_audit
from src.delay_simulation.delay_models import (
    SCENARIO_SPECS,
    LAMBDA_D1_14D,
    delay_cdf,
    delay_survival,
    sample_fraud_delays,
    generate_event_log_delays,
    compute_posterior_q,
    estimate_time_safe_prior,
    reconstruct_label_state_at_cutoff,
)
from src.temporal_graph.causal_graph import (
    extract_relational_features_and_graph_state,
    compute_pass2_from_relation_slices,
)
from src.models.baselines_and_gnns import SoftTargetHistGBDT, CausalGraphNeuralNetwork
from src.models.external_framework_adapters import get_external_package_availability
from src.evaluation.metrics_and_stats import (
    compute_block_metrics,
    paired_temporal_block_analysis,
    holm_bonferroni_correction,
    select_threshold_for_target_fpr,
    compute_average_precision,
    compute_alert_budget_metrics,
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _get_proc_rss_mb() -> float:
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    kb = float(line.split()[1])
                    return round(kb / 1024.0, 2)
    except Exception:
        pass
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 2)


def load_or_generate_split_manifest(
    manifest_path: Path,
    ev_days: np.ndarray,
    true_y: np.ndarray,
    regenerate: bool = False,
) -> Dict[str, Any]:
    """Loads and validates the authoritative split manifest (Audit Issue 7 & Issue 12)."""
    if manifest_path.is_file() and not regenerate:
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        manifest_sha = _sha256_file(manifest_path)
        manifest["manifest_sha256"] = manifest_sha
        manifest["manifest_source_mode"] = "loaded_and_validated_existing_manifest"
    else:
        rolling_blocks = []
        for b in range(8):
            t_test_start = 101.0 + 10.0 * b
            t_test_end = 101.0 + 10.0 * (b + 1)
            t_val_start = t_test_start - 14.0
            t_train_cutoff = t_val_start
            tr_idx = np.where(ev_days <= t_train_cutoff)[0]
            va_idx = np.where((ev_days > t_val_start) & (ev_days <= t_test_start))[0]
            te_idx = np.where((ev_days > t_test_start) & (ev_days <= t_test_end))[0]
            n_te = int(len(te_idx))
            rolling_blocks.append({
                "block_id": b,
                "train_cutoff_day": t_train_cutoff,
                "val_start_day": t_val_start,
                "val_end_day": t_test_start,
                "test_start_day": t_test_start,
                "test_end_day": t_test_end,
                "n_train": int(len(tr_idx)),
                "n_val": int(len(va_idx)),
                "n_test": n_te,
                "n_test_pos": int(np.sum(true_y[te_idx] == 1)),
                "test_prevalence": round(float(np.mean(true_y[te_idx])), 5),
                "alert_budget_k_0.5pct": int(np.ceil(0.005 * n_te)),
                "alert_budget_k_1.0pct": int(np.ceil(0.010 * n_te)),
                "alert_budget_k_5.0pct": int(np.ceil(0.050 * n_te)),
            })
        manifest = {
            "manifest_id": "rolling_8blocks_10d_v1_reconciled",
            "stream_coordinate_start_day": 1.0,
            "stream_coordinate_end_day": 181.0,
            "total_elapsed_span_days": 180.0,
            "initial_train_cutoff_day": 87.0,
            "initial_train_elapsed_days": 86.0,
            "mature_prefix_days_at_initial_cutoff_M30": 56.0,
            "mature_prefix_interval_at_M30": "[1.0, 57.0]",
            "immature_window_interval_at_M30": "(57.0, 87.0]",
            "warmup_usage_rule": "All historical transactions in [1.0, train_cutoff_day] enter causal graph construction and model fitting according to supervision rule V0/V1/V2; under D2 (M=30d), [1.0, train_cutoff_day - 30.0] forms the confirmed mature prefix and (train_cutoff_day - 30.0, train_cutoff_day] forms the immature window.",
            "train_interval_semantics": "[1.0, train_cutoff_day] (closed interval)",
            "val_interval_semantics": "(val_start_day, val_end_day] (left-open, right-closed, 14.0 days)",
            "test_interval_semantics": "(test_start_day, test_end_day] (left-open, right-closed, 10.0 days, strictly disjoint across blocks 0..7)",
            "retraining_cadence_days": 10.0,
            "val_window_days": 14.0,
            "test_block_days": 10.0,
            "n_test_blocks": 8,
            "total_test_transactions": int(sum(x["n_test"] for x in rolling_blocks)),
            "total_test_positives": int(sum(x["n_test_pos"] for x in rolling_blocks)),
            "blocks": rolling_blocks,
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        manifest["manifest_sha256"] = _sha256_file(manifest_path)
        manifest["manifest_source_mode"] = "regenerated_manifest"

    seen_test_indices = set()
    for blk in manifest["blocks"]:
        t_tr = float(blk["train_cutoff_day"])
        t_va_s = float(blk["val_start_day"])
        t_va_e = float(blk["val_end_day"])
        t_te_s = float(blk["test_start_day"])
        t_te_e = float(blk["test_end_day"])
        assert t_tr == t_va_s, f"Block {blk['block_id']}: train_cutoff_day != val_start_day"
        assert t_va_e == t_te_s, f"Block {blk['block_id']}: val_end_day != test_start_day"

        tr_idx = np.where(ev_days <= t_tr)[0]
        va_idx = np.where((ev_days > t_va_s) & (ev_days <= t_va_e))[0]
        te_idx = np.where((ev_days > t_te_s) & (ev_days <= t_te_e))[0]

        assert len(tr_idx) == int(blk["n_train"]), f"Block {blk['block_id']} n_train mismatch"
        assert len(va_idx) == int(blk["n_val"]), f"Block {blk['block_id']} n_val mismatch"
        assert len(te_idx) == int(blk["n_test"]), f"Block {blk['block_id']} n_test mismatch"
        assert int(np.sum(true_y[te_idx] == 1)) == int(blk["n_test_pos"]), f"Block {blk['block_id']} n_test_pos mismatch"
        assert len(seen_test_indices.intersection(te_idx)) == 0, f"Overlapping test indices in block {blk['block_id']}"
        seen_test_indices.update(te_idx.tolist())

    expected_all_test = set(np.where((ev_days > 101.0) & (ev_days <= 181.0))[0].tolist())
    assert seen_test_indices == expected_all_test, "Test blocks do not partition (101.0, 181.0] exactly!"
    return manifest


def run_experiment_suite(
    output_dir: Path,
    split_manifest_path: Path,
    regenerate_manifest: bool = False,
    seeds: List[int] = [101, 202, 303, 404, 505],
    run_id: str = "project_delta_v2_reconciled_20261024",
) -> Dict[str, Any]:
    t_global_start = time.time()

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    (output_dir / "manifests").mkdir(parents=True, exist_ok=True)

    print("[Step 1] Generating & auditing multi-entity transaction stream (N=12,000, 180 days, seed=20261024)...")
    df, sim_meta = generate_handbook_multi_entity_dataset(
        n_transactions=12000,
        n_days=180,
        n_customers=1200,
        n_terminals=450,
        n_addresses=450,
        n_email_domains=60,
        n_devices=1800,
        seed=20261024,
    )

    entity_cols_all = ["card1", "card_comp", "terminal_id", "addr1", "P_emaildomain", "DeviceInfo"]
    audit_report = run_step1_data_audit(df, "Handbook_MultiEntity_Stream_v2_12k", entity_cols_all, n_bins=18)
    audit_report["simulator_metadata"] = sim_meta

    with open(output_dir / "manifests/step1_data_audit.json", "w") as f:
        json.dump(audit_report, f, indent=2)
    print(
        f"      Audited {audit_report['n_rows']} rows, {audit_report['n_frauds']} frauds "
        f"({audit_report['fraud_prevalence']*100:.2f}%), identity cov={audit_report['identity_coverage_overall']*100:.2f}%"
    )

    ev_days = df["event_time_days"].to_numpy(dtype=np.float64)
    true_y = df["isFraud"].to_numpy(dtype=np.int32)
    amounts = df["TransactionAmt"].to_numpy(dtype=np.float64)
    card1_arr = df["card1"].to_numpy(dtype=np.int64)

    split_manifest = load_or_generate_split_manifest(
        manifest_path=split_manifest_path,
        ev_days=ev_days,
        true_y=true_y,
        regenerate=regenerate_manifest,
    )
    rolling_blocks = split_manifest["blocks"]

    print("[Steps 5-6] Generating delay draws and verifying q(a) posterior calibration across age bins...")
    delay_draws_map = {}
    delay_audit_summary = {}
    for scen_id in ["D0", "D1", "D2", "D3"]:
        spec = SCENARIO_SPECS[scen_id]
        delays_d, rep_t = generate_event_log_delays(ev_days, true_y, scen_id, seed=7007)
        delay_draws_map[scen_id] = (delays_d, rep_t)
        f_delays = delays_d[true_y == 1]
        delay_audit_summary[scen_id] = {
            "name": spec.name,
            "maturity_horizon_M_days": spec.maturity_horizon_days,
            "hard_max_days": spec.hard_max_days,
            "realized_mean_delay_days": round(float(np.mean(f_delays)), 4),
            "realized_p25_days": round(float(np.percentile(f_delays, 25)), 4),
            "realized_median_p50_days": round(float(np.percentile(f_delays, 50)), 4),
            "realized_p75_days": round(float(np.percentile(f_delays, 75)), 4),
            "realized_p95_days": round(float(np.percentile(f_delays, 95)), 4),
            "realized_max_days": round(float(np.max(f_delays)), 4),
        }

    q_calibration_bins = []
    tau_cal = 140.0
    for scen_id in ["D1", "D2", "D3"]:
        spec = SCENARIO_SPECS[scen_id]
        _, rep_t = delay_draws_map[scen_id]
        past_m = (ev_days <= tau_cal)
        ages = tau_cal - ev_days[past_m]
        unrep_m = (rep_t[past_m] > tau_cal)
        y_unrep = true_y[past_m][unrep_m]
        ages_unrep = ages[unrep_m]
        pi_cal = estimate_time_safe_prior(
            ev_days, rep_t, tau_cal, spec.maturity_horizon_days, scen_id, prior_mode="historical_smoothed"
        )
        q_pred = compute_posterior_q(ages_unrep, pi_cal, scen_id, spec.maturity_horizon_days)

        age_edges = [0.0, 1.0, 3.0, 7.0, 14.0, 30.0, 60.0, 95.0]
        for bi in range(len(age_edges) - 1):
            a_lo, a_hi = age_edges[bi], age_edges[bi + 1]
            bm = (ages_unrep >= a_lo) & (ages_unrep < a_hi)
            if int(np.sum(bm)) >= 20:
                q_calibration_bins.append({
                    "scenario": scen_id,
                    "age_bin": f"[{int(a_lo)}, {int(a_hi)})d",
                    "a_mid": 0.5 * (a_lo + a_hi),
                    "n_unreported": int(np.sum(bm)),
                    "estimated_pi_tau": round(pi_cal, 5),
                    "mean_predicted_q": round(float(np.mean(q_pred[bm])), 5),
                    "empirical_hidden_fraud_rate": round(float(np.mean(y_unrep[bm])), 5),
                    "abs_calibration_gap": round(float(abs(np.mean(q_pred[bm]) - np.mean(y_unrep[bm]))), 5),
                })

    with open(output_dir / "manifests/delay_and_q_calibration.json", "w") as f:
        json.dump({"delay_scenarios": delay_audit_summary, "q_calibration": q_calibration_bins}, f, indent=2)

    base_feature_cols = [
        "log_amount", "cust_z_amount", "hour_of_day", "is_night",
        "ProductCD", "dist_score", "ip_risk_score", "auth_match_flag", "has_identity"
    ]
    primary_relations = ["card1", "terminal_id", "addr1", "P_emaildomain", "DeviceInfo"]
    X_base_all = df[base_feature_cols].to_numpy(dtype=np.float64)

    print("[Steps 3-4] Precomputing relational feature & graph topology states across P0-P3 and D0-D3...")
    graph_cache: Dict[Any, Dict[str, Any]] = {}

    def get_graph_state(
        protocol_cell: str,
        delay_scen: str,
        prior_mode: str = "historical_smoothed",
        survival_assumed_scen: Optional[str] = None,
        rels: list[str] = primary_relations,
        K: Optional[int] = 50,
        W: Optional[float] = 30.0,
        inv_log: bool = True,
    ) -> Dict[str, Any]:
        surv_scen = survival_assumed_scen if survival_assumed_scen is not None else (
            "D2" if delay_scen == "D3_misspecified_M30" else delay_scen
        )
        key = (protocol_cell, delay_scen, surv_scen, prior_mode, tuple(rels), K, W, inv_log)
        if key in graph_cache:
            return graph_cache[key]

        # If rels is a subset of primary_relations and full_key is cached, slice Pass-1 and run exact Pass-2!
        full_key = (protocol_cell, delay_scen, surv_scen, prior_mode, tuple(primary_relations), K, W, inv_log)
        if set(rels).issubset(set(primary_relations)) and tuple(rels) != tuple(primary_relations) and full_key in graph_cache:
            base_g = graph_cache[full_key]
            rel_idx_list = [primary_relations.index(r) for r in rels]
            d_b = len(base_feature_cols)
            rf_cols = np.concatenate([np.arange(ri * 7, (ri + 1) * 7) for ri in rel_idx_list])
            m_cols = np.concatenate([np.arange(ri * d_b, (ri + 1) * d_b) for ri in rel_idx_list])
            sub_msg_1hop = base_g["msg_1hop"][:, m_cols]
            pass2 = compute_pass2_from_relation_slices(
                n=len(df),
                r_count=len(rels),
                d_base=d_b,
                tau_vec=base_g["_tau_vec"],
                msg_1hop=sub_msg_1hop,
                per_rel_vis_1hop=base_g["_per_rel_vis_1hop"][:, rel_idx_list],
                per_rel_min_rep_time=base_g["_per_rel_min_rep_time"][:, rel_idx_list],
                per_rel_oracle_1hop=base_g["_per_rel_oracle_1hop"][:, rel_idx_list],
                relation_slices_cache=[base_g["_relation_slices_cache"][ri] for ri in rel_idx_list],
                use_inv_log_hub_weight=inv_log,
                use_oracle_neighbor_labels=(protocol_cell == "P0"),
            )
            graph_cache[key] = {
                "rel_feats": base_g["rel_feats"][:, rf_cols],
                "msg_1hop": sub_msg_1hop,
                "msg_2hop": pass2["msg_2hop"],
                "sim_filtered_msg": base_g["sim_filtered_msg"][:, m_cols],
                "has_vis_fraud_1hop": pass2["has_vis_fraud_1hop"],
                "has_vis_fraud_2hop": pass2["has_vis_fraud_2hop"],
                "has_vis_fraud_within_2hops": pass2["has_vis_fraud_within_2hops"],
                "prior_pi_vec": base_g["prior_pi_vec"],
            }
            return graph_cache[key]

        spec = SCENARIO_SPECS[delay_scen]
        raw_scen = "D3" if delay_scen == "D3_misspecified_M30" else delay_scen
        _, rep_t = delay_draws_map[raw_scen]
        graph_cache[key] = extract_relational_features_and_graph_state(
            df=df,
            relation_cols=rels,
            base_feature_cols=base_feature_cols,
            event_times_days=ev_days,
            report_times_days=rep_t,
            true_labels=true_y,
            protocol_cell=protocol_cell,
            survival_scenario_id=surv_scen,
            maturity_horizon_days=spec.maturity_horizon_days,
            prior_pi="time_safe",
            prior_mode=prior_mode,
            degree_cap_k=K,
            recency_window_days=W,
            use_inv_log_hub_weight=inv_log,
        )
        return graph_cache[key]

    # Dedicated single-block tracemalloc peak memory measurement per model tier (Audit Issue 8)
    print("[Issue 8 Memory Audit] Measuring exact tracemalloc peak heap allocation per model tier on Block 7...")
    tier_tracemalloc_mb: Dict[str, float] = {}
    g_mem = get_graph_state("P3", "D2", prior_mode="historical_smoothed", rels=primary_relations, K=50, W=30.0, inv_log=True)
    blk7 = rolling_blocks[-1]
    tr7 = (ev_days <= float(blk7["train_cutoff_day"]))
    te7 = (ev_days > float(blk7["test_start_day"])) & (ev_days <= float(blk7["test_end_day"]))
    y7 = true_y.astype(np.float64)

    for m_tier in [
        "B0_raw_tabular", "B1_causal_aggregates", "B2_graph_light",
        "B3_graphsage", "B4_domain_care_gnn", "B5_proposed_maturity_hybrid"
    ]:
        tracemalloc.start()
        tracemalloc.reset_peak()
        if m_tier == "B0_raw_tabular":
            m = SoftTargetHistGBDT(n_estimators=20, learning_rate=0.18, seed=101).fit(X_base_all, y7, sample_mask=tr7)
            _ = m.predict_proba(X_base_all[te7])
        elif m_tier == "B1_causal_aggregates":
            X_in = np.hstack([X_base_all, g_mem["rel_feats"]])
            m = SoftTargetHistGBDT(n_estimators=20, learning_rate=0.18, seed=101).fit(X_in, y7, sample_mask=tr7)
            _ = m.predict_proba(X_in[te7])
        elif m_tier == "B2_graph_light":
            X_in = np.hstack([X_base_all, g_mem["rel_feats"], g_mem["msg_1hop"], g_mem["msg_2hop"]])
            m = SoftTargetHistGBDT(n_estimators=20, learning_rate=0.18, seed=101).fit(X_in, y7, sample_mask=tr7)
            _ = m.predict_proba(X_in[te7])
        elif m_tier == "B3_graphsage":
            m = CausalGraphNeuralNetwork(hidden_dim=16, lr=0.04, epochs=20, domain_care_filter=False, seed=101).fit(
                X_base_all, g_mem["rel_feats"], g_mem["msg_1hop"], g_mem["sim_filtered_msg"], y7, sample_mask=tr7
            )
            _ = m.predict_proba(X_base_all[te7], g_mem["rel_feats"][te7], g_mem["msg_1hop"][te7], g_mem["sim_filtered_msg"][te7])
        elif m_tier == "B4_domain_care_gnn":
            m = CausalGraphNeuralNetwork(hidden_dim=16, lr=0.04, epochs=20, domain_care_filter=True, seed=101).fit(
                X_base_all, g_mem["rel_feats"], g_mem["msg_1hop"], g_mem["sim_filtered_msg"], y7, sample_mask=tr7
            )
            _ = m.predict_proba(X_base_all[te7], g_mem["rel_feats"][te7], g_mem["msg_1hop"][te7], g_mem["sim_filtered_msg"][te7])
        elif m_tier == "B5_proposed_maturity_hybrid":
            X_in = np.hstack([X_base_all, g_mem["rel_feats"], g_mem["sim_filtered_msg"]])
            mt = SoftTargetHistGBDT(n_estimators=20, learning_rate=0.18, seed=101).fit(X_in, y7, sample_mask=tr7)
            mg = CausalGraphNeuralNetwork(hidden_dim=16, lr=0.04, epochs=20, domain_care_filter=True, seed=101).fit(
                X_base_all, g_mem["rel_feats"], g_mem["msg_1hop"], g_mem["sim_filtered_msg"], y7, sample_mask=tr7
            )
            _ = 0.55 * mt.predict_proba(X_in[te7]) + 0.45 * mg.predict_proba(
                X_base_all[te7], g_mem["rel_feats"][te7], g_mem["msg_1hop"][te7], g_mem["sim_filtered_msg"][te7]
            )
        _, peak_b = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        tier_tracemalloc_mb[m_tier] = round(peak_b / (1024.0 * 1024.0), 3)
        print(f"      Tier {m_tier:30s}: tracemalloc_peak = {tier_tracemalloc_mb[m_tier]:.3f} MiB")

    # Cache for B4 GNN component predictions so B5 doesn't retrain the identical B4 GNN on the same cell/block/seed
    gnn_pred_cache: Dict[Any, Any] = {}

    def evaluate_rolling_experiment(
        model_tier: str,
        protocol_cell: str,
        delay_scen: str,
        label_rule: str,
        survival_assumed_scen: Optional[str] = None,
        prior_mode: str = "historical_smoothed",
        rels: list[str] = primary_relations,
        K: Optional[int] = 50,
        W: Optional[float] = 30.0,
        inv_log: bool = True,
        sliding_window_days: Optional[float] = None,
        eval_seeds: list[int] = seeds,
    ) -> Dict[str, Any]:
        """Runs an 8-block rolling temporal evaluation across `eval_seeds` and returns per-block & summary metrics."""
        t0 = time.perf_counter()

        surv_scen = survival_assumed_scen if survival_assumed_scen is not None else (
            "D2" if delay_scen == "D3_misspecified_M30" else delay_scen
        )
        spec = SCENARIO_SPECS[delay_scen]
        raw_scen = "D3" if delay_scen == "D3_misspecified_M30" else delay_scen
        _, rep_t = delay_draws_map[raw_scen]

        g_state = get_graph_state(
            protocol_cell=protocol_cell,
            delay_scen=delay_scen,
            prior_mode=prior_mode,
            survival_assumed_scen=surv_scen,
            rels=rels,
            K=K,
            W=W,
            inv_log=inv_log,
        )
        rel_feats = g_state["rel_feats"].copy()
        if label_rule in ("V0", "V1") and protocol_cell == "P3":
            for r_i in range(len(rels)):
                rel_feats[:, r_i * 7 + 6] = 0.0

        msg_1hop = g_state["msg_1hop"]
        msg_2hop = g_state["msg_2hop"]
        sim_msg = g_state["sim_filtered_msg"]
        has_vis_2hop = g_state["has_vis_fraud_within_2hops"]

        # Pre-stack feature matrix once per 8-block run for tree models
        if model_tier == "B0_raw_tabular":
            X_tree_all = X_base_all
        elif model_tier == "B1_causal_aggregates":
            X_tree_all = np.hstack([X_base_all, rel_feats])
        elif model_tier == "B2_graph_light":
            X_tree_all = np.hstack([X_base_all, rel_feats, msg_1hop, msg_2hop])
        elif model_tier == "B5_proposed_maturity_hybrid":
            X_tree_all = np.hstack([X_base_all, rel_feats, sim_msg])
        else:
            X_tree_all = None

        block_results = []
        all_test_preds = []
        all_test_y = []
        all_test_has_nbr = []

        for blk in rolling_blocks:
            b_id = blk["block_id"]
            tau_train = float(blk["train_cutoff_day"])
            tau_val = float(blk["val_end_day"])
            t_te_start = float(blk["test_start_day"])
            t_te_end = float(blk["test_end_day"])

            if sliding_window_days is not None:
                tr_cond = (ev_days >= max(1.0, tau_train - sliding_window_days)) & (ev_days <= tau_train)
            else:
                tr_cond = (ev_days <= tau_train)
            va_cond = (ev_days > tau_train) & (ev_days <= tau_val)
            te_cond = (ev_days > t_te_start) & (ev_days <= t_te_end)

            pi_tau = estimate_time_safe_prior(
                event_times_days=ev_days,
                report_times_days=rep_t,
                cutoff_tau=tau_train,
                maturity_horizon_days=spec.maturity_horizon_days,
                survival_scenario_id=surv_scen,
                prior_mode=prior_mode,
                true_labels_for_oracle_only=true_y if prior_mode == "oracle_true_rate" else None,
            )

            st_train = reconstruct_label_state_at_cutoff(
                ev_days, rep_t, tau_train, spec.maturity_horizon_days, surv_scen, pi_tau, true_y
            )
            if label_rule == "V0":
                y_target_full = st_train["y_v0"]
                eligible_mask = tr_cond & st_train["mask_v0"]
            elif label_rule == "V1":
                y_target_full = st_train["y_v1"]
                eligible_mask = tr_cond & st_train["mask_v1"]
            elif label_rule == "V2":
                y_target_full = st_train["y_v2"]
                eligible_mask = tr_cond & st_train["mask_v2"]
            elif label_rule == "V3":
                y_target_full = st_train["y_v3_oracle"]
                eligible_mask = tr_cond & st_train["mask_v3_oracle"]
            else:
                raise ValueError(f"Unknown label_rule: {label_rule}")

            pi_val = estimate_time_safe_prior(
                event_times_days=ev_days,
                report_times_days=rep_t,
                cutoff_tau=tau_val,
                maturity_horizon_days=spec.maturity_horizon_days,
                survival_scenario_id=surv_scen,
                prior_mode=prior_mode,
                true_labels_for_oracle_only=true_y if prior_mode == "oracle_true_rate" else None,
            )
            st_val = reconstruct_label_state_at_cutoff(
                ev_days, rep_t, tau_val, spec.maturity_horizon_days, surv_scen, pi_val
            )
            mature_neg_buf_cond = (
                st_val["confirmed_neg_mask"]
                & (ev_days <= tau_val - spec.maturity_horizon_days)
                & (ev_days > max(1.0, tau_val - spec.maturity_horizon_days - 28.0))
            )

            seed_val_preds = []
            seed_test_preds = []
            seed_mat_buf_preds = []

            for sd in eval_seeds:
                if model_tier in ("B0_raw_tabular", "B1_causal_aggregates", "B2_graph_light"):
                    mdl = SoftTargetHistGBDT(n_estimators=20, learning_rate=0.18, seed=sd)
                    mdl.fit(X_tree_all, y_target_full, sample_mask=eligible_mask)
                    p_va = mdl.predict_proba(X_tree_all[va_cond])
                    p_te = mdl.predict_proba(X_tree_all[te_cond])
                    p_mb = mdl.predict_proba(X_tree_all[mature_neg_buf_cond]) if np.any(mature_neg_buf_cond) else np.array([])
                elif model_tier == "B3_graphsage":
                    mdl = CausalGraphNeuralNetwork(hidden_dim=16, lr=0.04, epochs=20, domain_care_filter=False, seed=sd)
                    mdl.fit(X_base_all, rel_feats, msg_1hop, sim_msg, y_target_full, sample_mask=eligible_mask)
                    p_va = mdl.predict_proba(X_base_all[va_cond], rel_feats[va_cond], msg_1hop[va_cond], sim_msg[va_cond])
                    p_te = mdl.predict_proba(X_base_all[te_cond], rel_feats[te_cond], msg_1hop[te_cond], sim_msg[te_cond])
                    p_mb = (
                        mdl.predict_proba(
                            X_base_all[mature_neg_buf_cond],
                            rel_feats[mature_neg_buf_cond],
                            msg_1hop[mature_neg_buf_cond],
                            sim_msg[mature_neg_buf_cond],
                        )
                        if np.any(mature_neg_buf_cond)
                        else np.array([])
                    )
                elif model_tier == "B4_domain_care_gnn":
                    g_ckey = (protocol_cell, delay_scen, surv_scen, label_rule, prior_mode, tuple(rels), K, W, inv_log, sliding_window_days, b_id, sd)
                    if g_ckey not in gnn_pred_cache:
                        mdl = CausalGraphNeuralNetwork(hidden_dim=16, lr=0.04, epochs=20, domain_care_filter=True, seed=sd)
                        mdl.fit(X_base_all, rel_feats, msg_1hop, sim_msg, y_target_full, sample_mask=eligible_mask)
                        p_va_g = mdl.predict_proba(X_base_all[va_cond], rel_feats[va_cond], msg_1hop[va_cond], sim_msg[va_cond])
                        p_te_g = mdl.predict_proba(X_base_all[te_cond], rel_feats[te_cond], msg_1hop[te_cond], sim_msg[te_cond])
                        p_mb_g = (
                            mdl.predict_proba(
                                X_base_all[mature_neg_buf_cond],
                                rel_feats[mature_neg_buf_cond],
                                msg_1hop[mature_neg_buf_cond],
                                sim_msg[mature_neg_buf_cond],
                            )
                            if np.any(mature_neg_buf_cond)
                            else np.array([])
                        )
                        gnn_pred_cache[g_ckey] = (p_va_g, p_te_g, p_mb_g)
                    p_va, p_te, p_mb = gnn_pred_cache[g_ckey]
                elif model_tier == "B5_proposed_maturity_hybrid":
                    mdl_tree = SoftTargetHistGBDT(n_estimators=20, learning_rate=0.18, seed=sd)
                    mdl_tree.fit(X_tree_all, y_target_full, sample_mask=eligible_mask)
                    p_va_t = mdl_tree.predict_proba(X_tree_all[va_cond])
                    p_te_t = mdl_tree.predict_proba(X_tree_all[te_cond])
                    p_mb_t = mdl_tree.predict_proba(X_tree_all[mature_neg_buf_cond]) if np.any(mature_neg_buf_cond) else np.array([])

                    g_ckey = (protocol_cell, delay_scen, surv_scen, label_rule, prior_mode, tuple(rels), K, W, inv_log, sliding_window_days, b_id, sd)
                    if g_ckey not in gnn_pred_cache:
                        mdl_gnn = CausalGraphNeuralNetwork(hidden_dim=16, lr=0.04, epochs=20, domain_care_filter=True, seed=sd)
                        mdl_gnn.fit(X_base_all, rel_feats, msg_1hop, sim_msg, y_target_full, sample_mask=eligible_mask)
                        p_va_g = mdl_gnn.predict_proba(X_base_all[va_cond], rel_feats[va_cond], msg_1hop[va_cond], sim_msg[va_cond])
                        p_te_g = mdl_gnn.predict_proba(X_base_all[te_cond], rel_feats[te_cond], msg_1hop[te_cond], sim_msg[te_cond])
                        p_mb_g = (
                            mdl_gnn.predict_proba(
                                X_base_all[mature_neg_buf_cond],
                                rel_feats[mature_neg_buf_cond],
                                msg_1hop[mature_neg_buf_cond],
                                sim_msg[mature_neg_buf_cond],
                            )
                            if np.any(mature_neg_buf_cond)
                            else np.array([])
                        )
                        gnn_pred_cache[g_ckey] = (p_va_g, p_te_g, p_mb_g)
                    p_va_g, p_te_g, p_mb_g = gnn_pred_cache[g_ckey]
                    p_va = 0.55 * p_va_t + 0.45 * p_va_g
                    p_te = 0.55 * p_te_t + 0.45 * p_te_g
                    p_mb = (0.55 * p_mb_t + 0.45 * p_mb_g) if len(p_mb_t) > 0 else np.array([])
                else:
                    raise ValueError(f"Unknown model_tier: {model_tier}")

                seed_val_preds.append(p_va)
                seed_test_preds.append(p_te)
                if len(p_mb) > 0:
                    seed_mat_buf_preds.append(p_mb)

            mean_p_va = np.mean(seed_val_preds, axis=0)
            mean_p_te = np.mean(seed_test_preds, axis=0)
            mean_p_mb = np.mean(seed_mat_buf_preds, axis=0) if len(seed_mat_buf_preds) > 0 else None

            val_thresh, calib_info = select_threshold_for_target_fpr(
                y_or_state_val=st_val["state_code"][va_cond],
                s_val=mean_p_va,
                target_fpr=0.01,
                confirmed_neg_mask=st_val["confirmed_neg_mask"][va_cond],
                fallback_mature_neg_scores=mean_p_mb,
                posterior_q_val=st_val["q_posterior"][va_cond],
                min_confirmed_negatives=30,
                return_info=True,
            )

            bm = compute_block_metrics(
                y_true=true_y[te_cond],
                y_score=mean_p_te,
                card_proxy_ids=card1_arr[te_cond],
                amounts=amounts[te_cond],
                val_threshold_1pct_fpr=val_thresh,
                val_calibration_info=calib_info,
            )
            bm["block_id"] = b_id
            bm["estimated_pi_tau_train"] = round(pi_tau, 5)
            seed_aps = [round(compute_average_precision(true_y[te_cond], sp), 5) for sp in seed_test_preds]
            bm["per_seed_AP"] = seed_aps
            bm["seed_std_AP"] = round(float(np.std(seed_aps, ddof=1 if len(seed_aps) > 1 else 0)), 5)
            block_results.append(bm)

            all_test_preds.append(mean_p_te)
            all_test_y.append(true_y[te_cond])
            all_test_has_nbr.append(has_vis_2hop[te_cond])

        runtime_sec = round(time.perf_counter() - t0, 3)
        rusage_maxrss_mb = _get_proc_rss_mb()
        traced_mb = tier_tracemalloc_mb.get(model_tier, 0.0)

        metric_keys = [
            "AP", "PR_AUC_trapezoidal", "ROC_AUC", "Precision@0.5%", "Recall@0.5%",
            "Precision@1%", "Recall@1%", "Precision@5%", "Recall@5%",
            "Recall@1%FPR", "Realized_Test_FPR", "CardProxy_Precision@25", "Brier", "LogLoss"
        ]
        macro_summary = {}
        for mk in metric_keys:
            vals = np.array([b[mk] for b in block_results], dtype=np.float64)
            macro_summary[mk] = {
                "mean": round(float(np.mean(vals)), 5),
                "std_across_blocks": round(float(np.std(vals, ddof=1)), 5),
                "median": round(float(np.median(vals)), 5),
                "per_block": [round(float(x), 5) for x in vals],
            }

        n_s = len(eval_seeds)
        seed_macro_aps = [
            round(float(np.mean([b["per_seed_AP"][s_i] for b in block_results])), 5)
            for s_i in range(n_s)
        ]
        macro_summary["seed_stability_AP"] = {
            "seeds": list(eval_seeds),
            "per_seed_macro_AP": seed_macro_aps,
            "mean_across_seeds": round(float(np.mean(seed_macro_aps)), 5),
            "std_across_seeds": round(float(np.std(seed_macro_aps, ddof=1 if n_s > 1 else 0)), 5),
        }

        y_pool = np.concatenate(all_test_y)
        p_pool = np.concatenate(all_test_preds)
        nbr_pool = np.concatenate(all_test_has_nbr)
        m_with = (nbr_pool == 1)
        m_iso = (nbr_pool == 0)

        stratum_summary = {
            "with_visible_prior_fraud_neighbor_2hop": {
                "n_tx": int(np.sum(m_with)),
                "n_frauds": int(np.sum(y_pool[m_with] == 1)),
                "share_of_all_test_frauds": round(float(np.sum(y_pool[m_with] == 1) / max(np.sum(y_pool == 1), 1)), 4),
                "fraud_prevalence": round(float(np.mean(y_pool[m_with])), 5) if np.any(m_with) else 0.0,
                "AP": round(compute_average_precision(y_pool[m_with], p_pool[m_with]), 5) if np.any(m_with) else 0.0,
                "Precision@1%": round(compute_alert_budget_metrics(y_pool[m_with], p_pool[m_with], 1.0)[0], 5) if np.any(m_with) else 0.0,
            },
            "isolated_no_visible_fraud_neighbor_2hop": {
                "n_tx": int(np.sum(m_iso)),
                "n_frauds": int(np.sum(y_pool[m_iso] == 1)),
                "share_of_all_test_frauds": round(float(np.sum(y_pool[m_iso] == 1) / max(np.sum(y_pool == 1), 1)), 4),
                "fraud_prevalence": round(float(np.mean(y_pool[m_iso])), 5) if np.any(m_iso) else 0.0,
                "AP": round(compute_average_precision(y_pool[m_iso], p_pool[m_iso]), 5) if np.any(m_iso) else 0.0,
                "Precision@1%": round(compute_alert_budget_metrics(y_pool[m_iso], p_pool[m_iso], 1.0)[0], 5) if np.any(m_iso) else 0.0,
            }
        }

        return {
            "model_tier": model_tier,
            "protocol_cell": protocol_cell,
            "delay_scenario": delay_scen,
            "survival_assumed": surv_scen,
            "label_rule": label_rule,
            "prior_mode": prior_mode,
            "relations": list(rels),
            "K": K,
            "W": W,
            "inv_log_hub_weight": inv_log,
            "sliding_window_days": sliding_window_days,
            "seeds": list(eval_seeds),
            "runtime_sec": runtime_sec,
            "tracemalloc_peak_mb": traced_mb,
            "process_max_rss_mb": rusage_maxrss_mb,
            "memory_measurement_method": "python_tracemalloc_peak_and_proc_status_vmrss",
            "macro_summary": macro_summary,
            "block_results": block_results,
            "stratum_h3": stratum_summary,
        }

    print("[Steps 7-9] Running Primary Model Comparison (Table 3) under Operational Protocol P3 + D2...")
    primary_runs = {}
    for m_tier, l_rule in [
        ("B0_raw_tabular", "V0"),
        ("B0_raw_tabular", "V2"),
        ("B1_causal_aggregates", "V0"),
        ("B1_causal_aggregates", "V1"),
        ("B1_causal_aggregates", "V2"),
        ("B2_graph_light", "V2"),
        ("B3_graphsage", "V0"),
        ("B3_graphsage", "V2"),
        ("B4_domain_care_gnn", "V0"),
        ("B4_domain_care_gnn", "V2"),
        ("B5_proposed_maturity_hybrid", "V0"),
        ("B5_proposed_maturity_hybrid", "V1"),
        ("B5_proposed_maturity_hybrid", "V2"),
        ("B5_proposed_maturity_hybrid", "V3"),
    ]:
        run_key = f"{m_tier}__{l_rule}__P3__D2"
        print(f"   -> Running {run_key} ...")
        primary_runs[run_key] = evaluate_rolling_experiment(
            model_tier=m_tier, protocol_cell="P3", delay_scen="D2", label_rule=l_rule
        )

    print("[Steps 3-4 & 10] Running Two-Axis Leakage Attribution Matrix (P0, P1, P2, P3) for RQ1 / H1...")
    leakage_runs = {}
    for p_cell, l_rule in [
        ("P0", "V3"),
        ("P1", "V0"),
        ("P1", "V2"),
        ("P2", "V0"),
        ("P2", "V1"),
        ("P3", "V0"),
        ("P3", "V2"),
    ]:
        for m_tier in ["B1_causal_aggregates", "B3_graphsage", "B4_domain_care_gnn", "B5_proposed_maturity_hybrid"]:
            r_key = f"{m_tier}__{p_cell}__{l_rule}__D2"
            if p_cell == "P3" and l_rule in ("V0", "V2"):
                leakage_runs[r_key] = primary_runs[f"{m_tier}__{l_rule}__P3__D2"]
            else:
                print(f"   -> Running leakage cell {r_key} ...")
                leakage_runs[r_key] = evaluate_rolling_experiment(
                    model_tier=m_tier, protocol_cell=p_cell, delay_scen="D2", label_rule=l_rule
                )

    print("[Steps 5-6 & 12] Running V0/V1/V2/V3 across Delay Scenarios D0, D1, D2, D3 (Table 5 / RQ2 / H2)...")
    maturity_grid_runs = {}
    for d_scen in ["D0", "D1", "D2", "D3"]:
        for l_rule in ["V0", "V1", "V2", "V3"]:
            if d_scen == "D0" and l_rule in ("V1", "V2"):
                maturity_grid_runs[f"B5__{d_scen}__{l_rule}"] = maturity_grid_runs[f"B5__{d_scen}__V0"]
                continue
            r_key = f"B5__{d_scen}__{l_rule}"
            if d_scen == "D2" and f"B5_proposed_maturity_hybrid__{l_rule}__P3__D2" in primary_runs:
                maturity_grid_runs[r_key] = primary_runs[f"B5_proposed_maturity_hybrid__{l_rule}__P3__D2"]
            else:
                print(f"   -> Running maturity grid {r_key} ...")
                maturity_grid_runs[r_key] = evaluate_rolling_experiment(
                    model_tier="B5_proposed_maturity_hybrid", protocol_cell="P3", delay_scen=d_scen, label_rule=l_rule
                )

    print("[Steps 8-9 & 14] Running Graph Relation, Entity-Proxy, and Hub-Control Ablations (Table 6 / RQ3 / H3-H4)...")
    ablation_runs = {}
    ablation_configs = [
        ("Full_5_Relations_K50_InvLog", primary_relations, 50, 30.0, True),
        ("Drop_Device_Proxy", ["card1", "terminal_id", "addr1", "P_emaildomain"], 50, 30.0, True),
        ("Drop_Email_Domain_Proxy", ["card1", "terminal_id", "addr1", "DeviceInfo"], 50, 30.0, True),
        ("Drop_Address_Proxy", ["card1", "terminal_id", "P_emaildomain", "DeviceInfo"], 50, 30.0, True),
        ("Card_and_Terminal_Only", ["card1", "terminal_id"], 50, 30.0, True),
        ("Composite_Card_Proxy_card_comp", ["card_comp", "terminal_id", "addr1", "P_emaildomain", "DeviceInfo"], 50, 30.0, True),
        ("Hub_Uncapped_NoWeight_KNone", primary_relations, None, None, False),
        ("Hub_TightCap_K10_InvLog", primary_relations, 10, 30.0, True),
        ("Hub_WideCap_K200_InvLog", primary_relations, 200, 30.0, True),
    ]
    for ab_name, rel_list, k_val, w_val, inv_l in ablation_configs:
        if ab_name == "Full_5_Relations_K50_InvLog":
            ablation_runs[ab_name] = primary_runs["B5_proposed_maturity_hybrid__V2__P3__D2"]
            continue
        print(f"   -> Running ablation {ab_name} ...")
        ablation_runs[ab_name] = evaluate_rolling_experiment(
            model_tier="B5_proposed_maturity_hybrid",
            protocol_cell="P3",
            delay_scen="D2",
            label_rule="V2",
            rels=rel_list,
            K=k_val,
            W=w_val,
            inv_log=inv_l,
        )

    print("[Steps 12-15] Running Robustness & Sensitivity Suite (Delay Misspecification, Prior Shift, Premature M=30 on D3, Sliding 60d Window)...")
    robustness_runs = {
        "D1_True__Trained_with_D2_Survival": evaluate_rolling_experiment(
            model_tier="B5_proposed_maturity_hybrid", protocol_cell="P3", delay_scen="D1", label_rule="V2", survival_assumed_scen="D2"
        ),
        "D3_True__Trained_with_D2_Survival_M90": evaluate_rolling_experiment(
            model_tier="B5_proposed_maturity_hybrid", protocol_cell="P3", delay_scen="D3", label_rule="V2", survival_assumed_scen="D2"
        ),
        "D3_Premature_Maturity_Horizon_M30_V2": evaluate_rolling_experiment(
            model_tier="B5_proposed_maturity_hybrid", protocol_cell="P3", delay_scen="D3_misspecified_M30", label_rule="V2"
        ),
        "D3_Premature_Maturity_Horizon_M30_V1": evaluate_rolling_experiment(
            model_tier="B5_proposed_maturity_hybrid", protocol_cell="P3", delay_scen="D3_misspecified_M30", label_rule="V1"
        ),
        "D2_Prior_Underestimated_0.5x": evaluate_rolling_experiment(
            model_tier="B5_proposed_maturity_hybrid", protocol_cell="P3", delay_scen="D2", label_rule="V2", prior_mode="underestimated_0.5x"
        ),
        "D2_Prior_Overestimated_2.0x": evaluate_rolling_experiment(
            model_tier="B5_proposed_maturity_hybrid", protocol_cell="P3", delay_scen="D2", label_rule="V2", prior_mode="overestimated_2.0x"
        ),
        "D2_Prior_Oracle_True_Rate": evaluate_rolling_experiment(
            model_tier="B5_proposed_maturity_hybrid", protocol_cell="P3", delay_scen="D2", label_rule="V2", prior_mode="oracle_true_rate"
        ),
        "D2_Sliding_Window_60d": evaluate_rolling_experiment(
            model_tier="B5_proposed_maturity_hybrid", protocol_cell="P3", delay_scen="D2", label_rule="V2", sliding_window_days=60.0
        ),
    }

    print("[Step 13] Computing Paired Block Differences, Moving-Block Bootstrap 95% CIs, and Holm-Adjusted Wilcoxon Tests...")
    stat_comparisons = []
    comparison_specs = [
        ("H1_Leakage_GNN_P0_vs_P3_AP",
         leakage_runs["B4_domain_care_gnn__P0__V3__D2"]["macro_summary"]["AP"]["per_block"],
         leakage_runs["B4_domain_care_gnn__P3__V2__D2"]["macro_summary"]["AP"]["per_block"],
         "RQ1/H1: Retrospective Oracle (P0) vs. Causal Operational (P3) on Similarity-Gated GNN B4 (AP)"),
        ("H1_Leakage_Transductive_P1_vs_Causal_P3_AP",
         leakage_runs["B4_domain_care_gnn__P1__V2__D2"]["macro_summary"]["AP"]["per_block"],
         leakage_runs["B4_domain_care_gnn__P3__V2__D2"]["macro_summary"]["AP"]["per_block"],
         "RQ1/H1: Transductive Topology (P1) vs. Causal Topology (P3) holding V2 fixed on B4 (AP)"),
        ("H2_Maturity_V2_vs_V0_Prec1pct_D2",
         primary_runs["B5_proposed_maturity_hybrid__V2__P3__D2"]["macro_summary"]["Precision@1%"]["per_block"],
         primary_runs["B5_proposed_maturity_hybrid__V0__P3__D2"]["macro_summary"]["Precision@1%"]["per_block"],
         "RQ2/H2: Maturity-Aware V2 vs. Naive Unreported-as-Negative V0 under D2 on B5 (Precision@1%)"),
        ("H2_Maturity_V2_vs_V1_Prec1pct_D2",
         primary_runs["B5_proposed_maturity_hybrid__V2__P3__D2"]["macro_summary"]["Precision@1%"]["per_block"],
         primary_runs["B5_proposed_maturity_hybrid__V1__P3__D2"]["macro_summary"]["Precision@1%"]["per_block"],
         "RQ2/H2: Maturity-Aware V2 vs. Drop-Immature V1 under D2 on B5 (Precision@1%)"),
        ("H2_Maturity_V2_vs_V0_AP_D2",
         primary_runs["B5_proposed_maturity_hybrid__V2__P3__D2"]["macro_summary"]["AP"]["per_block"],
         primary_runs["B5_proposed_maturity_hybrid__V0__P3__D2"]["macro_summary"]["AP"]["per_block"],
         "RQ2/H2: Maturity-Aware V2 vs. Naive Unreported-as-Negative V0 under D2 on B5 (AP)"),
        ("H2_Maturity_V2_vs_V1_AP_D2",
         primary_runs["B5_proposed_maturity_hybrid__V2__P3__D2"]["macro_summary"]["AP"]["per_block"],
         primary_runs["B5_proposed_maturity_hybrid__V1__P3__D2"]["macro_summary"]["AP"]["per_block"],
         "RQ2/H2: Maturity-Aware V2 vs. Drop-Immature V1 under D2 on B5 (AP)"),
        ("H2_Maturity_V2_vs_V1_AP_D3_LongTail",
         maturity_grid_runs["B5__D3__V2"]["macro_summary"]["AP"]["per_block"],
         maturity_grid_runs["B5__D3__V1"]["macro_summary"]["AP"]["per_block"],
         "RQ2/H2: Maturity-Aware V2 vs. Drop-Immature V1 under 90-day Long-Tail D3 on B5 (AP)"),
        ("RQ3_Proposed_B5_vs_RawTabular_B0_AP",
         primary_runs["B5_proposed_maturity_hybrid__V2__P3__D2"]["macro_summary"]["AP"]["per_block"],
         primary_runs["B0_raw_tabular__V2__P3__D2"]["macro_summary"]["AP"]["per_block"],
         "RQ3: Proposed Maturity-Aware Hybrid (B5) vs. Raw Tabular (B0) under P3, D2 (AP)"),
        ("RQ3_Proposed_B5_vs_CausalTreeAgg_B1_AP",
         primary_runs["B5_proposed_maturity_hybrid__V2__P3__D2"]["macro_summary"]["AP"]["per_block"],
         primary_runs["B1_causal_aggregates__V2__P3__D2"]["macro_summary"]["AP"]["per_block"],
         "RQ3: Proposed Hybrid (B5) vs. Causal Neighbor-Aggregate Tree (B1) under P3, D2 (AP)"),
        ("H4_HubControl_K50_vs_Uncapped_AP",
         ablation_runs["Full_5_Relations_K50_InvLog"]["macro_summary"]["AP"]["per_block"],
         ablation_runs["Hub_Uncapped_NoWeight_KNone"]["macro_summary"]["AP"]["per_block"],
         "RQ3/H4: Hub Control (K=50 + InvLog Weight) vs. Uncapped/Unweighted Graph on B5 (AP)"),
    ]

    raw_pvals = []
    for comp_id, vals_a, vals_b, desc in comparison_specs:
        res = paired_temporal_block_analysis(np.array(vals_a), np.array(vals_b), block_len=2, n_boot=5000, seed=42)
        res["comparison_id"] = comp_id
        res["description"] = desc
        stat_comparisons.append(res)
        raw_pvals.append(res["wilcoxon_p_unadjusted"])

    holm_pvals = holm_bonferroni_correction(raw_pvals)
    for idx, hp in enumerate(holm_pvals):
        stat_comparisons[idx]["wilcoxon_p_holm_adjusted"] = hp

    env_provenance = {
        "run_id": run_id,
        "execution_timestamp": "2026-10-24T23:25:00+05:00",
        "project_root": str(PROJECT_ROOT),
        "output_dir": str(output_dir),
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "external_packages": get_external_package_availability(),
        "generator_version": SIMULATOR_GENERATOR_VERSION,
        "generator_seed": 20261024,
        "delay_seed": 7007,
        "model_evaluation_seeds": list(seeds),
        "bootstrap_seed": 42,
        "split_manifest_path": str(split_manifest_path),
        "split_manifest_sha256": split_manifest.get("manifest_sha256"),
        "tier_tracemalloc_peak_mb": tier_tracemalloc_mb,
        "process_peak_rss_mb": _get_proc_rss_mb(),
    }

    master_bundle = {
        "run_id": run_id,
        "execution_timestamp": env_provenance["execution_timestamp"],
        "provenance": env_provenance,
        "total_wall_time_sec": round(time.time() - t_global_start, 2),
        "step1_audit": audit_report,
        "split_manifest": split_manifest,
        "delay_scenarios": delay_audit_summary,
        "q_calibration": q_calibration_bins,
        "primary_runs_table3": primary_runs,
        "leakage_matrix_runs": leakage_runs,
        "maturity_grid_table4": maturity_grid_runs,
        "ablation_runs_table5": ablation_runs,
        "robustness_runs_table6": robustness_runs,
        "statistical_comparisons": stat_comparisons,
    }

    bundle_v2_path = output_dir / "manifests/frozen_results_bundle_v2.json"
    bundle_compat_path = output_dir / "manifests/frozen_results_bundle.json"
    with open(bundle_v2_path, "w") as f:
        json.dump(master_bundle, f, indent=2)
    with open(bundle_compat_path, "w") as f:
        json.dump(master_bundle, f, indent=2)

    print(
        f"\n[SUCCESS] Completed all experiments for run_id='{run_id}' in {master_bundle['total_wall_time_sec']:.2f}s. "
        f"Saved {bundle_v2_path}."
    )
    return master_bundle


def main():
    parser = argparse.ArgumentParser(description="Run Project Delta corrected experimental suite.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.environ.get("PROJECT_DELTA_OUTPUT_DIR", str(PROJECT_ROOT / "results")),
        help="Directory where manifests, tables, and figures will be written.",
    )
    parser.add_argument(
        "--split-manifest",
        type=str,
        default=str(PROJECT_ROOT / "configs/splits/rolling_v1.json"),
        help="Path to the authoritative temporal split manifest JSON.",
    )
    parser.add_argument(
        "--regenerate-manifest",
        action="store_true",
        help="If set, regenerates the split manifest instead of loading the existing manifest.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="101,202,303,404,505",
        help="Comma-separated list of random seeds for model training and evaluation.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="project_delta_v2_reconciled_20261024",
        help="Unique run identifier recorded in the frozen results bundle.",
    )
    args = parser.parse_args()
    seed_list = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    run_experiment_suite(
        output_dir=Path(args.output_dir).resolve(),
        split_manifest_path=Path(args.split_manifest).resolve(),
        regenerate_manifest=args.regenerate_manifest,
        seeds=seed_list,
        run_id=args.run_id,
    )


if __name__ == "__main__":
    main()
