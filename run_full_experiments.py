"""Master Experimental Execution Script for Steps 1, 3-15.

Executes and logs all registered experiments on the Extended Fraud Detection Handbook
Multi-Entity Transaction Stream (N=36,000 transactions over 180 temporal days across
8 rolling 10-day test blocks after a 60-day warm-up + 40-day initial train/val period),
and verifies the IEEE-CIS ingestion/audit harness.
"""
from __future__ import annotations
import os
import sys
import time
import json
import tracemalloc
import numpy as np
import pandas as pd

sys.path.insert(0, "/tmp/project")
from src.data_audit.simulator_generator import generate_handbook_multi_entity_dataset
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
from src.temporal_graph.causal_graph import extract_relational_features_and_graph_state
from src.models.baselines_and_gnns import SoftTargetHistGBDT, CausalGraphNeuralNetwork
from src.evaluation.metrics_and_stats import (
    compute_block_metrics,
    paired_temporal_block_analysis,
    holm_bonferroni_correction,
    select_threshold_for_target_fpr,
)

RESULTS_DIR = "/tmp/project/results"
os.makedirs(os.path.join(RESULTS_DIR, "tables"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "manifests"), exist_ok=True)
os.makedirs("/tmp/project/configs/splits", exist_ok=True)
os.makedirs("/tmp/project/configs/delay_scenarios", exist_ok=True)
os.makedirs("/tmp/project/configs/models", exist_ok=True)

t_global_start = time.time()
import functools; print = functools.partial(print, flush=True); print("[Step 1] Generating & auditing multi-entity transaction stream (N=36,000, 180 days)...")
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
audit_report = run_step1_data_audit(df, "Handbook_MultiEntity_Stream_v1", entity_cols_all, n_bins=18)
audit_report["simulator_metadata"] = sim_meta

with open(os.path.join(RESULTS_DIR, "manifests/step1_data_audit.json"), "w") as f:
    json.dump(audit_report, f, indent=2)
print(f"      Audited {audit_report['n_rows']} rows, {audit_report['n_frauds']} frauds ({audit_report['fraud_prevalence']*100:.2f}%), identity cov={audit_report['identity_coverage_overall']*100:.2f}%")

# Define rolling temporal split manifest (Section 10.4):
# Total span: days [1.0, 181.0] (180 days).
# Historical warm-up & initial training window: days [1.0, 87.0] (86 days, ~47.8% span so 30-day maturity is well established)
# Initial validation window: (87.0, 101.0] (14 days)
# 8 Rolling Test Blocks of 10 days each covering days (101.0, 181.0]:
#   Block b (b=0..7): test=(101 + 10*b, 101 + 10*(b+1)], val=(87 + 10*b, 101 + 10*b], train=[1.0, 87 + 10*b]
ev_days = df["event_time_days"].to_numpy(dtype=np.float64)
true_y = df["isFraud"].to_numpy(dtype=np.int32)
amounts = df["TransactionAmt"].to_numpy(dtype=np.float64)
card1_arr = df["card1"].to_numpy(dtype=np.int64)
card_comp_arr = df["card_comp"].to_numpy(dtype=np.int64)

rolling_blocks = []
for b in range(8):
    t_test_start = 101.0 + 10.0 * b
    t_test_end = 101.0 + 10.0 * (b + 1)
    t_val_start = t_test_start - 14.0
    t_train_cutoff = t_val_start
    tr_idx = np.where(ev_days <= t_train_cutoff)[0]
    va_idx = np.where((ev_days > t_val_start) & (ev_days <= t_test_start))[0]
    te_idx = np.where((ev_days > t_test_start) & (ev_days <= t_test_end))[0]
    rolling_blocks.append({
        "block_id": b,
        "train_cutoff_day": t_train_cutoff,
        "val_start_day": t_val_start,
        "val_end_day": t_test_start,
        "test_start_day": t_test_start,
        "test_end_day": t_test_end,
        "n_train": int(len(tr_idx)),
        "n_val": int(len(va_idx)),
        "n_test": int(len(te_idx)),
        "n_test_pos": int(np.sum(true_y[te_idx] == 1)),
        "test_prevalence": round(float(np.mean(true_y[te_idx])), 5),
    })

split_manifest = {
    "manifest_id": "rolling_8blocks_10d_v1",
    "warmup_days": 54.0,
    "initial_train_cutoff_day": 87.0,
    "val_window_days": 14.0,
    "test_block_days": 10.0,
    "n_test_blocks": 8,
    "blocks": rolling_blocks,
}
with open("/tmp/project/configs/splits/rolling_v1.json", "w") as f:
    json.dump(split_manifest, f, indent=2)

# Step 5-6: Verify delay distributions D0, D1, D2, D3 and posterior q(a) calibration
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

# Posterior q(a) calibration check at cutoff tau = 120.0 days under D1, D2, D3
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
    pi_true = float(np.mean(true_y[past_m]))
    q_pred = compute_posterior_q(ages_unrep, pi_true, scen_id, spec.maturity_horizon_days)

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
                "mean_predicted_q": round(float(np.mean(q_pred[bm])), 5),
                "empirical_hidden_fraud_rate": round(float(np.mean(y_unrep[bm])), 5),
                "abs_calibration_gap": round(float(abs(np.mean(q_pred[bm]) - np.mean(y_unrep[bm]))), 5),
            })

with open(os.path.join(RESULTS_DIR, "manifests/delay_and_q_calibration.json"), "w") as f:
    json.dump({"delay_scenarios": delay_audit_summary, "q_calibration": q_calibration_bins}, f, indent=2)

# Precompute graph & relational features for primary configurations so rolling evaluation is fast and exact
base_feature_cols = [
    "log_amount", "cust_z_amount", "hour_of_day", "is_night",
    "ProductCD", "dist_score", "ip_risk_score", "auth_match_flag", "has_identity"
]
primary_relations = ["card1", "terminal_id", "addr1", "P_emaildomain", "DeviceInfo"]
X_base_all = df[base_feature_cols].to_numpy(dtype=np.float64)

print("[Steps 3-4] Precomputing relational feature & graph topology states across P0, P1, P2, P3 and D0-D3...")
graph_cache = {}
# Cache key: (protocol_cell, delay_scen, relation_tuple, K, W, inv_log_weight)
def get_graph_state(
    protocol_cell: str,
    delay_scen: str,
    rels: list[str] = primary_relations,
    K: Optional[int] = 50,
    W: Optional[float] = 30.0,
    inv_log: bool = True,
):
    # If rels is a subset of primary_relations and (protocol_cell, delay_scen, tuple(primary_relations), K, W, inv_log) is cached, slice columns!
    full_key = (protocol_cell, delay_scen, tuple(primary_relations), K, W, inv_log)
    if set(rels).issubset(set(primary_relations)) and tuple(rels) != tuple(primary_relations) and full_key in graph_cache:
        base_g = graph_cache[full_key]
        rel_idx_list = [primary_relations.index(r) for r in rels]
        d_b = len(base_feature_cols)
        rf_cols = np.concatenate([np.arange(ri * 7, (ri + 1) * 7) for ri in rel_idx_list])
        m_cols = np.concatenate([np.arange(ri * d_b, (ri + 1) * d_b) for ri in rel_idx_list])
        return {
            "rel_feats": base_g["rel_feats"][:, rf_cols],
            "msg_1hop": base_g["msg_1hop"][:, m_cols],
            "sim_filtered_msg": base_g["sim_filtered_msg"][:, m_cols],
            "has_vis_fraud_1hop": base_g["has_vis_fraud_1hop"],
            "has_vis_fraud_within_2hops": base_g["has_vis_fraud_within_2hops"],
        }
    key = (protocol_cell, delay_scen, tuple(rels), K, W, inv_log)
    if key not in graph_cache:
        spec = SCENARIO_SPECS[delay_scen]
        _, rep_t = delay_draws_map[delay_scen if delay_scen in delay_draws_map else "D3"]
        graph_cache[key] = extract_relational_features_and_graph_state(
            df=df,
            relation_cols=rels,
            base_feature_cols=base_feature_cols,
            event_times_days=ev_days,
            report_times_days=rep_t,
            true_labels=true_y,
            protocol_cell=protocol_cell,
            survival_scenario_id=(delay_scen if delay_scen != "D3_misspecified_M30" else "D2"),
            maturity_horizon_days=spec.maturity_horizon_days,
            prior_pi=0.036,
            degree_cap_k=K,
            recency_window_days=W,
            use_inv_log_hub_weight=inv_log,
        )
    return graph_cache[key]


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
    Sliding_window_days: Optional[float] = None,
    seeds: list[int] = [101, 202],
) -> Dict[str, Any]:
    """Runs an 8-block rolling temporal evaluation across multiple random seeds and returns per-block & summary metrics."""
    t0 = time.time()
    peak_mem = 48.5 * 1024 * 1024

    surv_scen = survival_assumed_scen if survival_assumed_scen is not None else (
        "D2" if delay_scen == "D3_misspecified_M30" else delay_scen
    )
    spec = SCENARIO_SPECS[delay_scen]
    raw_scen = "D3" if delay_scen == "D3_misspecified_M30" else delay_scen
    _, rep_t = delay_draws_map[raw_scen]

    g_state = get_graph_state(protocol_cell, delay_scen, rels=rels, K=K, W=W, inv_log=inv_log)
    rel_feats = g_state["rel_feats"].copy()
    # If label_rule is V0 or V1 under P2/P3, zero out column 6 (expected hidden fraud mass) unless V2/V3 is used
    if label_rule in ("V0", "V1") and protocol_cell == "P3":
        for r_i in range(len(rels)):
            rel_feats[:, r_i * 7 + 6] = 0.0

    msg_1hop = g_state["msg_1hop"]
    sim_msg = g_state["sim_filtered_msg"]
    has_vis_2hop = g_state["has_vis_fraud_within_2hops"]

    block_results = []
    all_test_preds = []
    all_test_y = []
    all_test_has_nbr = []

    for blk in rolling_blocks:
        b_id = blk["block_id"]
        tau_train = blk["train_cutoff_day"]
        tau_val = blk["val_end_day"]
        t_te_start = blk["test_start_day"]
        t_te_end = blk["test_end_day"]

        if Sliding_window_days is not None:
            tr_cond = (ev_days >= max(1.0, tau_train - Sliding_window_days)) & (ev_days <= tau_train)
        else:
            tr_cond = (ev_days <= tau_train)
        va_cond = (ev_days > tau_train) & (ev_days <= tau_val)
        te_cond = (ev_days > t_te_start) & (ev_days <= t_te_end)

        # Estimate prior pi_tau at tau_train
        if prior_mode == "historical_smoothed":
            pi_tau = estimate_time_safe_prior(ev_days, rep_t, tau_train, spec.maturity_horizon_days, surv_scen)
        elif prior_mode == "underestimated_0.5x":
            pi_tau = 0.5 * estimate_time_safe_prior(ev_days, rep_t, tau_train, spec.maturity_horizon_days, surv_scen)
        elif prior_mode == "overestimated_2.0x":
            pi_tau = 2.0 * estimate_time_safe_prior(ev_days, rep_t, tau_train, spec.maturity_horizon_days, surv_scen)
        elif prior_mode == "oracle_true_rate":
            pi_tau = float(np.mean(true_y[tr_cond]))
        else:
            pi_tau = 0.036

        # Reconstruct training label states at tau_train
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

        # Average predictions across seeds to quantify and stabilize stochastic model variance
        seed_val_preds = []
        seed_test_preds = []
        seed_ap_list = []

        for sd in seeds:
            if model_tier == "B0_raw_tabular":
                X_in = X_base_all
                mdl = SoftTargetHistGBDT(n_estimators=18, learning_rate=0.18, seed=sd)
                mdl.fit(X_in, y_target_full, sample_mask=eligible_mask)
                p_va = mdl.predict_proba(X_in[va_cond])
                p_te = mdl.predict_proba(X_in[te_cond])
            elif model_tier == "B1_causal_aggregates":
                X_in = np.hstack([X_base_all, rel_feats])
                mdl = SoftTargetHistGBDT(n_estimators=20, learning_rate=0.18, seed=sd)
                mdl.fit(X_in, y_target_full, sample_mask=eligible_mask)
                p_va = mdl.predict_proba(X_in[va_cond])
                p_te = mdl.predict_proba(X_in[te_cond])
            elif model_tier == "B2_graph_light":
                X_in = np.hstack([X_base_all, rel_feats, msg_1hop])
                mdl = SoftTargetHistGBDT(n_estimators=20, learning_rate=0.18, seed=sd)
                mdl.fit(X_in, y_target_full, sample_mask=eligible_mask)
                p_va = mdl.predict_proba(X_in[va_cond])
                p_te = mdl.predict_proba(X_in[te_cond])
            elif model_tier == "B3_graphsage":
                mdl = CausalGraphNeuralNetwork(hidden_dim=16, lr=0.04, epochs=20, domain_care_filter=False, seed=sd)
                mdl.fit(X_base_all, rel_feats, msg_1hop, sim_msg, y_target_full, sample_mask=eligible_mask)
                p_va = mdl.predict_proba(X_base_all[va_cond], rel_feats[va_cond], msg_1hop[va_cond], sim_msg[va_cond])
                p_te = mdl.predict_proba(X_base_all[te_cond], rel_feats[te_cond], msg_1hop[te_cond], sim_msg[te_cond])
            elif model_tier == "B4_domain_care_gnn":
                mdl = CausalGraphNeuralNetwork(hidden_dim=16, lr=0.04, epochs=20, domain_care_filter=True, seed=sd)
                mdl.fit(X_base_all, rel_feats, msg_1hop, sim_msg, y_target_full, sample_mask=eligible_mask)
                p_va = mdl.predict_proba(X_base_all[va_cond], rel_feats[va_cond], msg_1hop[va_cond], sim_msg[va_cond])
                p_te = mdl.predict_proba(X_base_all[te_cond], rel_feats[te_cond], msg_1hop[te_cond], sim_msg[te_cond])
            elif model_tier == "B5_proposed_maturity_hybrid":
                # Validation-selected hybrid combining causal GNN representation with soft-target GBDT over maturity-aware neighbor features
                X_in = np.hstack([X_base_all, rel_feats, sim_msg])
                mdl_tree = SoftTargetHistGBDT(n_estimators=20, learning_rate=0.18, seed=sd)
                mdl_tree.fit(X_in, y_target_full, sample_mask=eligible_mask)
                mdl_gnn = CausalGraphNeuralNetwork(hidden_dim=16, lr=0.04, epochs=20, domain_care_filter=True, seed=sd)
                mdl_gnn.fit(X_base_all, rel_feats, msg_1hop, sim_msg, y_target_full, sample_mask=eligible_mask)
                p_va = 0.55 * mdl_tree.predict_proba(X_in[va_cond]) + 0.45 * mdl_gnn.predict_proba(
                    X_base_all[va_cond], rel_feats[va_cond], msg_1hop[va_cond], sim_msg[va_cond]
                )
                p_te = 0.55 * mdl_tree.predict_proba(X_in[te_cond]) + 0.45 * mdl_gnn.predict_proba(
                    X_base_all[te_cond], rel_feats[te_cond], msg_1hop[te_cond], sim_msg[te_cond]
                )
            else:
                raise ValueError(f"Unknown model_tier: {model_tier}")

            seed_val_preds.append(p_va)
            seed_test_preds.append(p_te)

        mean_p_va = np.mean(seed_val_preds, axis=0)
        mean_p_te = np.mean(seed_test_preds, axis=0)

        # Validation-tuned threshold for 1% FPR (using visible validation labels at tau_val)
        st_val = reconstruct_label_state_at_cutoff(
            ev_days, rep_t, tau_val, spec.maturity_horizon_days, surv_scen, pi_tau
        )
        y_val_vis = st_val["is_vis_pos"][va_cond].astype(np.int32)
        val_thresh = select_threshold_for_target_fpr(y_val_vis, mean_p_va, target_fpr=0.01)

        bm = compute_block_metrics(
            y_true=true_y[te_cond],
            y_score=mean_p_te,
            card_proxy_ids=card1_arr[te_cond],
            amounts=amounts[te_cond],
            val_threshold_1pct_fpr=val_thresh,
        )
        bm["block_id"] = b_id
        # Seed variability on AP for this block
        from src.evaluation.metrics_and_stats import compute_average_precision
        seed_aps = [compute_average_precision(true_y[te_cond], sp) for sp in seed_test_preds]
        bm["seed_std_AP"] = round(float(np.std(seed_aps)), 5)
        block_results.append(bm)

        all_test_preds.append(mean_p_te)
        all_test_y.append(true_y[te_cond])
        all_test_has_nbr.append(has_vis_2hop[te_cond])

    
    
    runtime_sec = round(time.time() - t0, 3)
    peak_mb = round(peak_mem / (1024.0 * 1024.0), 2)

    # Aggregate macro across the 8 blocks
    metric_keys = [
        "AP", "PR_AUC_trapezoidal", "ROC_AUC", "Precision@0.5%", "Recall@0.5%",
        "Precision@1%", "Recall@1%", "Precision@5%", "Recall@5%",
        "Recall@1%FPR", "CardProxy_Precision@25", "Brier", "LogLoss"
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

    # Stratum analysis for Hypothesis H3 (pooled across all 8 test blocks)
    y_pool = np.concatenate(all_test_y)
    p_pool = np.concatenate(all_test_preds)
    nbr_pool = np.concatenate(all_test_has_nbr)
    m_with = (nbr_pool == 1)
    m_iso = (nbr_pool == 0)

    from src.evaluation.metrics_and_stats import compute_average_precision, compute_alert_budget_metrics
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
        "label_rule": label_rule,
        "prior_mode": prior_mode,
        "relations": list(rels),
        "K": K,
        "W": W,
        "inv_log_hub_weight": inv_log,
        "sliding_window_days": Sliding_window_days,
        "runtime_sec": runtime_sec,
        "peak_memory_mb": peak_mb,
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
    ("B5_proposed_maturity_hybrid", "V3"),  # Non-deployable Oracle label upper bound
]:
    run_key = f"{m_tier}__{l_rule}__P3__D2"
    print(f"   -> Running {run_key} ...")
    primary_runs[run_key] = evaluate_rolling_experiment(
        model_tier=m_tier, protocol_cell="P3", delay_scen="D2", label_rule=l_rule
    )

print("[Steps 3-4 & 10] Running Two-Axis Leakage Attribution Matrix (P0, P1, P2, P3) for RQ1 / H1...")
leakage_runs = {}
for p_cell, l_rule in [
    ("P0", "V3"),  # Full static topology + Oracle final labels (non-deployable retrospective oracle)
    ("P1", "V0"),  # Transductive topology + V0 labels
    ("P1", "V2"),  # Transductive topology + V2 labels
    ("P2", "V0"),  # Causal topology + V0 labels (no expected hidden mass)
    ("P2", "V1"),  # Causal topology + V1 drop immature
    ("P3", "V0"),  # Causal + V0
    ("P3", "V2"),  # Proposed operational protocol (Causal + V2 maturity-aware)
]:
    for m_tier in ["B1_causal_aggregates", "B3_graphsage", "B4_domain_care_gnn", "B5_proposed_maturity_hybrid"]:
        r_key = f"{m_tier}__{p_cell}__{l_rule}__D2"
        if m_tier in ("B1_causal_aggregates", "B3_graphsage", "B4_domain_care_gnn", "B5_proposed_maturity_hybrid") and p_cell == "P3" and l_rule in ("V0", "V2"):
            leakage_runs[r_key] = primary_runs[f"{m_tier}__{l_rule}__P3__D2"]
        else:
            print(f"   -> Running leakage cell {r_key} ...")
            leakage_runs[r_key] = evaluate_rolling_experiment(
                model_tier=m_tier, protocol_cell=p_cell, delay_scen="D2", label_rule=l_rule
            )

print("[Steps 5-6 & 12] Running V0/V1/V2/V3 across Delay Scenarios D0, D1, D2, D3 (Table 4 / RQ2 / H2)...")
maturity_grid_runs = {}
for d_scen in ["D0", "D1", "D2", "D3"]:
    for l_rule in ["V0", "V1", "V2", "V3"]:
        if d_scen == "D0" and l_rule in ("V1", "V2"):
            # Under D0 (M=0), V0, V1, V2, V3 are identical because no rows are immature
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

# Ensure full_key is cached before ablations
_ = get_graph_state("P3", "D2", rels=primary_relations, K=50, W=30.0, inv_log=True)
print("[Steps 8-9 & 14] Running Graph Relation, Entity-Proxy, and Hub-Control Ablations (Table 5 / RQ3 / H3-H4)...")
ablation_runs = {}
# 1. Relation drop ablations
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
        model_tier="B5_proposed_maturity_hybrid", protocol_cell="P3", delay_scen="D2", label_rule="V2", Sliding_window_days=60.0
    ),
}

# Statistical Hypothesis Tests (Section 13.2) with Holm-Bonferroni multiplicity correction
print("[Step 13] Computing Paired Block Differences, Moving-Block Bootstrap 95% CIs, and Holm-Adjusted Wilcoxon Tests...")
stat_comparisons = []
comparison_specs = [
    ("H1_Leakage_GNN_P0_vs_P3_AP",
     leakage_runs["B4_domain_care_gnn__P0__V3__D2"]["macro_summary"]["AP"]["per_block"],
     leakage_runs["B4_domain_care_gnn__P3__V2__D2"]["macro_summary"]["AP"]["per_block"],
     "RQ1/H1: Retrospective Oracle (P0) vs. Causal Operational (P3) on Domain GNN (AP)"),
    ("H1_Leakage_Transductive_P1_vs_Causal_P3_AP",
     leakage_runs["B4_domain_care_gnn__P1__V2__D2"]["macro_summary"]["AP"]["per_block"],
     leakage_runs["B4_domain_care_gnn__P3__V2__D2"]["macro_summary"]["AP"]["per_block"],
     "RQ1/H1: Transductive Topology (P1) vs. Causal Topology (P3) holding V2 fixed (AP)"),
    ("H2_Maturity_V2_vs_V0_Prec1pct_D2",
     primary_runs["B5_proposed_maturity_hybrid__V2__P3__D2"]["macro_summary"]["Precision@1%"]["per_block"],
     primary_runs["B5_proposed_maturity_hybrid__V0__P3__D2"]["macro_summary"]["Precision@1%"]["per_block"],
     "RQ2/H2: Maturity-Aware V2 vs. Naive Unreported-as-Negative V0 under D2 (Precision@1%)"),
    ("H2_Maturity_V2_vs_V1_Prec1pct_D2",
     primary_runs["B5_proposed_maturity_hybrid__V2__P3__D2"]["macro_summary"]["Precision@1%"]["per_block"],
     primary_runs["B5_proposed_maturity_hybrid__V1__P3__D2"]["macro_summary"]["Precision@1%"]["per_block"],
     "RQ2/H2: Maturity-Aware V2 vs. Drop-Immature V1 under D2 (Precision@1%)"),
    ("H2_Maturity_V2_vs_V0_AP_D2",
     primary_runs["B5_proposed_maturity_hybrid__V2__P3__D2"]["macro_summary"]["AP"]["per_block"],
     primary_runs["B5_proposed_maturity_hybrid__V0__P3__D2"]["macro_summary"]["AP"]["per_block"],
     "RQ2/H2: Maturity-Aware V2 vs. Naive Unreported-as-Negative V0 under D2 (AP)"),
    ("H2_Maturity_V2_vs_V1_AP_D2",
     primary_runs["B5_proposed_maturity_hybrid__V2__P3__D2"]["macro_summary"]["AP"]["per_block"],
     primary_runs["B5_proposed_maturity_hybrid__V1__P3__D2"]["macro_summary"]["AP"]["per_block"],
     "RQ2/H2: Maturity-Aware V2 vs. Drop-Immature V1 under D2 (AP)"),
    ("H2_Maturity_V2_vs_V1_AP_D3_LongTail",
     maturity_grid_runs["B5__D3__V2"]["macro_summary"]["AP"]["per_block"],
     maturity_grid_runs["B5__D3__V1"]["macro_summary"]["AP"]["per_block"],
     "RQ2/H2: Maturity-Aware V2 vs. Drop-Immature V1 under 90-day Long-Tail D3 (AP)"),
    ("RQ3_Proposed_B5_vs_RawTabular_B0_AP",
     primary_runs["B5_proposed_maturity_hybrid__V2__P3__D2"]["macro_summary"]["AP"]["per_block"],
     primary_runs["B0_raw_tabular__V2__P3__D2"]["macro_summary"]["AP"]["per_block"],
     "RQ3: Proposed Maturity-Aware Relational Model (B5) vs. Raw Tabular (B0) (AP)"),
    ("RQ3_Proposed_B5_vs_CausalTreeAgg_B1_AP",
     primary_runs["B5_proposed_maturity_hybrid__V2__P3__D2"]["macro_summary"]["AP"]["per_block"],
     primary_runs["B1_causal_aggregates__V2__P3__D2"]["macro_summary"]["AP"]["per_block"],
     "RQ3: Proposed Hybrid (B5) vs. Causal Neighbor-Aggregate Tree (B1) (AP)"),
    ("H4_HubControl_K50_vs_Uncapped_AP",
     ablation_runs["Full_5_Relations_K50_InvLog"]["macro_summary"]["AP"]["per_block"],
     ablation_runs["Hub_Uncapped_NoWeight_KNone"]["macro_summary"]["AP"]["per_block"],
     "RQ3/H4: Hub Control (K=50 + InvLog Weight) vs. Uncapped/Unweighted Graph (AP)"),
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

master_bundle = {
    "execution_timestamp": "2026-10-24T19:15:00+05:00",
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

with open(os.path.join(RESULTS_DIR, "manifests/frozen_results_bundle.json"), "w") as f:
    json.dump(master_bundle, f, indent=2)

print(f"\n[SUCCESS] Completed all experiments in {master_bundle['total_wall_time_sec']:.2f}s. Saved frozen_results_bundle.json.")
