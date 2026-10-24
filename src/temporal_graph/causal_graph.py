"""Time-Respecting Graph Construction and Maturity-Aware Neighbor Aggregation (Sections 8, 9.7, 10 & Audit Issues 3, 5, 13).

Implements a strict two-pass heterogeneous graph feature and message-passing builder:
  - Pass 1 computes per-relation 1-hop causal statistics (using cutoff-specific time-safe priors
    pi_tau for expected hidden fraud mass), 1-hop neighbor feature aggregations (`msg_1hop`),
    similarity-filtered neighbor aggregations (`sim_filtered_msg`), and per-relation 1-hop
    minimum report times (`per_rel_min_rep_time`).
  - Pass 2 computes exact, relation-order-invariant and row-order-invariant 2-hop visible fraud
    reachability (`has_vis_fraud_2hop`, `has_vis_fraud_within_2hops`) and 2-hop cross-relation
    feature diffusion (`msg_2hop`).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Any


def compute_pass2_from_relation_slices(
    n: int,
    r_count: int,
    d_base: int,
    tau_vec: np.ndarray,
    msg_1hop: np.ndarray,
    per_rel_vis_1hop: np.ndarray,
    per_rel_min_rep_time: np.ndarray,
    per_rel_oracle_1hop: np.ndarray,
    relation_slices_cache: List[Any],
    use_inv_log_hub_weight: bool,
    use_oracle_neighbor_labels: bool,
) -> Dict[str, np.ndarray]:
    """Executes Pass 2 (2-hop visible fraud reachability and 2-hop feature diffusion) across
    the active relations in `relation_slices_cache`. Completely invariant to relation order.
    """
    if r_count > 0:
        has_vis_fraud_1hop = np.max(per_rel_vis_1hop, axis=1).astype(np.int8)
        min_1hop_prior_rep_time = np.min(per_rel_min_rep_time, axis=1)
        has_1hop_oracle_fraud = np.max(per_rel_oracle_1hop, axis=1).astype(np.int8)
        cross_rel_msg_1hop = np.mean(msg_1hop.reshape(n, r_count, d_base), axis=1)
    else:
        has_vis_fraud_1hop = np.zeros(n, dtype=np.int8)
        min_1hop_prior_rep_time = np.full(n, np.inf, dtype=np.float64)
        has_1hop_oracle_fraud = np.zeros(n, dtype=np.int8)
        cross_rel_msg_1hop = np.zeros((n, d_base), dtype=np.float64)

    has_vis_fraud_2hop = np.zeros(n, dtype=np.int8)
    msg_2hop = np.zeros((n, r_count * d_base), dtype=np.float64)

    for r_idx, rel_groups in enumerate(relation_slices_cache):
        msg_offset = r_idx * d_base
        for g_rows, strict_end_ptr, eff_start_ptr, w_hub_static, is_future_top in rel_groups:
            m = len(g_rows)
            g_taus = tau_vec[g_rows]
            g_m1_all = cross_rel_msg_1hop[g_rows]
            csum_m1 = np.vstack([np.zeros((1, d_base)), np.cumsum(g_m1_all, axis=0)])

            if is_future_top:
                total_deg = m - 1
                sum_m1_all = csum_m1[-1]
                g_min_rep = min_1hop_prior_rep_time[g_rows]
                g_orc = has_1hop_oracle_fraud[g_rows]
                tot_orc = int(g_orc.sum())
                for loc_i in range(m):
                    i = g_rows[loc_i]
                    msg_2hop[i, msg_offset : msg_offset + d_base] = (
                        (sum_m1_all - g_m1_all[loc_i]) / total_deg
                    ) * w_hub_static
                    if use_oracle_neighbor_labels:
                        if (tot_orc - int(g_orc[loc_i])) > 0:
                            has_vis_fraud_2hop[i] = 1
                    else:
                        end_p = int(strict_end_ptr[loc_i])
                        if end_p > 0 and float(g_min_rep[:end_p].min()) <= g_taus[loc_i]:
                            has_vis_fraud_2hop[i] = 1
            else:
                g_min_rep = min_1hop_prior_rep_time[g_rows]
                for loc_i in range(m):
                    end_p = int(strict_end_ptr[loc_i])
                    start_p = int(eff_start_ptr[loc_i])
                    cnt_n = end_p - start_p
                    if cnt_n <= 0:
                        continue
                    i = g_rows[loc_i]
                    w_hub = (1.0 / np.log(2.0 + end_p)) if use_inv_log_hub_weight else 1.0
                    if float(g_min_rep[start_p:end_p].min()) <= g_taus[loc_i]:
                        has_vis_fraud_2hop[i] = 1
                    mean_m1 = (csum_m1[end_p] - csum_m1[start_p]) / cnt_n
                    msg_2hop[i, msg_offset : msg_offset + d_base] = mean_m1 * w_hub

    has_vis_fraud_within_2hops = np.maximum(has_vis_fraud_1hop, has_vis_fraud_2hop)
    return {
        "msg_2hop": msg_2hop,
        "has_vis_fraud_1hop": has_vis_fraud_1hop,
        "has_vis_fraud_2hop": has_vis_fraud_2hop,
        "has_vis_fraud_within_2hops": has_vis_fraud_within_2hops,
    }


def extract_relational_features_and_graph_state(
    df: pd.DataFrame,
    relation_cols: List[str],
    base_feature_cols: List[str],
    event_times_days: np.ndarray,
    report_times_days: np.ndarray,
    true_labels: np.ndarray,
    protocol_cell: str,
    survival_scenario_id: str,
    maturity_horizon_days: float,
    prior_pi: Union[float, np.ndarray, str] = "time_safe",
    prior_mode: str = "historical_smoothed",
    degree_cap_k: Optional[int] = 50,
    recency_window_days: Optional[float] = 30.0,
    use_inv_log_hub_weight: bool = True,
    cutoff_tau_override: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    from src.delay_simulation.delay_models import (
        LAMBDA_D1_14D,
        D2_KNOTS_DAYS,
        D2_KNOTS_CDF,
        compute_time_safe_prior_trajectory,
    )

    n = len(df)
    r_count = len(relation_cols)
    d_base = len(base_feature_cols)
    X_norm = df[base_feature_cols].to_numpy(dtype=np.float64)
    tau_vec = (
        np.asarray(cutoff_tau_override, dtype=np.float64)
        if cutoff_tau_override is not None
        else event_times_days.copy()
    )

    if isinstance(prior_pi, str):
        if prior_pi == "time_safe":
            pi_vec = compute_time_safe_prior_trajectory(
                event_times_days=event_times_days,
                report_times_days=report_times_days,
                maturity_horizon_days=maturity_horizon_days,
                survival_scenario_id=survival_scenario_id,
                prior_mode=prior_mode,
                true_labels_for_oracle_only=true_labels if prior_mode == "oracle_true_rate" else None,
            )
        else:
            raise ValueError(f"Unknown prior_pi mode string: {prior_pi}")
    else:
        pi_arr = np.asarray(prior_pi, dtype=np.float64)
        if pi_arr.ndim == 0:
            base_val = float(pi_arr)
            if prior_mode == "underestimated_0.5x":
                base_val *= 0.5
            elif prior_mode == "overestimated_2.0x":
                base_val *= 2.0
            pi_vec = np.full(n, np.clip(base_val, 1e-6, 1.0 - 1e-6), dtype=np.float64)
        else:
            pi_vec = np.clip(pi_arr, 1e-6, 1e-0 - 1e-6)

    if "TransactionID" in df.columns:
        tx_id_tie = df["TransactionID"].to_numpy(dtype=np.int64)
    else:
        tx_id_tie = np.arange(n, dtype=np.int64)

    rel_feats = np.zeros((n, r_count * 7), dtype=np.float64)
    msg_1hop = np.zeros((n, r_count * d_base), dtype=np.float64)
    sim_filtered_msg = np.zeros((n, r_count * d_base), dtype=np.float64)

    per_rel_vis_1hop = np.zeros((n, max(r_count, 1)), dtype=np.int8)
    per_rel_min_rep_time = np.full((n, max(r_count, 1)), np.inf, dtype=np.float64)
    per_rel_oracle_1hop = np.zeros((n, max(r_count, 1)), dtype=np.int8)

    allow_future_topology = protocol_cell in ("P0", "P1")
    use_oracle_neighbor_labels = (protocol_cell == "P0")
    use_maturity_hidden_mass = (protocol_cell == "P3")

    # Fast inline survival function for Pass 1 inner loop
    lmbd_d1 = LAMBDA_D1_14D
    denom_d1 = 1.0 - np.exp(-lmbd_d1 * 14.0)

    def _fast_q_sum(ages_imm: np.ndarray, pi_c: float) -> float:
        if survival_scenario_id == "D0":
            return 0.0
        elif survival_scenario_id == "D1":
            cdf = (1.0 - np.exp(-lmbd_d1 * np.clip(ages_imm, 0.0, 14.0))) / denom_d1
            s_a = np.where(ages_imm >= 14.0, 0.0, 1.0 - cdf)
        elif survival_scenario_id == "D2":
            s_a = 1.0 - np.interp(ages_imm, D2_KNOTS_DAYS, D2_KNOTS_CDF, left=0.0, right=1.0)
        else:
            cdf_d2 = np.interp(ages_imm, D2_KNOTS_DAYS, D2_KNOTS_CDF, left=0.0, right=1.0)
            cdf_tail = np.clip((ages_imm - 30.0) / 60.0, 0.0, 1.0)
            s_a = np.where(ages_imm >= 90.0, 0.0, 1.0 - (0.80 * cdf_d2 + 0.20 * cdf_tail))
        s_a = np.where(ages_imm >= maturity_horizon_days, 0.0, np.clip(s_a, 0.0, 1.0))
        num = pi_c * s_a
        return float((num / np.maximum(num + (1.0 - pi_c), 1e-12)).sum())

    relation_slices_cache = []

    for r_idx, col in enumerate(relation_cols):
        ent_vals = df[col].to_numpy(dtype=np.int64)
        col_offset = r_idx * 7
        msg_offset = r_idx * d_base

        valid_indices = np.where(ent_vals >= 0)[0]
        if len(valid_indices) == 0:
            relation_slices_cache.append([])
            continue

        v_ent = ent_vals[valid_indices]
        v_time = event_times_days[valid_indices]
        v_tid = tx_id_tie[valid_indices]
        order = np.lexsort((v_tid, v_time, v_ent))
        sorted_idx = valid_indices[order]
        sorted_ent = v_ent[order]

        split_pts = np.r_[0, np.where(np.diff(sorted_ent) != 0)[0] + 1, len(sorted_idx)]
        rel_groups = []

        for g_i in range(len(split_pts) - 1):
            g_rows = sorted_idx[split_pts[g_i] : split_pts[g_i + 1]]
            m = len(g_rows)
            if m <= 1:
                continue

            g_times = event_times_days[g_rows]
            g_reps = report_times_days[g_rows]
            g_y = true_labels[g_rows]
            g_X = X_norm[g_rows]
            g_taus = tau_vec[g_rows]
            g_pis = pi_vec[g_rows]
            csum_X = np.vstack([np.zeros((1, d_base)), np.cumsum(g_X, axis=0)])

            if allow_future_topology:
                total_deg = m - 1
                w_hub = (1.0 / np.log(2.0 + total_deg)) if use_inv_log_hub_weight else 1.0
                sum_X_all = csum_X[-1]
                sum_y_all = float((g_y == 1).sum())
                strict_end_ptr = np.searchsorted(g_times, g_times, side="left")
                vel_r = np.searchsorted(g_times, g_times + 1.0, side="right")
                vel_l = np.searchsorted(g_times, g_times - 1.0, side="left")
                log_deg = np.log1p(total_deg)

                for loc_i in range(m):
                    i = g_rows[loc_i]
                    mean_X = (sum_X_all - g_X[loc_i]) / total_deg
                    msg_1hop[i, msg_offset : msg_offset + d_base] = mean_X * w_hub
                    sim_filtered_msg[i, msg_offset : msg_offset + d_base] = mean_X * w_hub

                    if use_oracle_neighbor_labels:
                        oth_y = sum_y_all - float(g_y[loc_i] == 1)
                        vis_pos_cnt = oth_y * w_hub
                        vis_pos_rate = (oth_y / total_deg) * w_hub
                        mat_neg_cnt = (total_deg - oth_y) * w_hub
                        imm_cnt = 0.0
                        hid_mass = 0.0
                        if oth_y > 0:
                            per_rel_vis_1hop[i, r_idx] = 1
                            per_rel_oracle_1hop[i, r_idx] = 1
                    else:
                        end_p = int(strict_end_ptr[loc_i])
                        if end_p > 0:
                            t_curr = g_taus[loc_i]
                            ages = t_curr - g_times[:end_p]
                            reps_n = g_reps[:end_p]
                            vis_pos = (reps_n <= t_curr)
                            n_vp = int(vis_pos.sum())
                            unrep = ~vis_pos
                            n_mn = int((unrep & (ages >= maturity_horizon_days)).sum())
                            imm = unrep & (ages < maturity_horizon_days)
                            n_im = int(imm.sum())
                            vis_pos_cnt = float(n_vp) * w_hub
                            vis_pos_rate = float(n_vp / end_p) * w_hub
                            mat_neg_cnt = float(n_mn) * w_hub
                            imm_cnt = float(n_im) * w_hub
                            hid_mass = (_fast_q_sum(ages[imm], float(g_pis[loc_i])) * w_hub) if n_im > 0 else 0.0
                            per_rel_min_rep_time[i, r_idx] = float(reps_n.min())
                            if n_vp > 0:
                                per_rel_vis_1hop[i, r_idx] = 1
                        else:
                            vis_pos_cnt = vis_pos_rate = mat_neg_cnt = imm_cnt = hid_mass = 0.0

                    vel_24h = float(vel_r[loc_i] - vel_l[loc_i] - 1)
                    rel_feats[i, col_offset : col_offset + 7] = [
                        log_deg, vel_24h, vis_pos_cnt, vis_pos_rate, mat_neg_cnt, imm_cnt, hid_mass
                    ]
                rel_groups.append((g_rows, strict_end_ptr, np.zeros(m, dtype=np.int64), w_hub, True))
            else:
                strict_end_ptr = np.searchsorted(g_times, g_times, side="left")
                if recency_window_days is not None:
                    win_start_ptr = np.searchsorted(g_times, g_times - recency_window_days, side="left")
                else:
                    win_start_ptr = np.zeros(m, dtype=np.int64)
                vel_start_ptr = np.searchsorted(g_times, g_times - 1.0, side="left")
                eff_start_ptr = np.zeros(m, dtype=np.int64)

                for loc_i in range(m):
                    end_p = int(strict_end_ptr[loc_i])
                    if end_p == 0:
                        continue
                    i = g_rows[loc_i]
                    total_prior_deg = end_p
                    start_p = max(0, end_p - degree_cap_k) if degree_cap_k is not None else 0
                    if recency_window_days is not None and win_start_ptr[loc_i] > start_p:
                        start_p = int(win_start_ptr[loc_i])
                    eff_start_ptr[loc_i] = start_p
                    cnt_n = end_p - start_p
                    if cnt_n <= 0:
                        continue

                    t_curr = g_taus[loc_i]
                    pi_curr = float(g_pis[loc_i])
                    w_hub = (1.0 / np.log(2.0 + total_prior_deg)) if use_inv_log_hub_weight else 1.0
                    dt_nbrs = g_times[start_p:end_p]
                    ages = t_curr - dt_nbrs
                    reps_n = g_reps[start_p:end_p]
                    vis_pos = (reps_n <= t_curr)
                    n_vp = int(vis_pos.sum())
                    unrep = ~vis_pos
                    n_mn = int((unrep & (ages >= maturity_horizon_days)).sum())
                    imm = unrep & (ages < maturity_horizon_days)
                    n_im = int(imm.sum())

                    vis_pos_cnt = float(n_vp) * w_hub
                    mat_neg_cnt = float(n_mn) * w_hub
                    imm_cnt = float(n_im) * w_hub

                    if use_maturity_hidden_mass and n_im > 0:
                        hid_mass = _fast_q_sum(ages[imm], pi_curr) * w_hub
                    else:
                        hid_mass = 0.0

                    eff_pos = float(n_vp) + (hid_mass / max(w_hub, 1e-8) if use_maturity_hidden_mass else 0.0)
                    vis_pos_rate = (eff_pos / cnt_n) * w_hub
                    vel_24h = float(max(0, end_p - max(start_p, int(vel_start_ptr[loc_i]))))

                    per_rel_min_rep_time[i, r_idx] = float(reps_n.min())
                    if n_vp > 0:
                        per_rel_vis_1hop[i, r_idx] = 1

                    rel_feats[i, col_offset : col_offset + 7] = [
                        np.log1p(total_prior_deg), vel_24h, vis_pos_cnt, vis_pos_rate, mat_neg_cnt, imm_cnt, hid_mass
                    ]
                    mean_X = (csum_X[end_p] - csum_X[start_p]) / cnt_n
                    msg_1hop[i, msg_offset : msg_offset + d_base] = mean_X * w_hub

                    s_sim = max(start_p, end_p - 12)
                    X_sub = g_X[s_sim:end_p]
                    if len(X_sub) == 1:
                        sim_filtered_msg[i, msg_offset : msg_offset + d_base] = X_sub[0] * w_hub
                    else:
                        diff = X_sub - g_X[loc_i]
                        dists = (diff * diff).mean(axis=1)
                        sw = np.exp(-0.15 * dists)
                        sw /= max(float(sw.sum()), 1e-8)
                        sim_filtered_msg[i, msg_offset : msg_offset + d_base] = (sw @ X_sub) * w_hub

                rel_groups.append((g_rows, strict_end_ptr, eff_start_ptr, 0.0, False))

        relation_slices_cache.append(rel_groups)

    pass2 = compute_pass2_from_relation_slices(
        n=n,
        r_count=r_count,
        d_base=d_base,
        tau_vec=tau_vec,
        msg_1hop=msg_1hop,
        per_rel_vis_1hop=per_rel_vis_1hop,
        per_rel_min_rep_time=per_rel_min_rep_time,
        per_rel_oracle_1hop=per_rel_oracle_1hop,
        relation_slices_cache=relation_slices_cache,
        use_inv_log_hub_weight=use_inv_log_hub_weight,
        use_oracle_neighbor_labels=use_oracle_neighbor_labels,
    )

    return {
        "rel_feats": rel_feats,
        "msg_1hop": msg_1hop,
        "msg_2hop": pass2["msg_2hop"],
        "sim_filtered_msg": sim_filtered_msg,
        "has_vis_fraud_1hop": pass2["has_vis_fraud_1hop"],
        "has_vis_fraud_2hop": pass2["has_vis_fraud_2hop"],
        "has_vis_fraud_within_2hops": pass2["has_vis_fraud_within_2hops"],
        "prior_pi_vec": pi_vec,
        "_per_rel_vis_1hop": per_rel_vis_1hop,
        "_per_rel_min_rep_time": per_rel_min_rep_time,
        "_per_rel_oracle_1hop": per_rel_oracle_1hop,
        "_relation_slices_cache": relation_slices_cache,
        "_tau_vec": tau_vec,
    }
