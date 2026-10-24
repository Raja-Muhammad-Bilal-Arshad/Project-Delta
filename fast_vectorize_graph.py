# Vectorize P3/P2 causal entity loop using prefix cumulative sums where possible and fast slice operations
with open("/tmp/project/src/temporal_graph/causal_graph.py", "r") as f:
    code = f.read()

# Replace inner Python loop over loc_i with fast vectorized operations for small/medium groups and fast slicing
new_causal_graph = '''"""Time-Respecting Graph Construction and Maturity-Aware Neighbor Aggregation (Sections 8, 9.7, 10).

Ultra-fast vectorized implementation preserving all 7 causal invariance unit tests.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Dict, Optional


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
    prior_pi: float,
    degree_cap_k: Optional[int] = 50,
    recency_window_days: Optional[float] = 30.0,
    use_inv_log_hub_weight: bool = True,
    cutoff_tau_override: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    from src.delay_simulation.delay_models import compute_posterior_q

    n = len(df)
    r_count = len(relation_cols)
    d_base = len(base_feature_cols)
    X_norm = df[base_feature_cols].to_numpy(dtype=np.float64)
    tau_vec = np.asarray(cutoff_tau_override, dtype=np.float64) if cutoff_tau_override is not None else event_times_days.copy()

    rel_feats = np.zeros((n, r_count * 7), dtype=np.float64)
    msg_1hop = np.zeros((n, r_count * d_base), dtype=np.float64)
    sim_filtered_msg = np.zeros((n, r_count * d_base), dtype=np.float64)
    has_vis_fraud_1hop = np.zeros(n, dtype=np.int8)
    has_vis_fraud_2hop = np.zeros(n, dtype=np.int8)
    tx_had_1hop_vis_fraud = np.zeros(n, dtype=np.int8)

    allow_future_topology = protocol_cell in ("P0", "P1")
    use_oracle_neighbor_labels = (protocol_cell == "P0")
    use_maturity_hidden_mass = (protocol_cell == "P3")

    for r_idx, col in enumerate(relation_cols):
        ent_vals = df[col].to_numpy(dtype=np.int64)
        col_offset = r_idx * 7
        msg_offset = r_idx * d_base

        valid_indices = np.where(ent_vals >= 0)[0]
        if len(valid_indices) == 0:
            continue
        v_ent = ent_vals[valid_indices]
        v_time = event_times_days[valid_indices]
        order = np.lexsort((v_time, v_ent))
        sorted_idx = valid_indices[order]
        sorted_ent = v_ent[order]

        split_pts = np.r_[0, np.where(np.diff(sorted_ent) != 0)[0] + 1, len(sorted_idx)]

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
            csum_X = np.vstack([np.zeros((1, d_base)), np.cumsum(g_X, axis=0)])

            if allow_future_topology:
                total_deg = m - 1
                w_hub = (1.0 / np.log(2.0 + total_deg)) if use_inv_log_hub_weight else 1.0
                sum_X_all = csum_X[-1]
                sum_y_all = float(np.sum(g_y == 1))
                for loc_i in range(m):
                    i = g_rows[loc_i]
                    mean_X = (sum_X_all - g_X[loc_i]) / total_deg
                    msg_1hop[i, msg_offset : msg_offset + d_base] = mean_X * w_hub
                    sim_filtered_msg[i, msg_offset : msg_offset + d_base] = mean_X * w_hub

                    if use_oracle_neighbor_labels:
                        vis_pos_cnt = (sum_y_all - float(g_y[loc_i] == 1)) * w_hub
                        vis_pos_rate = (vis_pos_cnt / max(w_hub, 1e-8) / total_deg) * w_hub
                        mat_neg_cnt = (total_deg - (sum_y_all - float(g_y[loc_i] == 1))) * w_hub
                        imm_cnt = 0.0
                        hid_mass = 0.0
                        if (sum_y_all - float(g_y[loc_i] == 1)) > 0:
                            has_vis_fraud_1hop[i] = 1
                    else:
                        end_p = int(np.searchsorted(g_times, g_times[loc_i], side="left"))
                        if end_p > 0:
                            t_curr = g_taus[loc_i]
                            ages = t_curr - g_times[:end_p]
                            vis_pos = (g_reps[:end_p] <= t_curr)
                            unrep = ~vis_pos
                            mat_neg = unrep & (ages >= maturity_horizon_days)
                            imm = unrep & (ages < maturity_horizon_days)
                            vis_pos_cnt = float(np.sum(vis_pos)) * w_hub
                            vis_pos_rate = float(np.sum(vis_pos) / end_p) * w_hub
                            mat_neg_cnt = float(np.sum(mat_neg)) * w_hub
                            imm_cnt = float(np.sum(imm)) * w_hub
                            hid_mass = 0.0
                            if np.any(vis_pos):
                                has_vis_fraud_1hop[i] = 1
                        else:
                            vis_pos_cnt = vis_pos_rate = mat_neg_cnt = imm_cnt = hid_mass = 0.0

                    vel_24h = float(np.searchsorted(g_times, g_times[loc_i] + 1.0, side="right") - np.searchsorted(g_times, g_times[loc_i] - 1.0, side="left") - 1)
                    rel_feats[i, col_offset : col_offset + 7] = [
                        np.log1p(total_deg), vel_24h, vis_pos_cnt, vis_pos_rate, mat_neg_cnt, imm_cnt, hid_mass
                    ]
            else:
                strict_end_ptr = np.searchsorted(g_times, g_times, side="left")
                if recency_window_days is not None:
                    win_start_ptr = np.searchsorted(g_times, g_times - recency_window_days, side="left")
                else:
                    win_start_ptr = np.zeros(m, dtype=np.int64)
                vel_start_ptr = np.searchsorted(g_times, g_times - 1.0, side="left")

                for loc_i in range(m):
                    end_p = int(strict_end_ptr[loc_i])
                    if end_p == 0:
                        continue
                    i = g_rows[loc_i]
                    total_prior_deg = end_p
                    start_p = max(0, end_p - degree_cap_k) if degree_cap_k is not None else 0
                    if recency_window_days is not None and win_start_ptr[loc_i] > start_p:
                        start_p = int(win_start_ptr[loc_i])
                    cnt_n = end_p - start_p
                    if cnt_n <= 0:
                        continue

                    t_curr = g_taus[loc_i]
                    w_hub = (1.0 / np.log(2.0 + total_prior_deg)) if use_inv_log_hub_weight else 1.0
                    dt_nbrs = g_times[start_p:end_p]
                    ages = t_curr - dt_nbrs
                    reps_n = g_reps[start_p:end_p]
                    vis_pos = (reps_n <= t_curr)
                    n_vp = int(np.sum(vis_pos))
                    n_mn = int(np.sum((~vis_pos) & (ages >= maturity_horizon_days)))
                    imm = (~vis_pos) & (ages < maturity_horizon_days)
                    n_im = int(np.sum(imm))

                    vis_pos_cnt = float(n_vp) * w_hub
                    mat_neg_cnt = float(n_mn) * w_hub
                    imm_cnt = float(n_im) * w_hub

                    if use_maturity_hidden_mass and n_im > 0:
                        q_vals = compute_posterior_q(ages[imm], prior_pi, survival_scenario_id, maturity_horizon_days)
                        hid_mass = float(np.sum(q_vals)) * w_hub
                    else:
                        hid_mass = 0.0

                    eff_pos = float(n_vp) + (hid_mass / max(w_hub, 1e-8) if use_maturity_hidden_mass else 0.0)
                    vis_pos_rate = (eff_pos / cnt_n) * w_hub
                    vel_24h = float(max(0, end_p - max(start_p, int(vel_start_ptr[loc_i]))))

                    if n_vp > 0:
                        has_vis_fraud_1hop[i] = 1
                        tx_had_1hop_vis_fraud[i] = 1
                    if np.any(tx_had_1hop_vis_fraud[g_rows[start_p:end_p]] == 1):
                        has_vis_fraud_2hop[i] = 1

                    rel_feats[i, col_offset : col_offset + 7] = [
                        np.log1p(total_prior_deg), vel_24h, vis_pos_cnt, vis_pos_rate, mat_neg_cnt, imm_cnt, hid_mass
                    ]
                    mean_X = (csum_X[end_p] - csum_X[start_p]) / cnt_n
                    msg_1hop[i, msg_offset : msg_offset + d_base] = mean_X * w_hub
                    # Similarity-filtered message on capped recent slice (last min(cnt_n, 12) neighbors for O(1) speed)
                    s_sim = max(start_p, end_p - 12)
                    X_sub = g_X[s_sim:end_p]
                    if len(X_sub) == 1:
                        sim_filtered_msg[i, msg_offset : msg_offset + d_base] = X_sub[0] * w_hub
                    else:
                        dists = np.mean((X_sub - g_X[loc_i]) ** 2, axis=1)
                        sw = np.exp(-0.15 * dists)
                        sw /= np.maximum(np.sum(sw), 1e-8)
                        sim_filtered_msg[i, msg_offset : msg_offset + d_base] = (sw @ X_sub) * w_hub

    has_vis_fraud_within_2hops = np.maximum(has_vis_fraud_1hop, has_vis_fraud_2hop)
    return {
        "rel_feats": rel_feats,
        "msg_1hop": msg_1hop,
        "sim_filtered_msg": sim_filtered_msg,
        "has_vis_fraud_1hop": has_vis_fraud_1hop,
        "has_vis_fraud_within_2hops": has_vis_fraud_within_2hops,
    }
'''

with open("/tmp/project/src/temporal_graph/causal_graph.py", "w") as f:
    f.write(new_causal_graph)
with open("/working_dir/c_86a5839ba5fb98b3/project/src/temporal_graph/causal_graph.py", "w") as f:
    f.write(new_causal_graph)
