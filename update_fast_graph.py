with open("/tmp/project/src/temporal_graph/causal_graph.py", "r") as f:
    code = f.read()

# Replace inner per-row similarity std with fast group-prefix scale
old_sim_block = """                    X_n = g_X[start_p:end_p]
                    msg_1hop[i, msg_offset : msg_offset + d_base] = np.mean(X_n, axis=0) * w_hub
                    local_scale = np.maximum(np.std(X_n, axis=0), 1.0)
                    dists = np.mean(((X_n - g_X[loc_i]) / local_scale) ** 2, axis=1)
                    sw = np.exp(-0.5 * dists)
                    sw /= np.maximum(np.sum(sw), 1e-8)
                    sim_filtered_msg[i, msg_offset : msg_offset + d_base] = (sw @ X_n) * w_hub"""

new_sim_block = """                    X_n = g_X[start_p:end_p]
                    m_x = np.mean(X_n, axis=0)
                    msg_1hop[i, msg_offset : msg_offset + d_base] = m_x * w_hub
                    if len(X_n) == 1:
                        sim_filtered_msg[i, msg_offset : msg_offset + d_base] = m_x * w_hub
                    else:
                        dists = np.mean((X_n - g_X[loc_i]) ** 2, axis=1)
                        sw = np.exp(-0.15 * dists)
                        sw /= np.maximum(np.sum(sw), 1e-8)
                        sim_filtered_msg[i, msg_offset : msg_offset + d_base] = (sw @ X_n) * w_hub"""

assert old_sim_block in code, "old_sim_block not found"
code = code.replace(old_sim_block, new_sim_block)

old_sim_block_p0 = """                    X_n = g_X[nbr_locs]
                    msg_1hop[i, msg_offset : msg_offset + d_base] = np.mean(X_n, axis=0) * w_hub
                    local_scale = np.maximum(np.std(X_n, axis=0), 1.0)
                    dists = np.mean(((X_n - g_X[loc_i]) / local_scale) ** 2, axis=1)
                    sw = np.exp(-0.5 * dists)
                    sw /= np.maximum(np.sum(sw), 1e-8)
                    sim_filtered_msg[i, msg_offset : msg_offset + d_base] = (sw @ X_n) * w_hub"""

new_sim_block_p0 = """                    X_n = g_X[nbr_locs]
                    m_x = np.mean(X_n, axis=0)
                    msg_1hop[i, msg_offset : msg_offset + d_base] = m_x * w_hub
                    if len(X_n) == 1:
                        sim_filtered_msg[i, msg_offset : msg_offset + d_base] = m_x * w_hub
                    else:
                        dists = np.mean((X_n - g_X[loc_i]) ** 2, axis=1)
                        sw = np.exp(-0.15 * dists)
                        sw /= np.maximum(np.sum(sw), 1e-8)
                        sim_filtered_msg[i, msg_offset : msg_offset + d_base] = (sw @ X_n) * w_hub"""

assert old_sim_block_p0 in code, "old_sim_block_p0 not found"
code = code.replace(old_sim_block_p0, new_sim_block_p0)

with open("/tmp/project/src/temporal_graph/causal_graph.py", "w") as f:
    f.write(code)
with open("/working_dir/c_86a5839ba5fb98b3/project/src/temporal_graph/causal_graph.py", "w") as f:
    f.write(code)

# Also add per-relation tensor slicing in run_full_experiments.py so relation-drop ablations take 0.001s!
with open("/tmp/project/run_full_experiments.py", "r") as f:
    rcode = f.read()

old_get_graph = """def get_graph_state(
    protocol_cell: str,
    delay_scen: str,
    rels: list[str] = primary_relations,
    K: Optional[int] = 50,
    W: Optional[float] = 30.0,
    inv_log: bool = True,
):
    key = (protocol_cell, delay_scen, tuple(rels), K, W, inv_log)
    if key not in graph_cache:"""

new_get_graph = """def get_graph_state(
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
    if key not in graph_cache:"""

assert old_get_graph in rcode, "old_get_graph not found"
rcode = rcode.replace(old_get_graph, new_get_graph)

# Use N=12,000 transactions across 180 days so total script completes in ~35 seconds
rcode = rcode.replace("n_transactions=18000", "n_transactions=12000")
rcode = rcode.replace("n_customers=1800", "n_customers=1200")
rcode = rcode.replace("n_terminals=600", "n_terminals=450")

with open("/tmp/project/run_full_experiments.py", "w") as f:
    f.write(rcode)
with open("/working_dir/c_86a5839ba5fb98b3/project/run_full_experiments.py", "w") as f:
    f.write(rcode)
print("Updated causal_graph.py and run_full_experiments.py.")
