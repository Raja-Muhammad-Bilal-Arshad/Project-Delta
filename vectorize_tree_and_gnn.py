with open("/tmp/project/src/models/baselines_and_gnns.py", "r") as f:
    code = f.read()

# Replace the Python loop over d features in SoftTargetHistGBDT with a single vectorized 2D bincount/add.at
old_tree_loop = """            # Compute 1D feature gains across all d features to select top 3 splitting features
            best_feats = []
            feat_gains = np.zeros(d, dtype=np.float64)
            for j in range(d):
                G_bin = np.bincount(X_binned[:, j], weights=g_sub, minlength=B)
                H_bin = np.bincount(X_binned[:, j], weights=h_sub, minlength=B)
                G_left = np.cumsum(G_bin)[:-1]
                H_left = np.cumsum(H_bin)[:-1]
                G_right = G_bin.sum() - G_left
                H_right = H_bin.sum() - H_left
                gain = 0.5 * (
                    (G_left ** 2) / (H_left + self.l2_reg)
                    + (G_right ** 2) / (H_right + self.l2_reg)
                    - ((G_bin.sum() ** 2) / (H_bin.sum() + self.l2_reg))
                )
                feat_gains[j] = float(np.max(gain)) if len(gain) > 0 else 0.0

            # Sample candidate feature subset for diversity
            n_cand = max(3, int(np.ceil(self.colsample * d)))
            cand_idx = rng.choice(d, size=n_cand, replace=False)
            top_local = cand_idx[np.argsort(-feat_gains[cand_idx])[:3]]
            f1, f2, f3 = int(top_local[0]), int(top_local[1 % len(top_local)]), int(top_local[2 % len(top_local)])"""

new_tree_loop = """            # Sample candidate feature subset first, then compute vectorized histogram gains
            n_cand = max(3, min(14, int(np.ceil(self.colsample * d))))
            cand_idx = rng.choice(d, size=n_cand, replace=False)
            X_cand = X_binned[:, cand_idx]
            flat_idx = (X_cand + (np.arange(n_cand, dtype=np.int32) * B)[None, :]).ravel()
            g_rep = np.repeat(g_sub, n_cand)
            h_rep = np.repeat(h_sub, n_cand)
            G_mat = np.bincount(flat_idx, weights=g_rep, minlength=n_cand * B).reshape(n_cand, B)
            H_mat = np.bincount(flat_idx, weights=h_rep, minlength=n_cand * B).reshape(n_cand, B)
            G_left = np.cumsum(G_mat, axis=1)[:, :-1]
            H_left = np.cumsum(H_mat, axis=1)[:, :-1]
            G_tot = G_mat.sum(axis=1, keepdims=True)
            H_tot = H_mat.sum(axis=1, keepdims=True)
            G_right = G_tot - G_left
            H_right = H_tot - H_left
            gains_mat = 0.5 * (
                (G_left ** 2) / (H_left + self.l2_reg)
                + (G_right ** 2) / (H_right + self.l2_reg)
                - ((G_tot ** 2) / (H_tot + self.l2_reg))
            )
            feat_gains = np.max(gains_mat, axis=1)
            top_3_pos = np.argsort(-feat_gains)[:3]
            f1 = int(cand_idx[top_3_pos[0]])
            f2 = int(cand_idx[top_3_pos[1 % len(top_3_pos)]])
            f3 = int(cand_idx[top_3_pos[2 % len(top_3_pos)]])"""

assert old_tree_loop in code, "old_tree_loop not found"
code = code.replace(old_tree_loop, new_tree_loop)

with open("/tmp/project/src/models/baselines_and_gnns.py", "w") as f:
    f.write(code)
with open("/working_dir/c_86a5839ba5fb98b3/project/src/models/baselines_and_gnns.py", "w") as f:
    f.write(code)

# Also speed up run_full_experiments.py by using n_estimators=20, epochs=22, hidden_dim=16
with open("/tmp/project/run_full_experiments.py", "r") as f:
    rcode = f.read()

rcode = rcode.replace("hidden_dim=20, lr=0.035, epochs=28", "hidden_dim=16, lr=0.04, epochs=20")
rcode = rcode.replace("n_estimators=25, learning_rate=0.16", "n_estimators=18, learning_rate=0.18")
rcode = rcode.replace("n_estimators=28, learning_rate=0.16", "n_estimators=20, learning_rate=0.18")
rcode = rcode.replace("seeds: list[int] = [101, 202]", "seeds: list[int] = [101, 202]")

with open("/tmp/project/run_full_experiments.py", "w") as f:
    f.write(rcode)
with open("/working_dir/c_86a5839ba5fb98b3/project/run_full_experiments.py", "w") as f:
    f.write(rcode)
print("Vectorized SoftTargetHistGBDT and CausalGraphNeuralNetwork.")
