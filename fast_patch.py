with open("/tmp/project/src/models/baselines_and_gnns.py", "r") as f:
    mcode = f.read()

# Replace np.outer(d_logits, params["W2"]) and elementwise multiplies with fast BLAS matrix ops
old_grad = """            dh = np.outer(d_logits, params["W2"])
            d_relu_h = dh * gate
            d_pre_h = d_relu_h * (pre_h > 0.0)
            dW1 = Z.T @ d_pre_h + self.weight_decay * params["W1"]
            db1 = np.sum(d_pre_h, axis=0)

            grads = {"W1": dW1, "b1": db1, "W2": dW2, "w_skip": dw_skip, "b2": db2}
            if self.domain_care_filter:
                d_gate = dh * relu_h * gate * (1.0 - gate)
                grads["W_gate"] = Z.T @ d_gate + self.weight_decay * params["W_gate"]
                grads["b_gate"] = np.sum(d_gate, axis=0)"""

new_grad = """            dh = d_logits[:, None] * params["W2"][None, :]
            d_pre_h = dh * gate * (pre_h > 0.0)
            dW1 = Z.T @ d_pre_h + self.weight_decay * params["W1"]
            db1 = np.sum(d_pre_h, axis=0)

            grads = {"W1": dW1, "b1": db1, "W2": dW2, "w_skip": dw_skip, "b2": db2}
            if self.domain_care_filter:
                d_gate = dh * relu_h * gate * (1.0 - gate)
                grads["W_gate"] = Z.T @ d_gate + self.weight_decay * params["W_gate"]
                grads["b_gate"] = np.sum(d_gate, axis=0)"""

assert old_grad in mcode
mcode = mcode.replace(old_grad, new_grad)
mcode = mcode.replace("hidden_dim: int = 32", "hidden_dim: int = 20")
mcode = mcode.replace("epochs: int = 45", "epochs: int = 30")
mcode = mcode.replace("n_estimators: int = 35", "n_estimators: int = 25")

with open("/tmp/project/src/models/baselines_and_gnns.py", "w") as f:
    f.write(mcode)
with open("/working_dir/c_86a5839ba5fb98b3/project/src/models/baselines_and_gnns.py", "w") as f:
    f.write(mcode)

with open("/tmp/project/run_full_experiments.py", "r") as f:
    rcode = f.read()

rcode = rcode.replace("hidden_dim=32, lr=0.025, epochs=45", "hidden_dim=20, lr=0.035, epochs=28")
rcode = rcode.replace("n_estimators=35, learning_rate=0.14", "n_estimators=25, learning_rate=0.16")
rcode = rcode.replace("n_estimators=40, learning_rate=0.14", "n_estimators=28, learning_rate=0.16")
# Pre-warm primary_relations in graph_cache before ablations so relation-drop ablations use 0.001s slicing!
old_ab = 'print("[Steps 8-9 & 14] Running Graph Relation, Entity-Proxy, and Hub-Control Ablations (Table 5 / RQ3 / H3-H4)...")'
new_ab = '''# Ensure full_key is cached before ablations
_ = get_graph_state("P3", "D2", rels=primary_relations, K=50, W=30.0, inv_log=True)
print("[Steps 8-9 & 14] Running Graph Relation, Entity-Proxy, and Hub-Control Ablations (Table 5 / RQ3 / H3-H4)...")'''
rcode = rcode.replace(old_ab, new_ab)

with open("/tmp/project/run_full_experiments.py", "w") as f:
    f.write(rcode)
with open("/working_dir/c_86a5839ba5fb98b3/project/run_full_experiments.py", "w") as f:
    f.write(rcode)
print("Patched GNN & experiment script.")
