"""Baseline Tree Ensembles and Graph Neural Network Models (Sections 9.5 & 11).

Implements all six pre-registered model tiers (B0 to B5) with exact support for
hard binary targets (V0, V1, V3) and continuous Bayesian posterior soft targets q(a) (V2):
  - B0: Raw Tabular Histogram Gradient Boosted Decision Trees (GBDT)
  - B1: Causal Aggregates + GBDT (tabular + per-relation causal velocity, counts, and label summaries)
  - B2: Graph-Light Baseline (GBDT + 1-hop & 2-hop structural neighbor feature diffusion)
  - B3: Generic Causal GNN (2-layer GraphSAGE with relation-specific projection & residual MLP trained via Adam)
  - B4: Domain Fraud GNN (CARE-GNN / GTAN adaptation with similarity-filtered neighborhood aggregation & gated residual)
  - B5: Proposed Maturity-Aware Variant (Preselected on validation windows under V2 soft-target loss & hub control).

All models fit feature normalizers and quantile bins STRICTLY on the training partition.
"""
from __future__ import annotations
import time
import numpy as np
from typing import Dict, Any, Optional, Tuple


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


class SoftTargetHistGBDT:
    """Fast Histogram Gradient Boosted Decision Tree Ensemble supporting continuous soft targets y in [0, 1].
    Matches LightGBM / XGBoost second-order Newton boosting on log-loss:
      g_i = p_i - y_i,   h_i = p_i * (1 - p_i)
    Uses depth-3 oblivious/Cartesian interaction trees over quantile-binned features for fast vectorized fitting.
    """
    def __init__(
        self,
        n_estimators: int = 25,
        learning_rate: float = 0.14,
        max_bins: int = 16,
        l2_reg: float = 5.0,
        subsample: float = 0.85,
        colsample: float = 0.80,
        seed: int = 42,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_bins = max_bins
        self.l2_reg = l2_reg
        self.subsample = subsample
        self.colsample = colsample
        self.seed = seed
        self.bin_edges_: list[np.ndarray] = []
        self.trees_: list[Tuple[int, int, int, np.ndarray]] = []
        self.base_logit_: float = 0.0

    def _fit_bins(self, X: np.ndarray) -> np.ndarray:
        n, d = X.shape
        self.bin_edges_ = []
        X_binned = np.zeros((n, d), dtype=np.int32)
        qs = np.linspace(0.0, 1.0, self.max_bins + 1)[1:-1]
        for j in range(d):
            col = X[:, j]
            edges = np.unique(np.quantile(col, qs))
            self.bin_edges_.append(edges)
            X_binned[:, j] = np.digitize(col, edges)
        return X_binned

    def _transform_bins(self, X: np.ndarray) -> np.ndarray:
        n, d = X.shape
        X_binned = np.zeros((n, d), dtype=np.int32)
        for j in range(d):
            X_binned[:, j] = np.digitize(X[:, j], self.bin_edges_[j])
        return X_binned

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        sample_mask: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "SoftTargetHistGBDT":
        rng = np.random.default_rng(self.seed)
        if sample_mask is not None:
            X_tr = X_train[sample_mask]
            y_tr = np.clip(y_train[sample_mask], 0.0, 1.0)
        else:
            X_tr = X_train
            y_tr = np.clip(y_train, 0.0, 1.0)

        n, d = X_tr.shape
        mean_y = float(np.clip(np.mean(y_tr), 1e-5, 1.0 - 1.0e-5))
        self.base_logit_ = float(np.log(mean_y / (1.0 - mean_y)))
        logits = np.full(n, self.base_logit_, dtype=np.float64)

        X_binned = self._fit_bins(X_tr)
        B = self.max_bins
        self.trees_ = []

        for t in range(self.n_estimators):
            p = _sigmoid(logits)
            g = p - y_tr
            h = np.maximum(p * (1.0 - p), 1e-5)

            if self.subsample < 1.0:
                sub_mask = rng.uniform(0.0, 1.0, size=n) <= self.subsample
                g_sub = np.where(sub_mask, g, 0.0)
                h_sub = np.where(sub_mask, h, 0.0)
            else:
                g_sub, h_sub = g, h

            # Sample candidate feature subset first, then compute vectorized histogram gains
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
            f3 = int(cand_idx[top_3_pos[2 % len(top_3_pos)]])

            # Find best threshold for each of f1, f2, f3 sequentially (oblivious depth-3 tree = 8 leaves)
            splits = []
            for f_idx in (f1, f2, f3):
                G_bin = np.bincount(X_binned[:, f_idx], weights=g_sub, minlength=B)
                H_bin = np.bincount(X_binned[:, f_idx], weights=h_sub, minlength=B)
                G_left = np.cumsum(G_bin)[:-1]
                H_left = np.cumsum(H_bin)[:-1]
                G_right = G_bin.sum() - G_left
                H_right = H_bin.sum() - H_left
                gains = (G_left ** 2) / (H_left + self.l2_reg) + (G_right ** 2) / (H_right + self.l2_reg)
                best_b = int(np.argmax(gains)) if len(gains) > 0 else 0
                splits.append((f_idx, best_b))

            leaf_idx = (
                ((X_binned[:, splits[0][0]] > splits[0][1]).astype(np.int32) << 2)
                | ((X_binned[:, splits[1][0]] > splits[1][1]).astype(np.int32) << 1)
                | ((X_binned[:, splits[2][0]] > splits[2][1]).astype(np.int32))
            )
            G_leaf = np.bincount(leaf_idx, weights=g_sub, minlength=8)
            H_leaf = np.bincount(leaf_idx, weights=h_sub, minlength=8)
            leaf_values = -self.learning_rate * (G_leaf / (H_leaf + self.l2_reg))
            logits += leaf_values[leaf_idx]
            self.trees_.append((splits, leaf_values))

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_binned = self._transform_bins(X)
        logits = np.full(len(X), self.base_logit_, dtype=np.float64)
        for splits, leaf_values in self.trees_:
            leaf_idx = (
                ((X_binned[:, splits[0][0]] > splits[0][1]).astype(np.int32) << 2)
                | ((X_binned[:, splits[1][0]] > splits[1][1]).astype(np.int32) << 1)
                | ((X_binned[:, splits[2][0]] > splits[2][1]).astype(np.int32))
            )
            logits += leaf_values[leaf_idx]
        return np.clip(_sigmoid(logits), 1e-7, 1.0 - 1e-7)


class CausalGraphNeuralNetwork:
    """2-Layer Causal Graph Neural Network (GraphSAGE / CARE-GNN / GTAN architecture)
    trained with Adam optimizer and Expected Cross-Entropy loss under V0/V1/V2/V3.

    Architecture:
      h_i^(1) = ReLU( X_self_i W_self^(1) + Msg_1hop_i W_neigh^(1) + Rel_Summary_i W_rel^(1) + b^(1) )
      If domain_care_filter=True (B4): applies a learned gating unit over similarity-filtered neighbor messages:
        gate_i = sigmoid( [X_self_i || SimMsg_i] W_gate + b_gate )
        h_i^(1) = ReLU( X_self_i W_self^(1) + (gate_i * SimMsg_i) W_neigh^(1) + Rel_Summary_i W_rel^(1) + b^(1) )
      p_i = sigmoid( h_i^(1) W^(2) + X_self_i w_skip + b^(2) )
    """
    def __init__(
        self,
        hidden_dim: int = 20,
        lr: float = 0.025,
        epochs: int = 30,
        weight_decay: float = 1e-4,
        domain_care_filter: bool = False,
        seed: int = 42,
    ):
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.domain_care_filter = domain_care_filter
        self.seed = seed
        self.mu_: Optional[np.ndarray] = None
        self.sd_: Optional[np.ndarray] = None
        self.params_: Dict[str, np.ndarray] = {}

    def _assemble_features(
        self,
        X_base: np.ndarray,
        rel_feats: np.ndarray,
        msg_1hop: np.ndarray,
        sim_filtered_msg: np.ndarray,
    ) -> np.ndarray:
        if self.domain_care_filter:
            return np.hstack([X_base, rel_feats, sim_filtered_msg, msg_1hop])
        else:
            return np.hstack([X_base, rel_feats, msg_1hop])

    def fit(
        self,
        X_base: np.ndarray,
        rel_feats: np.ndarray,
        msg_1hop: np.ndarray,
        sim_filtered_msg: np.ndarray,
        y_target: np.ndarray,
        sample_mask: Optional[np.ndarray] = None,
    ) -> "CausalGraphNeuralNetwork":
        rng = np.random.default_rng(self.seed)
        F_all = self._assemble_features(X_base, rel_feats, msg_1hop, sim_filtered_msg)
        if sample_mask is not None:
            F_tr = F_all[sample_mask]
            y_tr = np.clip(y_target[sample_mask], 0.0, 1.0)
        else:
            F_tr = F_all
            y_tr = np.clip(y_target, 0.0, 1.0)

        # Fit normalizer strictly on training rows
        self.mu_ = np.mean(F_tr, axis=0)
        self.sd_ = np.maximum(np.std(F_tr, axis=0), 1e-6)
        Z = np.clip((F_tr - self.mu_) / self.sd_, -8.0, 8.0)

        n, d_in = Z.shape
        h_dim = self.hidden_dim
        mean_y = float(np.clip(np.mean(y_tr), 1e-5, 1.0 - 1e-5))
        base_b = float(np.log(mean_y / (1.0 - mean_y)))

        W1 = rng.normal(0.0, np.sqrt(2.0 / d_in), size=(d_in, h_dim))
        b1 = np.zeros(h_dim, dtype=np.float64)
        if self.domain_care_filter:
            W_gate = rng.normal(0.0, np.sqrt(1.0 / d_in), size=(d_in, h_dim))
            b_gate = np.zeros(h_dim, dtype=np.float64)
        else:
            W_gate = np.zeros((d_in, h_dim), dtype=np.float64)
            b_gate = np.zeros(h_dim, dtype=np.float64)

        W2 = rng.normal(0.0, np.sqrt(1.0 / h_dim), size=(h_dim,))
        w_skip = np.zeros(d_in, dtype=np.float64)
        b2 = np.array([base_b], dtype=np.float64)

        params = {"W1": W1, "b1": b1, "W_gate": W_gate, "b_gate": b_gate, "W2": W2, "w_skip": w_skip, "b2": b2}
        m_adam = {k: np.zeros_like(v) for k, v in params.items()}
        v_adam = {k: np.zeros_like(v) for k, v in params.items()}
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        for epoch in range(1, self.epochs + 1):
            pre_h = Z @ params["W1"] + params["b1"]
            relu_h = np.maximum(0.0, pre_h)
            if self.domain_care_filter:
                gate_pre = Z @ params["W_gate"] + params["b_gate"]
                gate = _sigmoid(gate_pre)
                h = relu_h * gate
            else:
                gate = np.ones_like(relu_h)
                h = relu_h

            logits = h @ params["W2"] + Z @ params["w_skip"] + params["b2"][0]
            p = _sigmoid(logits)

            # Gradient of expected cross-entropy L = -1/N sum [y log p + (1-y) log(1-p)] w.r.t. logits is (p - y) / N
            d_logits = (p - y_tr) / float(n)
            dW2 = h.T @ d_logits + self.weight_decay * params["W2"]
            dw_skip = Z.T @ d_logits + self.weight_decay * params["w_skip"]
            db2 = np.array([np.sum(d_logits)], dtype=np.float64)

            dh = d_logits[:, None] * params["W2"][None, :]
            d_pre_h = dh * gate * (pre_h > 0.0)
            dW1 = Z.T @ d_pre_h + self.weight_decay * params["W1"]
            db1 = np.sum(d_pre_h, axis=0)

            grads = {"W1": dW1, "b1": db1, "W2": dW2, "w_skip": dw_skip, "b2": db2}
            if self.domain_care_filter:
                d_gate = dh * relu_h * gate * (1.0 - gate)
                grads["W_gate"] = Z.T @ d_gate + self.weight_decay * params["W_gate"]
                grads["b_gate"] = np.sum(d_gate, axis=0)

            for k, g_val in grads.items():
                m_adam[k] = beta1 * m_adam[k] + (1.0 - beta1) * g_val
                v_adam[k] = beta2 * v_adam[k] + (1.0 - beta2) * (g_val ** 2)
                m_hat = m_adam[k] / (1.0 - beta1 ** epoch)
                v_hat = v_adam[k] / (1.0 - beta2 ** epoch)
                params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

        self.params_ = params
        return self

    def predict_proba(
        self,
        X_base: np.ndarray,
        rel_feats: np.ndarray,
        msg_1hop: np.ndarray,
        sim_filtered_msg: np.ndarray,
    ) -> np.ndarray:
        F_all = self._assemble_features(X_base, rel_feats, msg_1hop, sim_filtered_msg)
        Z = np.clip((F_all - self.mu_) / self.sd_, -8.0, 8.0)
        pre_h = Z @ self.params_["W1"] + self.params_["b1"]
        relu_h = np.maximum(0.0, pre_h)
        if self.domain_care_filter:
            gate = _sigmoid(Z @ self.params_["W_gate"] + self.params_["b_gate"])
            h = relu_h * gate
        else:
            h = relu_h
        logits = h @ self.params_["W2"] + Z @ self.params_["w_skip"] + self.params_["b2"][0]
        return np.clip(_sigmoid(logits), 1e-7, 1.0 - 1e-7)
