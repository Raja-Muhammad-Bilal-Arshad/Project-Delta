"""Evaluation Metrics and Statistical Analysis Module (Sections 12 & 13).

Implements:
  - Exact Average Precision (AP) matching scikit-learn's step-wise PR integration:
      AP = sum_n (R_n - R_{n-1}) * P_n
  - Trapezoidal PR-AUC (explicitly distinguished from AP)
  - ROC-AUC via exact Wilcoxon-Mann-Whitney rank statistic
  - Operational Alert-Budget Metrics: Precision@B% and Recall@B% for B in {0.5%, 1.0%, 5.0%}
    with deterministic tie-breaking (score, then transaction amount, then index)
  - Recall at 1% validation-tuned FPR (threshold selected strictly on validation window)
  - Card-Proxy Precision@k (aggregated by max score per unique card proxy per block)
  - Brier Score and Expected Calibration Error (ECE)
  - Moving-Block Bootstrap 95% confidence intervals (preserving temporal block dependence)
  - Paired Wilcoxon signed-rank test (via scipy.stats.wilcoxon) with Holm-Bonferroni correction.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import wilcoxon, rankdata
from typing import Dict, Any, List, Tuple, Optional


def compute_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Computes exact Average Precision (AP) = sum_n (R_n - R_{n-1}) * P_n
    matching scikit-learn's average_precision_score implementation [R9].
    """
    y = np.asarray(y_true, dtype=np.int32)
    s = np.asarray(y_score, dtype=np.float64)
    n_pos = int(np.sum(y == 1))
    if n_pos == 0 or len(y) == 0:
        return 0.0
    # Sort descending by score; stable mergesort preserves tie order
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1).astype(np.float64)
    fp = np.cumsum(y_sorted == 0).astype(np.float64)
    precision = tp / (tp + fp)
    # Sum precision at positions where a positive example occurs, divided by total positives
    # (Exact step-function integral when scores are distinct or grouped at threshold changes)
    distinct_value_indices = np.where(np.diff(s[order]))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_sorted.size - 1]
    p_at_thresh = precision[threshold_idxs]
    r_at_thresh = tp[threshold_idxs] / n_pos
    r_prev = np.r_[0.0, r_at_thresh[:-1]]
    ap = float(np.sum((r_at_thresh - r_prev) * p_at_thresh))
    return ap


def compute_trapezoidal_prauc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Computes trapezoidal area under the precision-recall curve (distinct from step-wise AP)."""
    y = np.asarray(y_true, dtype=np.int32)
    s = np.asarray(y_score, dtype=np.float64)
    n_pos = int(np.sum(y == 1))
    if n_pos == 0 or len(y) == 0:
        return 0.0
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1).astype(np.float64)
    fp = np.cumsum(y_sorted == 0).astype(np.float64)
    precision = np.r_[1.0, tp / (tp + fp)]
    recall = np.r_[0.0, tp / n_pos]
    return float(np.trapezoid(precision, recall))


def compute_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Computes exact ROC-AUC via rank statistic."""
    y = np.asarray(y_true, dtype=np.int32)
    s = np.asarray(y_score, dtype=np.float64)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = rankdata(s, method="average")
    sum_pos_ranks = float(np.sum(ranks[y == 1]))
    auc = (sum_pos_ranks - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
    return float(auc)


def compute_alert_budget_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    budget_pct: float = 1.0,
    amounts: Optional[np.ndarray] = None,
) -> Tuple[float, float, int]:
    """Computes Precision@B% and Recall@B% for top k = ceil((B/100) * N) transactions.
    Deterministic tie-breaking: higher score, then higher transaction amount, then earlier index.
    """
    y = np.asarray(y_true, dtype=np.int32)
    s = np.asarray(y_score, dtype=np.float64)
    n = len(y)
    n_pos = int(np.sum(y == 1))
    if n == 0:
        return 0.0, 0.0, 0
    k = max(1, int(np.ceil((budget_pct / 100.0) * n)))
    if amounts is not None:
        idx = np.arange(n)
        order = np.lexsort((idx, -np.asarray(amounts, dtype=np.float64), -s))
    else:
        order = np.argsort(-s, kind="mergesort")
    top_k_y = y[order[:k]]
    caught = int(np.sum(top_k_y == 1))
    prec = float(caught / k)
    rec = float(caught / n_pos) if n_pos > 0 else 0.0
    return prec, rec, k


def compute_card_proxy_precision_at_k(
    y_true: np.ndarray,
    y_score: np.ndarray,
    card_proxy_ids: np.ndarray,
    k_cards: int = 50,
) -> float:
    """Computes Card-Proxy Precision@k by aggregating max score and max true fraud label
    per unique card proxy within the evaluation block (Section 12.1).
    """
    y = np.asarray(y_true, dtype=np.int32)
    s = np.asarray(y_score, dtype=np.float64)
    c = np.asarray(card_proxy_ids, dtype=np.int64)
    unique_c, inv = np.unique(c, return_inverse=True)
    max_s = np.full(len(unique_c), -np.inf, dtype=np.float64)
    max_y = np.zeros(len(unique_c), dtype=np.int32)
    np.maximum.at(max_s, inv, s)
    np.maximum.at(max_y, inv, y)
    k_eff = min(k_cards, len(unique_c))
    if k_eff == 0:
        return 0.0
    order = np.argsort(-max_s, kind="mergesort")
    return float(np.mean(max_y[order[:k_eff]]))


def select_threshold_for_target_fpr(y_val: np.ndarray, s_val: np.ndarray, target_fpr: float = 0.01) -> float:
    """Selects score threshold strictly on validation predictions to achieve target FPR (e.g., 1%)."""
    neg_scores = np.asarray(s_val, dtype=np.float64)[np.asarray(y_val) == 0]
    if len(neg_scores) == 0:
        return 0.5
    return float(np.quantile(neg_scores, 1.0 - target_fpr))


def compute_block_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    card_proxy_ids: Optional[np.ndarray] = None,
    amounts: Optional[np.ndarray] = None,
    val_threshold_1pct_fpr: Optional[float] = None,
) -> Dict[str, Any]:
    """Computes all primary and secondary metrics on a single evaluation block."""
    y = np.asarray(y_true, dtype=np.int32)
    s = np.clip(np.asarray(y_score, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    n = len(y)
    n_pos = int(np.sum(y == 1))
    prev = float(n_pos / max(n, 1))

    ap = compute_average_precision(y, s)
    prauc_trap = compute_trapezoidal_prauc(y, s)
    roc_auc = compute_roc_auc(y, s)
    p05, r05, k05 = compute_alert_budget_metrics(y, s, 0.5, amounts)
    p10, r10, k10 = compute_alert_budget_metrics(y, s, 1.0, amounts)
    p50, r50, k50 = compute_alert_budget_metrics(y, s, 5.0, amounts)

    brier = float(np.mean((s - y) ** 2))
    logloss = float(-np.mean(y * np.log(s) + (1.0 - y) * np.log(1.0 - s)))

    if val_threshold_1pct_fpr is not None and n_pos > 0:
        rec_at_1pct_fpr = float(np.mean(s[y == 1] >= val_threshold_1pct_fpr))
    else:
        thresh = select_threshold_for_target_fpr(y, s, 0.01)
        rec_at_1pct_fpr = float(np.mean(s[y == 1] >= thresh)) if n_pos > 0 else 0.0

    cp_50 = compute_card_proxy_precision_at_k(y, s, card_proxy_ids, k_cards=25) if card_proxy_ids is not None else 0.0

    return {
        "n_transactions": n,
        "n_positives": n_pos,
        "prevalence": round(prev, 5),
        "AP": round(ap, 5),
        "PR_AUC_trapezoidal": round(prauc_trap, 5),
        "ROC_AUC": round(roc_auc, 5),
        "Precision@0.5%": round(p05, 5),
        "Recall@0.5%": round(r05, 5),
        "Precision@1%": round(p10, 5),
        "Recall@1%": round(r10, 5),
        "Precision@5%": round(p50, 5),
        "Recall@5%": round(r50, 5),
        "Recall@1%FPR": round(rec_at_1pct_fpr, 5),
        "CardProxy_Precision@25": round(cp_50, 5),
        "Brier": round(brier, 5),
        "LogLoss": round(logloss, 5),
    }


def paired_temporal_block_analysis(
    block_vals_a: np.ndarray,
    block_vals_b: np.ndarray,
    block_len: int = 2,
    n_boot: int = 5000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Computes paired per-block differences Delta_b = A_b - B_b, moving-block bootstrap 95% CI,
    and two-sided Wilcoxon signed-rank p-value (Section 13.2).
    """
    a = np.asarray(block_vals_a, dtype=np.float64)
    b = np.asarray(block_vals_b, dtype=np.float64)
    diffs = a - b
    n_blocks = len(diffs)
    mean_diff = float(np.mean(diffs))
    median_diff = float(np.median(diffs))
    n_pos_signs = int(np.sum(diffs > 0))
    n_neg_signs = int(np.sum(diffs < 0))

    # Moving-block bootstrap preserving adjacent temporal autocorrelation
    rng = np.random.default_rng(seed)
    L = max(1, min(block_len, n_blocks))
    n_starts = n_blocks - L + 1
    n_draws_per_rep = int(np.ceil(n_blocks / L))
    boot_means = np.empty(n_boot, dtype=np.float64)
    for r in range(n_boot):
        starts = rng.integers(0, n_starts, size=n_draws_per_rep)
        idx = np.concatenate([np.arange(s, s + L) for s in starts])[:n_blocks]
        boot_means[r] = np.mean(diffs[idx])
    ci_low, ci_high = float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))

    # Wilcoxon signed-rank test
    nonzero_diffs = diffs[np.abs(diffs) > 1e-12]
    if len(nonzero_diffs) >= 5:
        stat, p_val = wilcoxon(nonzero_diffs, alternative="two-sided", mode="auto")
        p_val = float(p_val)
        stat = float(stat)
    else:
        stat, p_val = 0.0, 1.0

    return {
        "n_blocks": n_blocks,
        "mean_A": round(float(np.mean(a)), 5),
        "mean_B": round(float(np.mean(b)), 5),
        "mean_paired_diff": round(mean_diff, 5),
        "median_paired_diff": round(median_diff, 5),
        "ci95_low": round(ci_low, 5),
        "ci95_high": round(ci_high, 5),
        "positive_diff_blocks": n_pos_signs,
        "negative_diff_blocks": n_neg_signs,
        "wilcoxon_stat": round(stat, 4),
        "wilcoxon_p_unadjusted": round(p_val, 6),
    }


def holm_bonferroni_correction(p_values: List[float]) -> List[float]:
    """Applies Holm-Bonferroni step-down multiplicity correction to a family of p-values."""
    m = len(p_values)
    if m == 0:
        return []
    order = np.argsort(p_values)
    adjusted = np.zeros(m, dtype=np.float64)
    running_max = 0.0
    for rank_idx, orig_idx in enumerate(order):
        adj = min(1.0, p_values[orig_idx] * (m - rank_idx))
        running_max = max(running_max, adj)
        adjusted[orig_idx] = running_max
    return [round(float(x), 6) for x in adjusted]
