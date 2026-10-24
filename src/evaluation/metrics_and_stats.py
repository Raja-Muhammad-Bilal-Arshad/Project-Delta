"""Evaluation Metrics and Statistical Analysis Module (Sections 12 & 13, Audit Issue 6, Test G).

Implements:
  - Exact Average Precision (AP) matching scikit-learn's step-wise PR integration:
      AP = sum_n (R_n - R_{n-1}) * P_n
  - Trapezoidal PR-AUC (explicitly distinguished from step-wise AP)
  - ROC-AUC via exact Wilcoxon-Mann-Whitney rank statistic
  - Operational Alert-Budget Metrics: Precision@B% and Recall@B% for B in {0.5%, 1.0%, 5.0%}
    with deterministic tie-breaking (higher score, then higher transaction amount, then earlier index)
  - Delay-aware validation threshold calibration for 1% target FPR using ONLY confirmed mature
    negatives (state_code == 0), never treating immature/unreported rows (state_code == 2) as negatives,
    with documented fallback when confirmed mature negatives in the validation window are below threshold
  - Card-Proxy Precision@k (aggregated by max score per unique card proxy per block)
  - Brier Score and Log-Loss
  - Moving-Block Bootstrap 95% confidence intervals (preserving temporal block dependence)
  - Paired Wilcoxon signed-rank test (via scipy.stats.wilcoxon) with Holm-Bonferroni correction.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import wilcoxon, rankdata
from typing import Dict, Any, List, Tuple, Optional, Union


def compute_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Computes exact Average Precision (AP) = sum_n (R_n - R_{n-1}) * P_n
    matching scikit-learn's average_precision_score implementation [R9].
    """
    y = np.asarray(y_true, dtype=np.int32)
    s = np.asarray(y_score, dtype=np.float64)
    n_pos = int(np.sum(y == 1))
    if n_pos == 0 or len(y) == 0:
        return 0.0
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1).astype(np.float64)
    fp = np.cumsum(y_sorted == 0).astype(np.float64)
    precision = tp / (tp + fp)
    distinct_value_indices = np.where(np.diff(s[order]))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_sorted.size - 1]
    p_at_thresh = precision[threshold_idxs]
    r_at_thresh = tp[threshold_idxs] / n_pos
    r_prev = np.r_[0.0, r_at_thresh[:-1]]
    ap = float(np.sum((r_at_thresh - r_prev) * p_at_thresh))
    return ap


def compute_trapezoidal_prauc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Computes trapezoidal area under the precision-recall curve (distinct from step-wise AP).
    Evaluates at distinct score thresholds so tied scores are handled invariantly.
    """
    y = np.asarray(y_true, dtype=np.int32)
    s = np.asarray(y_score, dtype=np.float64)
    n_pos = int(np.sum(y == 1))
    if n_pos == 0 or len(y) == 0:
        return 0.0
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1).astype(np.float64)
    fp = np.cumsum(y_sorted == 0).astype(np.float64)
    distinct_value_indices = np.where(np.diff(s[order]))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_sorted.size - 1]
    precision = np.r_[1.0, (tp / (tp + fp))[threshold_idxs]]
    recall = np.r_[0.0, (tp / n_pos)[threshold_idxs]]
    return float(np.sum(0.5 * (precision[1:] + precision[:-1]) * (recall[1:] - recall[:-1])))


def compute_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Computes exact ROC-AUC via rank statistic (handling ties via average rank)."""
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
    if n == 0 or budget_pct <= 0.0:
        return 0.0, 0.0, 0
    k = min(n, max(1, int(np.ceil((budget_pct / 100.0) * n))))
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
    k_cards: int = 25,
) -> float:
    """Computes Card-Proxy Precision@k by aggregating max score and max true fraud label
    per unique card proxy within the evaluation block (Section 12.1).
    """
    y = np.asarray(y_true, dtype=np.int32)
    s = np.asarray(y_score, dtype=np.float64)
    c = np.asarray(card_proxy_ids, dtype=np.int64)
    if len(y) == 0:
        return 0.0
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


def _weighted_quantile(values: np.ndarray, quantile: float, weights: np.ndarray) -> float:
    """Computes weighted quantile for fallback delay-aware negative score distribution."""
    v = np.asarray(values, dtype=np.float64)
    w = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    if len(v) == 0 or np.sum(w) <= 0.0:
        return 0.5
    sorter = np.argsort(v, kind="mergesort")
    v_sorted = v[sorter]
    w_sorted = w[sorter]
    cdf = np.cumsum(w_sorted) / np.sum(w_sorted)
    idx = int(np.searchsorted(cdf, quantile, side="left"))
    idx = min(max(idx, 0), len(v_sorted) - 1)
    return float(v_sorted[idx])


def select_threshold_for_target_fpr(
    y_or_state_val: np.ndarray,
    s_val: np.ndarray,
    target_fpr: float = 0.01,
    confirmed_neg_mask: Optional[np.ndarray] = None,
    fallback_mature_neg_scores: Optional[np.ndarray] = None,
    posterior_q_val: Optional[np.ndarray] = None,
    min_confirmed_negatives: int = 30,
    return_info: bool = False,
) -> Union[float, Tuple[float, Dict[str, Any]]]:
    """Selects score threshold strictly on time-safe validation data to achieve target FPR (Audit Issue 6).

    Crucial Invariant:
      Immature / unreported transactions (`state_code == 2`) are NEVER treated as confirmed negatives.
      1. Primary: Uses confirmed mature negatives (`confirmed_neg_mask` or `y_or_state_val == 0`) inside
         the validation window whenever `n_confirmed_neg_in_val >= min_confirmed_negatives`.
      2. Fallback A: When the validation window is shorter than maturity horizon M so that
         `n_confirmed_neg_in_val < min_confirmed_negatives`, uses `fallback_mature_neg_scores`
         (the most recent confirmed mature negative cohort at cutoff tau_val, where age >= M and
         report_time > tau_val) if it contains at least `min_confirmed_negatives` observations.
      3. Fallback B: If even the historical mature cohort has `< min_confirmed_negatives` observations
         (e.g., if tau_val < M), uses a survival-posterior-weighted quantile over unreported validation
         rows weighted by `w_i = 1 - q(a_i; pi_tau)`.
    """
    s = np.asarray(s_val, dtype=np.float64)
    st = np.asarray(y_or_state_val)

    if confirmed_neg_mask is not None:
        c_neg_mask = np.asarray(confirmed_neg_mask, dtype=bool)
    else:
        # In reconstruct_label_state_at_cutoff, state_code == 0 is confirmed mature negative,
        # state_code == 1 is visible positive, and state_code == 2 is immature unreported.
        c_neg_mask = (st == 0)

    n_vis_pos = int(np.sum(st == 1))
    n_mature_neg = int(np.sum(c_neg_mask))
    n_immature = int(np.sum(st == 2))

    if n_mature_neg >= min_confirmed_negatives or (
        n_mature_neg > 0 and fallback_mature_neg_scores is None and posterior_q_val is None
    ):
        neg_scores = s[c_neg_mask]
        thresh = float(np.quantile(neg_scores, 1.0 - target_fpr))
        calib_mode = "confirmed_mature_negatives_in_window"
        n_used = int(len(neg_scores))
    elif fallback_mature_neg_scores is not None and len(fallback_mature_neg_scores) >= min_confirmed_negatives:
        fb_scores = np.asarray(fallback_mature_neg_scores, dtype=np.float64)
        thresh = float(np.quantile(fb_scores, 1.0 - target_fpr))
        calib_mode = "recent_confirmed_mature_negative_buffer"
        n_used = int(len(fb_scores))
    elif posterior_q_val is not None and len(s) > 0:
        unrep_mask = (st != 1)
        if np.any(unrep_mask):
            q_u = np.clip(np.asarray(posterior_q_val, dtype=np.float64)[unrep_mask], 0.0, 1.0)
            w_neg = 1.0 - q_u
            thresh = _weighted_quantile(s[unrep_mask], 1.0 - target_fpr, w_neg)
            calib_mode = "posterior_weighted_unreported_fallback"
            n_used = int(np.sum(unrep_mask))
        else:
            thresh = 0.5
            calib_mode = "constant_fallback_no_negatives"
            n_used = 0
    elif n_mature_neg > 0:
        neg_scores = s[c_neg_mask]
        thresh = float(np.quantile(neg_scores, 1.0 - target_fpr))
        calib_mode = "confirmed_mature_negatives_small_sample"
        n_used = int(len(neg_scores))
    else:
        thresh = 0.5
        calib_mode = "constant_fallback_no_negatives"
        n_used = 0

    info = {
        "n_val_total": int(len(s)),
        "n_val_vis_pos": n_vis_pos,
        "n_val_mature_neg_in_window": n_mature_neg,
        "n_val_immature_excluded": n_immature,
        "n_confirmed_neg_used": n_used,
        "calibration_mode": calib_mode,
        "threshold_1pct_fpr": round(thresh, 6),
    }
    if return_info:
        return thresh, info
    return thresh


def compute_block_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    card_proxy_ids: Optional[np.ndarray] = None,
    amounts: Optional[np.ndarray] = None,
    val_threshold_1pct_fpr: Optional[float] = None,
    val_calibration_info: Optional[Dict[str, Any]] = None,
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

    brier = float(np.mean((s - y) ** 2)) if n > 0 else 0.0
    logloss = float(-np.mean(y * np.log(s) + (1.0 - y) * np.log(1.0 - s))) if n > 0 else 0.0

    if val_threshold_1pct_fpr is not None:
        thresh = val_threshold_1pct_fpr
    else:
        thresh = select_threshold_for_target_fpr(y, s, 0.01)

    rec_at_1pct_fpr = float(np.mean(s[y == 1] >= thresh)) if n_pos > 0 else 0.0
    realized_test_fpr = float(np.mean(s[y == 0] >= thresh)) if int(np.sum(y == 0)) > 0 else 0.0

    cp_25 = compute_card_proxy_precision_at_k(y, s, card_proxy_ids, k_cards=25) if card_proxy_ids is not None else 0.0

    out = {
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
        "Realized_Test_FPR": round(realized_test_fpr, 5),
        "CardProxy_Precision@25": round(cp_25, 5),
        "Brier": round(brier, 5),
        "LogLoss": round(logloss, 5),
    }
    if val_calibration_info is not None:
        out["val_threshold_calibration"] = val_calibration_info
    return out


def paired_temporal_block_analysis(
    block_vals_a: np.ndarray,
    block_vals_b: np.ndarray,
    block_len: int = 2,
    n_boot: int = 5000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Computes paired per-block differences Delta_b = A_b - B_b, moving-block bootstrap 95% CI,
    standardized paired effect size (Cohen's d_z), and two-sided Wilcoxon signed-rank p-value (Section 13.2).
    """
    a = np.asarray(block_vals_a, dtype=np.float64)
    b = np.asarray(block_vals_b, dtype=np.float64)
    diffs = a - b
    n_blocks = len(diffs)
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1)) if n_blocks > 1 else 0.0
    cohens_dz = float(mean_diff / std_diff) if std_diff > 1e-12 else 0.0
    median_diff = float(np.median(diffs))
    n_pos_signs = int(np.sum(diffs > 1e-12))
    n_neg_signs = int(np.sum(diffs < -1e-12))
    n_ties = int(n_blocks - n_pos_signs - n_neg_signs)

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
        "std_paired_diff": round(std_diff, 5),
        "cohens_dz": round(cohens_dz, 4),
        "median_paired_diff": round(median_diff, 5),
        "ci95_low": round(ci_low, 5),
        "ci95_high": round(ci_high, 5),
        "positive_diff_blocks": n_pos_signs,
        "negative_diff_blocks": n_neg_signs,
        "tied_blocks": n_ties,
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
