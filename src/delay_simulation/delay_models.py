"""Label-Delay Models and Bayesian Maturity-Aware Posterior Computation (Section 9).

Implements exact bounded/truncated delay distributions (D0, D1, D2, D3),
event-log label visibility state reconstruction at cutoff tau, time-safe
fraud prior estimation, and Bayesian posterior soft targets q(a).
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import brentq
from dataclasses import dataclass
from typing import Dict, Tuple, Optional


@dataclass(frozen=True)
class DelayScenarioSpec:
    scenario_id: str
    name: str
    maturity_horizon_days: float
    hard_max_days: float
    description: str


def solve_truncated_exponential_lambda(median_days: float = 1.0, max_days: float = 14.0) -> float:
    """Solves for rate lambda > 0 of a truncated exponential on [0, max_days]
    such that F_T(median_days) = (1 - exp(-lambda * median_days)) / (1 - exp(-lambda * max_days)) = 0.5.
    """
    def obj(lmbd: float) -> float:
        if lmbd <= 1e-12:
            return (median_days / max_days) - 0.5
        return (1.0 - np.exp(-lmbd * median_days)) / (1.0 - np.exp(-lmbd * max_days)) - 0.5

    return float(brentq(obj, 1e-6, 50.0))


LAMBDA_D1_14D = solve_truncated_exponential_lambda(1.0, 14.0)

# Piecewise-linear knot points for D2 (inspired by Nguyen et al., AISTATS 2022 compromised-card profile)
# Knots in days: (0.0, 0.0), (1/24 day, 0.01), (1.0 day, 0.30), (7.0 days, 0.60), (10.0 days, 0.80), (30.0 days, 1.00)
D2_KNOTS_DAYS = np.array([0.0, 1.0 / 24.0, 1.0, 7.0, 10.0, 30.0], dtype=np.float64)
D2_KNOTS_CDF = np.array([0.0, 0.01, 0.30, 0.60, 0.80, 1.00], dtype=np.float64)

SCENARIO_SPECS: Dict[str, DelayScenarioSpec] = {
    "D0": DelayScenarioSpec("D0", "Immediate-Label Diagnostic", 0.0, 0.0, "Zero delay (D=0); all fraud labels visible immediately."),
    "D1": DelayScenarioSpec("D1", "Bounded Short Delay (Truncated Exp)", 14.0, 14.0, f"Truncated exponential on [0, 14] days with exact median 1.0 day (lambda={LAMBDA_D1_14D:.6f})."),
    "D2": DelayScenarioSpec("D2", "Profile-Inspired Delay (Piecewise-Linear)", 30.0, 30.0, "Piecewise-linear CDF through (1h, 1%), (1d, 30%), (7d, 60%), (10d, 80%), (30d, 100%); M=30 days."),
    "D3": DelayScenarioSpec("D3", "Long-Tail Stress Test Mixture", 90.0, 90.0, "80% sampled from D2 on [0, 30] days + 20% sampled uniformly on [30, 90] days; M=90 days."),
    "D3_misspecified_M30": DelayScenarioSpec("D3_misspecified_M30", "Premature Maturity Stress Test (D3 with M=30)", 30.0, 90.0, "Deliberately misspecified M=30 days under D3 90-day tail to measure premature negative labeling distortion.")
}


def delay_cdf(age_days: np.ndarray | float, scenario_id: str) -> np.ndarray:
    """Evaluates F_D(a) = P(D <= a | Y=1) for age_days >= 0."""
    a = np.asarray(age_days, dtype=np.float64)
    if scenario_id == "D0":
        return np.where(a >= 0.0, 1.0, 0.0)
    elif scenario_id == "D1":
        M = 14.0
        lmbd = LAMBDA_D1_14D
        denom = 1.0 - np.exp(-lmbd * M)
        cdf = (1.0 - np.exp(-lmbd * np.clip(a, 0.0, M))) / denom
        return np.where(a <= 0.0, 0.0, np.where(a >= M, 1.0, cdf))
    elif scenario_id == "D2":
        return np.interp(a, D2_KNOTS_DAYS, D2_KNOTS_CDF, left=0.0, right=1.0)
    elif scenario_id in ("D3", "D3_misspecified_M30"):
        cdf_d2 = np.interp(a, D2_KNOTS_DAYS, D2_KNOTS_CDF, left=0.0, right=1.0)
        cdf_tail = np.clip((a - 30.0) / 60.0, 0.0, 1.0)
        cdf = 0.80 * cdf_d2 + 0.20 * cdf_tail
        return np.where(a <= 0.0, 0.0, np.where(a >= 90.0, 1.0, cdf))
    else:
        raise ValueError(f"Unknown delay scenario_id: {scenario_id}")


def delay_survival(age_days: np.ndarray | float, scenario_id: str) -> np.ndarray:
    """Evaluates survival function S(a) = P(D > a | Y=1) = 1 - F_D(a)."""
    return np.clip(1.0 - delay_cdf(age_days, scenario_id), 0.0, 1.0)


def sample_fraud_delays(n_samples: int, scenario_id: str, seed: int = 42) -> np.ndarray:
    """Samples exact fraud reporting delays D_i (in days) via inverse-CDF."""
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 1.0, size=n_samples)
    if scenario_id == "D0":
        return np.zeros(n_samples, dtype=np.float64)
    elif scenario_id == "D1":
        M = 14.0
        lmbd = LAMBDA_D1_14D
        denom = 1.0 - np.exp(-lmbd * M)
        return -np.log(1.0 - u * denom) / lmbd
    elif scenario_id == "D2":
        return np.interp(u, D2_KNOTS_CDF, D2_KNOTS_DAYS)
    elif scenario_id in ("D3", "D3_misspecified_M30"):
        is_tail = rng.uniform(0.0, 1.0, size=n_samples) >= 0.80
        d_d2 = np.interp(u, D2_KNOTS_CDF, D2_KNOTS_DAYS)
        d_tail = 30.0 + 60.0 * u
        return np.where(is_tail, d_tail, d_d2)
    else:
        raise ValueError(f"Unknown delay scenario_id: {scenario_id}")


def generate_event_log_delays(
    event_times_days: np.ndarray,
    true_labels: np.ndarray,
    scenario_id: str,
    seed: int = 42,
    card_ids: Optional[np.ndarray] = None,
    card_correlated: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates simulated report arrival times t_i^rep = t_i + D_i for Y_i=1, and +inf for Y_i=0.
    If card_correlated=True, compromised cards share a card-blocking timestamp t_block = min(t_i + D_i)
    across fraud events on that card proxy.
    """
    n = len(event_times_days)
    delays = np.full(n, np.inf, dtype=np.float64)
    report_times = np.full(n, np.inf, dtype=np.float64)
    fraud_mask = (true_labels == 1)
    n_frauds = int(np.sum(fraud_mask))
    if n_frauds > 0:
        drawn_delays = sample_fraud_delays(n_frauds, scenario_id, seed=seed)
        delays[fraud_mask] = drawn_delays
        report_times[fraud_mask] = event_times_days[fraud_mask] + drawn_delays
        if card_correlated and card_ids is not None:
            # Compromised card blocking correlation: once the earliest fraud report on a card arrives,
            # all prior fraudulent transactions on that card become reported at max(t_i, min_report_time).
            fraud_indices = np.where(fraud_mask)[0]
            card_min_rep: Dict[int, float] = {}
            for idx in fraud_indices:
                c = int(card_ids[idx])
                rt = float(report_times[idx])
                if c not in card_min_rep or rt < card_min_rep[c]:
                    card_min_rep[c] = rt
            for idx in fraud_indices:
                c = int(card_ids[idx])
                report_times[idx] = max(float(event_times_days[idx]) + 0.01, card_min_rep[c])
                delays[idx] = report_times[idx] - event_times_days[idx]
    return delays, report_times


def compute_posterior_q(
    age_days: np.ndarray | float,
    prior_pi: float,
    survival_scenario_id: str,
    maturity_horizon_days: Optional[float] = None,
) -> np.ndarray:
    """Computes Bayesian posterior q(a) = [pi * S(a)] / [pi * S(a) + (1 - pi)]
    for an unreported transaction of age a >= 0 (Section 9.2).
    """
    a = np.asarray(age_days, dtype=np.float64)
    pi = float(np.clip(prior_pi, 1e-6, 1.0 - 1e-6))
    s_a = delay_survival(a, survival_scenario_id)
    if maturity_horizon_days is not None:
        s_a = np.where(a >= maturity_horizon_days, 0.0, s_a)
    num = pi * s_a
    den = num + (1.0 - pi)
    q = np.where(a < 0.0, pi, num / np.maximum(den, 1e-12))
    return np.clip(q, 0.0, 1.0)


def estimate_time_safe_prior(
    event_times_days: np.ndarray,
    report_times_days: np.ndarray,
    cutoff_tau: float,
    maturity_horizon_days: float,
    survival_scenario_id: str,
    default_prior: float = 0.035,
) -> float:
    """Estimates fraud prevalence pi_tau at cutoff tau without peeking at future or hidden labels.
    Uses mature historical cohort (a_i >= M) if non-empty; otherwise applies IPW/Horvitz-Thompson
    survival adjustment on visible reports with Laplace smoothing.
    """
    hist_mask = (event_times_days <= cutoff_tau)
    if not np.any(hist_mask):
        return default_prior
    ages = cutoff_tau - event_times_days[hist_mask]
    vis_pos = (report_times_days[hist_mask] <= cutoff_tau)
    mature_mask = (ages >= maturity_horizon_days)
    if np.sum(mature_mask) >= 200:
        mature_rate = float(np.mean(vis_pos[mature_mask]))
        # Smooth slightly with recent IPW estimate to handle mild drift
        cdf_vals = np.maximum(delay_cdf(ages, survival_scenario_id), 0.05)
        ipw_rate = float(np.sum(vis_pos / cdf_vals) / len(ages))
        est = 0.75 * mature_rate + 0.25 * ipw_rate
        return float(np.clip(est, 0.002, 0.50))
    else:
        cdf_vals = np.maximum(delay_cdf(ages, survival_scenario_id), 0.10)
        ipw_rate = float((np.sum(vis_pos / cdf_vals) + 2.0 * default_prior) / (len(ages) + 2.0))
        return float(np.clip(ipw_rate, 0.002, 0.50))


def reconstruct_label_state_at_cutoff(
    event_times_days: np.ndarray,
    report_times_days: np.ndarray,
    cutoff_tau: float,
    maturity_horizon_days: float,
    survival_scenario_id: str,
    prior_pi: float,
    true_labels_for_oracle_only: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Reconstructs observable label state at decision/retraining cutoff tau (Section 9.4).
    Crucially, model-facing states (V0, V1, V2) NEVER inspect true_labels_for_oracle_only;
    they rely exclusively on event_times_days and report_times_days <= cutoff_tau.
    
    State codes:
      1 = Visible positive (report_time <= cutoff_tau and event_time <= cutoff_tau)
      0 = Mature negative (report_time > cutoff_tau and age >= maturity_horizon_days)
      2 = Immature / unreported (report_time > cutoff_tau and 0 <= age < maturity_horizon_days)
     -1 = Future transaction (event_time > cutoff_tau; excluded from state)
    """
    ages = cutoff_tau - event_times_days
    is_past = (event_times_days <= cutoff_tau)
    is_vis_pos = is_past & (report_times_days <= cutoff_tau)
    is_unreported = is_past & (~is_vis_pos)
    is_mature_neg = is_unreported & (ages >= maturity_horizon_days)
    is_immature = is_unreported & (ages < maturity_horizon_days)

    state_code = np.full(len(event_times_days), -1, dtype=np.int8)
    state_code[is_vis_pos] = 1
    state_code[is_mature_neg] = 0
    state_code[is_immature] = 2

    q_imm = compute_posterior_q(ages, prior_pi, survival_scenario_id, maturity_horizon_days)
    
    # Targets for V0, V1, V2
    y_v0 = np.where(is_vis_pos, 1.0, 0.0)
    mask_v0 = is_past

    y_v1 = np.where(is_vis_pos, 1.0, 0.0)
    mask_v1 = is_vis_pos | is_mature_neg  # Drops immature rows from training loss

    y_v2 = np.where(is_vis_pos, 1.0, np.where(is_mature_neg, 0.0, q_imm))
    mask_v2 = is_past

    res = {
        "state_code": state_code,
        "ages": ages,
        "is_past": is_past,
        "is_vis_pos": is_vis_pos,
        "is_mature_neg": is_mature_neg,
        "is_immature": is_immature,
        "q_posterior": np.where(is_immature, q_imm, np.where(is_vis_pos, 1.0, 0.0)),
        "y_v0": y_v0,
        "mask_v0": mask_v0,
        "y_v1": y_v1,
        "mask_v1": mask_v1,
        "y_v2": y_v2,
        "mask_v2": mask_v2,
    }
    if true_labels_for_oracle_only is not None:
        res["y_v3_oracle"] = true_labels_for_oracle_only.astype(np.float64)
        res["mask_v3_oracle"] = is_past
    return res
