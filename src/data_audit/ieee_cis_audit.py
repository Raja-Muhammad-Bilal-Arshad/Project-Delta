"""Data Acquisition, Schema Verification, and Step 1 Audit Pipeline (Section 7).

Provides:
  1. `audit_transaction_dataset(df, dataset_name)`: Computes the full Step 1 audit
     deliverables (time units & span, label prevalence by bin, identity coverage,
     entity cardinality & degree quantiles, hub concentration share, and integrity assertions).
  2. `load_and_join_ieee_cis(train_tx_path, train_id_path)`: Joins IEEE-CIS
     `train_transaction.csv` and `train_identity.csv` on `TransactionID` with
     strict key uniqueness and row-preservation assertions.
"""
from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional


def load_and_join_ieee_cis(train_tx_path: str, train_id_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Loads and merges official IEEE-CIS train_transaction and train_identity CSVs."""
    if not os.path.isfile(train_tx_path):
        raise FileNotFoundError(
            f"IEEE-CIS train_transaction file not found at '{train_tx_path}'. "
            "Per Kaggle Competition Rules, download train_transaction.csv and train_identity.csv "
            "from https://www.kaggle.com/c/ieee-fraud-detection/data and supply their path."
        )
    tx = pd.read_csv(train_tx_path)
    n_tx_before = len(tx)
    assert "TransactionID" in tx.columns and "TransactionDT" in tx.columns and "isFraud" in tx.columns
    assert tx["TransactionID"].is_unique, "Duplicate TransactionID detected in train_transaction!"

    if os.path.isfile(train_id_path):
        ident = pd.read_csv(train_id_path)
        assert ident["TransactionID"].is_unique, "Duplicate TransactionID detected in train_identity!"
        ident["has_identity"] = 1
        merged = tx.merge(ident, on="TransactionID", how="left")
        merged["has_identity"] = merged["has_identity"].fillna(0).astype(np.int8)
        n_id_rows = len(ident)
        unmatched_id = int(len(set(ident["TransactionID"]) - set(tx["TransactionID"])))
    else:
        merged = tx.copy()
        merged["has_identity"] = 0
        n_id_rows = 0
        unmatched_id = 0

    assert len(merged) == n_tx_before, "Row count changed after left join on TransactionID!"
    # Sort chronologically by TransactionDT, breaking ties deterministically by TransactionID
    merged = merged.sort_values(["TransactionDT", "TransactionID"], kind="mergesort").reset_index(drop=True)
    join_report = {
        "train_transaction_rows": int(n_tx_before),
        "train_identity_rows": int(n_id_rows),
        "merged_rows": int(len(merged)),
        "unmatched_identity_rows": int(unmatched_id),
        "identity_join_coverage": float(merged["has_identity"].mean()),
    }
    return merged, join_report


def audit_entity_degrees(df: pd.DataFrame, entity_cols: List[str]) -> Dict[str, Any]:
    """Computes unique values, missing rate, degree quantiles, max degree, and top-1% hub share."""
    entity_stats: Dict[str, Any] = {}
    n_total = len(df)
    for col in entity_cols:
        if col not in df.columns:
            continue
        s = df[col]
        valid_mask = s.notna() & (s != -1) & (s != "")
        s_valid = s[valid_mask]
        missing_rate = float(1.0 - (len(s_valid) / max(n_total, 1)))
        if len(s_valid) == 0:
            continue
        vc = s_valid.value_counts()
        degrees = vc.to_numpy(dtype=np.float64)
        top_1pct_k = max(1, int(np.ceil(0.01 * len(degrees))))
        top_1pct_tx_share = float(np.sum(degrees[:top_1pct_k]) / len(s_valid))
        deg_gt_50_tx_share = float(np.sum(degrees[degrees > 50]) / len(s_valid))
        entity_stats[col] = {
            "unique_entities": int(len(degrees)),
            "missing_or_unlinked_rate": round(missing_rate, 4),
            "degree_p50": float(np.percentile(degrees, 50)),
            "degree_p90": float(np.percentile(degrees, 90)),
            "degree_p99": float(np.percentile(degrees, 99)),
            "max_degree": int(np.max(degrees)),
            "top_1pct_entity_tx_share": round(top_1pct_tx_share, 4),
            "tx_share_in_hubs_deg_gt_50": round(deg_gt_50_tx_share, 4),
        }
    return entity_stats


def run_step1_data_audit(
    df: pd.DataFrame,
    dataset_name: str,
    entity_cols: List[str],
    n_bins: int = 18,
) -> Dict[str, Any]:
    """Runs the complete Step 1 Data Audit and Integrity Check (Section 7.2)."""
    assert "TransactionID" in df.columns, "Missing TransactionID"
    assert df["TransactionID"].is_unique, "Duplicate TransactionIDs found"
    assert "TransactionDT" in df.columns, "Missing TransactionDT"
    assert "isFraud" in df.columns, "Missing isFraud"
    assert df["isFraud"].isin([0, 1]).all(), "Invalid values in isFraud"

    dt = df["TransactionDT"].to_numpy(dtype=np.float64)
    is_monotonic = bool(np.all(np.diff(dt) >= 0))
    dt_min, dt_max = float(np.min(dt)), float(np.max(dt))
    dt_span_sec = dt_max - dt_min
    dt_span_days_equiv = dt_span_sec / 86400.0

    # Bin statistics over time
    bin_edges = np.linspace(dt_min, dt_max + 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(dt, bin_edges) - 1, 0, n_bins - 1)
    bin_records = []
    for b in range(n_bins):
        m = (bin_idx == b)
        cnt = int(np.sum(m))
        pos = int(np.sum(df["isFraud"].to_numpy()[m] == 1)) if cnt > 0 else 0
        id_cov = float(np.mean(df["has_identity"].to_numpy()[m])) if (cnt > 0 and "has_identity" in df.columns) else 0.0
        bin_records.append({
            "bin_id": int(b),
            "dt_start": float(bin_edges[b]),
            "dt_end": float(bin_edges[b + 1]),
            "n_transactions": cnt,
            "n_frauds": pos,
            "fraud_rate": round(pos / max(cnt, 1), 5),
            "identity_coverage": round(id_cov, 4),
            "low_positive_warning": bool(pos < 15),
        })

    entity_stats = audit_entity_degrees(df, entity_cols)

    return {
        "dataset_name": dataset_name,
        "n_rows": int(len(df)),
        "n_frauds": int(df["isFraud"].sum()),
        "n_legitimate": int((df["isFraud"] == 0).sum()),
        "missing_targets": int(df["isFraud"].isna().sum()),
        "fraud_prevalence": round(float(df["isFraud"].mean()), 5),
        "identity_coverage_overall": round(float(df["has_identity"].mean()), 4) if "has_identity" in df.columns else 0.0,
        "time_audit": {
            "min_TransactionDT": dt_min,
            "max_TransactionDT": dt_max,
            "span_raw_units": dt_span_sec,
            "span_86400_day_units": round(dt_span_days_equiv, 2),
            "is_monotonic_non_decreasing": is_monotonic,
            "unique_timestamps": int(df["TransactionDT"].nunique()),
        },
        "temporal_bins": bin_records,
        "entity_degree_audit": entity_stats,
        "integrity_assertions_passed": True,
    }
