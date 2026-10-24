"""Data Acquisition, Schema Verification, and Step 1 Audit Pipeline (Section 7 & Audit Issue 9).

Provides:
  1. `audit_transaction_dataset(df, dataset_name)`: Computes the full Step 1 audit
     deliverables (time units & span, label prevalence by bin, identity coverage,
     entity cardinality & degree quantiles, hub concentration share, and integrity assertions).
  2. `load_and_join_ieee_cis(train_tx_path, train_id_path, allow_transaction_only=False)`:
     Joins official IEEE-CIS `train_transaction.csv` and `train_identity.csv` on `TransactionID`
     with fail-closed file checks, duplicate-key rejection, 1:1 left-join cardinality verification,
     and SHA-256 file provenance logging.
"""
from __future__ import annotations
import os
import hashlib
import warnings
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple


def _compute_file_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_and_join_ieee_cis(
    train_tx_path: str,
    train_id_path: Optional[str] = None,
    allow_transaction_only: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Loads and merges official IEEE-CIS train_transaction and train_identity CSVs.

    By default (`allow_transaction_only=False`), BOTH `train_transaction.csv` and
    `train_identity.csv` are strictly required; missing either file raises `FileNotFoundError`.
    If `allow_transaction_only=True` is explicitly passed and `train_id_path` is missing,
    emits a visible `UserWarning` and tags `dataset_identity` as `"IEEE_CIS_TRANSACTION_ONLY_ABLATION"`.
    """
    if not train_tx_path or not os.path.isfile(train_tx_path):
        raise FileNotFoundError(
            f"IEEE-CIS train_transaction file not found at '{train_tx_path}'. "
            "Per Kaggle Competition Rules, download train_transaction.csv and train_identity.csv "
            "from https://www.kaggle.com/c/ieee-fraud-detection/data and supply their paths."
        )

    has_id_file = bool(train_id_path and os.path.isfile(train_id_path))
    if not has_id_file and not allow_transaction_only:
        raise FileNotFoundError(
            f"IEEE-CIS train_identity file not found at '{train_id_path}'. "
            "Both train_transaction.csv and train_identity.csv are required for the full joined benchmark. "
            "To run an explicit transaction-only ablation, pass allow_transaction_only=True."
        )

    tx_sha256 = _compute_file_sha256(train_tx_path)
    tx = pd.read_csv(train_tx_path)
    n_tx_before = len(tx)

    required_tx_cols = {"TransactionID", "TransactionDT", "isFraud", "TransactionAmt"}
    missing_tx_cols = required_tx_cols - set(tx.columns)
    if missing_tx_cols:
        raise ValueError(f"Missing required columns in train_transaction: {sorted(missing_tx_cols)}")

    if not tx["TransactionID"].is_unique:
        n_dups = int(tx["TransactionID"].duplicated().sum())
        raise ValueError(f"Duplicate TransactionID keys ({n_dups} duplicates) detected in train_transaction!")

    if not tx["isFraud"].isin([0, 1]).all():
        raise ValueError("Invalid target values in train_transaction['isFraud']; expected binary {0, 1}.")

    if has_id_file:
        assert train_id_path is not None
        id_sha256 = _compute_file_sha256(train_id_path)
        ident = pd.read_csv(train_id_path)
        if "TransactionID" not in ident.columns:
            raise ValueError("Missing required column 'TransactionID' in train_identity!")
        if not ident["TransactionID"].is_unique:
            n_id_dups = int(ident["TransactionID"].duplicated().sum())
            raise ValueError(f"Duplicate TransactionID keys ({n_id_dups} duplicates) detected in train_identity!")

        ident = ident.copy()
        ident["has_identity"] = np.int8(1)
        merged = tx.merge(ident, on="TransactionID", how="left", validate="one_to_one")
        merged["has_identity"] = merged["has_identity"].fillna(0).astype(np.int8)
        n_id_rows = len(ident)
        matched_id = int(merged["has_identity"].sum())
        unmatched_id = int(n_id_rows - matched_id)
        dataset_identity = "IEEE_CIS_FULL_JOINED_BENCHMARK"
    else:
        warnings.warn(
            "Running in explicit transaction-only ablation mode without train_identity.csv! "
            "Output must NOT be reported as the full joined IEEE-CIS benchmark.",
            UserWarning,
            stacklevel=2,
        )
        id_sha256 = None
        merged = tx.copy()
        merged["has_identity"] = np.int8(0)
        n_id_rows = 0
        matched_id = 0
        unmatched_id = 0
        dataset_identity = "IEEE_CIS_TRANSACTION_ONLY_ABLATION"

    if len(merged) != n_tx_before:
        raise RuntimeError(f"Row count changed after left join on TransactionID ({n_tx_before} -> {len(merged)})!")

    # Sort chronologically by TransactionDT, breaking ties deterministically by TransactionID
    merged = merged.sort_values(["TransactionDT", "TransactionID"], kind="mergesort").reset_index(drop=True)
    join_report = {
        "dataset_identity": dataset_identity,
        "allow_transaction_only": bool(allow_transaction_only),
        "train_transaction_path": os.path.abspath(train_tx_path),
        "train_transaction_sha256": tx_sha256,
        "train_identity_path": os.path.abspath(train_id_path) if has_id_file and train_id_path else None,
        "train_identity_sha256": id_sha256,
        "train_transaction_rows": int(n_tx_before),
        "train_identity_rows": int(n_id_rows),
        "merged_rows": int(len(merged)),
        "matched_identity_rows": int(matched_id),
        "unmatched_identity_rows": int(unmatched_id),
        "identity_join_coverage": round(float(merged["has_identity"].mean()), 6),
        "total_columns": int(len(merged.columns)),
        "join_cardinality_verified": "one_to_one",
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
