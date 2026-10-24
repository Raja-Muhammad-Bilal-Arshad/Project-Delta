"""Reproducible Multi-Entity Payment Transaction Simulator (Section 14 & Step 10-11).

Implements the canonical 3-scenario customer-terminal transaction generator from
the Fraud Detection Handbook (Le Borgne et al., 2022) and extends it with:
  - Card proxy (card1) vs. composite card proxy (card_comp) displaying realistic splits/merges
  - Billing address proxy (addr1) with regional hubs
  - Purchaser email domain proxy (P_emaildomain) with extreme power-law public hubs
  - Device fingerprint proxy (DeviceInfo) with partial identity coverage (~26.5% matching IEEE-CIS)
  - Coordinated fraud rings & card-compromise bursts across shared entities so graph topology
    contains genuine 1-hop and 2-hop signal that degrades under hub dilution or label delay.

Authoritative study configuration (reconciled per Audit Issue 1):
  n_transactions = 12,000, n_days = 180, n_customers = 1,200, n_terminals = 450,
  n_addresses = 450, n_email_domains = 60, n_devices = 1,800, seed = 20261024
  -> Produces 12,000 transactions, 208 frauds (1.7333% prevalence), 26.49% identity coverage.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

SIMULATOR_GENERATOR_VERSION = "handbook_multi_entity_v2.0_reconciled_12k"


def generate_handbook_multi_entity_dataset(
    n_transactions: int = 12000,
    n_days: int = 180,
    n_customers: int = 1200,
    n_terminals: int = 450,
    n_addresses: int = 450,
    n_email_domains: int = 60,
    n_devices: int = 1800,
    seed: int = 20261024,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Generates a chronologically ordered multi-entity payment card dataset."""
    rng = np.random.default_rng(seed)

    # Customer spending profiles (mean amount, std amount, frequency weight)
    cust_mean_amt = rng.uniform(15.0, 140.0, size=n_customers)
    cust_std_amt = cust_mean_amt * rng.uniform(0.25, 0.55, size=n_customers)
    cust_freq = rng.gamma(shape=2.0, scale=1.0, size=n_customers)
    cust_prob = cust_freq / np.sum(cust_freq)

    # Assign primary entities to customers
    # card1 has slight collisions (false merges ~10% of customers share a card1 BIN/sub-range)
    cust_card1 = rng.integers(1000, 1000 + int(n_customers * 0.90), size=n_customers)
    cust_card2 = rng.integers(100, 600, size=n_customers)
    # Composite card key is higher precision
    cust_card_comp = cust_card1 * 1000 + cust_card2

    # Address proxy with power-law urban hubs
    addr_weights = 1.0 / (np.arange(1, n_addresses + 1) ** 1.15)
    addr_weights /= np.sum(addr_weights)
    cust_addr = rng.choice(n_addresses, size=n_customers, p=addr_weights)

    # Email domain proxy with extreme public-domain hubs (top 5 domains account for ~68% of volume)
    email_weights = 1.0 / (np.arange(1, n_email_domains + 1) ** 1.65)
    email_weights /= np.sum(email_weights)
    cust_email = rng.choice(n_email_domains, size=n_customers, p=email_weights)

    # Sample transaction timestamps continuously over [1.0, n_days + 1.0] days
    # (Matching IEEE-CIS convention where day 1 starts at 86,400 seconds)
    raw_days = np.sort(rng.uniform(1.0, float(n_days) + 1.0, size=n_transactions))
    transaction_dt_sec = np.round(raw_days * 86400.0).astype(np.int64)
    event_time_days = transaction_dt_sec / 86400.0

    cust_ids = rng.choice(n_customers, size=n_transactions, p=cust_prob)

    # Terminal selection: each customer mostly uses 3-5 local terminals, occasionally random
    cust_home_terms = rng.integers(0, n_terminals, size=(n_customers, 4))
    term_choice_idx = rng.integers(0, 4, size=n_transactions)
    term_ids = cust_home_terms[cust_ids, term_choice_idx]
    random_term_mask = rng.uniform(0.0, 1.0, size=n_transactions) < 0.15
    term_ids[random_term_mask] = rng.integers(0, n_terminals, size=int(np.sum(random_term_mask)))

    # Transaction amounts
    amounts = rng.normal(loc=cust_mean_amt[cust_ids], scale=cust_std_amt[cust_ids])
    amounts = np.clip(np.round(amounts, 2), 2.50, 2500.0)

    # Entity attributes per transaction (with realistic attack/session overrides)
    card1 = cust_card1[cust_ids].copy()
    card_comp = cust_card_comp[cust_ids].copy()
    addr1 = cust_addr[cust_ids].copy()
    email_dom = cust_email[cust_ids].copy()

    # Partial identity / device coverage (~26.49% overall after attack overrides, matching IEEE-CIS 24.42%)
    has_identity = rng.uniform(0.0, 1.0, size=n_transactions) < 0.255
    cust_device = rng.integers(1, n_devices + 1, size=n_customers)
    device_id = np.where(has_identity, cust_device[cust_ids], -1)

    # Auxiliary tabular features (analogous to IEEE-CIS C, D, M, V features)
    hour_of_day = ((transaction_dt_sec // 3600) % 24).astype(np.float64)
    is_night = ((hour_of_day >= 1) & (hour_of_day <= 5)).astype(np.float64)
    product_cd = rng.choice([0, 1, 2, 3, 4], size=n_transactions, p=[0.74, 0.10, 0.08, 0.05, 0.03])
    dist_score = rng.exponential(scale=1.0, size=n_transactions)
    ip_risk_tabular = rng.normal(0.0, 1.0, size=n_transactions)
    auth_match_flag = (rng.uniform(0.0, 1.0, size=n_transactions) < 0.85).astype(np.float64)

    # Fraud generation: 3 Handbook Scenarios + Coordinated Multi-Entity Fraud Rings
    is_fraud = np.zeros(n_transactions, dtype=np.int8)
    fraud_scenario = np.zeros(n_transactions, dtype=np.int8)

    # Scenario 1: Baseline high-amount threshold / anomaly fraud (tabular + slight night elevation)
    s1_mask = (amounts > 210.0) & (rng.uniform(0.0, 1.0, size=n_transactions) < 0.22)
    is_fraud[s1_mask] = 1
    fraud_scenario[s1_mask] = 1

    # Scenario 2: Compromised terminals (batches of terminals compromised for 10-day windows)
    n_comp_term_waves = 55
    for _ in range(n_comp_term_waves):
        t_start = rng.uniform(2.0, float(n_days) - 10.0)
        t_end = t_start + rng.uniform(6.0, 14.0)
        target_term = int(rng.integers(0, n_terminals))
        wave_mask = (term_ids == target_term) & (event_time_days >= t_start) & (event_time_days <= t_end)
        hit = wave_mask & (rng.uniform(0.0, 1.0, size=n_transactions) < 0.62)
        is_fraud[hit] = 1
        fraud_scenario[hit] = 2
        ip_risk_tabular[hit] += rng.normal(0.85, 0.4, size=int(np.sum(hit)))

    # Scenario 3: Compromised cards & coordinated fraud rings (shared device/address/card bursts)
    n_comp_rings = 140
    for r_idx in range(n_comp_rings):
        t_start = rng.uniform(2.0, float(n_days) - 8.0)
        t_end = t_start + rng.uniform(3.0, 10.0)
        ring_cards = rng.choice(n_customers, size=rng.integers(2, 6), replace=False)
        shared_attack_addr = int(rng.integers(50, n_addresses))
        shared_attack_dev = int(rng.integers(1, 350))
        ring_mask = np.isin(cust_ids, ring_cards) & (event_time_days >= t_start) & (event_time_days <= t_end)
        idxs = np.where(ring_mask)[0]
        if len(idxs) > 0:
            for idx in idxs:
                if rng.uniform(0.0, 1.0) < 0.72:
                    is_fraud[idx] = 1
                    fraud_scenario[idx] = 3
                    amounts[idx] = np.round(amounts[idx] * rng.uniform(1.6, 3.8), 2)
                    addr1[idx] = shared_attack_addr
                    has_identity[idx] = True
                    device_id[idx] = shared_attack_dev
                    ip_risk_tabular[idx] += rng.normal(1.15, 0.5)
                    auth_match_flag[idx] = 0.0 if rng.uniform(0, 1) < 0.65 else 1.0

    # Inject mild tabular signal for all frauds so B0 raw tabular is a realistic competitive baseline
    fraud_indices = np.where(is_fraud == 1)[0]
    ip_risk_tabular[fraud_indices] += rng.normal(0.75, 0.6, size=len(fraud_indices))
    dist_score[fraud_indices] *= rng.uniform(1.1, 2.4, size=len(fraud_indices))

    df = pd.DataFrame({
        "TransactionID": np.arange(3000000, 3000000 + n_transactions, dtype=np.int64),
        "TransactionDT": transaction_dt_sec,
        "event_time_days": event_time_days,
        "day_Block": np.floor(event_time_days - 1.0).astype(np.int32),
        "TransactionAmt": amounts,
        "log_amount": np.log1p(amounts),
        "cust_z_amount": (amounts - cust_mean_amt[cust_ids]) / np.maximum(cust_std_amt[cust_ids], 1.0),
        "hour_of_day": hour_of_day,
        "is_night": is_night,
        "ProductCD": product_cd,
        "dist_score": dist_score,
        "ip_risk_score": ip_risk_tabular,
        "auth_match_flag": auth_match_flag,
        "has_identity": has_identity.astype(np.int8),
        "card1": card1.astype(np.int32),
        "card_comp": card_comp.astype(np.int64),
        "terminal_id": term_ids.astype(np.int32),
        "addr1": addr1.astype(np.int32),
        "P_emaildomain": email_dom.astype(np.int32),
        "DeviceInfo": device_id.astype(np.int32),
        "fraud_scenario": fraud_scenario,
        "isFraud": is_fraud,
    })

    meta = {
        "generator_version": SIMULATOR_GENERATOR_VERSION,
        "seed": int(seed),
        "n_transactions": int(n_transactions),
        "n_days": int(n_days),
        "n_customers": int(n_customers),
        "n_terminals": int(n_terminals),
        "n_addresses": int(n_addresses),
        "n_email_domains": int(n_email_domains),
        "n_devices": int(n_devices),
        "fraud_count": int(np.sum(is_fraud)),
        "fraud_prevalence": float(np.mean(is_fraud)),
        "identity_coverage": float(np.mean(has_identity)),
        "unique_card1": int(df["card1"].nunique()),
        "unique_card_comp": int(df["card_comp"].nunique()),
        "unique_addr1": int(df["addr1"].nunique()),
        "unique_email": int(df["P_emaildomain"].nunique()),
        "unique_device": int(df.loc[df["DeviceInfo"] >= 0, "DeviceInfo"].nunique()),
        "scenario_counts": {
            "legitimate": int(np.sum(fraud_scenario == 0)),
            "S1_amount_anomaly": int(np.sum(fraud_scenario == 1)),
            "S2_compromised_terminal": int(np.sum(fraud_scenario == 2)),
            "S3_coordinated_ring_card": int(np.sum(fraud_scenario == 3)),
        }
    }
    return df, meta
