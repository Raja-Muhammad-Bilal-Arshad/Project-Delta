# Reproducibility Package: Time-Respecting Evaluation and Label-Maturity-Aware Learning for Graph-Based Credit Card Fraud Detection

## 1. Overview and Repository Structure
This repository implements the complete software, automated temporal/label leakage invariance test suite, multi-entity transaction simulator, IEEE-CIS data audit pipeline, and rolling experimental evaluation framework specified in the 20-step *Research Execution Handbook*.

```
project/
  README.md
  run_full_experiments.py            # Master end-to-end experiment runner
  generate_figures.py                # Reproducible plotting script from frozen JSON
  protocol/
    protocol_v1.md                   # Gate 0 frozen protocol specification
    deviations_log.md                # Documented environment constraints & pre-test decisions
    data_dictionary.md               # Dataset provenance & IEEE-CIS feature-availability ledger
    literature_search_log.csv        # Systematic search queries & screening log
    literature_evidence_matrix.csv   # 15-study verified literature evidence matrix
    claim_evidence_ledger.csv        # Claim-to-source verification ledger
  configs/
    splits/rolling_v1.json           # Frozen 8-block rolling temporal split manifest
  src/
    data_audit/
      simulator_generator.py         # Extended Handbook multi-entity transaction generator
      ieee_cis_audit.py              # IEEE-CIS join verification & Step 1 audit pipeline
    delay_simulation/
      delay_models.py                # Bounded D0/D1/D2/D3 delay CDFs, survival S(a), posterior q(a)
    temporal_graph/
      causal_graph.py                # Causal event-history graph builder (P0, P1, P2, P3)
    models/
      baselines_and_gnns.py          # SoftTargetHistGBDT (B0-B2) & CausalGraphNeuralNetwork (B3-B5)
    evaluation/
      metrics_and_stats.py           # Exact step-wise AP, Precision@budget, block bootstrap, Wilcoxon
  tests/
    test_no_future_edges.py          # Future-edge, future-feature, & batch permutation invariance
    test_label_visibility.py         # Label state reconstruction & target isolation checks
    test_split_integrity.py          # Rolling window disjointness & hand-worked AP tests
    test_reproducible_delay_draws.py # Truncated exponential median, knots, & q(a) calibration
  results/
    manifests/
      step1_data_audit.json
      delay_and_q_calibration.json
      frozen_results_bundle.json
    tables/
      table3_primary_comparison.csv
      table_leakage_matrix.csv
      table4_maturity_rules.csv
      table5_graph_ablations.csv
      table6_robustness.csv
    figures/
      fig1_protocol_timeline.png .. fig7_efficiency_and_robustness.png
  manuscript/
    references.bib
```

## 2. Environment and Deterministic Execution Instructions
* **Verified Python Environment:** Python 3.11.2, `numpy 2.1.3`, `scipy 1.15.2`, `pandas 2.2.3`, `matplotlib 3.10.1`, `seaborn 0.13.2`.
* **Run Automated Leakage & Invariance Unit Tests (12/12 tests):**
  ```bash
  for f in tests/test_*.py; do PYTHONPATH=. python3 "$f" -v; done
  ```
* **Run Full Experimental Suite (Steps 1–15):**
  ```bash
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. python3 run_full_experiments.py
  ```
* **Regenerate All Manuscript Figures from Saved Predictions/Manifests:**
  ```bash
  PYTHONPATH=. python3 generate_figures.py
  ```
* **Run Official IEEE-CIS Benchmark Evaluation (when Kaggle CSVs are mounted):**
  Download `train_transaction.csv` and `train_identity.csv` from the official [Kaggle IEEE-CIS Fraud Detection competition](https://www.kaggle.com/c/ieee-fraud-detection/data) in compliance with competition rules, and invoke `load_and_join_ieee_cis(train_tx_path, train_id_path)` in `src/data_audit/ieee_cis_audit.py`.
