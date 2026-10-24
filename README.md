# Reproducibility Package: Time-Respecting Evaluation and Label-Maturity-Aware Learning for Graph-Based Credit Card Fraud Detection (v2.0 Corrected & Validated)

## 1. Overview and Repository Structure
This repository implements the corrected, validated software, 10-category temporal/causal/maturity invariance test suite, reproducible multi-entity transaction simulator (`N = 12,000` transactions, `208` frauds, `1.7333%` prevalence over `180` days), fail-closed IEEE-CIS data audit pipeline, and 8-block rolling temporal evaluation framework.

```
project_delta_v2/
  README.md
  run_full_experiments.py            # Portable master experiment runner (CLI configurable)
  export_tables_and_summary.py       # Deterministic CSV table generator & bundle verifier
  generate_figures.py                # Reproducible plotting script from frozen JSON
  protocol/
    protocol_v1.md                   # Pre-specified frozen protocol specification
    deviations_log.md                # Documented environment constraints & pre-test decisions
    data_dictionary.md               # Dataset provenance & IEEE-CIS feature-availability ledger
    literature_search_log.csv        # Search queries, overlap counts, & net unique inclusions
    literature_deduplication_ledger.csv # 26-hit to 15-unique-study deduplication ledger
    literature_evidence_matrix.csv   # 15-study verified literature evidence matrix
    claim_evidence_ledger.csv        # Claim-to-source verification ledger
  configs/
    splits/rolling_v1.json           # Authoritative 8-block rolling temporal split manifest
  src/
    data_audit/
      simulator_generator.py         # Extended Handbook multi-entity transaction generator (N=12,000)
      ieee_cis_audit.py              # Fail-closed IEEE-CIS join verification & Step 1 audit pipeline
    delay_simulation/
      delay_models.py                # Bounded D0/D1/D2/D3 delay CDFs, survival S(a), time-safe prior, posterior q(a)
    temporal_graph/
      causal_graph.py                # Two-pass causal event-history graph builder (P0, P1, P2, P3)
    models/
      baselines_and_gnns.py          # Custom SoftTargetHistGBDT (B0-B2) & CausalGraphNeuralNetwork (B3-B5)
      external_framework_adapters.py # Optional LightGBM / XGBoost / PyTorch wrappers for external environments
    evaluation/
      metrics_and_stats.py           # Step-wise AP, trap PR-AUC, mature-negative FPR threshold, bootstrap, Wilcoxon
  tests/
    test_no_future_edges.py          # Future-edge, future-feature, & batch permutation invariance
    test_label_visibility.py         # Label state reconstruction & target isolation checks
    test_split_integrity.py          # Rolling window disjointness & hand-worked AP tests
    test_reproducible_delay_draws.py # Truncated exponential median, knots, & q(a) calibration
    test_mandatory_validation_suite.py # Mandatory validation categories A through J (Audit Amendment Section 4)
  results/
    manifests/
      step1_data_audit.json
      delay_and_q_calibration.json
      frozen_results_bundle_v2.json
      frozen_results_bundle.json
    tables/
      table1_entity_degree_audit.csv
      table2_split_manifest.csv
      table3_primary_comparison.csv
      table_leakage_matrix.csv
      table4_maturity_rules.csv
      table5_graph_ablations.csv
      table6_robustness.csv
      table7_statistical_comparisons.csv
    figures/
      fig1_protocol_timeline.png .. fig7_efficiency_and_robustness.png
  manuscript/
    references.bib
```

## 2. Environment and Deterministic Execution Instructions
* **Verified Python Environment:** Python 3.11.2, `numpy 2.1.3`, `scipy 1.15.2`, `pandas 2.2.3`, `matplotlib 3.10.1`. (External C++/CUDA packages `lightgbm`, `xgboost`, `scikit-learn`, and `torch` are not pre-installed in the air-gapped VM; models B0–B5 use the documented custom NumPy estimators in `src/models/baselines_and_gnns.py`.)
* **Run Complete Automated Unit & Audit Validation Test Suite (5 test files, 24 tests across Categories A–J):**
  ```bash
  for f in tests/test_*.py; do PYTHONPATH=. python3 "$f" -v; done
  ```
* **Run Full Experimental Suite (portable from any checkout path and custom output directory):**
  ```bash
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 run_full_experiments.py \
    --output-dir ./results \
    --split-manifest ./configs/splits/rolling_v1.json \
    --seeds 101,202,303,404,505 \
    --run-id project_delta_v2_reconciled_20261024
  ```
* **Export & Verify All CSV Tables from Frozen Results Bundle:**
  ```bash
  python3 export_tables_and_summary.py --results-dir ./results
  ```
* **Regenerate All Manuscript Figures from Frozen Results Bundle:**
  ```bash
  python3 generate_figures.py --results-dir ./results
  ```
* **Run Official IEEE-CIS Benchmark Evaluation (when Kaggle CSVs are mounted):**
  Both `train_transaction.csv` and `train_identity.csv` are strictly required by default (`load_and_join_ieee_cis(train_tx_path, train_id_path, allow_transaction_only=False)`). Missing files or duplicate `TransactionID` keys raise explicit errors.
