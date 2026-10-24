import os, json, zipfile

out_dir = "/working_dir/c_86a5839ba5fb98b3/artifacts/file_generation/ttl=63d/output"
proj_tmp = "/tmp/project"
proj_ws = "/working_dir/c_86a5839ba5fb98b3/project"

# 1. Write project/README.md
readme_text = """# Reproducibility Package: Time-Respecting Evaluation and Label-Maturity-Aware Learning for Graph-Based Credit Card Fraud Detection

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
"""

with open(os.path.join(proj_tmp, "README.md"), "w") as f:
    f.write(readme_text)
with open(os.path.join(proj_ws, "README.md"), "w") as f:
    f.write(readme_text)
with open(os.path.join(proj_ws, "generate_figures.py"), "w") as f:
    with open(os.path.join(proj_tmp, "generate_figures.py")) as fr:
        f.write(fr.read())

# 2. Deliverable 4: Research Implementation & Test Audit Report
deliv4_text = """# Research Implementation Architecture, Codebase Documentation, and Automated Leakage Test Report

**Deliverable ID:** Deliverable 4 — Research Implementation & Software Verification  
**Associated Execution Plan Steps:** Steps 3–6, Step 15 (Implementation Architecture, CI Checks, and Unit Tests)  

## 1. Modular Software Architecture
In accordance with Section 6.2 and Section 15 of the *Research Execution Handbook*, the codebase is organized into five strictly decoupled modules under `project/src/` and verified by four automated unit-test suites under `project/tests/`:

| Module Path | Core Classes & Functions | Methodological Responsibility |
| :--- | :--- | :--- |
| `src/delay_simulation/delay_models.py` | `solve_truncated_exponential_lambda`, `delay_cdf`, `delay_survival`, `sample_fraud_delays`, `generate_event_log_delays`, `compute_posterior_q`, `estimate_time_safe_prior`, `reconstruct_label_state_at_cutoff` | Implements exact bounded/truncated delay laws ($D_0, D_1, D_2, D_3$), solves $\lambda = 0.693122\text{ day}^{-1}$ via Brent's method so $F_T(1\text{ d}) = 0.50$ on $[0, 14]\text{ d}$, enforces event-log label masking at cutoff $\tau$, and computes the Bayesian posterior $q(a_i) = \frac{\pi_\tau S(a_i)}{\pi_\tau S(a_i) + (1-\pi_\tau)}$. |
| `src/temporal_graph/causal_graph.py` | `extract_relational_features_and_graph_state` | Constructs multi-relational transaction graphs across 5 entity relations (`card1`/`card_comp`, `terminal_id`, `addr1`, `P_emaildomain`, `DeviceInfo`). Supports all four visibility cells ($P_0, P_1, P_2, P_3$), enforces causal prefix ordering ($t_j < t_i$), degree capping $K \in \{10, 50, 200\}$, recency window $W$, inverse-log hub weighting $1/\ln(2+\text{deg})$, and computes 4 distinct maturity-aware neighbor signals plus 1-hop and similarity-filtered structural messages. |
| `src/models/baselines_and_gnns.py` | `SoftTargetHistGBDT`, `CausalGraphNeuralNetwork` | Implements second-order Newton histogram gradient boosted decision trees (`B0`, `B1`, `B2`) and 2-layer causal GraphSAGE / gated similarity-filtered CARE-GNN architectures (`B3`, `B4`, `B5`) trained with Adam under expected cross-entropy loss supporting continuous soft targets $y \in [0, 1]$. All feature normalizers and quantile bin edges are fit **strictly on the eligible training partition**. |
| `src/evaluation/metrics_and_stats.py` | `compute_average_precision`, `compute_trapezoidal_prauc`, `compute_alert_budget_metrics`, `compute_card_proxy_precision_at_k`, `paired_temporal_block_analysis`, `holm_bonferroni_correction` | Implements exact step-wise Average Precision ($\text{AP} = \sum (R_n - R_{n-1})P_n$), trapezoidal PR-AUC, deterministic top-$B\%$ alert budget precision/recall, card-proxy precision@$k$, moving-block bootstrap $95\%$ confidence intervals, and paired Wilcoxon signed-rank tests with Holm-Bonferroni adjustment. |
| `src/data_audit/ieee_cis_audit.py` & `simulator_generator.py` | `load_and_join_ieee_cis`, `run_step1_data_audit`, `generate_handbook_multi_entity_dataset` | Implements the official IEEE-CIS two-table join and integrity audit pipeline alongside the reproducible multi-entity extension of the Fraud Detection Handbook simulator ([Le Borgne et al., 2022](https://fraud-detection-handbook.github.io/fraud-detection-handbook/)). |

## 2. Continuous Integration & Leakage Invariance Unit Test Audit (12 / 12 Passed)
Every unit test required by Section 8.4 and Section 15.3 was executed and verified prior to freezing experimental results:

| Test Suite File | Test Case Name | Invariant Verified | Status |
| :--- | :--- | :--- | :--- |
| `tests/test_no_future_edges.py` | `test_future_edge_insertion_invariance_p3` | Appending future transactions at $t > t_i$ leaves all earlier relational features and graph embeddings bit-for-bit identical ($\text{atol} = 10^{-12}$). | **PASS** |
| `tests/test_no_future_edges.py` | `test_future_feature_mutation_invariance_p3` | Mutating transaction amounts or risk features of future nodes ($t > t_i$) cannot alter any earlier node's message-passing state or score. *(Caught and fixed a prefix-normalization bug during Step 4).* | **PASS** |
| `tests/test_no_future_edges.py` | `test_simultaneous_timestamp_permutation_invariance` | Permuting simultaneous transactions ($t_j = t_i$) produces identical features without intra-batch leakage. | **PASS** |
| `tests/test_label_visibility.py` | `test_posterior_edge_cases_and_monotonicity` | Verifies $q(0) = \pi_\tau$, $q(a \ge M) = 0$, exact analytical match at $a = 7\text{ d}$ under $D_2$, and strict monotonic decay on $[0, M]$. | **PASS** |
| `tests/test_label_visibility.py` | `test_hand_worked_label_states_and_future_label_invariance` | Confirms exact classification into Visible Positive ($1$), Mature Negative ($0$), Immature ($2$), and Future ($-1$), and proves flipping hidden final labels $Y_i$ cannot change $V_0, V_1, V_2$ targets. | **PASS** |
| `tests/test_label_visibility.py` | `test_target_isolation_in_neighbor_features` | Proves that transaction $i$'s own final fraud label $Y_i$ never enters its own neighbor aggregate features. | **PASS** |
| `tests/test_split_integrity.py` | `test_hand_worked_average_precision_and_alert_budget` | Verifies step-wise $\text{AP} = 5/6 = 0.833333333$ and $\text{Precision}@40\% = 0.50$ on a 5-transaction hand-worked benchmark. | **PASS** |
| `tests/test_split_integrity.py` | `test_chronological_rolling_window_disjointness` | Verifies zero index overlap and strict chronological ordering ($\max t_{\text{train}} < \min t_{\text{val}} < \max t_{\text{val}} < \min t_{\text{test}}$) across all rolling blocks. | **PASS** |
| `tests/test_reproducible_delay_draws.py` | `test_d1_truncated_exponential_bounds_and_median` | Verifies $F_T(1\text{ d}) = 0.50$, $S_{D1}(14\text{ d}) = 0.0$, and Monte Carlo sample median within $\pm 0.025\text{ d}$ of $1.0\text{ d}$. | **PASS** |
| `tests/test_reproducible_delay_draws.py` | `test_d2_piecewise_linear_knots` | Verifies exact piecewise-linear CDF values at $(1\text{ h}, 1\%)$, $(1\text{ d}, 30\%)$, $(7\text{ d}, 60\%)$, $(10\text{ d}, 80\%)$, and $(30\text{ d}, 100\%)$. | **PASS** |
| `tests/test_reproducible_delay_draws.py` | `test_d3_long_tail_survival` | Verifies $F_{D3}(30\text{ d}) = 0.80$, $S_{D3}(30\text{ d}) = 0.20 > 0$, and $S_{D3}(90\text{ d}) = 0.0$. | **PASS** |
| `tests/test_reproducible_delay_draws.py` | `test_posterior_q_empirical_calibration` | Verifies that mean predicted posterior $\mathbb{E}[q(a)]$ over $200,000$ synthetic transactions matches the empirical unreported fraud rate within $< 0.002$. | **PASS** |
"""

with open(os.path.join(out_dir, "Deliverable_4_Research_Implementation_and_Tests.md"), "w") as f:
    f.write(deliv4_text)

# 3. Zip the entire /tmp/project into Deliverable_7_Reproducibility_Package.zip
zip_path = os.path.join(out_dir, "Deliverable_7_Reproducibility_Package.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(proj_tmp):
        for file in files:
            if file.endswith(".pyc") or "__pycache__" in root:
                continue
            full_p = os.path.join(root, file)
            rel_p = os.path.relpath(full_p, os.path.dirname(proj_tmp))
            zf.write(full_p, rel_p)

print("Saved Deliverable 4 and Deliverable 7 ZIP archive:", os.path.getsize(zip_path), "bytes")
