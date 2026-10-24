import os, json, csv
import numpy as np

with open("/tmp/project/results/manifests/frozen_results_bundle.json") as f:
    b = json.load(f)

tables_dir = "/tmp/project/results/tables"
os.makedirs(tables_dir, exist_ok=True)
ws_tables_dir = "/working_dir/c_86a5839ba5fb98b3/project/results/tables"
os.makedirs(ws_tables_dir, exist_ok=True)

print("=== 1. STEP 1 DATA AUDIT ===")
a = b["step1_audit"]
print(f"Rows: {a['n_rows']}, Frauds: {a['n_frauds']} ({a['fraud_prevalence']*100:.2f}%), Identity Cov: {a['identity_coverage_overall']*100:.2f}%")
for k, v in a["entity_degree_audit"].items():
    print(f"  Entity {k:15s}: unique={v['unique_entities']:5d}, miss={v['missing_or_unlinked_rate']*100:5.1f}%, p50={v['degree_p50']:4.1f}, p90={v['degree_p90']:5.1f}, p99={v['degree_p99']:6.1f}, max={v['max_degree']:5d}, top1%_share={v['top_1pct_entity_tx_share']*100:5.1f}%")

print("\n=== 2. DELAY SCENARIOS & Q(A) CALIBRATION ===")
for k, v in b["delay_scenarios"].items():
    print(f"  {k}: M={v['maturity_horizon_M_days']}d, mean={v['realized_mean_delay_days']:.2f}d, median={v['realized_median_p50_days']:.2f}d, p95={v['realized_p95_days']:.2f}d, max={v['realized_max_days']:.2f}d")
for qc in b["q_calibration"]:
    print(f"  q_cal {qc['scenario']} {qc['age_bin']:10s}: n={qc['n_unreported']:5d}, pred_q={qc['mean_predicted_q']:.4f}, emp_rate={qc['empirical_hidden_fraud_rate']:.4f}, gap={qc['abs_calibration_gap']:.4f}")

print("\n=== 3. PRIMARY COMPARISON (TABLE 3: P3 + D2) ===")
t3_rows = [["Run_Key", "Model_Tier", "Label_Rule", "AP_Mean", "AP_Std", "PR_AUC_Trap", "ROC_AUC", "Prec@0.5%", "Prec@1%", "Recall@1%", "Prec@5%", "Recall@1%FPR", "CardProxy_P@25", "Brier", "Runtime_s"]]
for k, r in b["primary_runs_table3"].items():
    ms = r["macro_summary"]
    row = [
        k, r["model_tier"], r["label_rule"],
        ms["AP"]["mean"], ms["AP"]["std_across_blocks"],
        ms["PR_AUC_trapezoidal"]["mean"], ms["ROC_AUC"]["mean"],
        ms["Precision@0.5%"]["mean"], ms["Precision@1%"]["mean"], ms["Recall@1%"]["mean"],
        ms["Precision@5%"]["mean"], ms["Recall@1%FPR"]["mean"],
        ms["CardProxy_Precision@25"]["mean"], ms["Brier"]["mean"], r["runtime_sec"]
    ]
    t3_rows.append(row)
    print(f"  {k:45s}: AP={ms['AP']['mean']:.4f}±{ms['AP']['std_across_blocks']:.4f} | P@0.5%={ms['Precision@0.5%']['mean']:.4f} | P@1%={ms['Precision@1%']['mean']:.4f} | R@1%={ms['Recall@1%']['mean']:.4f} | P@5%={ms['Precision@5%']['mean']:.4f} | Brier={ms['Brier']['mean']:.4f} | t={r['runtime_sec']:.2f}s")

with open(os.path.join(tables_dir, "table3_primary_comparison.csv"), "w", newline="") as f:
    csv.writer(f).writerows(t3_rows)
with open(os.path.join(ws_tables_dir, "table3_primary_comparison.csv"), "w", newline="") as f:
    csv.writer(f).writerows(t3_rows)

print("\n=== 4. LEAKAGE MATRIX (P0, P1, P2, P3) ===")
t_leak_rows = [["Run_Key", "Model_Tier", "Protocol_Cell", "Label_Rule", "AP_Mean", "AP_Std", "Prec@1%", "Recall@1%", "ROC_AUC"]]
for k, r in b["leakage_matrix_runs"].items():
    ms = r["macro_summary"]
    t_leak_rows.append([k, r["model_tier"], r["protocol_cell"], r["label_rule"], ms["AP"]["mean"], ms["AP"]["std_across_blocks"], ms["Precision@1%"]["mean"], ms["Recall@1%"]["mean"], ms["ROC_AUC"]["mean"]])
    print(f"  {k:48s}: AP={ms['AP']['mean']:.4f}±{ms['AP']['std_across_blocks']:.4f} | P@1%={ms['Precision@1%']['mean']:.4f} | R@1%={ms['Recall@1%']['mean']:.4f}")

with open(os.path.join(tables_dir, "table_leakage_matrix.csv"), "w", newline="") as f:
    csv.writer(f).writerows(t_leak_rows)
with open(os.path.join(ws_tables_dir, "table_leakage_matrix.csv"), "w", newline="") as f:
    csv.writer(f).writerows(t_leak_rows)

print("\n=== 5. MATURITY RULES V0/V1/V2/V3 ACROSS D0-D3 (TABLE 4) ===")
t4_rows = [["Run_Key", "Delay_Scenario", "Label_Rule", "AP_Mean", "AP_Std", "Prec@1%", "Recall@1%", "Brier", "LogLoss"]]
for k, r in b["maturity_grid_table4"].items():
    ms = r["macro_summary"]
    t4_rows.append([k, r["delay_scenario"], r["label_rule"], ms["AP"]["mean"], ms["AP"]["std_across_blocks"], ms["Precision@1%"]["mean"], ms["Recall@1%"]["mean"], ms["Brier"]["mean"], ms["LogLoss"]["mean"]])
    print(f"  {k:22s}: AP={ms['AP']['mean']:.4f}±{ms['AP']['std_across_blocks']:.4f} | P@1%={ms['Precision@1%']['mean']:.4f} | R@1%={ms['Recall@1%']['mean']:.4f} | Brier={ms['Brier']['mean']:.4f}")

with open(os.path.join(tables_dir, "table4_maturity_rules.csv"), "w", newline="") as f:
    csv.writer(f).writerows(t4_rows)
with open(os.path.join(ws_tables_dir, "table4_maturity_rules.csv"), "w", newline="") as f:
    csv.writer(f).writerows(t4_rows)

print("\n=== 6. GRAPH & HUB ABLATIONS + H3 STRATUM (TABLE 5) ===")
t5_rows = [["Ablation_Name", "AP_Mean", "AP_Std", "Prec@1%", "Recall@1%", "WithVisFraudNbr_AP", "Isolated_AP", "WithVisFraudNbr_ShareOfFrauds"]]
for k, r in b["ablation_runs_table5"].items():
    ms = r["macro_summary"]
    st = r["stratum_h3"]
    w_ap = st["with_visible_prior_fraud_neighbor_2hop"]["AP"]
    i_ap = st["isolated_no_visible_fraud_neighbor_2hop"]["AP"]
    f_sh = st["with_visible_prior_fraud_neighbor_2hop"]["share_of_all_test_frauds"]
    t5_rows.append([k, ms["AP"]["mean"], ms["AP"]["std_across_blocks"], ms["Precision@1%"]["mean"], ms["Recall@1%"]["mean"], w_ap, i_ap, f_sh])
    print(f"  {k:34s}: AP={ms['AP']['mean']:.4f}±{ms['AP']['std_across_blocks']:.4f} | P@1%={ms['Precision@1%']['mean']:.4f} | VisNbr_AP={w_ap:.4f} (fraud_share={f_sh*100:.1f}%) | Iso_AP={i_ap:.4f}")

with open(os.path.join(tables_dir, "table5_graph_ablations.csv"), "w", newline="") as f:
    csv.writer(f).writerows(t5_rows)
with open(os.path.join(ws_tables_dir, "table5_graph_ablations.csv"), "w", newline="") as f:
    csv.writer(f).writerows(t5_rows)

print("\n=== 7. ROBUSTNESS & SENSITIVITY (TABLE 6) ===")
t6_rows = [["Robustness_Scenario", "AP_Mean", "AP_Std", "Prec@1%", "Recall@1%", "Brier"]]
for k, r in b["robustness_runs_table6"].items():
    ms = r["macro_summary"]
    t6_rows.append([k, ms["AP"]["mean"], ms["AP"]["std_across_blocks"], ms["Precision@1%"]["mean"], ms["Recall@1%"]["mean"], ms["Brier"]["mean"]])
    print(f"  {k:40s}: AP={ms['AP']['mean']:.4f}±{ms['AP']['std_across_blocks']:.4f} | P@1%={ms['Precision@1%']['mean']:.4f} | Brier={ms['Brier']['mean']:.4f}")

with open(os.path.join(tables_dir, "table6_robustness.csv"), "w", newline="") as f:
    csv.writer(f).writerows(t6_rows)
with open(os.path.join(ws_tables_dir, "table6_robustness.csv"), "w", newline="") as f:
    csv.writer(f).writerows(t6_rows)

print("\n=== 8. STATISTICAL COMPARISONS (PAIRED BLOCKS & HOLM-BONFERRONI) ===")
for sc in b["statistical_comparisons"]:
    print(f"  {sc['comparison_id']:42s}: A={sc['mean_A']:.4f}, B={sc['mean_B']:.4f}, diff={sc['mean_paired_diff']:+.4f} [{sc['ci95_low']:+.4f}, {sc['ci95_high']:+.4f}], signs=+{sc['positive_diff_blocks']}/-{sc['negative_diff_blocks']}, p_raw={sc['wilcoxon_p_unadjusted']:.4f}, p_holm={sc['wilcoxon_p_holm_adjusted']:.4f}")
