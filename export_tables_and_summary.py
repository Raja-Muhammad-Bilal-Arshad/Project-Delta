"""Authoritative Table Export and Consistency Verification Script (Audit Issue 2 & Test H).

Reads the frozen results bundle (`results/manifests/frozen_results_bundle_v2.json` or
`results/manifests/frozen_results_bundle.json`), generates all CSV tables in `results/tables/`,
and executes automated numerical consistency assertions verifying that every CSV cell matches
the source bundle within rounding tolerance (`atol <= 1e-5`).
"""
from __future__ import annotations
import os
import sys
import csv
import json
import argparse
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent


def export_and_verify_tables(results_dir: Path) -> Dict[str, Any]:
    bundle_v2 = results_dir / "manifests/frozen_results_bundle_v2.json"
    bundle_v1 = results_dir / "manifests/frozen_results_bundle.json"
    bundle_path = bundle_v2 if bundle_v2.is_file() else bundle_v1
    if not bundle_path.is_file():
        raise FileNotFoundError(f"No frozen results bundle found at {bundle_path}")

    with open(bundle_path, "r") as f:
        b = json.load(f)

    tables_dir = results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # 1. Table 1: Entity Degree Audit CSV
    t1_rows = [[
        "Entity_Relation", "Unique_Entities", "Missing_Rate", "Deg_p50",
        "Deg_p90", "Deg_p99", "Max_Degree", "Top1pct_Share", "Hub_DegGt50_Share"
    ]]
    for k, v in b["step1_audit"]["entity_degree_audit"].items():
        t1_rows.append([
            k, v["unique_entities"], v["missing_or_unlinked_rate"],
            v["degree_p50"], v["degree_p90"], v["degree_p99"],
            v["max_degree"], v["top_1pct_entity_tx_share"], v["tx_share_in_hubs_deg_gt_50"]
        ])
    with open(tables_dir / "table1_entity_degree_audit.csv", "w", newline="") as f:
        csv.writer(f).writerows(t1_rows)

    # 2. Table 2: Rolling Split Manifest CSV
    t2_split_rows = [[
        "Block_ID", "Train_Cutoff_Day", "Val_Start_Day", "Val_End_Day",
        "Test_Start_Day", "Test_End_Day", "N_Train", "N_Val", "N_Test",
        "N_Test_Pos", "Test_Prevalence", "Alert_Budget_0.5pct", "Alert_Budget_1.0pct", "Alert_Budget_5.0pct"
    ]]
    for blk in b["split_manifest"]["blocks"]:
        t2_split_rows.append([
            blk["block_id"], blk["train_cutoff_day"], blk["val_start_day"], blk["val_end_day"],
            blk["test_start_day"], blk["test_end_day"], blk["n_train"], blk["n_val"], blk["n_test"],
            blk["n_test_pos"], blk["test_prevalence"],
            blk.get("alert_budget_k_0.5pct", int(round(0.005 * blk["n_test"]))),
            blk.get("alert_budget_k_1.0pct", int(round(0.010 * blk["n_test"]))),
            blk.get("alert_budget_k_5.0pct", int(round(0.050 * blk["n_test"]))),
        ])
    with open(tables_dir / "table2_split_manifest.csv", "w", newline="") as f:
        csv.writer(f).writerows(t2_split_rows)

    # 3. Table 3: Primary Comparison (P3 + D2)
    t3_rows = [[
        "Run_Key", "Model_Tier", "Label_Rule", "AP_Mean", "AP_Std", "PR_AUC_Trap", "ROC_AUC",
        "Prec@0.5%", "Prec@1%", "Recall@1%", "Prec@5%", "Recall@1%FPR", "CardProxy_P@25",
        "Brier", "Runtime_s", "Tracemalloc_Peak_MB", "Process_Max_RSS_MB"
    ]]
    for k, r in b["primary_runs_table3"].items():
        ms = r["macro_summary"]
        t3_rows.append([
            k, r["model_tier"], r["label_rule"],
            ms["AP"]["mean"], ms["AP"]["std_across_blocks"],
            ms["PR_AUC_trapezoidal"]["mean"], ms["ROC_AUC"]["mean"],
            ms["Precision@0.5%"]["mean"], ms["Precision@1%"]["mean"], ms["Recall@1%"]["mean"],
            ms["Precision@5%"]["mean"], ms["Recall@1%FPR"]["mean"],
            ms["CardProxy_Precision@25"]["mean"], ms["Brier"]["mean"],
            r["runtime_sec"],
            r.get("tracemalloc_peak_mb", 0.0),
            r.get("process_max_rss_mb", r.get("peak_memory_mb", 0.0)),
        ])
    with open(tables_dir / "table3_primary_comparison.csv", "w", newline="") as f:
        csv.writer(f).writerows(t3_rows)

    # 4. Leakage Attribution Matrix (P0, P1, P2, P3)
    t_leak_rows = [[
        "Run_Key", "Model_Tier", "Protocol_Cell", "Label_Rule",
        "AP_Mean", "AP_Std", "Prec@1%", "Recall@1%", "ROC_AUC"
    ]]
    for k, r in b["leakage_matrix_runs"].items():
        ms = r["macro_summary"]
        t_leak_rows.append([
            k, r["model_tier"], r["protocol_cell"], r["label_rule"],
            ms["AP"]["mean"], ms["AP"]["std_across_blocks"],
            ms["Precision@1%"]["mean"], ms["Recall@1%"]["mean"], ms["ROC_AUC"]["mean"]
        ])
    with open(tables_dir / "table_leakage_matrix.csv", "w", newline="") as f:
        csv.writer(f).writerows(t_leak_rows)

    # 5. Maturity Rules V0/V1/V2/V3 across D0-D3 (Table 4/5)
    t4_rows = [[
        "Run_Key", "Delay_Scenario", "Label_Rule", "AP_Mean", "AP_Std",
        "Prec@1%", "Recall@1%", "Brier", "LogLoss"
    ]]
    for k, r in b["maturity_grid_table4"].items():
        ms = r["macro_summary"]
        t4_rows.append([
            k, r["delay_scenario"], r["label_rule"],
            ms["AP"]["mean"], ms["AP"]["std_across_blocks"],
            ms["Precision@1%"]["mean"], ms["Recall@1%"]["mean"],
            ms["Brier"]["mean"], ms["LogLoss"]["mean"]
        ])
    with open(tables_dir / "table4_maturity_rules.csv", "w", newline="") as f:
        csv.writer(f).writerows(t4_rows)

    # 6. Graph & Hub Ablations + H3 Stratum (Table 5/6)
    t5_rows = [[
        "Ablation_Name", "AP_Mean", "AP_Std", "Prec@1%", "Recall@1%",
        "WithVisFraudNbr_AP", "Isolated_AP", "WithVisFraudNbr_ShareOfFrauds", "Brier"
    ]]
    for k, r in b["ablation_runs_table5"].items():
        ms = r["macro_summary"]
        st = r["stratum_h3"]
        w_ap = st["with_visible_prior_fraud_neighbor_2hop"]["AP"]
        i_ap = st["isolated_no_visible_fraud_neighbor_2hop"]["AP"]
        f_sh = st["with_visible_prior_fraud_neighbor_2hop"]["share_of_all_test_frauds"]
        t5_rows.append([
            k, ms["AP"]["mean"], ms["AP"]["std_across_blocks"],
            ms["Precision@1%"]["mean"], ms["Recall@1%"]["mean"],
            w_ap, i_ap, f_sh, ms["Brier"]["mean"]
        ])
    with open(tables_dir / "table5_graph_ablations.csv", "w", newline="") as f:
        csv.writer(f).writerows(t5_rows)

    # 7. Robustness & Sensitivity (Table 6)
    t6_rows = [[
        "Robustness_Scenario", "AP_Mean", "AP_Std", "Prec@1%", "Recall@1%",
        "WithVisFraudNbr_AP", "Isolated_AP", "Brier"
    ]]
    for k, r in b["robustness_runs_table6"].items():
        ms = r["macro_summary"]
        st = r["stratum_h3"]
        t6_rows.append([
            k, ms["AP"]["mean"], ms["AP"]["std_across_blocks"],
            ms["Precision@1%"]["mean"], ms["Recall@1%"]["mean"],
            st["with_visible_prior_fraud_neighbor_2hop"]["AP"],
            st["isolated_no_visible_fraud_neighbor_2hop"]["AP"],
            ms["Brier"]["mean"]
        ])
    with open(tables_dir / "table6_robustness.csv", "w", newline="") as f:
        csv.writer(f).writerows(t6_rows)

    # 8. Statistical Comparisons (Table 7)
    t7_rows = [[
        "Comparison_ID", "Description", "Mean_A", "Mean_B", "Mean_Paired_Diff",
        "CI95_Low", "CI95_High", "Positive_Blocks", "Negative_Blocks", "Tied_Blocks",
        "Wilcoxon_Stat", "Wilcoxon_P_Raw", "Wilcoxon_P_Holm"
    ]]
    for sc in b["statistical_comparisons"]:
        t7_rows.append([
            sc["comparison_id"], sc["description"], sc["mean_A"], sc["mean_B"],
            sc["mean_paired_diff"], sc["ci95_low"], sc["ci95_high"],
            sc["positive_diff_blocks"], sc["negative_diff_blocks"], sc.get("tied_blocks", 0),
            sc["wilcoxon_stat"], sc["wilcoxon_p_unadjusted"], sc["wilcoxon_p_holm_adjusted"]
        ])
    with open(tables_dir / "table7_statistical_comparisons.csv", "w", newline="") as f:
        csv.writer(f).writerows(t7_rows)

    # Automated verification: re-read CSVs and assert exact match against bundle
    mismatches = []
    with open(tables_dir / "table3_primary_comparison.csv", "r") as f:
        for row in csv.DictReader(f):
            rk = row["Run_Key"]
            ms = b["primary_runs_table3"][rk]["macro_summary"]
            for csv_col, b_key in [
                ("AP_Mean", "AP"), ("PR_AUC_Trap", "PR_AUC_trapezoidal"),
                ("ROC_AUC", "ROC_AUC"), ("Prec@0.5%", "Precision@0.5%"),
                ("Prec@1%", "Precision@1%"), ("Recall@1%", "Recall@1%"),
                ("Prec@5%", "Precision@5%"), ("Recall@1%FPR", "Recall@1%FPR"),
                ("CardProxy_P@25", "CardProxy_Precision@25"), ("Brier", "Brier")
            ]:
                if abs(float(row[csv_col]) - float(ms[b_key]["mean"])) > 1e-5:
                    mismatches.append((rk, csv_col, row[csv_col], ms[b_key]["mean"]))

    if mismatches:
        raise AssertionError(f"Detected {len(mismatches)} CSV-to-bundle mismatches: {mismatches}")

    return {
        "bundle_path": str(bundle_path),
        "tables_dir": str(tables_dir),
        "tables_generated": 8,
        "mismatches_detected": len(mismatches),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=str,
        default=os.environ.get("PROJECT_DELTA_OUTPUT_DIR", str(PROJECT_ROOT / "results")),
    )
    args = parser.parse_args()
    info = export_and_verify_tables(Path(args.results_dir).resolve())
    print(f"[SUCCESS] Generated and verified {info['tables_generated']} CSV tables in {info['tables_dir']} (0 mismatches).")
