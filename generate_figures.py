"""Publication-Quality Figure Generation Script (Appendix C2 & Section 16.3).

Generates all 7 required figures directly from `results/manifests/frozen_results_bundle.json`:
  - fig1_protocol_timeline.png: Study timeline & event-time information boundary diagram
  - fig2_data_audit_and_hubs.png: Temporal fraud prevalence, identity coverage, and entity degree distributions
  - fig3_leakage_protocol_matrix.png: Two-axis protocol leakage comparison (P0, P1, P2, P3 across models)
  - fig4_rolling_block_performance.png: Per-block AP and Precision@1% across rolling temporal test windows
  - fig5_maturity_rules_and_calibration.png: V0/V1/V2/V3 across D0-D3 and Bayesian q(a) calibration vs empirical truth
  - fig6_graph_ablations_and_stratum.png: Relation drop ablations, hub caps K, and visible fraud-neighbor stratum (H3)
  - fig7_efficiency_and_robustness.png: Computational runtime vs AP tradeoff and delay/prior misspecification sensitivity
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Palette: Deep Navy (#1B365D), Teal (#008080), Coral/Crimson (#C0392B), Amber (#D68910), Slate (#5D6D7E), Purple (#6C3483)
NAVY = "#1B365D"
TEAL = "#008080"
CRIMSON = "#C0392B"
AMBER = "#D68910"
SLATE = "#5D6D7E"
PURPLE = "#6C3483"
LIGHT_BG = "#F8F9FA"

from pathlib import Path
import argparse
PROJECT_ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--results-dir", type=str, default=os.environ.get("PROJECT_DELTA_OUTPUT_DIR", str(PROJECT_ROOT / "results")))
args, _ = parser.parse_known_args()
res_dir = Path(args.results_dir).resolve()
b_v2 = res_dir / "manifests/frozen_results_bundle_v2.json"
b_v1 = res_dir / "manifests/frozen_results_bundle.json"
with open(b_v2 if b_v2.is_file() else b_v1) as f:
    bundle = json.load(f)

fig_dirs = [str(res_dir / "figures")]
for d in fig_dirs:
    os.makedirs(d, exist_ok=True)

def save_all(fig, filename):
    for d in fig_dirs:
        fig.savefig(os.path.join(d, filename), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved figure: {filename}")

# -------------------------------------------------------------------------
# FIGURE 1: Study Timeline & Decision-Time Information Boundary (C2.1)
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10.5, 4.4))
ax.set_xlim(0, 105)
ax.set_ylim(0, 10)
ax.axis("off")

# Title banner
ax.text(52.5, 9.5, "Event-Time Causal Graph Visibility & Label-Maturity Protocol (P3 vs. Leaked P0/P1)",
        ha="center", va="center", fontsize=12, fontweight="bold", color=NAVY)

# Main time axis arrow
ax.annotate("", xy=(102, 1.3), xytext=(4, 1.3),
            arrowprops=dict(arrowstyle="->", lw=2.0, color=NAVY))
for x_pos, lbl in [(8, "t = 0\n(Warm-up Start)"), (42, "τ - M\n(Maturity Cutoff)"), (68, "τ\n(Retraining Cutoff)"), (86, "t_i\n(Test Scoring)"), (98, "t > t_i\n(Future)")]:
    ax.plot([x_pos, x_pos], [1.1, 1.5], color=NAVY, lw=1.8)
    ax.text(x_pos, 0.45, lbl, ha="center", va="center", fontsize=8.5, fontweight="bold", color=NAVY)

# Vertical dashed lines for τ - M, τ, and t_i
ax.axvline(42, ymin=0.15, ymax=0.88, color=TEAL, ls="--", lw=1.4, alpha=0.8)
ax.axvline(68, ymin=0.15, ymax=0.88, color=CRIMSON, ls="--", lw=1.8, alpha=0.9)
ax.axvline(86, ymin=0.15, ymax=0.88, color=PURPLE, ls=":", lw=1.5, alpha=0.85)

# Shaded zones
ax.add_patch(mpatches.Rectangle((8, 1.7), 34, 6.8, facecolor="#E8F6F3", edgecolor="none", alpha=0.7))
ax.add_patch(mpatches.Rectangle((42, 1.7), 26, 6.8, facecolor="#FEF9E7", edgecolor="none", alpha=0.8))
ax.add_patch(mpatches.Rectangle((68, 1.7), 32, 6.8, facecolor="#FDEDEC", edgecolor="none", alpha=0.6))

ax.text(25, 8.1, "Mature Historical Cohort\n(Age a_i ≥ M: S(a_i) = 0)", ha="center", va="center", fontsize=9, fontweight="bold", color=TEAL)
ax.text(55, 8.1, "Immature Window [τ - M, τ]\n(Age 0 ≤ a_i < M: S(a_i) > 0)", ha="center", va="center", fontsize=9, fontweight="bold", color=AMBER)
ax.text(84, 8.1, "Future / Evaluation Window (t > τ)\nExcluded from Training State at τ", ha="center", va="center", fontsize=9, fontweight="bold", color=CRIMSON)

# Four illustrative transaction trajectories
# Case 1: Visible Positive (t_1 = 22, report at 31 <= tau)
ax.plot([18, 30], [6.6, 6.6], color=CRIMSON, lw=2.5)
ax.scatter([18], [6.6], color=NAVY, s=45, zorder=5)
ax.scatter([30], [6.6], color=CRIMSON, marker="*", s=110, zorder=5)
ax.text(32, 6.6, "Visible Positive (Z_i(τ)=1): Report t_i + D_i ≤ τ  →  Target y = 1", va="center", fontsize=8.5, color="#1C2833", fontweight="bold")

# Case 2: Mature Negative (t_2 = 16, no report, age >= M at tau)
ax.plot([16, 68], [5.2, 5.2], color=TEAL, lw=2.2, ls="-")
ax.scatter([16], [5.2], color=TEAL, s=45, zorder=5)
ax.text(20, 5.5, "Mature Negative (Z_i(τ)=0): No report & a_i ≥ M  →  Target y = 0", va="center", fontsize=8.5, color=TEAL, fontweight="bold")

# Case 3: Immature Unreported Fraud (t_3 = 50, report arrives at 75 > tau)
ax.plot([48, 68], [3.6, 3.6], color=AMBER, lw=2.5)
ax.plot([68, 77], [3.6, 3.6], color=CRIMSON, lw=2.0, ls="--")
ax.scatter([48], [3.6], color=AMBER, s=45, zorder=5)
ax.scatter([77], [3.6], color=CRIMSON, marker="*", s=100, zorder=5)
ax.text(43, 2.7, "Immature / Unreported at τ (Z_i(τ)=imm): V0→0 (Biased), V1→Drop, V2→Soft Target q(a_i)=πS(a_i)/[πS(a_i)+(1-π)]",
        va="center", fontsize=8.2, color="#7E5109", fontweight="bold", bbox=dict(boxstyle="round,pad=0.25", facecolor="#FCF3CF", edgecolor=AMBER, lw=0.8))

save_all(fig, "fig1_protocol_timeline.png")

# -------------------------------------------------------------------------
# FIGURE 2: Step 1 Data Audit & Entity Hub Degree Distributions (C2.2)
# -------------------------------------------------------------------------
audit = bundle["step1_audit"]
bins = audit["temporal_bins"]
bin_ids = [b["bin_id"] + 1 for b in bins]
f_rates = [b["fraud_rate"] * 100.0 for b in bins]
id_covs = [b["identity_coverage"] * 100.0 for b in bins]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2))
ax1_twin = ax1.twinx()
ax1_twin.spines["top"].set_visible(False)

l1 = ax1.plot(bin_ids, f_rates, marker="o", color=CRIMSON, lw=2.0, label="Fraud Prevalence (%)")
l2 = ax1_twin.plot(bin_ids, id_covs, marker="s", color=NAVY, lw=1.8, ls="--", label="Identity Join Coverage (%)")
ax1.set_xlabel("Chronological 10-Day Temporal Bin (Days 1–180)")
ax1.set_ylabel("Fraud Prevalence (%)", color=CRIMSON, fontweight="bold")
ax1_twin.set_ylabel("Identity Join Coverage (%)", color=NAVY, fontweight="bold")
ax1.set_ylim(0, 4.0)
ax1_twin.set_ylim(0, 50.0)
ax1.set_title("(a) Fraud Rate & Partial Identity Coverage Across Time")
lines = l1 + l2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="upper right", frameon=True, facecolor=LIGHT_BG)
ax1.grid(axis="y", ls=":", alpha=0.5)

# Panel (b): Entity degree quantiles & max degree (Log scale)
ent_audit = audit["entity_degree_audit"]
ent_names = ["card1", "card_comp", "terminal_id", "addr1", "P_emaildomain", "DeviceInfo"]
ent_labels = ["card1\n(Proxy)", "card_comp\n(Composite)", "terminal_id\n(Merchant)", "addr1\n(Billing)", "P_email\n(Domain Hub)", "DeviceInfo\n(26.5% Cov)"]
p50_vals = [ent_audit[k]["degree_p50"] for k in ent_names]
p90_vals = [ent_audit[k]["degree_p90"] for k in ent_names]
p99_vals = [ent_audit[k]["degree_p99"] for k in ent_names]
max_vals = [ent_audit[k]["max_degree"] for k in ent_names]

x = np.arange(len(ent_names))
w = 0.20
ax2.bar(x - 1.5*w, p50_vals, width=w, color=SLATE, label="Median (p50)")
ax2.bar(x - 0.5*w, p90_vals, width=w, color=TEAL, label="90th Pct (p90)")
ax2.bar(x + 0.5*w, p99_vals, width=w, color=AMBER, label="99th Pct (p99)")
ax2.bar(x + 1.5*w, max_vals, width=w, color=CRIMSON, label="Max Hub Degree")
ax2.set_yscale("log")
ax2.set_xticks(x)
ax2.set_xticklabels(ent_labels, fontsize=8.5)
ax2.set_ylabel("Transactions per Entity Node (Log Scale)", fontweight="bold")
ax2.set_title("(b) Entity Degree Quantiles & Extreme Hub Concentration")
ax2.legend(loc="upper left", ncol=2, frameon=True, facecolor=LIGHT_BG)
ax2.grid(axis="y", ls=":", alpha=0.5)

fig.tight_layout()
save_all(fig, "fig2_data_audit_and_hubs.png")

# -------------------------------------------------------------------------
# FIGURE 3: Two-Axis Protocol Leakage Matrix (P0, P1, P2, P3) (C2.3)
# -------------------------------------------------------------------------
leak = bundle["leakage_matrix_runs"]
models_l = [
    ("B1_causal_aggregates", "B1: Tree + Neighbor Agg."),
    ("B3_graphsage", "B3: Causal Agg. MLP (SAGE-Style)"),
    ("B4_domain_care_gnn", "B4: Sim-Gated MLP (CARE-Insp.)"),
    ("B5_proposed_maturity_hybrid", "B5: Proposed Hybrid"),
]
protocols_l = [
    ("P0__V3", "P0: Static Oracle\n(Non-Deployable)", CRIMSON),
    ("P1__V2", "P1: Transductive\nTopology + V2", AMBER),
    ("P2__V0", "P2: Causal Graph\n+ Naive V0", SLATE),
    ("P3__V2", "P3: Causal Graph\n+ Maturity V2 (Ours)", TEAL),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.3))
x = np.arange(len(models_l))
w = 0.19

for p_i, (p_code, p_label, col) in enumerate(protocols_l):
    ap_means = [leak[f"{m_code}__{p_code}__D2"]["macro_summary"]["AP"]["mean"] for m_code, _ in models_l]
    p1_means = [leak[f"{m_code}__{p_code}__D2"]["macro_summary"]["Precision@1%"]["mean"] for m_code, _ in models_l]
    offset = (p_i - 1.5) * w
    bars1 = ax1.bar(x + offset, ap_means, width=w, color=col, label=p_label, edgecolor="white", lw=0.6)
    bars2 = ax2.bar(x + offset, p1_means, width=w, color=col, label=p_label, edgecolor="white", lw=0.6)
    if p_code == "P0__V3":
        for b_rect in list(bars1) + list(bars2):
            b_rect.set_hatch("//")

ax1.set_xticks(x)
ax1.set_xticklabels([m_lbl for _, m_lbl in models_l], fontsize=8.5)
ax1.set_ylim(0.45, 0.82)
ax1.set_ylabel("Macro Average Precision (AP)", fontweight="bold")
ax1.set_title("(a) Ranking Performance Across Visibility Protocols (D2)")
ax1.grid(axis="y", ls=":", alpha=0.5)
ax1.legend(loc="upper right", fontsize=8, frameon=True, facecolor=LIGHT_BG)

ax2.set_xticks(x)
ax2.set_xticklabels([m_lbl for _, m_lbl in models_l], fontsize=8.5)
ax2.set_ylim(0.45, 0.82)
ax2.set_ylabel("Alert-Budget Precision@1%", fontweight="bold")
ax2.set_title("(b) Top-1% Alert Precision: Oracle Inflation vs. Causal P3")
ax2.grid(axis="y", ls=":", alpha=0.5)
ax2.text(0.02, 0.93, "Hatched bars (//) denote non-deployable Oracle P0", transform=ax2.transAxes,
         fontsize=8, style="italic", color=CRIMSON)

fig.tight_layout()
save_all(fig, "fig3_leakage_protocol_matrix.png")

# -------------------------------------------------------------------------
# FIGURE 4: Per-Block Rolling Temporal Performance (AP & Precision@1%) (C2.4)
# -------------------------------------------------------------------------
prim = bundle["primary_runs_table3"]
blk_nums = np.arange(1, 9)
blk_labels = [f"B{i}\n(d{101+10*(i-1)}-{101+10*i})" for i in blk_nums]

series_cfg = [
    ("B0_raw_tabular__V2__P3__D2", "B0: Raw Tabular (V2)", SLATE, "s", "--"),
    ("B1_causal_aggregates__V2__P3__D2", "B1: Tree + Causal Neighbor Agg. (V2)", NAVY, "^", "-"),
    ("B4_domain_care_gnn__V2__P3__D2", "B4: Sim-Gated MLP (CARE-Insp., V2)", PURPLE, "d", "-."),
    ("B5_proposed_maturity_hybrid__V2__P3__D2", "B5: Proposed Maturity Hybrid (V2)", TEAL, "o", "-"),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.2))
for r_key, lbl, col, mk, ls in series_cfg:
    ap_b = prim[r_key]["macro_summary"]["AP"]["per_block"]
    p1_b = prim[r_key]["macro_summary"]["Precision@1%"]["per_block"]
    ax1.plot(blk_nums, ap_b, marker=mk, color=col, ls=ls, lw=2.0, label=lbl)
    ax2.plot(blk_nums, p1_b, marker=mk, color=col, ls=ls, lw=2.0, label=lbl)

ax1.set_xticks(blk_nums)
ax1.set_xticklabels(blk_labels, fontsize=8)
ax1.set_ylabel("Per-Block Average Precision (AP)", fontweight="bold")
ax1.set_title("(a) Rolling Temporal Block AP Trajectory (Protocol P3, D2)")
ax1.grid(True, ls=":", alpha=0.5)
ax1.legend(loc="lower left", fontsize=8, frameon=True, facecolor=LIGHT_BG)

ax2.set_xticks(blk_nums)
ax2.set_xticklabels(blk_labels, fontsize=8)
ax2.set_ylabel("Per-Block Alert Precision@1%", fontweight="bold")
ax2.set_title("(b) Rolling Temporal Block Precision@1% Trajectory")
ax2.grid(True, ls=":", alpha=0.5)

fig.tight_layout()
save_all(fig, "fig4_rolling_block_performance.png")

# -------------------------------------------------------------------------
# FIGURE 5: Maturity Rules V0/V1/V2/V3 across D0-D3 & Posterior Calibration (C2.5)
# -------------------------------------------------------------------------
mat_grid = bundle["maturity_grid_table4"]
scens = ["D0", "D1", "D2", "D3"]
scen_labels = ["D0: Immediate\n(M=0d)", "D1: Short Exp.\n(M=14d)", "D2: Profile\n(M=30d)", "D3: Long-Tail\n(M=90d)"]
rules = [
    ("V0", "V0: Unreported as Negative", SLATE),
    ("V1", "V1: Drop Immature Rows", AMBER),
    ("V2", "V2: Bayesian Soft Target q(a)", TEAL),
    ("V3", "V3: Oracle Upper Bound (Non-Deployable)", CRIMSON),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.3))
x = np.arange(len(scens))
w = 0.19

for r_i, (r_code, r_lbl, col) in enumerate(rules):
    aps = [mat_grid[f"B5__{sc}__{r_code}"]["macro_summary"]["AP"]["mean"] for sc in scens]
    bars = ax1.bar(x + (r_i - 1.5)*w, aps, width=w, color=col, label=r_lbl, edgecolor="white")
    if r_code == "V3":
        for br in bars:
            br.set_hatch("//")

ax1.set_xticks(x)
ax1.set_xticklabels(scen_labels, fontsize=8.5)
ax1.set_ylim(0.35, 0.68)
ax1.set_ylabel("Macro Average Precision (AP)", fontweight="bold")
ax1.set_title("(a) Supervision Rules V0–V3 Across Latency Regimes (D0–D3)")
ax1.legend(loc="lower left", fontsize=8, frameon=True, facecolor=LIGHT_BG)
ax1.grid(axis="y", ls=":", alpha=0.5)

# Panel (b): Theoretical Posterior q(a) vs Empirical Unreported Fraud Rate across Age Bins (D1, D2, D3)
ages_grid = np.linspace(0.0, 90.0, 300)
from src.delay_simulation.delay_models import compute_posterior_q
for sc, col, ls, lbl in [
    ("D1", NAVY, "--", "D1 Theoretical q(a) (M=14d)"),
    ("D2", TEAL, "-", "D2 Theoretical q(a) (M=30d)"),
    ("D3", CRIMSON, "-.", "D3 Theoretical q(a) (M=90d)"),
]:
    M_val = 14.0 if sc == "D1" else (30.0 if sc == "D2" else 90.0)
    q_curve = compute_posterior_q(ages_grid, 0.0173, sc, M_val) * 100.0
    ax2.plot(ages_grid, q_curve, color=col, ls=ls, lw=2.2, label=lbl)

# Overlay empirical hidden fraud rate points from q_calibration for D2 and D3
q_cal = bundle["q_calibration"]
for sc, col, mk in [("D2", TEAL, "o"), ("D3", CRIMSON, "s")]:
    pts = [row for row in q_cal if row["scenario"] == sc and row["a_mid"] >= 1.5]
    ax2.scatter([p["a_mid"] for p in pts], [p["empirical_hidden_fraud_rate"]*100.0 for p in pts],
                color=col, marker=mk, s=45, zorder=5, label=f"{sc} Empirical Unreported Fraud % (a ≥ 1d)")

ax2.set_xlabel("Transaction Age a = τ - t_i at Cutoff τ (Days)", fontweight="bold")
ax2.set_ylabel("Hidden Fraud Probability Among Unreported (%)", fontweight="bold")
ax2.set_title("(b) Bayesian Posterior q(a) Decay & Empirical Calibration")
ax2.set_xlim(0, 90)
ax2.set_ylim(-0.1, 2.2)
ax2.legend(loc="upper right", fontsize=7.8, frameon=True, facecolor=LIGHT_BG)
ax2.grid(True, ls=":", alpha=0.5)

fig.tight_layout()
save_all(fig, "fig5_maturity_rules_and_calibration.png")

# -------------------------------------------------------------------------
# FIGURE 6: Graph Relation Ablations, Hub Caps K, and Stratum H3 (C2.6)
# -------------------------------------------------------------------------
abl = bundle["ablation_runs_table5"]
ab_keys = [
    ("Hub_WideCap_K200_InvLog", "K=200 + InvLog Weight"),
    ("Hub_TightCap_K10_InvLog", "K=10 + InvLog Weight"),
    ("Full_5_Relations_K50_InvLog", "Full 5 Rels (K=50, InvLog)"),
    ("Hub_Uncapped_NoWeight_KNone", "Uncapped / Unweighted"),
    ("Composite_Card_Proxy_card_comp", "Composite Card (card_comp)"),
    ("Card_and_Terminal_Only", "Card + Terminal Only"),
    ("Drop_Email_Domain_Proxy", "Drop Email-Domain Proxy"),
    ("Drop_Device_Proxy", "Drop Device Proxy"),
    ("Drop_Address_Proxy", "Drop Address Proxy"),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.4))
plt.subplots_adjust(wspace=0.34)
y_pos = np.arange(len(ab_keys))
ap_vals = [abl[k]["macro_summary"]["AP"]["mean"] for k, _ in ab_keys]
colors_ab = [TEAL if "Cap" in k or "Full" in k else (NAVY if "Card" in k else AMBER) for k, _ in ab_keys]

ax1.barh(y_pos, ap_vals, color=colors_ab, height=0.62, edgecolor="white")
ax1.set_yticks(y_pos)
ax1.set_yticklabels([lbl for _, lbl in ab_keys], fontsize=8.5)
ax1.set_xlim(0.50, 0.66)
ax1.set_xlabel("Macro Average Precision (AP)", fontweight="bold")
ax1.set_title("(a) Relation & Hub-Cap Ablations", fontsize=10.2)
for idx, v in enumerate(ap_vals):
    ax1.text(v + 0.002, idx, f"{v:.4f}", va="center", fontsize=8)
ax1.grid(axis="x", ls=":", alpha=0.5)

# Panel (b): Hypothesis H3 Stratum Comparison (Visible Prior Fraud Neighbor within 2 Hops vs. Isolated)
# Compare under K=10 (where 78.1% of frauds have visible recent neighbor vs 21.9% isolated) and B0 vs B1 vs B5
stratum_models = [
    ("B0_raw_tabular__V2__P3__D2", "B0: Raw\nTabular"),
    ("B1_causal_aggregates__V2__P3__D2", "B1: Tree +\nCausal Agg."),
    ("B4_domain_care_gnn__V2__P3__D2", "B4: Domain\nGNN (CARE)"),
    ("B5_proposed_maturity_hybrid__V2__P3__D2", "B5: Proposed\nHybrid"),
]
vis_nbr_aps = [prim[k]["stratum_h3_specific_2hop"]["with_visible_prior_fraud_neighbor"]["AP"] for k, _ in stratum_models]
iso_aps = [prim[k]["stratum_h3_specific_2hop"]["isolated_no_visible_fraud_neighbor"]["AP"] for k, _ in stratum_models]

x_s = np.arange(len(stratum_models))
w_s = 0.32
ax2.bar(x_s - w_s/2, vis_nbr_aps, width=w_s, color=TEAL, label="Stratum 1: ≥1 Vis. Prior Fraud Nbr (2-Hop Specific, 49.3% frauds)")
ax2.bar(x_s + w_s/2, iso_aps, width=w_s, color=SLATE, label="Stratum 2: Isolated on Specific Entities (50.7% frauds)")
ax2.set_xticks(x_s)
ax2.set_xticklabels([lbl for _, lbl in stratum_models], fontsize=8.5)
ax2.set_ylim(0.35, 0.74)
ax2.set_ylabel("Pooled Test Average Precision (AP)", fontweight="bold")
ax2.set_title("(b) H3: Lift by Specific-Entity 2-Hop Stratum", fontsize=10.2)
ax2.legend(loc="upper left", fontsize=8, frameon=True, facecolor=LIGHT_BG)
ax2.grid(axis="y", ls=":", alpha=0.5)

fig.tight_layout()
save_all(fig, "fig6_graph_ablations_and_stratum.png")

# -------------------------------------------------------------------------
# FIGURE 7: Computational Efficiency Tradeoff & Misspecification Robustness (C2.7)
# -------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.2))

eff_models = [
    ("B0_raw_tabular__V2__P3__D2", "B0: Raw Tabular", SLATE, "s"),
    ("B1_causal_aggregates__V2__P3__D2", "B1: Tree + Causal Agg.", NAVY, "^"),
    ("B2_graph_light__V2__P3__D2", "B2: Graph-Light Tree", AMBER, "P"),
    ("B3_graphsage__V2__P3__D2", "B3: Causal Agg. MLP", PURPLE, "D"),
    ("B4_domain_care_gnn__V2__P3__D2", "B4: Sim-Gated MLP (CARE-Insp.)", CRIMSON, "X"),
    ("B5_proposed_maturity_hybrid__V2__P3__D2", "B5: Proposed Hybrid", TEAL, "o"),
]
for k, lbl, col, mk in eff_models:
    rt = prim[k]["runtime_sec"]
    ap = prim[k]["macro_summary"]["AP"]["mean"]
    ax1.scatter([rt], [ap], color=col, marker=mk, s=110, label=f"{lbl} ({rt:.1f}s)")
    ax1.annotate(lbl.split(":")[0], (rt, ap), xytext=(5, 4), textcoords="offset points", fontsize=8.5, fontweight="bold", color=col)

ax1.set_xlabel("8-Block Rolling Wall Time Across 5 Seeds (Seconds)", fontweight="bold")
ax1.set_ylabel("Macro Average Precision (AP)", fontweight="bold")
ax1.set_title("(a) Accuracy vs. Computational Runtime Pareto Frontier")
ax1.set_xlim(0, 31.0)
ax1.set_ylim(0.50, 0.65)
ax1.legend(loc="lower right", fontsize=8, frameon=True, facecolor=LIGHT_BG)
ax1.grid(True, ls=":", alpha=0.5)

# Panel (b): Robustness & Sensitivity Grid (Table 6)
rob = bundle["robustness_runs_table6"]
rob_items = [
    ("D2 Baseline (Matched Survival)", prim["B5_proposed_maturity_hybrid__V2__P3__D2"]["macro_summary"]["AP"]["mean"], TEAL),
    ("D1 True / D2 Survival Assumed", rob["D1_True__Trained_with_D2_Survival"]["macro_summary"]["AP"]["mean"], NAVY),
    ("D3 True / D2 Survival (M=90d)", rob["D3_True__Trained_with_D2_Survival_M90"]["macro_summary"]["AP"]["mean"], NAVY),
    ("D3 Premature M=30d (V2 Soft)", rob["D3_Premature_Maturity_Horizon_M30_V2"]["macro_summary"]["AP"]["mean"], AMBER),
    ("D3 Premature M=30d (V1 Drop)", rob["D3_Premature_Maturity_Horizon_M30_V1"]["macro_summary"]["AP"]["mean"], CRIMSON),
    ("Prior π Underestimated (0.5x)", rob["D2_Prior_Underestimated_0.5x"]["macro_summary"]["AP"]["mean"], SLATE),
    ("Prior π Overestimated (2.0x)", rob["D2_Prior_Overestimated_2.0x"]["macro_summary"]["AP"]["mean"], SLATE),
    ("60-Day Sliding Window (vs. Expanding)", rob["D2_Sliding_Window_60d"]["macro_summary"]["AP"]["mean"], PURPLE),
]
y_r = np.arange(len(rob_items))
r_vals = [v for _, v, _ in rob_items]
r_cols = [c for _, _, c in rob_items]
ax2.barh(y_r, r_vals, color=r_cols, height=0.60, edgecolor="white")
ax2.set_yticks(y_r)
ax2.set_yticklabels([lbl for lbl, _, _ in rob_items], fontsize=8.3)
ax2.set_xlim(0.50, 0.65)
ax2.set_xlabel("Macro Average Precision (AP)", fontweight="bold")
ax2.set_title("(b) Sensitivity to Delay, Prior, Maturity & Window Shifts")
for idx, v in enumerate(r_vals):
    ax2.text(v + 0.002, idx, f"{v:.4f}", va="center", fontsize=8)
ax2.grid(axis="x", ls=":", alpha=0.5)

fig.tight_layout()
save_all(fig, "fig7_efficiency_and_robustness.png")
print("All 7 figures generated and saved successfully.")
