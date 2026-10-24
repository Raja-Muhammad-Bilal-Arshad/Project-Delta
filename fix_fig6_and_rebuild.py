with open("/tmp/project/generate_figures.py", "r") as f:
    code = f.read()

old_f6 = """stratum_models = [
    ("B0_raw_tabular__V2__P3__D2", "B0: Raw Tabular"),
    ("B1_causal_aggregates__V2__P3__D2", "B1: Tree + Causal Agg."),
    ("B4_domain_care_gnn__V2__P3__D2", "B4: Domain GNN"),
    ("B5_proposed_maturity_hybrid__V2__P3__D2", "B5: Proposed Hybrid"),
]"""

new_f6 = """stratum_models = [
    ("B0_raw_tabular__V2__P3__D2", "B0: Raw\\nTabular"),
    ("B1_causal_aggregates__V2__P3__D2", "B1: Tree +\\nCausal Agg."),
    ("B4_domain_care_gnn__V2__P3__D2", "B4: Domain\\nGNN (CARE)"),
    ("B5_proposed_maturity_hybrid__V2__P3__D2", "B5: Proposed\\nHybrid"),
]"""

code = code.replace(old_f6, new_f6)
code = code.replace(
    'ax1.set_title("(a) Relation Drops, Card Proxy & Hub-Control Ablations")',
    'ax1.set_title("(a) Relation & Hub-Cap Ablations", fontsize=10.2)'
)
code = code.replace(
    'ax2.set_title("(b) Hypothesis H3: Relational Lift by Fraud-Neighbor Visibility")',
    'ax2.set_title("(b) H3: Lift by Fraud-Neighbor Stratum", fontsize=10.2)'
)
code = code.replace(
    "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.3))\ny_pos = np.arange(len(ab_keys))",
    "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.4))\nplt.subplots_adjust(wspace=0.34)\ny_pos = np.arange(len(ab_keys))"
)

with open("/tmp/project/generate_figures.py", "w") as f:
    f.write(code)
