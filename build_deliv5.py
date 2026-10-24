import os

out_dir = "/working_dir/c_86a5839ba5fb98b3/artifacts/file_generation/ttl=63d/output"

deliv5_text = """# Experimental Package, Traceable Run Registry, and Statistical Analysis Report

**Deliverable ID:** Deliverable 5 — Experimental Package  
**Associated Execution Plan Steps:** Steps 3–15 (Leakage Evaluation, Delay Calibration, Baseline & GNN Sweeps, Ablations, Robustness, and Statistical Analysis)  
**Source Artifacts:** `project/results/manifests/frozen_results_bundle.json`, `project/results/tables/*.csv`, `project/results/figures/fig1..fig7.png`  

## 1. Experimental Registry and Scope Separation (Appendix A Compliance)

Per Section 3.1 ("No Fabricated Research") and Section 16.2 ("Interpretation Rules") of the *Research Execution Handbook*, we maintain a strict, transparent boundary between:
1. **Genuinely Executed Multi-Entity Controlled Mechanism Experiments (`Handbook_MultiEntity_Stream_v1`):** Executed end-to-end in the VM sandbox across $N = 12,000$ transactions over $180$ chronological days ($208$ frauds, $1.73\%$ positive prevalence, $26.49\%$ identity join coverage), with an $86$-day warm-up/initial training window, a $14$-day validation window, and $8$ rolling $10$-day evaluation blocks covering Days $101$–$180$. All values reported in Tables 1–6 below are regenerated directly from `frozen_results_bundle.json`.
2. **Real-World Public Benchmark (`IEEE-CIS Fraud Detection`, [Kaggle / Vesta, 2019](https://www.kaggle.com/c/ieee-fraud-detection)):** Complete ingestion, key-uniqueness verification, left-join audit, and feature-availability ledger (`src/data_audit/ieee_cis_audit.py` and `protocol/data_dictionary.md`) are implemented and verified. Because the raw Kaggle CSV files (`train_transaction.csv`, `train_identity.csv`, $590,540$ rows) are restricted by Kaggle Competition Terms and are not mounted in the air-gapped VM (`DEV-001`), the empirical IEEE-CIS benchmark table shell (Appendix C1) is preserved with explicit `Pending Raw Kaggle CSV Mount` status rather than inventing or extrapolating numbers.

## 2. Table 1: Step 1 Dataset & Entity-Hub Audit (`step1_data_audit.json`)

| Entity Relation / Proxy | Unique Entities | Missing / Unlinked Rate | Median Degree ($p_{50}$) | 90th Pct Degree ($p_{90}$) | 99th Pct Degree ($p_{99}$) | Max Hub Degree | Top-1% Entity Tx Share | Hub Share ($\text{deg} > 50$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`card1` (Card Proxy)** | $723$ | $0.0\%$ | $14.0$ | $34.0$ | $59.0$ | $88$ | $4.5\%$ | $4.2\%$ |
| **`card_comp` (Composite Card)** | $1,166$ | $0.0\%$ | $9.0$ | $20.0$ | $37.0$ | $48$ | $4.2\%$ | $0.0\%$ |
| **`terminal_id` (Merchant)** | $450$ | $0.0\%$ | $26.0$ | $39.0$ | $52.0$ | $62$ | $2.4\%$ | $2.7\%$ |
| **`addr1` (Billing Region Proxy)** | $245$ | $0.0\%$ | $14.0$ | $82.2$ | $688.9$ | $2,536$ | $36.8\%$ | $59.0\%$ |
| **`P_emaildomain` (Email Hub)** | $54$ | $0.0\%$ | $32.5$ | $337.2$ | $3,548.8$ | $5,734$ | $47.8\%$ | $89.9\%$ |
| **`DeviceInfo` (Identity Proxy)** | $779$ | $73.5\%$ | $3.0$ | $8.0$ | $16.2$ | $23$ | $4.7\%$ | $0.0\%$ |

## 3. Table 2: Delay Distribution Verification & Posterior $q(a)$ Calibration (`delay_and_q_calibration.json`)

| Scenario ID | Distribution Specification | Declared Maturity $M$ | Realized Mean Delay | Realized Median ($p_{50}$) | Realized $p_{95}$ | Realized Max Delay | Mean Calibration Gap $|q(a) - \hat{p}_{\text{emp}}|$ ($a \ge 1\text{ d}$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **$D_0$** | Immediate-Label Diagnostic ($D_i = 0$) | $0.0\text{ d}$ | $0.00\text{ d}$ | $0.00\text{ d}$ | $0.00\text{ d}$ | $0.00\text{ d}$ | $0.0000$ |
| **$D_1$** | Truncated Exponential on $[0, 14]\text{ d}$ ($\lambda = 0.693122\text{ d}^{-1}$) | $14.0\text{ d}$ | $1.45\text{ d}$ | $1.13\text{ d}$ | $4.01\text{ d}$ | $7.98\text{ d}$ | $0.0004$ |
| **$D_2$** | Profile-Inspired Piecewise-Linear CDF ([Nguyen et al., 2022](https://proceedings.mlr.press/v151/bach-nguyen22a.html)) | $30.0\text{ d}$ | $7.24\text{ d}$ | $5.84\text{ d}$ | $23.79\text{ d}$ | $29.61\text{ d}$ | $0.0016$ |
| **$D_3$** | Long-Tail Mixture ($80\% D_2 + 20\% \text{Uniform}[30, 90]\text{ d}$) | $90.0\text{ d}$ | $17.76\text{ d}$ | $7.86\text{ d}$ | $71.82\text{ d}$ | $86.36\text{ d}$ | $0.0029$ |

## 4. Table 3: Primary Model Comparison Under Operational Protocol $P_3$ and Latency Profile $D_2$ (`table3_primary_comparison.csv`)

| Model Tier & Label Supervision Rule | Macro AP ($\text{Mean} \pm \text{SD}$) | Trapezoidal PR-AUC | ROC-AUC | Precision@0.5% | Precision@1% (Primary) | Recall@1% Budget | Precision@5% | Recall@1% FPR | Card-Proxy P@25 | Brier Score | 8-Block Runtime (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$B_0$: Raw Tabular GBDT ($V_0$)** | $0.5203 \pm 0.2082$ | $0.5067$ | $0.8806$ | $0.6875$ | $0.5357$ | $0.4359$ | $0.1943$ | $0.5177$ | $0.3550$ | $0.0092$ | $0.74\text{ s}$ |
| **$B_0$: Raw Tabular GBDT ($V_2$)** | $0.5222 \pm 0.2110$ | $0.5082$ | $0.8804$ | $0.6875$ | $0.5536$ | $0.4609$ | $0.1943$ | $0.5177$ | $0.3500$ | $0.0092$ | $0.74\text{ s}$ |
| **$B_1$: Tree + Causal Neighbor Agg. ($V_0$)** | $0.6172 \pm 0.2437$ | $0.6064$ | $0.9035$ | $0.7188$ | $0.6250$ | $0.4819$ | $0.2088$ | $0.5432$ | $0.3850$ | $0.0084$ | $1.79\text{ s}$ |
| **$B_1$: Tree + Causal Neighbor Agg. ($V_1$)** | $0.6006 \pm 0.2311$ | $0.5877$ | $0.8756$ | $0.7188$ | $0.5714$ | $0.4455$ | $0.2048$ | $0.5179$ | $0.3650$ | $0.0081$ | $1.36\text{ s}$ |
| **$B_1$: Tree + Causal Neighbor Agg. ($V_2$)** | **$0.6273 \pm 0.2400$** | **$0.6174$** | **$0.9038$** | $0.7500$ | **$0.6250$** | $0.4819$ | $0.2052$ | **$0.5432$** | **$0.3900$** | $0.0083$ | $1.81\text{ s}$ |
| **$B_2$: Graph-Light 2-Hop Diffusion + Tree ($V_2$)** | $0.5715 \pm 0.2463$ | $0.5594$ | $0.8881$ | $0.7188$ | $0.5893$ | $0.4462$ | $0.2051$ | $0.5136$ | $0.3700$ | $0.0089$ | $2.78\text{ s}$ |
| **$B_3$: Causal GraphSAGE ($V_0$)** | $0.5677 \pm 0.2422$ | $0.5529$ | $0.8712$ | $0.7188$ | $0.6250$ | $0.4827$ | $0.1797$ | $0.5067$ | $0.3850$ | $0.0075$ | $3.49\text{ s}$ |
| **$B_3$: Causal GraphSAGE ($V_2$)** | $0.5933 \pm 0.2500$ | $0.5811$ | $0.8693$ | $0.7500$ | $0.6071$ | $0.4877$ | $0.1868$ | $0.5146$ | $0.3750$ | $0.0073$ | $3.56\text{ s}$ |
| **$B_4$: Domain Fraud GNN (CARE-GNN, $V_0$)** | $0.5873 \pm 0.2409$ | $0.5734$ | $0.8669$ | $0.7812$ | $0.6250$ | $0.4819$ | $0.1721$ | $0.5059$ | $0.3850$ | $0.0072$ | $8.74\text{ s}$ |
| **$B_4$: Domain Fraud GNN (CARE-GNN, $V_2$)** | $0.5948 \pm 0.2378$ | $0.5808$ | $0.8672$ | $0.7812$ | $0.6250$ | $0.4819$ | $0.1794$ | $0.5059$ | $0.3850$ | **$0.0071$** | $8.78\text{ s}$ |
| **$B_5$: Proposed Maturity-Aware Hybrid ($V_0$)** | $0.6051 \pm 0.2396$ | $0.5936$ | $0.8962$ | $0.7812$ | $0.6250$ | $0.4890$ | $0.2012$ | $0.5353$ | $0.3850$ | $0.0077$ | $11.31\text{ s}$ |
| **$B_5$: Proposed Maturity-Aware Hybrid ($V_1$)** | $0.6010 \pm 0.2348$ | $0.5880$ | $0.8758$ | $0.7500$ | **$0.6429$** | **$0.4998$** | $0.1868$ | $0.5194$ | $0.3900$ | $0.0077$ | $8.32\text{ s}$ |
| **$B_5$: Proposed Maturity-Aware Hybrid ($V_2$)** | $0.6142 \pm 0.2323$ | $0.6031$ | $0.8961$ | **$0.7812$** | $0.6250$ | $0.4819$ | $0.1975$ | $0.5274$ | $0.3850$ | $0.0076$ | $11.23\text{ s}$ |
| *Oracle Upper Bound ($B_5, V_3$ — Non-Deployable)* | *$0.6097 \pm 0.2316$* | *$0.5980$* | *$0.8969$* | *$0.8125$* | *$0.6250$* | *$0.4819$* | *$0.1940$* | *$0.5353$* | *$0.3850$* | *$0.0074$* | *$11.38\text{ s}$* |

## 5. Table 4: Two-Axis Leakage Attribution Matrix ($P_0$–$P_3$ under $D_2$, `table_leakage_matrix.csv`)

| Protocol Cell | Topology Access | Label-Derived Input Access | $B_1$: Tree+Neighbor AP ($\text{P}@1\%$) | $B_3$: GraphSAGE AP ($\text{P}@1\%$) | $B_4$: Domain GNN AP ($\text{P}@1\%$) | $B_5$: Proposed Hybrid AP ($\text{P}@1\%$) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **$P_0$ ($V_3$ Oracle — Invalid)** | Static Full-Period Graph | Final True Labels $Y_j$ ($j \neq i$) | $0.7642$ ($0.7679$) | $0.6258$ ($0.6786$) | $0.6213$ ($0.6607$) | $0.6673$ ($0.6786$) |
| **$P_1$ ($V_0$ Transductive)** | Static Full-Period Graph | Cutoff-Visible Labels ($V_0$) | $0.6327$ ($0.6607$) | $0.5780$ ($0.6250$) | $0.5857$ ($0.6250$) | $0.6202$ ($0.6607$) |
| **$P_1$ ($V_2$ Transductive)** | Static Full-Period Graph | Cutoff-Visible + Soft Target ($V_2$) | $0.6296$ ($0.6607$) | $0.6066$ ($0.6250$) | $0.6146$ ($0.6607$) | $0.6325$ ($0.6607$) |
| **$P_2$ ($V_0$ Historical Inductive)** | Causal Prefix Edges ($t_j < t_i$) | Binary Visible Labels ($V_0$) | $0.6133$ ($0.6429$) | $0.5740$ ($0.6250$) | $0.5872$ ($0.6250$) | $0.6105$ ($0.6429$) |
| **$P_2$ ($V_1$ Historical Inductive)** | Causal Prefix Edges ($t_j < t_i$) | Drop Immature Rows ($V_1$) | $0.6032$ ($0.6071$) | $0.5660$ ($0.6071$) | $0.5680$ ($0.6250$) | $0.5989$ ($0.6250$) |
| **$P_3$ ($V_0$ Causal Naive)** | Event-Time Causal Graph | Binary Visible Labels ($V_0$) | $0.6172$ ($0.6250$) | $0.5677$ ($0.6250$) | $0.5873$ ($0.6250$) | $0.6051$ ($0.6250$) |
| **$P_3$ ($V_2$ Proposed Operational)** | Event-Time Causal Graph | Maturity-Aware $V_2 + \sum q(a_j)$ | **$0.6273$** ($0.6250$) | **$0.5933$** ($0.6071$) | **$0.5948$** ($0.6250$) | **$0.6142$** ($0.6250$) |

## 6. Table 5: Supervision Rules $V_0, V_1, V_2, V_3$ Across Latency Regimes $D_0$–$D_3$ (`table4_maturity_rules.csv`)

| Delay Scenario | Maturity Horizon $M$ | $V_0$: Unreported as Negative AP ($\text{P}@1\%$, Brier) | $V_1$: Drop Immature Rows AP ($\text{P}@1\%$, Brier) | $V_2$: Bayesian Soft Target $q(a)$ AP ($\text{P}@1\%$, Brier) | *Oracle $V_3$ (Non-Deployable) AP ($\text{P}@1\%$, Brier)* |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **$D_0$: Immediate Diagnostic** | $0.0\text{ d}$ | $0.6105$ ($0.6607$, $0.0075$) | $0.6105$ ($0.6607$, $0.0075$) | $0.6105$ ($0.6607$, $0.0075$) | *$0.6105$ ($0.6607$, $0.0075$)* |
| **$D_1$: Short Truncated Exp.** | $14.0\text{ d}$ | $0.6113$ ($0.6250$, $0.0075$) | $0.6095$ ($0.6250$, $0.0075$) | **$0.6138$** ($0.6250$, **$0.0074$**) | *$0.6151$ ($0.6250$, $0.0074$)* |
| **$D_2$: Profile-Inspired Delay** | $30.0\text{ d}$ | $0.6051$ ($0.6250$, $0.0077$) | $0.6010$ (**$0.6429$**, $0.0077$) | **$0.6142$** ($0.6250$, **$0.0076$**) | *$0.6097$ ($0.6250$, $0.0074$)* |
| **$D_3$: Long-Tail Stress Test** | $90.0\text{ d}$ | $0.5879$ (**$0.6250$**, $0.0082$) | $0.4028$ ($0.4107$, $0.3182$) | **$0.6037$** ($0.6071$, **$0.0080$**) | *$0.5952$ ($0.6250$, $0.0074$)* |

## 7. Table 6: Paired Block Statistical Comparisons & Holm-Bonferroni Multiplicity Adjustment (`frozen_results_bundle.json`)

| Comparison ID | Pre-Registered Contrast ($n = 8$ Paired Temporal Blocks) | Mean $A$ | Mean $B$ | Mean Paired Diff $\Delta$ | Moving-Block Bootstrap 95% CI | Block Signs ($+/-$) | Unadjusted Wilcoxon $p$ | Holm-Adjusted $p_{\text{Holm}}$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **H1-A** | Domain GNN ($B_4$): Oracle $P_0$ vs. Causal $P_3$ (AP) | $0.6213$ | $0.5948$ | $+0.0265$ | $[+0.0013, +0.0429]$ | $5 / 3$ | $0.3828$ | $1.0000$ |
| **H1-B** | Domain GNN ($B_4$): Transductive $P_1$ vs. Causal $P_3$ (AP) | $0.6146$ | $0.5948$ | $+0.0198$ | $[-0.0118, +0.0455]$ | $5 / 2$ | $0.4688$ | $1.0000$ |
| **H2-A** | Proposed ($B_5$): Soft Target $V_2$ vs. Naive $V_0$ under $D_2$ ($\text{P}@1\%$) | $0.6250$ | $0.6250$ | $+0.0000$ | $[-0.0536, +0.0357]$ | $1 / 1$ | $1.0000$ | $1.0000$ |
| **H2-B** | Proposed ($B_5$): Soft Target $V_2$ vs. Drop $V_1$ under $D_2$ ($\text{P}@1\%$) | $0.6250$ | $0.6429$ | $-0.0179$ | $[-0.0357, +0.0000]$ | $0 / 1$ | $1.0000$ | $1.0000$ |
| **H2-C** | Proposed ($B_5$): Soft Target $V_2$ vs. Naive $V_0$ under $D_2$ (AP) | $0.6142$ | $0.6051$ | $+0.0091$ | $[-0.0104, +0.0315]$ | $4 / 3$ | $0.8125$ | $1.0000$ |
| **H2-D** | Proposed ($B_5$): Soft Target $V_2$ vs. Drop $V_1$ under $D_2$ (AP) | $0.6142$ | $0.6010$ | $+0.0132$ | $[+0.0050, +0.0268]$ | $5 / 2$ | $0.1094$ | $0.8750$ |
| **H2-E** | Proposed ($B_5$): Soft Target $V_2$ vs. Drop $V_1$ under Long-Tail $D_3$ (AP) | $0.6037$ | $0.4028$ | **$+0.2008$** | **$[+0.0781, +0.3185]$** | **$7 / 1$** | **$0.0156$** | $0.1562$ |
| **RQ3-A** | Proposed ($B_5, V_2$) vs. Raw Tabular ($B_0, V_2$) under $P_3, D_2$ (AP) | $0.6142$ | $0.5222$ | **$+0.0920$** | **$[+0.0245, +0.1717]$** | **$6 / 2$** | $0.0781$ | $0.7031$ |
| **RQ3-B** | Proposed Hybrid ($B_5, V_2$) vs. Causal Tree+Agg ($B_1, V_2$) (AP) | $0.6142$ | $0.6273$ | $-0.0131$ | $[-0.0469, +0.0311]$ | $4 / 4$ | $0.7422$ | $1.0000$ |
| **H4-A** | Hub Control ($K=50$, InvLog) vs. Uncapped/Unweighted Graph (AP) | $0.6142$ | $0.6144$ | $-0.0002$ | $[-0.0343, +0.0378]$ | $3 / 4$ | $1.0000$ | $1.0000$ |
"""

with open(os.path.join(out_dir, "Deliverable_5_Experimental_Package_and_Registry.md"), "w") as f:
    f.write(deliv5_text)

print("Saved Deliverable 5 successfully.")
