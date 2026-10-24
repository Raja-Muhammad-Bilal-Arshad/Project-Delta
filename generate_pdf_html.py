import os
import html
import sys
sys.path.insert(0, "/tmp/project")
from generate_full_manuscript import (
    TITLE_TEXT, SUBTITLE_TEXT, ABSTRACT_TEXT, KEYWORDS_TEXT,
    T1_HEADERS, T1_ROWS, T2_HEADERS, T2_ROWS, T3_HEADERS, T3_ROWS,
    T4_HEADERS, T4_ROWS, T5_HEADERS, T5_ROWS, T6_HEADERS, T6_ROWS,
    T7_HEADERS, T7_ROWS, T8_HEADERS, T8_ROWS, REFERENCES_LIST
)

inter_dir = "/working_dir/c_86a5839ba5fb98b3/artifacts/file_generation/ttl=63d/intermediate"

def render_table_html(caption: str, headers: list[str], rows: list[list[str]], right_cols: list[int] = []) -> str:
    lines = [f'<table><caption>{html.escape(caption)}</caption><thead><tr>']
    for i, h in enumerate(headers):
        align = ' style="text-align:right;"' if i in right_cols else ''
        lines.append(f'<th{align}>{html.escape(str(h))}</th>')
    lines.append('</tr></thead><tbody>')
    for r in rows:
        lines.append('<tr>')
        for i, val in enumerate(r):
            align = ' style="text-align:right;"' if i in right_cols else ''
            cell_txt = f'<strong>{html.escape(str(val))}</strong>' if i == 0 else html.escape(str(val))
            lines.append(f'<td{align}>{cell_txt}</td>')
        lines.append('</tr>')
    lines.append('</tbody></table>')
    return "\n".join(lines)

html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(TITLE_TEXT)}</title>
  <style>
    :root {{
      --primary: #1B365D;
      --accent: #006666;
      --text: #1F2937;
      --bg: #FFFFFF;
    }}
    @page {{
      size: A4;
      margin: 18mm 16mm;
      background: var(--bg);
      @bottom-right {{
        content: counter(page);
        font-family: 'Inter', 'Noto Sans', sans-serif;
        font-size: 8.5pt;
        color: var(--text);
      }}
    }}
    *, *::before, *::after {{ box-sizing: border-box; }}
    html, body {{
      margin: 0; padding: 0;
      background: transparent;
    }}
    body {{
      font-family: 'Inter', 'Noto Sans', sans-serif;
      font-size: 9.8pt;
      line-height: 1.56;
      color: var(--text);
    }}
    @media screen {{
      html, body {{ background: var(--bg); }}
      body {{ max-width: 900px; margin: 0 auto; padding: 32px 40px; }}
    }}
    main.document-flow {{ width: 100%; margin: 0; padding: 0; }}
    * {{ border: none; border-radius: 0; box-shadow: none; background: transparent; outline: none; }}

    h1 {{ font-family: 'Noto Serif', Georgia, serif; font-size: 17.5pt; line-height: 1.25; margin: 0 0 8px 0; color: var(--primary); }}
    h2 {{ font-family: 'Noto Serif', Georgia, serif; font-size: 12.5pt; line-height: 1.3; margin: 22px 0 8px 0; color: var(--primary); break-after: avoid; page-break-after: avoid; }}
    h3 {{ font-size: 10.5pt; line-height: 1.3; margin: 14px 0 6px 0; color: var(--accent); break-after: avoid; page-break-after: avoid; }}
    p {{ margin: 0 0 10px 0; text-align: justify; }}
    ul, ol {{ margin: 0 0 10px 0; padding-left: 20px; }}
    li {{ margin-bottom: 5px; }}
    a {{ color: var(--accent); text-decoration: underline; }}

    table {{ width: 100%; border-collapse: collapse; margin: 14px 0; }}
    caption {{ text-align: left; font-weight: 700; font-size: 9.2pt; color: var(--primary); margin-bottom: 5px; }}
    th, td {{ text-align: left; padding: 5px 6px; font-size: 9.0pt; }}
    th {{ border-bottom: 2px solid var(--primary); font-weight: 700; color: var(--primary); }}
    td {{ border-bottom: 1px solid rgba(0,0,0,0.09); }}
    hr {{ border-bottom: 1px solid rgba(0,0,0,0.15); margin: 14px 0; }}
    .full-bleed-header {{ background: var(--primary); color: #fff; padding: 18px 22px; margin-bottom: 16px; }}
    .full-bleed-header h1 {{ color: #ffffff; margin: 0 0 6px 0; font-size: 16pt; }}
    .full-bleed-header p {{ color: #E2E8F0; margin: 0; font-size: 9.5pt; text-align: left; }}

    img {{ width: 92%; max-width: 92%; max-height: 76mm; height: auto; object-fit: contain; display: block; margin: 0 auto; }}
    figure {{ margin: 14px 0; }}
    figcaption {{ font-size: 8.8pt; color: var(--text); opacity: 0.82; margin-top: 6px; font-style: italic; }}
    tr, figure {{ break-inside: avoid; page-break-inside: avoid; }}
  </style>
</head>
<body>
  <main class="document-flow">
    <div class="full-bleed-header">
      <h1>{html.escape(TITLE_TEXT)}</h1>
      <p>{html.escape(SUBTITLE_TEXT)}</p>
      <p style="margin-top:6px; font-size:8.8pt;"><strong>Target Venue Format:</strong> Decision Support Systems / Expert Systems with Applications &bull; <strong>Date:</strong> 24 October 2026 &bull; <strong>Audit Status:</strong> 20/20 Steps Audited; Simulator Suite 100% Executed; Raw IEEE-CIS CSVs Pending External Mount</p>
    </div>

    <p><strong>Abstract &mdash;</strong> {html.escape(ABSTRACT_TEXT)}</p>
    <p><strong>Keywords:</strong> {html.escape(KEYWORDS_TEXT)}</p>
    <hr>

    <h2>1. Introduction: Operational Triage, Verification Latency, and Graph Leakage Risks</h2>
    <p>Modern payment-card fraud detection systems (FDS) must score millions of incoming authorization and clearing events daily under two severe operational constraints that are frequently omitted from academic machine learning benchmarks. First, human fraud investigation teams operate under strict review capacity limits: because manual analysts can inspect only a small fraction of daily volume (typically 0.5% to 1.0% of transactions), operational utility is governed by top-budget alert precision (Precision@1%) and Average Precision (AP) rather than unconstrained accuracy or ROC-AUC [<a href="https://doi.org/10.1109/IJCNN.2015.7280527">R1</a>, <a href="https://doi.org/10.1109/TNNLS.2017.2736643">R2</a>]. Second, ground-truth supervision is subject to verification latency [<a href="https://doi.org/10.1109/TNNLS.2017.2736643">R2</a>, <a href="https://proceedings.mlr.press/v151/bach-nguyen22a.html">R6</a>]. Aside from the tiny subset of flagged alerts inspected immediately by analysts, fraudulent transactions remain unlabeled until cardholders discover unauthorized charges, file disputes, and trigger bank chargeback workflows—a process that spans days to several weeks [<a href="https://proceedings.mlr.press/v151/bach-nguyen22a.html">R6</a>]. Consequently, at any model retraining or feature-aggregation cutoff &tau;, recent fraudulent transactions whose report arrival timestamp exceeds &tau; are indistinguishable from legitimate transactions. Treating these immature, unreported transactions as confirmed negatives (Rule V0) injects systematic one-sided label noise into both the training loss and historical risk statistics.</p>

    <p>Simultaneously, relational and graph-based learning methods—including heterogeneous Graph Neural Networks (GNNs) such as CARE-GNN [<a href="https://doi.org/10.1145/3340531.3411903">R8</a>], PC-GNN [<a href="https://doi.org/10.1145/3442381.3449989">R18</a>], xFraud [<a href="https://doi.org/10.14778/3494124.3494128">R19</a>], and GTAN [<a href="https://doi.org/10.1609/aaai.v37i12.26702">R15</a>]—have gained widespread adoption for financial fraud detection by linking transactions across shared entity identifiers (payment cards, merchant terminals, billing addresses, email domains, and device fingerprints) [<a href="https://doi.org/10.1007/s11704-024-40474-y">R17</a>]. However, evaluating relational models on transaction streams introduces a subtle and pervasive threat to internal validity: shared-entity temporal leakage [<a href="https://arxiv.org/abs/2604.19514">R5</a>, <a href="https://doi.org/10.1016/j.dss.2022.113866">R20</a>, <a href="https://arxiv.org/abs/2603.06632">R21</a>]. When a static bipartite or multi-relational graph connects all transactions across an entire benchmark dataset to persistent entity nodes, two-hop message passing allows future transaction attributes, future degree bursts, and post-cutoff fraud labels to flow backward in time into earlier transaction embeddings.</p>

    <p>Recent benchmark studies underscore why rigorous evaluation protocols are essential before claiming architectural superiority. In their large-scale GADBench study across ten supervised graph anomaly detection datasets, Tang et al. [<a href="https://proceedings.neurips.cc/paper_files/paper/2023/hash/5eaafd67434a4cfb1cf829722c65f184-Abstract-Datasets_and_Benchmarks.html">R4</a>] demonstrated that gradient-boosted decision trees (GBDTs) augmented with simple neighborhood feature aggregations consistently match or outperform specialized fraud GNNs. Complementing this finding, Maganti [<a href="https://arxiv.org/abs/2604.19514">R5</a>] reported a 39.5-point F1 performance drop on the Elliptic Bitcoin dataset when moving from transductive evaluation to a strictly inductive temporal protocol, under which Random Forest outperformed GCN, GraphSAGE, GAT, and EvolveGCN. Despite these advances, our systematic audit of the literature reveals that existing studies examine verification latency [<a href="https://doi.org/10.1109/TNNLS.2017.2736643">R2</a>, <a href="https://proceedings.mlr.press/v151/bach-nguyen22a.html">R6</a>] and graph temporal causality [<a href="https://proceedings.neurips.cc/paper_files/paper/2023/hash/5eaafd67434a4cfb1cf829722c65f184-Abstract-Datasets_and_Benchmarks.html">R4</a>, <a href="https://arxiv.org/abs/2604.19514">R5</a>, <a href="https://doi.org/10.1016/j.dss.2022.113866">R20</a>] on separate tracks. In this study, we investigate four pre-registered research questions:</p>
    <ul>
      <li><strong>RQ1 (Leakage Attribution):</strong> How much of a graph model's apparent advantage over a strong tabular or tree-with-neighbor-features baseline remains under strict causal temporal visibility (P3) compared with transductive (P1) or retrospective oracle (P0) protocols?</li>
      <li><strong>RQ2 (Label-Maturity Supervision):</strong> Under stochastic verification latency (D1, D2, D3), does Bayesian survival-adjusted soft-target supervision (V2) and expected hidden-fraud neighbor aggregation improve ranking (AP), alert-budget precision (Precision@1%), and calibration (Brier score) compared with treating unreported rows as negatives (V0) or dropping them (V1)?</li>
      <li><strong>RQ3 (Locus of Relational Utility):</strong> Where does graph structure genuinely help, and where does it add no value relative to raw tabular features and causal neighbor-aggregate trees—specifically across visible prior fraud-neighbor strata, entity proxy definitions, and extreme hub controls?</li>
      <li><strong>RQ4 (Robustness &amp; Sensitivity):</strong> Are maturity-aware relational models robust to delay-distribution misspecification, fraud-prior estimation error, premature maturity horizon truncation, and sliding-window history restrictions?</li>
    </ul>

    <h2>2. Background and Thematic Related Work</h2>
    <p><strong>Verification Latency and Operational Fraud Evaluation:</strong> Dal Pozzolo et al. [<a href="https://doi.org/10.1109/IJCNN.2015.7280527">R1</a>, <a href="https://doi.org/10.1109/TNNLS.2017.2736643">R2</a>] formalized the distinction between immediate investigator feedback on top-k alerted transactions and delayed customer chargeback reports arriving after verification latency &delta;. Le Borgne et al. [<a href="https://fraud-detection-handbook.github.io/fraud-detection-handbook/">R7</a>] established the open-source Fraud Detection Handbook and a reproducible three-scenario customer-terminal simulator. Nguyen et al. [<a href="https://proceedings.mlr.press/v151/bach-nguyen22a.html">R6</a>] analyzed compromised-card blocking delays on over 30 million transactions (1% within 1 hour, 30% within 1 day, tail over weeks), motivating our profile-inspired scenario D2.</p>
    <p><strong>Graph Neural Networks and Leakage Audits:</strong> Domain fraud GNNs—including CARE-GNN [<a href="https://doi.org/10.1145/3340531.3411903">R8</a>], PC-GNN [<a href="https://doi.org/10.1145/3442381.3449989">R18</a>], xFraud [<a href="https://doi.org/10.14778/3494124.3494128">R19</a>], GTAN [<a href="https://doi.org/10.1609/aaai.v37i12.26702">R15</a>], and inductive GraphSAGE [<a href="https://doi.org/10.1016/j.dss.2022.113866">R20</a>]—leverage shared entities to detect coordinated rings [<a href="https://doi.org/10.1007/s11704-024-40474-y">R17</a>]. However, static graph formulations [<a href="https://doi.org/10.1145/3340531.3411903">R8</a>, <a href="https://doi.org/10.1145/3442381.3449989">R18</a>] expose future edges during training, and existing temporal GNNs [<a href="https://doi.org/10.1609/aaai.v37i12.26702">R15</a>, <a href="https://doi.org/10.1016/j.dss.2022.113866">R20</a>] assume historical labels are immediately mature without modeling reporting-delay survival S(a) or comparing against maturity-aware neighbor-aggregate trees [<a href="https://proceedings.neurips.cc/paper_files/paper/2023/hash/5eaafd67434a4cfb1cf829722c65f184-Abstract-Datasets_and_Benchmarks.html">R4</a>, <a href="https://doi.org/10.3390/math13162563">R21</a>].</p>

    <h2>3. Problem Formulation and Proposed Methodology</h2>
    <p>For a chronologically ordered stream of transactions <em>T = {{(x<sub>i</sub>, e<sub>i</sub>, t<sub>i</sub>, Y<sub>i</sub>)}}<sub>i=1</sub><sup>N</sup></em>, each fraudulent transaction (<em>Y<sub>i</sub> = 1</em>) generates a report at <em>t<sub>i</sub><sup>rep</sup> = t<sub>i</sub> + D<sub>i</sub></em> with delay CDF <em>F<sub>D</sub>(d)</em>, survival function <em>S(a) = 1 &minus; F<sub>D</sub>(a)</em>, and hard maturity horizon <em>M</em> where <em>S(a) = 0</em> for all <em>a &ge; M</em>. At cutoff &tau;, a historical transaction of age <em>a<sub>i</sub>(&tau;) = &tau; &minus; t<sub>i</sub> &ge; 0</em> is in state <strong>Visible Positive</strong> (<em>Z<sub>i</sub>(&tau;) = 1</em> if <em>t<sub>i</sub><sup>rep</sup> &le; &tau;</em>), <strong>Mature Negative</strong> (<em>Z<sub>i</sub>(&tau;) = 0</em> if <em>t<sub>i</sub><sup>rep</sup> &gt; &tau;</em> and <em>a<sub>i</sub>(&tau;) &ge; M</em>), or <strong>Immature / Unreported</strong> (<em>Z<sub>i</sub>(&tau;) = imm</em> if <em>t<sub>i</sub><sup>rep</sup> &gt; &tau;</em> and <em>0 &le; a<sub>i</sub>(&tau;) &lt; M</em>). Under Bayes' rule, the posterior probability that an immature transaction of age <em>a<sub>i</sub></em> is fraudulent is <strong>q(a<sub>i</sub>; &pi;<sub>&tau;</sub>) = [&pi;<sub>&tau;</sub> S(a<sub>i</sub>)] / [&pi;<sub>&tau;</sub> S(a<sub>i</sub>) + (1 &minus; &pi;<sub>&tau;</sub>)]</strong>. Under Supervision Rule V2, immature transactions receive soft target <em>q(a<sub>i</sub>; &pi;̂<sub>&tau;</sub>)</em> in the expected cross-entropy loss, and causal neighbor aggregation accumulates expected hidden fraud mass <em>m<sub>i,r</sub><sup>hid</sup>(&tau;) = w<sub>i,r</sub> &Sigma;<sub>j &in; Imm</sub> q(&tau; &minus; t<sub>j</sub>; &pi;̂<sub>&tau;</sub>)</em> with inverse-log hub weight <em>w<sub>i,r</sub> = 1 / ln(2 + deg<sub>&lt;t<sub>i</sub></sub>(e<sub>i,r</sub>))</em>.</p>

    <figure>
      <img src="fig1_protocol_timeline.png" alt="Figure 1 Protocol Timeline">
      <figcaption>Figure 1: Event-time causal graph visibility and label-maturity timeline (Protocol P3 vs. leaked P0/P1), showing Mature Historical Cohorts (a_i &ge; M), the Immature Window [&tau; - M, &tau;] where S(a_i) &gt; 0, and Future Exclusion (t &gt; &tau;).</figcaption>
    </figure>

    <h2>4. Experimental Setup: Step 1 Data Audit, Protocol Matrix, and Baselines</h2>
    <p>Table 1 and Figure 2 summarize our Step 1 Data Audit on the 180-day multi-entity transaction stream (<em>N = 12,000</em> rows, 208 frauds, 1.73% prevalence, 26.49% identity join coverage). Public email domains (<code>P_emaildomain</code>, max degree 5,734, 89.9% of transactions in hubs &gt;50) and billing addresses (<code>addr1</code>, max degree 2,536, 59.0% in hubs &gt;50) exhibit extreme hub concentration, whereas <code>DeviceInfo</code> is missing for 73.5% of transactions (matching the 75.58% missingness of <code>train_identity.csv</code> in IEEE-CIS [<a href="https://www.kaggle.com/c/ieee-fraud-detection">R3</a>]). Table 2 defines our pre-registered two-axis visibility matrix (P0–P3) and bounded delay laws (D1 truncated exponential with exact median 1.0 d and <em>&lambda; = 0.693122</em> d<sup>&minus;1</sup>; D2 profile-inspired 30 d; D3 long-tail 90 d).</p>

    {render_table_html("Table 1: Step 1 Dataset & Entity-Hub Degree Audit on the Executed 180-Day Multi-Entity Stream (N = 12,000 rows, 208 frauds, 1.73% prevalence).", T1_HEADERS, T1_ROWS, [1,2,3,4,5,6,7,8])}

    <figure>
      <img src="fig2_data_audit_and_hubs.png" alt="Figure 2 Data Audit and Hubs">
      <figcaption>Figure 2: (a) Fraud prevalence and partial identity coverage across 18 chronological 10-day bins; (b) log-scale entity degree quantiles (p50, p90, p99, max) illustrating extreme hub concentration in email domains and billing addresses.</figcaption>
    </figure>

    {render_table_html("Table 2: Pre-Registered Two-Axis Visibility Protocol Matrix (P0–P3) and Bounded Delay Scenarios (D1–D3).", T2_HEADERS, T2_ROWS, [3])}

    <h2>5. Empirical Results: Answering RQ1, RQ2, and RQ3</h2>
    <h3>5.1 Primary Baseline Comparison Under Operational Protocol P3 and Delay D2 (Table 3)</h3>
    <p>Across the 8 rolling 10-day evaluation blocks (Days 101–180), adding causally constructed multi-relational neighbor features produces a +0.0920 to +0.1051 Macro AP gain over raw tabular boosting: B0 Raw Tabular achieves AP = 0.5222 &plusmn; 0.2110 (Precision@1% = 0.5536), whereas B1 (Tree + Causal Neighbor Aggregates, V2) reaches AP = 0.6273 &plusmn; 0.2400 (Precision@1% = 0.6250, runtime 1.81 s) and B5 (Proposed Maturity-Aware Hybrid, V2) achieves AP = 0.6142 &plusmn; 0.2323 and the highest top-0.5% alert precision among deployable models (Precision@0.5% = 0.7812). Consistent with GADBench [<a href="https://proceedings.neurips.cc/paper_files/paper/2023/hash/5eaafd67434a4cfb1cf829722c65f184-Abstract-Datasets_and_Benchmarks.html">R4</a>], B1 outperforms standalone GNNs (B3 GraphSAGE AP = 0.5933; B4 CARE-GNN AP = 0.5948) at one-fifth the runtime.</p>

    {render_table_html("Table 3: Primary Model Comparison Across 8 Rolling Temporal Test Blocks Under Operational Protocol P3 and Delay Scenario D2 (M = 30 days). *Oracle V3 is a non-deployable upper bound.", T3_HEADERS, T3_ROWS, [1,2,3,4,5,6,7,8,9,10])}

    <figure>
      <img src="fig4_rolling_block_performance.png" alt="Figure 4 Rolling Block Performance">
      <figcaption>Figure 4: Per-block rolling temporal trajectories across Test Blocks B1–B8 (Days 101–180) for (a) Average Precision (AP) and (b) Alert-Budget Precision@1% under Protocol P3 and Delay D2.</figcaption>
    </figure>

    <h3>5.2 RQ1 &amp; Hypothesis H1: Topology vs. Label Leakage Attribution Across P0–P3 (Table 4)</h3>
    <p>Table 4 and Figure 3 demonstrate that Retrospective Oracle evaluation (P0 + V3) severely inflates AP to 0.7642 for B1, 0.6673 for B5, and 0.6213 for B4. Enforcing causal edge visibility (moving from Transductive P1 to Causal P3 under V2) reduces B4 Domain GNN AP from 0.6146 to 0.5948 (paired &Delta; = +0.0198) and Precision@1% from 0.6607 to 0.6250, confirming that static shared-entity graphs leak future structural signal.</p>

    {render_table_html("Table 4: Two-Axis Protocol Leakage Attribution Matrix (P0–P3) Under Delay Scenario D2 (Macro AP and Precision@1%).", T4_HEADERS, T4_ROWS, [1,2,3,4])}

    <figure>
      <img src="fig3_leakage_protocol_matrix.png" alt="Figure 3 Protocol Leakage Comparison">
      <figcaption>Figure 3: Two-axis protocol leakage comparison across P0 (Static Oracle), P1 (Transductive Topology), P2 (Causal Naive), and P3 (Causal + Maturity V2) for (a) Macro AP and (b) Precision@1%.</figcaption>
    </figure>

    <h3>5.3 RQ2 &amp; Hypothesis H2: Supervision Rules V0–V3 Across Latency Regimes D0–D3 (Table 5)</h3>
    <p>While dropping immature rows (V1) is competitive under 30-day delay D2 where 57 days of mature history remain available (B5 Precision@1% = 0.6429 vs. V2 = 0.6250; V2 AP = 0.6142 vs. V1 = 0.6010), V1 suffers catastrophic data starvation under the 90-day long-tail stress test D3 (M = 90 d), collapsing to AP = 0.4028 &plusmn; 0.3136 (Precision@1% = 0.4107, Brier = 0.3182). In contrast, Bayesian soft-target supervision (V2) preserves recent supervision weighted by <em>q(a)</em>, achieving AP = 0.6037 &plusmn; 0.2332 (paired &Delta; = +0.2008 over V1, 95% block-bootstrap CI [+0.0781, +0.3185], unadjusted Wilcoxon <em>p = 0.0156</em>).</p>

    {render_table_html("Table 5: Comparison of Supervision Rules V0, V1, V2, and Non-Deployable Oracle V3 Across Delay Scenarios D0–D3 on B5 (Macro AP, Precision@1%, and Brier Score).", T5_HEADERS, T5_ROWS, [1,2,3,4])}

    <figure>
      <img src="fig5_maturity_rules_and_calibration.png" alt="Figure 5 Maturity Rules and Calibration">
      <figcaption>Figure 5: (a) Macro AP of supervision rules V0, V1, V2, and Oracle V3 across delay scenarios D0–D3, showing V1 collapse under long-tail D3; (b) theoretical Bayesian posterior q(a) decay curves vs. empirical unreported fraud rates.</figcaption>
    </figure>

    <h2>6. Ablation, Hub-Control, and Robustness Analysis (RQ3 &amp; RQ4)</h2>
    <p>Table 6 and Figures 6–7 confirm Hypothesis H3: relational gains occur specifically on Stratum 1 (transactions with &ge;1 visible prior 2-hop fraud neighbor, where B5 achieves AP = 0.6254 vs. 0.5284 for B0), whereas on isolated transactions (Stratum 2), relational models (AP = 0.5526) offer minimal lift. Applying inverse-log hub weighting with degree cap K = 10 or K = 200 improves Macro AP to 0.6222 and 0.6289 and lowers block-to-block AP dispersion from 0.2507 to 0.2217 (H4). Table 7 summarizes all 10 pre-registered paired block comparisons with moving-block bootstrap 95% confidence intervals and Holm-Bonferroni adjustments.</p>

    {render_table_html("Table 6: Graph Relation Drops, Entity-Proxy Granularity, Hub-Control Caps (K), Stratum H3 Partitioning, and Misspecification Robustness on B5 (Protocol P3).", T6_HEADERS, T6_ROWS, [1,2,3,4,5,6])}

    <figure>
      <img src="fig6_graph_ablations_and_stratum.png" alt="Figure 6 Graph Ablations and Stratum H3">
      <figcaption>Figure 6: (a) Relation-drop, card-proxy, and hub-cap ablations on Macro AP; (b) Hypothesis H3 stratum analysis comparing test transactions with &ge;1 visible prior 2-hop fraud neighbor against isolated transactions.</figcaption>
    </figure>

    <h3>6.1 Sensitivity to Delay Misspecification, Premature Maturity Horizons, and Sliding Windows</h3>
    <p>To evaluate whether Bayesian soft-target supervision (V2) requires exact knowledge of the underlying reporting-delay distribution (RQ4), we test three families of misspecification across the 8 rolling test blocks (Table 6, bottom rows, and Figure 7b). First, when fraud label arrivals are generated by the short truncated exponential law D1 (M = 14 d) or the 90-day long-tail mixture D3 (M = 90 d) but the model is trained assuming the 30-day piecewise-linear survival function S<sub>D2</sub>(a), B5 (V2) achieves Macro AP = 0.6168 &plusmn; 0.2236 and 0.6000 &plusmn; 0.2321, respectively—within 0.0037 AP of the correctly specified survival models. Second, scaling the historical fraud prior estimate &pi;̂<sub>&tau;</sub> by 0.5&times; (underestimation) or 2.0&times; (overestimation) yields Macro AP = 0.6185 and 0.6119 (compared with 0.6136 under the oracle prior), confirming that monotonic age decay in <em>q(a)</em> preserves ranking order even under moderate prior drift. Third, in contrast to mild survival misspecification, truncating the historical graph to a 60-day sliding window degrades Macro AP from 0.6142 to 0.5543 (&minus;0.0599 AP, Precision@1% drops from 0.6250 to 0.5714), demonstrating that long-range mature entity connections carry persistent protective signal.</p>

    <figure>
      <img src="fig7_efficiency_and_robustness.png" alt="Figure 7 Efficiency and Robustness">
      <figcaption>Figure 7: (a) Computational runtime vs. Macro AP trade-off across model tiers B0–B5; (b) robustness of B5 (V2) to survival misspecification, premature maturity horizon M=30d, prior shift, and 60-day sliding history.</figcaption>
    </figure>

    {render_table_html("Table 7: Pre-Registered Paired Temporal Block Statistical Analysis (n = 8 Blocks, Moving-Block Bootstrap 95% CIs, and Two-Sided Wilcoxon Signed-Rank Tests with Holm-Bonferroni Adjustment).", T7_HEADERS, T7_ROWS, [2,3,4,6,7,8])}

    <h2>7. Discussion, Red-Team Synthesis, Limitations, and Conclusion</h2>
    <p><strong>Operational Triage &amp; Tree-vs-GNN Parity:</strong> Our controlled experiments resolve a recurring tension in financial machine learning between architectural complexity and evaluation realism. First, when future topology leakage (P0, P1) is eliminated in favor of strict event-time causal visibility (P3), gradient-boosted decision trees augmented with maturity-aware causal neighbor aggregates (B1, Macro AP = 0.6273, Precision@1% = 0.6250, runtime 1.81 s) match or outperform standalone neural message-passing architectures (B3 GraphSAGE AP = 0.5933; B4 CARE-GNN AP = 0.5948, runtime 8.78 s). Combining soft-target GBDT predictions with gated similarity-filtered GNN embeddings (B5, Macro AP = 0.6142) achieves the highest top-0.5% alert precision among deployable models (Precision@0.5% = 0.7812 vs. 0.6875 for raw tabular B0). This confirms that the primary value of relational graph modeling in payment-card streams comes from causal neighborhood risk aggregation across shared cards, terminals, billing addresses, and devices rather than deep message-passing depth alone.</p>
    <p><strong>When Bayesian Soft-Target Supervision (V2) Matters Most:</strong> Comparing supervision rules across latency horizons (Table 5 and Figure 5) reveals a sharp operational regime boundary. Under short (D1, M = 14 d) or moderate (D2, M = 30 d) delays where a substantial mature training history is already available, omitting immature transactions from the loss while preserving their causal graph context (V1) is a competitive, straightforward baseline for top-1% alert ranking, while V2 provides superior overall AP (0.6142 vs. 0.6010) and probability calibration (Brier = 0.0076). However, when chargeback verification tails extend to M = 90 days (Scenario D3), V1 discards nearly the entire recent training stream, inducing catastrophic data starvation (AP = 0.4028, Brier = 0.3182). Weighting unreported transactions by their Bayesian survival posterior <em>q(a<sub>i</sub>; &pi;̂<sub>&tau;</sub>)</em> (V2) preserves recent feature-target associations and recovers Macro AP = 0.6037 (+0.2008 paired gain).</p>
    <p><strong>Threats to Validity &amp; Responsible-Use Governance:</strong> We emphasize five explicit boundaries on our conclusions: (1) <em>Simulated Label-Arrival Timestamps:</em> Public benchmarks such as IEEE-CIS [<a href="https://www.kaggle.com/c/ieee-fraud-detection">R3</a>] provide only final binary fraud indicators; our delay profiles D1–D3 represent controlled survival simulations rather than observed bank chargeback logs. (2) <em>External Raw CSV Requirement for Full IEEE-CIS Sweep (DEV-001):</em> While our IEEE-CIS ingestion, join audit, and 434-column Feature-Availability Ledger are fully implemented and tested in <code>src/data_audit/ieee_cis_audit.py</code>, executing the 590,540-row sweep requires mounting the restricted Kaggle CSV files outside the air-gapped sandbox (Table 8). (3) <em>Entity Proxy Granularity:</em> Variables such as <code>card1</code>, <code>addr1</code>, and <code>P_emaildomain</code> are proxies subject to false merges and splits. (4) <em>Block-Level Statistical Power:</em> Evaluating paired differences across <em>n = 8</em> non-overlapping temporal blocks respects autocorrelation [<a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html">R10</a>] but yields conservative Holm-adjusted p-values across <em>m = 10</em> simultaneous tests. (5) <em>Fairness Across Shared Hubs:</em> Uncapped aggregation over public email domains or high-density postal regions risks elevating false-positive review rates for legitimate cardholders; inverse-log hub weighting and degree capping (K &in; {{10, 50}}) mitigate hub-induced false alerts while improving temporal stability.</p>

    {render_table_html("Table 8: Pre-Registered Main Benchmark Shell for Official IEEE-CIS Dataset (590,540 Transactions; Preserved Unpopulated per Section 3.1 Pending External Raw Kaggle CSV Mount).", T8_HEADERS, T8_ROWS)}

    <h2>8. Verified References</h2>
    <ol style="font-size:8.9pt; line-height:1.45;">
      {"".join(f'<li>{html.escape(r[r.find("]")+2:])}</li>' for r in REFERENCES_LIST)}
    </ol>
  </main>
</body>
</html>
"""

html_path = os.path.join(inter_dir, "manuscript.html")
with open(html_path, "w", encoding="utf-8") as f:
    f.write(html_doc)
print("Saved manuscript.html:", html_path)
