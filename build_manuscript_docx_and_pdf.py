import os
import json
import docx
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls
from docx.shared import Inches, Pt, RGBColor

inter_dir = "/working_dir/c_86a5839ba5fb98b3/artifacts/file_generation/ttl=63d/intermediate"
out_dir = "/working_dir/c_86a5839ba5fb98b3/artifacts/file_generation/ttl=63d/output"
fig_dir = "/tmp/project/results/figures"
os.makedirs(inter_dir, exist_ok=True)
os.makedirs(out_dir, exist_ok=True)

# 1. Save Phase 1 Content & Layout Plan to draft.md
draft_plan = """# Manuscript Content & Design Plan (Phase 1)

## Title
Time-Respecting Evaluation and Label-Maturity-Aware Learning for Graph-Based Credit Card Fraud Detection

## Section Structure (Matching Master Prompt Phase M & Handbook Section 17)
1. Title, Abstract (Five-Part Structure), and Keywords
2. Section 1: Introduction — Operational Alert Budgets, Verification Latency, and Graph Leakage Risks
3. Section 2: Background and Thematic Related Work — Classical Latency, Fraud GNNs, and Protocol Benchmarks
4. Section 3: Problem Formulation — Event-Time Causal Graph Visibility and Asynchronous Label Maturity
5. Section 4: Proposed Methodology — Bounded Delay Laws, Bayesian Soft-Target Supervision, and Maturity-Aware Neighbor Aggregation
6. Section 5: Experimental Setup — Dataset Audit, Feature-Availability Governance, Rolling Splits, and Two-Axis Protocol Matrix
7. Section 6: Empirical Results — Leakage Attribution (RQ1), Maturity Supervision (RQ2), and Relational Utility (RQ3)
8. Section 7: Ablation, Sensitivity, and Hub-Control Analysis — Degree Caps, Proxy Definitions, and Misspecification Robustness (RQ4)
9. Section 8: Discussion — Operational Triage Implications, Tree-vs-GNN Parity, and Red-Team Objections
10. Section 9: Limitations, Threats to Validity, and Responsible-Use Considerations
11. Section 10: Conclusion and Future Work
12. Section 11: Verified References ([R1]–[R21])
13. Appendix A: IEEE-CIS Real-Data Benchmark Execution Shell and Pre-Registered Protocol Matrix

## Design & Layout Plan
- **Word (.docx) Design**: Path B Structured Technical/Academic Manuscript (`Calibri` 11pt body, 1.15 line spacing, 6pt space after; `10pt` table text floor with `#1B365D` deep navy header row and `#F4F6F9` alternating row shading; 1-inch margins; all 7 figures embedded at 5.8-inch width with 9.5pt italic captions; `w:tblHeader` and `w:cantSplit` on all tables).
- **PDF (.pdf) Design**: Archetype 1 Continuous Flow inside `<main class="document-flow">` with an Archetype 2 Cover Page (`.cover-page` with `page-break-after: always`). Two-tone palette (`--primary: #1B365D`, `--accent: #008080`, `--text: #1F2937`, `--bg: #FFFFFF`), clean booktabs tables (`9.5pt` text floor), and all 7 high-resolution figures embedded.
"""

with open(os.path.join(inter_dir, "draft.md"), "w", encoding="utf-8") as f:
    f.write(draft_plan)

# 2. Define Complete Manuscript Sections & Tables Data
 TITLE_TEXT = "Time-Respecting Evaluation and Label-Maturity-Aware Learning for Graph-Based Credit Card Fraud Detection"
SUBTITLE_TEXT = "A Pre-Registered Two-Axis Leakage Attribution Protocol, Bayesian Survival-Adjusted Supervision Framework, and Controlled Multi-Entity Empirical Study"

ABSTRACT_TEXT = (
    "Payment-card fraud detection systems (FDS) operate under strict event-time information boundaries and asynchronous verification latency: "
    "at the moment a transaction is scored, future network interactions are unobserved, and historical transactions whose chargeback or customer-dispute "
    "windows have not yet elapsed remain in an immature, unreported state. Standard graph neural network (GNN) benchmarks routinely violate both constraints "
    "by constructing static multi-relational graphs across entire observation periods and treating final retrospective fraud indicators as immediately known. "
    "Conducting a systematic review of 15 foundational and recent studies (2015–2026), we identify a critical methodological gap: prior work addresses either "
    "tabular verification latency or static/inductive graph learning in isolation, without orthogonalizing shared-entity topology leakage from label-maturity censoring. "
    "To resolve this gap, we formulate (C1) a four-cell, two-axis evaluation protocol (P0–P3) that independently controls future topology visibility and label-derived input access; "
    "(C2) a Bayesian survival-adjusted supervision rule (V2) that assigns age-dependent posterior soft targets q(a) = πS(a)/[πS(a) + (1-π)] to unreported transactions "
    "and aggregates expected hidden-fraud mass across causally prior two-hop neighborhoods under inverse-log hub weighting; and (C3) a pre-registered 8-block rolling temporal "
    "evaluation across six model tiers (B0–B5) and four bounded latency scenarios (D0–D3). Across N = 12,000 transactions spanning 180 chronological days in our reproducible "
    "multi-entity extension of the Fraud Detection Handbook simulator (208 frauds, 1.73% positive prevalence, 26.49% identity coverage), retrospective oracle evaluation (P0) "
    "inflates Average Precision (AP) to 0.7642 for causal neighbor-aggregate trees (B1) and 0.6213 for domain fraud GNNs (B4, CARE-GNN), whereas enforcing strict event-time "
    "causal visibility (P3) reduces B4 AP to 0.5948 (paired Δ = +0.0265, 95% moving-block bootstrap CI [+0.0013, +0.0429]). Under causal visibility (P3) and 30-day profile latency (D2), "
    "gradient-boosted trees with maturity-aware causal neighbor aggregates (B1, AP = 0.6273 ± 0.2400, Precision@1% = 0.6250, runtime 1.81 s) and our maturity-aware hybrid (B5, AP = 0.6142 ± 0.2323, "
    "Precision@0.5% = 0.7812, Brier = 0.0076) substantially outperform raw tabular boosting (B0, AP = 0.5222 ± 0.2110, Δ = +0.0920) while matching or outperforming standalone GNN message passing "
    "(B3 GraphSAGE AP = 0.5933; B4 CARE-GNN AP = 0.5948) at one-fifth the computational cost. Crucially, while dropping immature examples (V1) is competitive under moderate 30-day delays "
    "(B5 Precision@1% = 0.6429 vs. V2 = 0.6250), V1 suffers catastrophic data starvation under 90-day long-tail chargeback latency (D3, AP = 0.4028, Brier = 0.3182), where Bayesian soft-target "
    "supervision (V2) preserves ranking and calibration (AP = 0.6037, paired Δ = +0.2008, 95% CI [+0.0781, +0.3185], unadjusted Wilcoxon p = 0.0156). Stratum partitioning proves that relational "
    "gains occur specifically when a scored transaction shares at least one causally prior, visibly reported fraud neighbor within two hops (AP = 0.6254 vs. 0.5526 isolated). "
    "We provide a complete open-source reproducibility package and an audited feature-availability and execution harness for the 590,540-row IEEE-CIS benchmark, noting explicitly that "
    "simulated delay profiles test algorithmic mechanisms under stated survival assumptions and require raw Kaggle CSV mounting for full real-benchmark validation."
)

KEYWORDS_TEXT = (
    "Credit card fraud detection; Graph neural networks; Verification latency; Label maturity; "
    "Temporal data leakage; Gradient boosted decision trees; Bayesian soft-target learning; Operational alert budget"
)

# Helper to build the Word document (.docx)
doc = docx.Document()
PRIMARY_HEX = "1B365D"
ALT_ROW_HEX = "F2F5F9"
CALLOUT_HEX = "EBF3F5"
TEXT_DARK = RGBColor(0x1F, 0x29, 0x37)
NAVY_RGB = RGBColor(0x1B, 0x36, 0x5D)
TEAL_RGB = RGBColor(0x00, 0x66, 0x66)

for section in doc.sections:
    section.top_margin = Inches(1.0)
    section.bottom_margin = Inches(1.0)
    section.left_margin = Inches(1.0)
    section.right_margin = Inches(1.0)
    footer = section.footer
    fp = footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    fr = fp.add_run("Time-Respecting, Label-Maturity-Aware Graph Fraud Detection | Manuscript & Reproducible Evidence")
    fr.font.name = "Calibri"
    fr.font.size = Pt(9.5)
    fr.font.color.rgb = RGBColor(0x64, 0x74, 0x8B)

style_normal = doc.styles["Normal"]
style_normal.font.name = "Calibri"
style_normal.font.size = Pt(11)
style_normal.font.color.rgb = TEXT_DARK

def add_title_block():
    tp = doc.add_paragraph()
    tp.alignment = WD_ALIGN_PARAGRAPH.LEFT
    tr = tp.add_run(TITLE_TEXT)
    tr.font.name = "Calibri"
    tr.font.size = Pt(20)
    tr.font.bold = True
    tr.font.color.rgb = NAVY_RGB
    tp.paragraph_format.space_after = Pt(6)

    sp = doc.add_paragraph()
    sr = sp.add_run(SUBTITLE_TEXT)
    sr.font.name = "Calibri"
    sr.font.size = Pt(12)
    sr.font.italic = True
    sr.font.color.rgb = TEAL_RGB
    sp.paragraph_format.space_after = Pt(10)

    mp = doc.add_paragraph()
    mr = mp.add_run("Document Type: Peer-Reviewed Journal Manuscript Draft (Decision Support Systems / Expert Systems with Applications Format) • Execution Date: 24 October 2026 • Reproducibility Status: All Reported Simulator Metrics & Figures Deterministically Verified; Full 590,540-Row IEEE-CIS Sweep Pending External Raw CSV Mount")
    mr.font.size = Pt(10)
    mr.font.bold = True
    mr.font.color.rgb = RGBColor(0x47, 0x55, 0x69)
    mp.paragraph_format.space_after = Pt(12)

def add_h1(text: str):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.font.name = "Calibri"
    r.font.size = Pt(15)
    r.font.bold = True
    r.font.color.rgb = NAVY_RGB
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True

def add_h2(text: str):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.font.name = "Calibri"
    r.font.size = Pt(12.5)
    r.font.bold = True
    r.font.color.rgb = TEAL_RGB
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.keep_with_next = True

def add_p(text: str, bold_prefix: str = ""):
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(6)
    if bold_prefix:
        rb = p.add_run(bold_prefix)
        rb.font.size = Pt(11)
        rb.font.bold = True
        rb.font.color.rgb = NAVY_RGB
    r = p.add_run(text)
    r.font.size = Pt(11)
    r.font.color.rgb = TEXT_DARK
    return p

def add_equation_box(eq_text: str):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.rows[0].cells[0]
    shd = parse_xml(r'<w:shd {} w:fill="{}"/>'.format(nsdecls("w"), CALLOUT_HEX))
    cell._tc.get_or_add_tcPr().append(shd)
    trPr = tbl.rows[0]._tr.get_or_add_trPr()
    trPr.append(parse_xml(r'<w:cantSplit {}/>'.format(nsdecls("w"))))
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(eq_text)
    r.font.name = "Cambria"
    r.font.size = Pt(10.5)
    r.font.bold = True
    r.font.color.rgb = NAVY_RGB
    sp = doc.add_paragraph()
    sp.paragraph_format.space_after = Pt(4)

def add_styled_table(caption_text: str, headers: list[str], rows: list[list[str]], right_align_cols: list[int] = []):
    cp = doc.add_paragraph()
    cp.paragraph_format.space_before = Pt(8)
    cp.paragraph_format.space_after = Pt(3)
    cp.paragraph_format.keep_with_next = True
    cr = cp.add_run(caption_text)
    cr.font.size = Pt(10)
    cr.font.bold = True
    cr.font.color.rgb = NAVY_RGB

    tbl = doc.add_table(rows=1, cols=len(headers))
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER

    hdr_row = tbl.rows[0]
    hdr_tr = hdr_row._tr.get_or_add_trPr()
    hdr_tr.append(parse_xml(r'<w:tblHeader {}/>'.format(nsdecls("w"))))
    hdr_tr.append(parse_xml(r'<w:cantSplit {}/>'.format(nsdecls("w"))))

    for idx, h_text in enumerate(headers):
        cell = hdr_row.cells[idx]
        cell.text = str(h_text)
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        shd = parse_xml(r'<w:shd {} w:fill="{}"/>'.format(nsdecls("w"), PRIMARY_HEX))
        cell._tc.get_or_add_tcPr().append(shd)
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.RIGHT if idx in right_align_cols else WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(3)
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.keep_with_next = True
            for run in p.runs:
                run.font.name = "Calibri"
                run.font.size = Pt(10)
                run.font.bold = True
                run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    for r_i, r_vals in enumerate(rows):
        row = tbl.add_row()
        trPr = row._tr.get_or_add_trPr()
        trPr.append(parse_xml(r'<w:cantSplit {}/>'.format(nsdecls("w"))))
        for c_i, val in enumerate(r_vals):
            cell = row.cells[c_i]
            cell.text = str(val)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            if r_i % 2 == 1:
                shd = parse_xml(r'<w:shd {} w:fill="{}"/>'.format(nsdecls("w"), ALT_ROW_HEX))
                cell._tc.get_or_add_tcPr().append(shd)
            for p in cell.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT if c_i in right_align_cols else WD_ALIGN_PARAGRAPH.LEFT
                p.paragraph_format.space_before = Pt(2.5)
                p.paragraph_format.space_after = Pt(2.5)
                if len(rows) <= 5 and r_i < len(rows) - 1:
                    p.paragraph_format.keep_with_next = True
                for run in p.runs:
                    run.font.name = "Calibri"
                    run.font.size = Pt(10)
                    run.font.bold = (c_i == 0)
                    run.font.color.rgb = TEXT_DARK

    sp = doc.add_paragraph()
    sp.paragraph_format.space_after = Pt(4)

def add_figure(img_filename: str, caption_text: str):
    img_path = os.path.join(fig_dir, img_filename)
    if not os.path.isfile(img_path):
        return
    ip = doc.add_paragraph()
    ip.alignment = WD_ALIGN_PARAGRAPH.CENTER
    ip.paragraph_format.space_before = Pt(6)
    ip.paragraph_format.space_after = Pt(2)
    ip.paragraph_format.keep_with_next = True
    ip.add_run().add_picture(img_path, width=Inches(6.0))

    cp = doc.add_paragraph()
    cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cp.paragraph_format.space_after = Pt(8)
    cr = cp.add_run(caption_text)
    cr.font.name = "Calibri"
    cr.font.size = Pt(9.5)
    cr.font.italic = True
    cr.font.color.rgb = RGBColor(0x47, 0x55, 0x69)

print("Helper functions initialized.")
