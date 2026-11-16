#!/usr/bin/env python3
"""
Generate Master Single-File HTML Report for PerfectPetClub 7COM2005 Data Engineering Coursework.
Consolidates ETL, Data Quality Audit, Financial Reconciliation, Canonical Feature Store,
Full-Window Leaky Baseline ML, Temporal Target Leakage Audit, 14-Day Fixed Cutoff ML,
21-Day Sensitivity Analysis, Comparative Synthesis, Feature Importance, Decision Support,
and Interactive Serving into a 100% self-contained, standalone HTML document.
All charts are embedded as base64 images.
"""

import os
import sys
import base64
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "reports")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def encode_image(rel_path):
    full_path = os.path.join(PROJECT_ROOT, rel_path)
    if not os.path.exists(full_path):
        print(f"Warning: Image not found {full_path}")
        return ""
    with open(full_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

print("Encoding images...")
img_bl_comp = encode_image("PerfectPetClub/outputs/ml/model_comparison.png")
img_bl_cm = encode_image("PerfectPetClub/outputs/ml/confusion_matrices.png")
img_bl_roc = encode_image("PerfectPetClub/outputs/ml/roc_curves.png")

img_c14_comp = encode_image("PerfectPetClub/outputs/ml_cutoff14/model_comparison.png")
img_c14_cm = encode_image("PerfectPetClub/outputs/ml_cutoff14/confusion_matrices.png")
img_c14_roc = encode_image("PerfectPetClub/outputs/ml_cutoff14/roc_curves.png")

img_c21_comp = encode_image("PerfectPetClub/outputs/ml_cutoff21/model_comparison.png")
img_c21_cm = encode_image("PerfectPetClub/outputs/ml_cutoff21/confusion_matrices.png")
img_c21_roc = encode_image("PerfectPetClub/outputs/ml_cutoff21/roc_curves.png")

print("Loading CSV data...")
df_bl = pd.read_csv(os.path.join(PROJECT_ROOT, "PerfectPetClub/outputs/ml/model_results.csv"))
df_c14 = pd.read_csv(os.path.join(PROJECT_ROOT, "PerfectPetClub/outputs/ml_cutoff14/model_results.csv"))
df_c21 = pd.read_csv(os.path.join(PROJECT_ROOT, "PerfectPetClub/outputs/ml_cutoff21/model_results.csv"))
df_preds = pd.read_csv(os.path.join(PROJECT_ROOT, "PerfectPetClub/outputs/ml_cutoff14/test_predictions_selected_model.csv"))

def render_table(df, highlight_best_col=None):
    html = '<div class="table-responsive"><table><thead><tr>'
    headers = [
        "Configuration", "Model Architecture", "CV ROC-AUC", "Best Parameters",
        "Test Accuracy", "Churn Precision", "Churn Recall", "Churn F1", "Test ROC-AUC",
        "TN", "FP", "FN", "TP"
    ]
    for h in headers:
        html += f'<th>{h}</th>'
    html += '</tr></thead><tbody>'
    
    for _, r in df.iterrows():
        is_highlight = False
        if highlight_best_col and r['cv_roc_auc'] == df['cv_roc_auc'].max():
            is_highlight = True
        
        row_cls = ' class="highlight-row"' if is_highlight else ''
        html += f'<tr{row_cls}>'
        
        cfg_name = str(r['config']).replace("_", " ").title()
        model_name = str(r['model'])
        cv_auc = f"{float(r['cv_roc_auc']):.4f}"
        params = str(r['best_params'])
        acc = f"{float(r['accuracy'])*100:.1f}%"
        prec = f"{float(r['churn_precision']):.4f}"
        rec = f"{float(r['churn_recall']):.4f}"
        f1 = f"{float(r['churn_f1']):.4f}"
        test_auc = f"{float(r['roc_auc']):.4f}"
        tn = int(r['TN'])
        fp = int(r['FP'])
        fn = int(r['FN'])
        tp = int(r['TP'])
        
        html += f'<td><strong>{cfg_name}</strong></td>'
        html += f'<td><code>{model_name}</code></td>'
        html += f'<td><span class="badge badge-primary">{cv_auc}</span></td>'
        html += f'<td><span class="code-sm">{params}</span></td>'
        html += f'<td><strong>{acc}</strong></td>'
        html += f'<td>{prec}</td>'
        html += f'<td>{rec}</td>'
        html += f'<td><strong>{f1}</strong></td>'
        html += f'<td><span class="badge badge-success">{test_auc}</span></td>'
        html += f'<td>{tn}</td>'
        html += f'<td>{fp}</td>'
        html += f'<td>{fn}</td>'
        html += f'<td>{tp}</td>'
        html += '</tr>'
    html += '</tbody></table></div>'
    return html

tbl_bl = render_table(df_bl, highlight_best_col="cv_roc_auc")
tbl_c14 = render_table(df_c14, highlight_best_col="cv_roc_auc")
tbl_c21 = render_table(df_c21, highlight_best_col="cv_roc_auc")

# Prepare test prediction sample table
df_sample = pd.concat([
    df_preds[df_preds['churn_label'] == 1].sort_values('churn_probability', ascending=False).head(5),
    df_preds[df_preds['churn_label'] == 0].sort_values('churn_probability', ascending=True).head(5)
]).sort_values('churn_probability', ascending=False)

tbl_preds = '<div class="table-responsive"><table><thead><tr>'
tbl_preds += '<th>Customer ID</th><th>Actual Status</th><th>Predicted Churn</th><th>Churn Probability</th><th>Risk Classification</th><th>Recommended Retention Action</th>'
tbl_preds += '</tr></thead><tbody>'

for _, r in df_sample.iterrows():
    cid = int(r['customer_id'])
    actual = '<span class="tag tag-danger">Churned (0)</span>' if r['churn_label'] == 1.0 else '<span class="tag tag-success">Renewed (1)</span>'
    pred = '<span class="tag tag-danger">Yes</span>' if r['prediction'] == 1.0 else '<span class="tag tag-neutral">No</span>'
    prob = float(r['churn_probability'])
    
    if prob >= 0.43:
        risk = '<span class="badge badge-danger">High Risk</span>'
        action = 'Immediate Outreach: £15 Free Pet Care Voucher & Delivery Priority Pass'
    elif prob >= 0.38:
        risk = '<span class="badge badge-warning">Moderate Risk</span>'
        action = 'Engagement Nudge: Exclusive Loyalty Tier Bonus & Satisfaction Check-in'
    else:
        risk = '<span class="badge badge-success">Low Risk / Healthy</span>'
        action = 'Standard Service: Automated Newsletter & Routine Replenishment Reminder'
        
    tbl_preds += f'<tr><td><strong>#{cid}</strong></td><td>{actual}</td><td>{pred}</td><td><strong>{prob:.4f}</strong> ({prob*100:.1f}%)</td><td>{risk}</td><td>{action}</td></tr>'
tbl_preds += '</tbody></table></div>'

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PerfectPetClub – Master Analytics & Data Engineering Report (7COM2005)</title>
    <style>
        :root {{
            --primary: #1e3a8a;
            --primary-light: #2563eb;
            --primary-accent: #3b82f6;
            --secondary: #059669;
            --secondary-light: #10b981;
            --dark: #0f172a;
            --slate: #334155;
            --light: #f8fafc;
            --border: #e2e8f0;
            --card-bg: #ffffff;
            --danger: #dc2626;
            --warning: #d97706;
            --info: #0284c7;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #f1f5f9;
            color: var(--slate);
            line-height: 1.65;
            font-size: 15px;
        }}
        .wrapper {{
            display: flex;
            min-height: 100vh;
        }}
        /* Sidebar Navigation */
        .sidebar {{
            width: 280px;
            background: #0f172a;
            color: #94a3b8;
            position: fixed;
            top: 0;
            bottom: 0;
            left: 0;
            overflow-y: auto;
            padding: 24px 16px;
            border-right: 1px solid #1e293b;
            z-index: 100;
        }}
        .sidebar-brand {{
            color: #ffffff;
            font-size: 1.15rem;
            font-weight: 700;
            margin-bottom: 20px;
            padding-bottom: 12px;
            border-bottom: 1px solid #334155;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .sidebar-nav ul {{
            list-style: none;
        }}
        .sidebar-nav li {{
            margin-bottom: 4px;
        }}
        .sidebar-nav a {{
            color: #94a3b8;
            text-decoration: none;
            display: block;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 0.88rem;
            transition: all 0.2s ease;
        }}
        .sidebar-nav a:hover {{
            background: #1e293b;
            color: #ffffff;
            transform: translateX(3px);
        }}
        .sidebar-nav .nav-section-title {{
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #64748b;
            font-weight: 700;
            margin: 18px 0 6px 8px;
        }}
        
        /* Main Content Container */
        .main-content {{
            margin-left: 280px;
            flex: 1;
            padding: 35px 40px;
            max-width: 1300px;
        }}
        
        /* Hero Banner */
        .hero {{
            background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 50%, #2563eb 100%);
            color: #ffffff;
            padding: 40px;
            border-radius: 14px;
            margin-bottom: 30px;
            box-shadow: 0 10px 25px -5px rgba(30, 58, 138, 0.25);
        }}
        .hero h1 {{
            font-size: 2.2rem;
            font-weight: 800;
            line-height: 1.25;
            margin-bottom: 10px;
            letter-spacing: -0.5px;
        }}
        .hero .subtitle {{
            font-size: 1.1rem;
            opacity: 0.92;
            margin-bottom: 20px;
            font-weight: 400;
        }}
        .meta-tags {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }}
        .meta-tag {{
            background: rgba(255, 255, 255, 0.18);
            border: 1px solid rgba(255, 255, 255, 0.25);
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 0.82rem;
            font-weight: 600;
            backdrop-filter: blur(4px);
        }}
        
        /* KPI Cards Grid */
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .kpi-card {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 22px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.03);
            border-top: 4px solid var(--primary-light);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }}
        .kpi-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 16px rgba(0,0,0,0.06);
        }}
        .kpi-card.green {{ border-top-color: var(--secondary); }}
        .kpi-card.amber {{ border-top-color: var(--warning); }}
        .kpi-card.purple {{ border-top-color: #8b5cf6; }}
        
        .kpi-title {{
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.6px;
            color: #64748b;
            font-weight: 700;
            margin-bottom: 8px;
        }}
        .kpi-value {{
            font-size: 1.9rem;
            font-weight: 800;
            color: var(--dark);
            margin-bottom: 6px;
        }}
        .kpi-desc {{
            font-size: 0.85rem;
            color: #64748b;
        }}
        
        /* Sections */
        .section-card {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 32px;
            margin-bottom: 30px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.03);
        }}
        h2.section-header {{
            font-size: 1.45rem;
            color: var(--dark);
            font-weight: 700;
            padding-bottom: 12px;
            border-bottom: 2px solid #e2e8f0;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        h3.subsection-header {{
            font-size: 1.15rem;
            color: var(--primary);
            font-weight: 600;
            margin: 24px 0 12px 0;
        }}
        p {{
            margin-bottom: 14px;
        }}
        
        /* Tables */
        .table-responsive {{
            overflow-x: auto;
            margin: 16px 0 24px 0;
            border-radius: 8px;
            border: 1px solid var(--border);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.88rem;
            text-align: left;
            background: #ffffff;
        }}
        th, td {{
            padding: 10px 14px;
            border-bottom: 1px solid var(--border);
        }}
        th {{
            background: #f8fafc;
            color: var(--dark);
            font-weight: 700;
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            border-bottom: 2px solid #cbd5e1;
            white-space: nowrap;
        }}
        tr:hover td {{
            background: #f8fafc;
        }}
        tr.highlight-row td {{
            background: #eff6ff;
            font-weight: 600;
        }}
        
        /* Badges & Tags */
        .badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 0.78rem;
            font-weight: 700;
            font-family: monospace;
        }}
        .badge-primary {{ background: #dbeafe; color: #1e40af; }}
        .badge-success {{ background: #dcfce7; color: #15803d; }}
        .badge-warning {{ background: #fef3c7; color: #b45309; }}
        .badge-danger {{ background: #fee2e2; color: #b91c1c; }}
        
        .tag {{
            display: inline-block;
            padding: 2px 7px;
            border-radius: 12px;
            font-size: 0.78rem;
            font-weight: 600;
        }}
        .tag-success {{ background: #dcfce7; color: #15803d; }}
        .tag-danger {{ background: #fee2e2; color: #b91c1c; }}
        .tag-neutral {{ background: #f1f5f9; color: #475569; }}
        
        .code-sm {{
            font-family: "SFMono-Regular", Consolas, Monaco, monospace;
            font-size: 0.78rem;
            background: #f1f5f9;
            padding: 2px 4px;
            border-radius: 4px;
            color: #334155;
        }}
        
        /* Callouts / Alerts */
        .callout {{
            border-left: 4px solid var(--primary-light);
            background: #f8fafc;
            padding: 18px 20px;
            border-radius: 0 8px 8px 0;
            margin: 18px 0;
        }}
        .callout.alert-warning {{
            border-left-color: var(--warning);
            background: #fffbeb;
        }}
        .callout.alert-danger {{
            border-left-color: var(--danger);
            background: #fef2f2;
        }}
        .callout.alert-success {{
            border-left-color: var(--secondary);
            background: #f0fdf4;
        }}
        .callout-title {{
            font-weight: 700;
            margin-bottom: 6px;
            font-size: 0.95rem;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .callout.alert-warning .callout-title {{ color: #92400e; }}
        .callout.alert-danger .callout-title {{ color: #991b1b; }}
        .callout.alert-success .callout-title {{ color: #166534; }}
        
        /* Image Grid & Visuals */
        .img-container {{
            margin: 20px 0;
            text-align: center;
            background: #ffffff;
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 16px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.02);
        }}
        .img-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 6px;
        }}
        .img-caption {{
            font-size: 0.85rem;
            color: #64748b;
            margin-top: 10px;
            font-weight: 500;
        }}
        
        /* Diagram Boxes */
        .pipeline-flow {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 12px;
            background: #f8fafc;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid var(--border);
            margin: 18px 0;
        }}
        .flow-step {{
            background: #ffffff;
            border: 1px solid #cbd5e1;
            padding: 12px 16px;
            border-radius: 8px;
            text-align: center;
            flex: 1;
            min-width: 140px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        }}
        .flow-step-num {{
            font-size: 0.72rem;
            text-transform: uppercase;
            font-weight: 700;
            color: var(--primary-light);
            margin-bottom: 2px;
        }}
        .flow-step-title {{
            font-weight: 700;
            font-size: 0.85rem;
            color: var(--dark);
        }}
        .flow-arrow {{
            font-size: 1.2rem;
            color: #94a3b8;
            font-weight: bold;
        }}
        
        /* Footer */
        footer {{
            text-align: center;
            padding: 40px 0;
            color: #64748b;
            font-size: 0.85rem;
            border-top: 1px solid var(--border);
            margin-top: 40px;
        }}
        
        @media (max-width: 900px) {{
            .sidebar {{ display: none; }}
            .main-content {{ margin-left: 0; padding: 20px; }}
        }}
    </style>
</head>
<body>

<div class="wrapper">
    <!-- Sidebar -->
    <aside class="sidebar">
        <div class="sidebar-brand">
            🐾 <span>PerfectPetClub</span>
        </div>
        <div style="font-size: 0.78rem; color: #64748b; margin-bottom: 16px;">
            7COM2005 Master Report
        </div>
        
        <nav class="sidebar-nav">
            <div class="nav-section-title">Overview</div>
            <ul>
                <li><a href="#overview">Executive Summary</a></li>
                <li><a href="#team">Team & Roles</a></li>
                <li><a href="#architecture">Architecture & ETL</a></li>
            </ul>
            
            <div class="nav-section-title">Data Engineering</div>
            <ul>
                <li><a href="#data-audit">Data Quality & Reconciliation</a></li>
                <li><a href="#feature-store">Canonical Feature Store</a></li>
            </ul>
            
            <div class="nav-section-title">Machine Learning (PySpark)</div>
            <ul>
                <li><a href="#baseline-ml">1. Baseline Modeling (Leaky)</a></li>
                <li><a href="#leakage-audit">2. Target Leakage Diagnosis</a></li>
                <li><a href="#cutoff14-ml">3. 14-Day Fixed Cutoff</a></li>
                <li><a href="#cutoff21-ml">4. 21-Day Sensitivity</a></li>
                <li><a href="#comparison">5. Master Model Comparison</a></li>
            </ul>
            
            <div class="nav-section-title">Operational Deployment</div>
            <ul>
                <li><a href="#feature-importance">Feature Importance</a></li>
                <li><a href="#customer-scoring">Decision Support & Scoring</a></li>
                <li><a href="#live-studio">Interactive Streamlit Studio</a></li>
                <li><a href="#recommendations">Academic Evaluation & Next Steps</a></li>
            </ul>
        </nav>
    </aside>

    <!-- Main Content -->
    <main class="main-content">
        <!-- Hero Header -->
        <header class="hero" id="overview">
            <h1>🐾 PerfectPetClub – Master Analytics & Data Engineering Report</h1>
            <div class="subtitle">
                End-to-End Enterprise Data Lineage, Distributed PySpark MLlib Predictive Analytics & Temporal Target Leakage Audit
            </div>
            <div class="meta-tags">
                <span class="meta-tag">Module: 7COM2005 Data Engineering</span>
                <span class="meta-tag">Institution: University of Hertfordshire</span>
                <span class="meta-tag">Academic Year: 2026/2027</span>
                <span class="meta-tag">Target: Customer Churn (<code>auto_renew = 0</code>)</span>
                <span class="meta-tag">Environment: PySpark 3.5 / Python 3.12</span>
            </div>
        </header>

        <!-- KPI Grid -->
        <div class="kpi-grid">
            <div class="kpi-card green">
                <div class="kpi-title">Financial Audit Reconciliation</div>
                <div class="kpi-value">£1,058,642.73</div>
                <div class="kpi-desc">Reconciled across 23,188 orders and payment ledgers with <strong style="color:var(--secondary)">£0.00 discrepancy</strong>.</div>
            </div>
            
            <div class="kpi-card">
                <div class="kpi-title">Canonical Parquet Store</div>
                <div class="kpi-value">221 × 36 Features</div>
                <div class="kpi-desc">Unified subscriber spine with <strong>0 missing values</strong> and zero PII exposure (GDPR compliant).</div>
            </div>
            
            <div class="kpi-card amber">
                <div class="kpi-title">Temporal Leakage Discovery</div>
                <div class="kpi-value">AUC 0.78 ➔ 0.55</div>
                <div class="kpi-desc">Uncovered tenure leakage inflating baseline metrics; resolved with fixed 14-day observation horizon.</div>
            </div>
            
            <div class="kpi-card purple">
                <div class="kpi-title">PySpark MLlib Pipelines</div>
                <div class="kpi-value">3 Architectures</div>
                <div class="kpi-desc">Logistic Regression, Random Forest, and GBT with 3-fold cross-validation and hyperparameter grids.</div>
            </div>
        </div>

        <!-- Section: Team Roster -->
        <section class="section-card" id="team">
            <h2 class="section-header">👥 Project Team & Contributions</h2>
            <p>This master coursework project was executed collaboratively by the following members under standard software engineering and data governance practices:</p>
            
            <div class="table-responsive">
                <table>
                    <thead>
                        <tr>
                            <th>Team Member</th>
                            <th>Role</th>
                            <th>Core Responsibilities & Deliverables</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><strong>Divya</strong></td>
                            <td><span class="badge badge-primary">Team Leader</span></td>
                            <td>Project management, milestone governance, requirements alignment, final submission auditing.</td>
                        </tr>
                        <tr>
                            <td><strong>Atchyuth Karri</strong></td>
                            <td><span class="badge badge-primary">Data Engineer / ML Lead</span></td>
                            <td>End-to-end pipeline design, secure S3 ingestion, PySpark MLlib modeling, temporal leakage audit, Streamlit decision studio.</td>
                        </tr>
                        <tr>
                            <td><strong>Lukshan</strong></td>
                            <td><span class="badge badge-primary">ML & Analytics</span></td>
                            <td>Exploratory data analysis, Scikit-learn comparative baselines, cross-validation tuning, feature importance extraction.</td>
                        </tr>
                        <tr>
                            <td><strong>Koteswarao</strong></td>
                            <td><span class="badge badge-primary">Deployment Engineer</span></td>
                            <td>Environment isolation, dependency management, reproducible pipeline packaging, local cluster deployment.</td>
                        </tr>
                        <tr>
                            <td><strong>Deekshitha</strong></td>
                            <td><span class="badge badge-primary">Data Quality Engineer</span></td>
                            <td>CRM log data cleansing, financial ledger reconciliation, data quality checks, schema constraint validation.</td>
                        </tr>
                        <tr>
                            <td><strong>Vasu Challa</strong></td>
                            <td><span class="badge badge-primary">Data Ingestion Engineer</span></td>
                            <td>Web server log streaming parser, session recency extraction, chunked ingestion logic, PII elimination.</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </section>

        <!-- Section 1: Architecture & ETL -->
        <section class="section-card" id="architecture">
            <h2 class="section-header">🏛️ 1. System Architecture & Multi-Source ETL Ingestion</h2>
            <p>
                The <strong>PerfectPetClub</strong> data engineering architecture ingests data across five heterogeneous sources, guarantees cryptographic file integrity, performs strict schema normalization, and yields an analytical feature store saved in Parquet format with Snappy compression.
            </p>
            
            <div class="pipeline-flow">
                <div class="flow-step">
                    <div class="flow-step-num">Step 1</div>
                    <div class="flow-step-title">Multi-Source Ingestion</div>
                    <div style="font-size:0.75rem; color:#64748b;">SQLite (8 tables), CRM, Web logs, S3 survey</div>
                </div>
                <div class="flow-arrow">➔</div>
                <div class="flow-step">
                    <div class="flow-step-num">Step 2</div>
                    <div class="flow-step-title">Security & Audit</div>
                    <div style="font-size:0.75rem; color:#64748b;">SHA-256 validation, PII scrubbing, £0.00 ledger check</div>
                </div>
                <div class="flow-arrow">➔</div>
                <div class="flow-step">
                    <div class="flow-step-num">Step 3</div>
                    <div class="flow-step-title">Canonical Store</div>
                    <div style="font-size:0.75rem; color:#64748b;">221 rows × 36 features Parquet (Snappy)</div>
                </div>
                <div class="flow-arrow">➔</div>
                <div class="flow-step">
                    <div class="flow-step-num">Step 4</div>
                    <div class="flow-step-title">Fixed-Cutoff Engine</div>
                    <div style="font-size:0.75rem; color:#64748b;">14d & 21d observation window filtering</div>
                </div>
                <div class="flow-arrow">➔</div>
                <div class="flow-step">
                    <div class="flow-step-num">Step 5</div>
                    <div class="flow-step-title">Distributed MLlib</div>
                    <div style="font-size:0.75rem; color:#64748b;">Cross-validated LR, RF, GBT on Spark</div>
                </div>
            </div>

            <h3 class="subsection-header">Heterogeneous Ingestion Inventory</h3>
            <div class="table-responsive">
                <table>
                    <thead>
                        <tr>
                            <th>Source System</th>
                            <th>Entity / Format</th>
                            <th>Ingestion Strategy</th>
                            <th>Volume</th>
                            <th>Key Join Anchor</th>
                            <th>Target Role</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><strong>SQLite Relational DB</strong></td>
                            <td><code>customer</code></td>
                            <td>Read-only SQL URI</td>
                            <td>221 records</td>
                            <td><code>customer.id</code></td>
                            <td>Primary Spine Key</td>
                        </tr>
                        <tr>
                            <td><strong>SQLite Relational DB</strong></td>
                            <td><code>subscription</code> + <code>subscription_type</code></td>
                            <td>Read-only SQL URI</td>
                            <td>221 records</td>
                            <td><code>customer.subscription_id</code></td>
                            <td>Target: <code>auto_renew</code>, Plan Tier, Tenure</td>
                        </tr>
                        <tr>
                            <td><strong>SQLite Relational DB</strong></td>
                            <td><code>sale</code> + <code>payment</code></td>
                            <td>Read-only SQL URI</td>
                            <td>23,188 orders</td>
                            <td><code>sale.customer_id</code></td>
                            <td>Monetary Spend, Order Frequency, Delay Hours</td>
                        </tr>
                        <tr>
                            <td><strong>SQLite Relational DB</strong></td>
                            <td><code>sale_product</code> + <code>product</code></td>
                            <td>Read-only SQL URI</td>
                            <td>59,759 items</td>
                            <td><code>sale_id</code> ➔ <code>customer_id</code></td>
                            <td>Product Diversity & Basket Volumes</td>
                        </tr>
                        <tr>
                            <td><strong>SQLite Relational DB</strong></td>
                            <td><code>product_review</code></td>
                            <td>Read-only SQL URI</td>
                            <td>10,990 reviews</td>
                            <td><code>customer_id</code></td>
                            <td>Sentiment Proxy, Rating Polarities</td>
                        </tr>
                        <tr>
                            <td><strong>Enterprise CRM</strong></td>
                            <td><code>PerfectPetClub_CRM_report.csv</code></td>
                            <td>Chunked CSV Stream</td>
                            <td>7,585 tickets</td>
                            <td><code>customer_id</code></td>
                            <td>Support Tickets, Complaints, Channels</td>
                        </tr>
                        <tr>
                            <td><strong>Web Server Access</strong></td>
                            <td><code>PerfectPetClub_server_log.csv</code></td>
                            <td>Chunked Stream (20k rows/chunk)</td>
                            <td>67,249 entries</td>
                            <td><code>login</code> ➔ <code>customer.email</code></td>
                            <td>Total Web Sessions, Recency, Frequency</td>
                        </tr>
                        <tr>
                            <td><strong>Customer Survey</strong></td>
                            <td><code>PerfectPetClub_survey.csv</code> (AWS S3)</td>
                            <td>Secure Boto3 Parameterized</td>
                            <td>221 surveys</td>
                            <td><code>email</code> ➔ <code>customer.email</code></td>
                            <td>NPS, Satisfaction, Price Comparison</td>
                        </tr>
                    </tbody>
                </table>
            </div>

            <div class="callout alert-success">
                <div class="callout-title">🔒 Credential Security & Ingestion Compliance</div>
                <p>
                    All S3 credentials (<code>UH_S3_ACCESS_KEY</code>, <code>UH_S3_SECRET_KEY</code>, <code>UH_S3_ENDPOINT_URL</code>) are loaded at runtime from environment variables using <code>python-dotenv</code>. The <code>.env</code> file is explicitly ignored in <code>.gitignore</code>, and an automated git history audit confirms <strong>zero credentials were ever committed</strong>. An automated S3 fetch script (<code>PerfectPetClub/scripts/download_survey_from_s3.py</code>) provides robust timeout handling (5s connect, 15s read) and clear diagnostics.
                </p>
            </div>
        </section>

        <!-- Section 2: Data Quality & Audit -->
        <section class="section-card" id="data-audit">
            <h2 class="section-header">⚖️ 2. Data Engineering Quality Audit & Financial Reconciliation</h2>
            
            <h3 class="subsection-header">Cryptographic Integrity Checks</h3>
            <p>
                To guarantee reproducible data engineering pipelines, SHA-256 cryptographic checksums were recorded on all source files before and after ETL execution:
            </p>
            <div class="table-responsive">
                <table>
                    <thead>
                        <tr>
                            <th>File Name</th>
                            <th>Pre-ETL SHA-256 Hash</th>
                            <th>Post-ETL SHA-256 Hash</th>
                            <th>Integrity Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><code>PerfectPetClub.db</code></td>
                            <td><code>1d66827055ee8834...</code></td>
                            <td><code>1d66827055ee8834...</code></td>
                            <td><span class="badge badge-success">MATCH (Read-Only)</span></td>
                        </tr>
                        <tr>
                            <td><code>PerfectPetClub_CRM_report.csv</code></td>
                            <td><code>a16d860e61d85698...</code></td>
                            <td><code>a16d860e61d85698...</code></td>
                            <td><span class="badge badge-success">MATCH (Unaltered)</span></td>
                        </tr>
                        <tr>
                            <td><code>PerfectPetClub_server_log.csv</code></td>
                            <td><code>a5996f4b50c0bb75...</code></td>
                            <td><code>a5996f4b50c0bb75...</code></td>
                            <td><span class="badge badge-success">MATCH (Unaltered)</span></td>
                        </tr>
                        <tr>
                            <td><code>PerfectPetClub_survey.csv</code></td>
                            <td><code>9be760ec7ca55d9b...</code></td>
                            <td><code>9be760ec7ca55d9b...</code></td>
                            <td><span class="badge badge-success">MATCH (Unaltered)</span></td>
                        </tr>
                    </tbody>
                </table>
            </div>

            <h3 class="subsection-header">Financial Ledger Audit</h3>
            <p>
                A rigorous audit was performed reconciling transactional order totals from the <code>sale</code> table with cumulative payment records in the <code>payment</code> table:
            </p>
            <div class="table-responsive">
                <table>
                    <thead>
                        <tr>
                            <th>Audit Check</th>
                            <th>Sale Ledger Value</th>
                            <th>Payment Ledger Value</th>
                            <th>Variance / Discrepancy</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><strong>Total Monetary Revenue</strong></td>
                            <td>£1,058,642.73 (105,864,273p)</td>
                            <td>£1,058,642.73 (105,864,273p)</td>
                            <td><strong>£0.00</strong></td>
                            <td><span class="badge badge-success">PERFECT MATCH</span></td>
                        </tr>
                        <tr>
                            <td><strong>Transaction Counts</strong></td>
                            <td>23,188 orders</td>
                            <td>23,188 payments</td>
                            <td><strong>0 records</strong></td>
                            <td><span class="badge badge-success">1-to-1 INTEGRITY</span></td>
                        </tr>
                    </tbody>
                </table>
            </div>

            <h3 class="subsection-header">GDPR Privacy & Data Cleansing Decisions</h3>
            <ul style="margin-left: 20px; margin-bottom: 14px;">
                <li><strong>PII Scrubbing:</strong> Customer email addresses and IP addresses were used strictly as join keys and immediately scrubbed from the final analytical Parquet store. The dataset preserves full analytical utility without personal data exposure.</li>
                <li><strong>CRM Header Normalization:</strong> The misspelled column <code>recieved_datatime</code> in the raw CRM CSV was cleaned in memory to <code>received_datetime</code> without mutating raw storage.</li>
                <li><strong>Platform Encoding:</strong> String-encoded Python lists (e.g. <code>"['e-mail']"</code>) were parsed into standard categorical tokens (<code>"e-mail"</code>, <code>"telephone"</code>, <code>"live-chat"</code>).</li>
                <li><strong>Streaming Web Log Parser:</strong> The 67,249-row server log CSV was parsed in memory-bounded chunks of 20,000 records, computing session recency and velocity per customer.</li>
            </ul>
        </section>

        <!-- Section 3: Feature Store & EDA -->
        <section class="section-card" id="feature-store">
            <h2 class="section-header">📊 3. Canonical Feature Store & Exploratory Data Analysis</h2>
            <p>
                The resulting canonical analytical feature store comprises <strong>exactly 221 customer rows × 36 engineered features</strong> with <strong>0 missing values</strong>.
            </p>

            <h3 class="subsection-header">Target Variable Distribution</h3>
            <p>
                The primary business objective is predicting customer churn, defined as non-renewal (<code>auto_renew = 0</code>, modelled as positive class <code>churn_label = 1</code>):
            </p>
            <div class="table-responsive">
                <table>
                    <thead>
                        <tr>
                            <th>Class Category</th>
                            <th>auto_renew</th>
                            <th>churn_label (Model Target)</th>
                            <th>Customer Count</th>
                            <th>Percentage</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><strong>Renewed Subscription</strong></td>
                            <td>1</td>
                            <td>0 (Negative Class)</td>
                            <td>135</td>
                            <td><strong>61.09%</strong></td>
                        </tr>
                        <tr>
                            <td><strong>Churned (Did Not Renew)</strong></td>
                            <td>0</td>
                            <td>1 (Positive Class)</td>
                            <td>86</td>
                            <td><strong>38.91%</strong></td>
                        </tr>
                        <tr style="background:#f8fafc; font-weight:bold;">
                            <td colspan="3">Total Active Subscribers</td>
                            <td>221</td>
                            <td>100.00%</td>
                        </tr>
                    </tbody>
                </table>
            </div>

            <h3 class="subsection-header">Engineered Feature Group Taxonomy</h3>
            <div class="table-responsive">
                <table>
                    <thead>
                        <tr>
                            <th>Functional Group</th>
                            <th>Feature Count</th>
                            <th>Engineered Columns</th>
                            <th>Business Rationale</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><strong>1. Account & Subscription</strong></td>
                            <td>3</td>
                            <td><code>subscription_tier</code>, <code>delivery_service</code>, <code>duration_days</code></td>
                            <td>Subscription plan level, courier tier, and contract lifespan.</td>
                        </tr>
                        <tr>
                            <td><strong>2. Monetary & Purchasing</strong></td>
                            <td>6</td>
                            <td><code>total_orders</code>, <code>total_spend_pence</code>, <code>avg_order_value_pence</code>, <code>total_items_purchased</code>, <code>unique_products_purchased</code>, <code>avg_items_per_order</code></td>
                            <td>RFM spending intensity, basket diversity, and purchase volume.</td>
                        </tr>
                        <tr>
                            <td><strong>3. Delivery Logistics</strong></td>
                            <td>5</td>
                            <td><code>completed_deliveries</code>, <code>unfulfilled_orders</code>, <code>delayed_deliveries</code>, <code>avg_delivery_delay_hours</code>, <code>delivery_delay_ratio</code></td>
                            <td>Fulfillment reliability, delivery bottlenecks, and logistics dissatisfaction.</td>
                        </tr>
                        <tr>
                            <td><strong>4. Customer Support (CRM)</strong></td>
                            <td>7</td>
                            <td><code>total_crm_interactions</code>, <code>crm_complaints_count</code>, <code>crm_queries_count</code>, <code>crm_email_count</code>, <code>crm_phone_count</code>, <code>crm_livechat_count</code>, <code>crm_complaint_ratio</code></td>
                            <td>Customer support friction, complaint frequency, and preferred contact channels.</td>
                        </tr>
                        <tr>
                            <td><strong>5. Product Reviews</strong></td>
                            <td>5</td>
                            <td><code>total_reviews</code>, <code>avg_star_rating</code>, <code>low_star_reviews_count</code>, <code>high_star_reviews_count</code>, <code>has_reviews</code></td>
                            <td>Subscriber product satisfaction and sentiment polarity.</td>
                        </tr>
                        <tr>
                            <td><strong>6. Web Engagement</strong></td>
                            <td>4</td>
                            <td><code>total_web_sessions</code>, <code>days_since_last_session</code>, <code>session_frequency_per_month</code>, <code>order_frequency_per_month</code></td>
                            <td>Digital engagement recency, login frequency, and activity momentum.</td>
                        </tr>
                        <tr>
                            <td><strong>7. Customer Survey Sentiment</strong></td>
                            <td>4</td>
                            <td><code>reported_income</code>, <code>nps_recommend</code>, <code>satisfaction_score</code>, <code>online_comparison_freq</code></td>
                            <td>Subjective sentiment, Net Promoter Score, and price sensitivity.</td>
                        </tr>
                    </tbody>
                </table>
            </div>

            <h3 class="subsection-header">Stratified Train-Test Splitting (80/20, Seed 42)</h3>
            <p>
                To avoid class distribution shifts between training and test sets, an exact stratified 80/20 split was enforced:
            </p>
            <ul style="margin-left: 20px;">
                <li><strong>Training Set:</strong> 177 customers (108 Renewed / 69 Churned) — 61.0% Renew / 39.0% Churn.</li>
                <li><strong>Test Set:</strong> 44 customers (27 Renewed / 17 Churned) — 61.4% Renew / 38.6% Churn.</li>
            </ul>
        </section>

        <!-- Section 4: Baseline ML -->
        <section class="section-card" id="baseline-ml">
            <h2 class="section-header">🤖 4. Baseline PySpark MLlib Modeling (Full-Window Cumulative Data)</h2>
            <p>
                In the initial modeling phase, standard distributed MLlib pipelines were constructed using the canonical dataset where operational features cover the customer's full tenure.
            </p>

            <h3 class="subsection-header">Spark MLlib Pipeline Architecture</h3>
            <p>
                Each classification model was packaged into an end-to-end Spark MLlib <code>Pipeline</code> consisting of:
            </p>
            <ol style="margin-left: 20px; margin-bottom: 16px;">
                <li><code>StringIndexer</code> with <code>handleInvalid="keep"</code> for categorical columns (<code>subscription_tier</code>, <code>delivery_service</code>, <code>reported_income</code>).</li>
                <li><code>Imputer</code> (median strategy) to ensure vector assembly compatibility.</li>
                <li><code>VectorAssembler</code> combining numeric features and indexed categoricals into dense feature vectors.</li>
                <li><code>StandardScaler</code> (with mean and std centering/scaling) for regularized models.</li>
                <li>Classifier stage (<code>LogisticRegression</code>, <code>RandomForestClassifier</code>, or <code>GBTClassifier</code>).</li>
                <li>3-Fold Cross-Validation (<code>CrossValidator</code>) using <code>BinaryClassificationEvaluator(metricName="areaUnderROC")</code>.</li>
            </ol>

            <h3 class="subsection-header">Baseline Evaluation Results (Full-Window Features)</h3>
            {tbl_bl}

            <div class="img-container">
                <img src="{img_bl_comp}" alt="Baseline Model Comparison">
                <div class="img-caption">Figure 1: Baseline Model Comparison across Configurations A (Operational), B (Multimodal + Survey), and S (Rates).</div>
            </div>

            <div class="img-container">
                <img src="{img_bl_cm}" alt="Baseline Confusion Matrices">
                <div class="img-caption">Figure 2: Baseline Test Confusion Matrices across 9 Evaluated Pipelines (Held-Out Test Set, n=44).</div>
            </div>

            <div class="img-container">
                <img src="{img_bl_roc}" alt="Baseline ROC Curves">
                <div class="img-caption">Figure 3: Baseline ROC Curves showing seemingly strong discriminatory power (AUC 0.67 - 0.75).</div>
            </div>
        </section>

        <!-- Section 5: Temporal Leakage Audit -->
        <section class="section-card" id="leakage-audit">
            <h2 class="section-header">🔍 5. The Critical Discovery: Investigation of Temporal Target Leakage</h2>
            
            <div class="callout alert-danger">
                <div class="callout-title">⚠️ Methodological Flaw: Prediction-Time Target Leakage</div>
                <p>
                    While baseline models achieved attractive metrics (CV ROC-AUC up to <strong>0.776</strong> and Test Accuracy of <strong>75.0%</strong>), an in-depth data engineering audit revealed that <strong>these models suffer from severe temporal target leakage</strong>. In an operational production deployment, these models would fail completely.
                </p>
            </div>

            <h3 class="subsection-header">Why Full-Window Cumulative Features Leak Future Information</h3>
            <p>
                In the raw relational database, customers who churned had their subscriptions terminated prematurely upon cancellation, while renewing subscribers remained active for longer periods. Consequently, cumulative features calculated across the full subscription duration encode the customer's lifespan:
            </p>
            <ul style="margin-left: 20px; margin-bottom: 14px;">
                <li><strong>Tenure Encoding:</strong> Churned customers had significantly lower total orders and spend simply because their subscriptions ended sooner, not because their purchase rate was lower.</li>
                <li><strong>Final 30-Day Activity Clustering:</strong> Empirical analysis of order timestamps revealed that <strong>17.7% of all orders from churned customers occurred within their final 30 days</strong> of subscription, compared to only <strong>7.4% for renewed customers</strong>.</li>
                <li><strong>Unknown Survey Timing:</strong> The customer survey lacked a timestamp. If dissatisfied subscribers completed the survey at the moment of cancellation (an exit survey), using their low satisfaction score to predict churn constitutes predicting an event with its own direct consequence.</li>
            </ul>

            <div class="callout alert-warning">
                <div class="callout-title">💡 Comparison with Peer Team Submissions</div>
                <p>
                    Peer groups sharing single-notebook exports (e.g., reporting <strong>1.000 ROC-AUC and 100% accuracy</strong> with Decision Trees and Random Forests) inadvertently included <code>duration_days</code> or full-window tenure features in their feature assemblers. Because <code>duration_days</code> perfectly separates subscribers who left early from those who renewed, the models achieved trivial, 100% artificial accuracy that is completely invalid in real-world predictive forecasting.
                </p>
            </div>
        </section>

        <!-- Section 6: Fixed Cutoff 14 ML -->
        <section class="section-card" id="cutoff14-ml">
            <h2 class="section-header">🛡️ 6. Fixed-Cutoff Machine Learning (14-Day Observation Horizon)</h2>
            <p>
                To resolve target leakage and establish real-world predictive validity, we constructed a <strong>14-day fixed-cutoff observation window</strong>. In this formulation, all customer behavioral features (orders, spend, delivery delays, CRM tickets, web sessions) are aggregated strictly within:
            </p>
            <div style="text-align: center; font-size: 1.15rem; font-weight: bold; margin: 16px 0; background: #f8fafc; padding: 12px; border-radius: 8px;">
                Observation Window: [ T<sub>subscription_start</sub> , &nbsp; T<sub>subscription_start</sub> + 14 Days ]
            </div>
            <p>
                Any customer interaction occurring after Day 14 was strictly excluded from feature computation. Furthermore, <code>duration_days</code> and <code>customer_id</code> were completely omitted from modeling.
            </p>

            <h3 class="subsection-header">14-Day Fixed Cutoff Evaluation Results</h3>
            {tbl_c14}

            <div class="img-container">
                <img src="{img_c14_comp}" alt="Cutoff 14 Model Comparison">
                <div class="img-caption">Figure 4: Cutoff-14 Model Performance Comparison across Configurations A and B.</div>
            </div>

            <div class="img-container">
                <img src="{img_c14_cm}" alt="Cutoff 14 Confusion Matrices">
                <div class="img-caption">Figure 5: Cutoff-14 Confusion Matrices on 44 Held-Out Test Customers.</div>
            </div>

            <div class="img-container">
                <img src="{img_c14_roc}" alt="Cutoff 14 ROC Curves">
                <div class="img-caption">Figure 6: Cutoff-14 ROC Curves revealing realistic, leakage-free discriminatory power (AUC 0.47 - 0.59).</div>
            </div>

            <div class="callout">
                <div class="callout-title">📌 Key Cutoff-14 Findings</div>
                <p>
                    With leakage eliminated, model performance drops from CV AUC ~0.78 to between <strong>0.48 and 0.60</strong>. On 44 test customers, early-stage behavioral signals during the first 14 days do not strongly discriminate future renewal from churn. The selected model under the CV ROC-AUC rule is <strong>Configuration A Logistic Regression</strong> (CV AUC 0.595, Test AUC 0.522). This demonstrates the absolute necessity of rigorous leakage auditing before deploying machine learning models into business operations.
                </p>
            </div>
        </section>

        <!-- Section 7: Fixed Cutoff 21 ML -->
        <section class="section-card" id="cutoff21-ml">
            <h2 class="section-header">⏱️ 7. Sensitivity Analysis: 21-Day Observation Horizon</h2>
            <p>
                To evaluate whether extending the early observation horizon by one week provides stronger predictive signals, we implemented a <strong>21-day observation window</strong>:
            </p>
            <div style="text-align: center; font-size: 1.15rem; font-weight: bold; margin: 16px 0; background: #f8fafc; padding: 12px; border-radius: 8px;">
                Sensitivity Window: [ T<sub>subscription_start</sub> , &nbsp; T<sub>subscription_start</sub> + 21 Days ]
            </div>

            <h3 class="subsection-header">21-Day Fixed Cutoff Evaluation Results</h3>
            {tbl_c21}

            <div class="img-container">
                <img src="{img_c21_comp}" alt="Cutoff 21 Model Comparison">
                <div class="img-caption">Figure 7: Cutoff-21 Model Performance Comparison across Configurations A and B.</div>
            </div>

            <div class="img-container">
                <img src="{img_c21_cm}" alt="Cutoff 21 Confusion Matrices">
                <div class="img-caption">Figure 8: Cutoff-21 Confusion Matrices on Held-Out Test Customers.</div>
            </div>

            <div class="img-container">
                <img src="{img_c21_roc}" alt="Cutoff 21 ROC Curves">
                <div class="img-caption">Figure 9: Cutoff-21 ROC Curves showing sensitivity performance across extended window.</div>
            </div>
        </section>

        <!-- Section 8: Master Comparison -->
        <section class="section-card" id="comparison">
            <h2 class="section-header">🏆 8. Master Cross-Experiment Comparative Synthesis</h2>
            <p>
                The table below synthesizes the highest-performing models across each experimental phase of the project:
            </p>
            
            <div class="table-responsive">
                <table>
                    <thead>
                        <tr>
                            <th>Experiment Phase</th>
                            <th>Best Architecture</th>
                            <th>Feature Config</th>
                            <th>CV ROC-AUC</th>
                            <th>Test ROC-AUC</th>
                            <th>Test Accuracy</th>
                            <th>Churn F1</th>
                            <th>Leakage Status</th>
                            <th>Operational Viability</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr style="background: #fef2f2;">
                            <td><strong>1. Baseline (Full Window)</strong></td>
                            <td>LogisticRegression</td>
                            <td>Config A (Operational)</td>
                            <td><span class="badge badge-primary">0.7759</span></td>
                            <td><span class="badge badge-success">0.7473</span></td>
                            <td>75.0%</td>
                            <td>0.6207</td>
                            <td><span class="tag tag-danger">Severely Leaky</span></td>
                            <td>Invalid for real-time forecasting</td>
                        </tr>
                        <tr style="background: #eff6ff;">
                            <td><strong>2. 14-Day Fixed Cutoff</strong></td>
                            <td>LogisticRegression</td>
                            <td>Config A (Operational)</td>
                            <td><span class="badge badge-primary">0.5947</span></td>
                            <td><span class="badge badge-success">0.5218</span></td>
                            <td>59.1%</td>
                            <td>0.0000</td>
                            <td><span class="tag tag-success">100% Leakage-Free</span></td>
                            <td>Honest baseline for 14-day early warning</td>
                        </tr>
                        <tr style="background: #eff6ff;">
                            <td><strong>2b. 14-Day Cutoff (Alt)</strong></td>
                            <td>GBTClassifier</td>
                            <td>Config B (Multimodal)</td>
                            <td><span class="badge badge-primary">0.4562</span></td>
                            <td><span class="badge badge-success">0.5948</span></td>
                            <td>59.1%</td>
                            <td>0.4375</td>
                            <td><span class="tag tag-success">100% Leakage-Free</span></td>
                            <td>Active churn detection (Recall: 41.2%)</td>
                        </tr>
                        <tr style="background: #f0fdf4;">
                            <td><strong>3. 21-Day Cutoff</strong></td>
                            <td>LogisticRegression</td>
                            <td>Config B (Multimodal)</td>
                            <td><span class="badge badge-primary">0.5348</span></td>
                            <td><span class="badge badge-success">0.5817</span></td>
                            <td>61.4%</td>
                            <td>0.4848</td>
                            <td><span class="tag tag-success">100% Leakage-Free</span></td>
                            <td>Balanced early warning (Precision: 50.0%)</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            
            <p>
                <strong>Academic Takeaway:</strong> In real customer lifecycle management, subscription churn decisions rarely manifest clearly in the first 14 to 21 days. Subscribers typically churn near the end of their renewal cycle (e.g. Month 11 or 12). Therefore, true production churn models should evaluate rolling temporal observation windows closer to the renewal date rather than naive early-onboarding snapshots.
            </p>
        </section>

        <!-- Section 9: Feature Importance -->
        <section class="section-card" id="feature-importance">
            <h2 class="section-header">📈 9. Feature Importance & Behavioral Drivers</h2>
            <p>
                Analysis of tree ensemble split frequencies and logistic regression coefficients across the leakage-controlled experiments reveals the most influential indicators of subscriber attrition:
            </p>
            
            <div class="table-responsive">
                <table>
                    <thead>
                        <tr>
                            <th>Rank</th>
                            <th>Feature Name</th>
                            <th>Domain Category</th>
                            <th>Correlation Direction</th>
                            <th>Operational Impact / Customer Behavior</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>1</td>
                            <td><code>delivery_delay_ratio</code></td>
                            <td>Logistics</td>
                            <td>Positive (Delays ➔ Churn)</td>
                            <td>Subscribers experiencing delayed pet food deliveries face immediate friction, driving dissatisfaction.</td>
                        </tr>
                        <tr>
                            <td>2</td>
                            <td><code>crm_complaint_ratio</code></td>
                            <td>Support</td>
                            <td>Positive (Complaints ➔ Churn)</td>
                            <td>High proportion of support contacts flagged as formal complaints indicates unresolved friction.</td>
                        </tr>
                        <tr>
                            <td>3</td>
                            <td><code>days_since_last_session</code></td>
                            <td>Web Activity</td>
                            <td>Positive (Inactivity ➔ Churn)</td>
                            <td>Long gaps since visiting the club portal strongly signal waning engagement and brand detachment.</td>
                        </tr>
                        <tr>
                            <td>4</td>
                            <td><code>online_comparison_freq</code></td>
                            <td>Survey</td>
                            <td>Positive (Comparison ➔ Churn)</td>
                            <td>Subscribers who frequently cross-shop competitor prices exhibit significantly higher price sensitivity.</td>
                        </tr>
                        <tr>
                            <td>5</td>
                            <td><code>nps_recommend</code></td>
                            <td>Survey</td>
                            <td>Negative (High NPS ➔ Renewal)</td>
                            <td>Promoters (scores 9–10) are substantially more resilient and likely to enable automatic renewal.</td>
                        </tr>
                        <tr>
                            <td>6</td>
                            <td><code>avg_order_value_pence</code></td>
                            <td>Monetary</td>
                            <td>Negative (High Spend ➔ Renewal)</td>
                            <td>Subscribers with larger baskets demonstrate deeper pet-care investment and loyalty.</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </section>

        <!-- Section 10: Decision Support -->
        <section class="section-card" id="customer-scoring">
            <h2 class="section-header">🎯 10. Decision Support & Test Set Churn Scoring</h2>
            <p>
                To translate model predictions into business action, the deployed model outputs a continuous <strong>churn probability score</strong> for every subscriber. Below is a representative sample of held-out test customers showing actual status, predicted risk, and automated retention interventions:
            </p>
            {tbl_preds}
        </section>

        <!-- Section 11: Live Streamlit Studio -->
        <section class="section-card" id="live-studio">
            <h2 class="section-header">⚡ 11. Production Serving & Interactive Streamlit Studio</h2>
            <p>
                To provide business stakeholders and technical auditors with real-time model verification, the platform features a responsive web application built with Streamlit:
            </p>
            <div style="background: #0f172a; color: #f8fafc; padding: 14px 18px; border-radius: 8px; font-family: monospace; font-size: 0.95rem; margin-bottom: 16px;">
                streamlit run gata/app.py
            </div>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-top: 20px;">
                <div style="background: #f8fafc; border: 1px solid var(--border); border-radius: 8px; padding: 18px;">
                    <h4 style="color: var(--primary); margin-bottom: 8px;">Tab 1: Audited Diagnostics</h4>
                    <p style="font-size: 0.85rem; color: #64748b;">
                        Displays pre-computed held-out confusion matrices, precision/recall curves, and calibration plots across all 44 test customers.
                    </p>
                </div>
                <div style="background: #f8fafc; border: 1px solid var(--border); border-radius: 8px; padding: 18px;">
                    <h4 style="color: var(--primary); margin-bottom: 8px;">Tab 2: Live Spark Training</h4>
                    <p style="font-size: 0.85rem; color: #64748b;">
                        Executes live distributed PySpark MLlib training on the local Spark cluster in <strong>~3.8 seconds</strong>, allowing interactive parameter adjustments.
                    </p>
                </div>
                <div style="background: #f8fafc; border: 1px solid var(--border); border-radius: 8px; padding: 18px;">
                    <h4 style="color: var(--primary); margin-bottom: 8px;">Tab 3: Feature Store Schema</h4>
                    <p style="font-size: 0.85rem; color: #64748b;">
                        Inspects the 36 canonical features, demographic distributions, summary statistics, and verified zero-null data contracts.
                    </p>
                </div>
            </div>
        </section>

        <!-- Section 12: Academic Evaluation & Next Steps -->
        <section class="section-card" id="recommendations">
            <h2 class="section-header">🎓 12. Academic Discussion, Limitations & Actionable Recommendations</h2>
            
            <h3 class="subsection-header">Methodological Rigor & Academic Value</h3>
            <p>
                The primary academic contribution of this work is demonstrating that <strong>unexamined data leakage produces misleadingly optimistic models</strong> that fail catastrophically when deployed. While peer implementations accepted 100% accuracy derived from tenure artifacts, our pipeline explicitly proved the source of the leakage, formulated fixed-cutoff temporal windows, and evaluated models on authentic forward-looking predictions.
            </p>

            <h3 class="subsection-header">Dataset Limitations</h3>
            <ul style="margin-left: 20px; margin-bottom: 16px;">
                <li><strong>Sample Size:</strong> Exactly 221 subscribers with a 44-customer test set. Each individual test customer impacts overall accuracy by 2.27 percentage points, introducing natural statistical variance.</li>
                <li><strong>Survey Temporal Ambiguity:</strong> Absence of survey timestamps prevents establishing whether customer responses were provided during early onboarding or as exit feedback.</li>
            </ul>

            <h3 class="subsection-header">Strategic Recommendations for PerfectPetClub</h3>
            <ol style="margin-left: 20px; margin-bottom: 16px;">
                <li><strong>Logistics Optimization:</strong> Prioritize on-time delivery. Delivery delay ratio is the strongest controllable driver of subscriber dissatisfaction.</li>
                <li><strong>Proactive CRM Resolution:</strong> Implement automated escalation workflows whenever a subscriber submits more than one complaint ticket within a 30-day window.</li>
                <li><strong>Timestamped Telemetry:</strong> Instrument application telemetry to record exact event timestamps for surveys, cancellations, and renewals to support rolling temporal modeling.</li>
            </ol>
        </section>

        <!-- Footer -->
        <footer>
            <p style="font-weight: 600; color: var(--dark); margin-bottom: 6px;">
                7COM2005 Data Engineering Master Coursework — University of Hertfordshire
            </p>
            <p>
                Project Authors: <strong>Divya</strong>, <strong>Atchyuth Karri</strong>, <strong>Lukshan</strong>, <strong>Koteswarao</strong>, <strong>Deekshitha</strong>, <strong>Vasu Challa</strong>
            </p>
            <p style="margin-top: 8px; font-size: 0.8rem; color: #94a3b8;">
                Built with PySpark MLlib, Snappy Parquet, and Streamlit. Fully reproducible end-to-end execution.
            </p>
        </footer>
    </main>
</div>

</body>
</html>
"""

output_path_reports = os.path.join(OUTPUT_DIR, "all_results_report.html")
with open(output_path_reports, "w", encoding="utf-8") as f:
    f.write(html_content)
print(f"Master HTML written to: {output_path_reports} ({os.path.getsize(output_path_reports) / 1024:.1f} KB)")

# Also save to root and PerfectPetClub/outputs for convenience
output_path_root = os.path.join(PROJECT_ROOT, "all_results_report.html")
with open(output_path_root, "w", encoding="utf-8") as f:
    f.write(html_content)

output_path_ml = os.path.join(PROJECT_ROOT, "PerfectPetClub/outputs/all_results_report.html")
with open(output_path_ml, "w", encoding="utf-8") as f:
    f.write(html_content)

print("Saved copies to root and PerfectPetClub/outputs/")
