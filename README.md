# PerfectPetClub – Scalable Data Engineering, PySpark MLlib Churn Prediction & Analytics Platform

### University of Hertfordshire | Module: 7COM2005 – Data Engineering
**Academic Year:** 2026/2027  
**Project Type:** Collaborative Team Coursework  
**Repository:** `7COM2005-Data-Engineering`  

---

## 1. Executive Summary

**PerfectPetClub** is an end-to-end, enterprise-grade data engineering, big data analytics, and machine learning platform built for a subscription-based pet e-commerce enterprise. The platform addresses critical real-world business challenges: extracting and harmonizing heterogeneous operational datasets, guaranteeing data quality and financial reconciliation, training scalable machine learning models using **Apache Spark (PySpark MLlib)** for customer churn prediction, rigorously auditing temporal data leakage, and serving operational predictions via an interactive **Streamlit** dashboard.

### Key Highlights & Achievements:
* **Multi-Source Ingestion**: Scalable ingestion across relational databases (SQLite), customer relationship management (CRM) logs, web server access logs, and an encrypted **Amazon S3** cloud object store.
* **Flawless Financial Reconciliation**: Reconciled £1,058,642.73 (105,864,273 pence) in customer transactions with a **£0.00 discrepancy** across transactional orders and payment ledgers.
* **Canonical Parquet Feature Store**: Created a single source of truth containing 221 validated customer subscribers across 36 engineered features with 0 null values and 0 duplicate IDs.
* **Full-Window vs. Fixed-Cutoff ML Auditing**: Implemented PySpark MLlib classification pipelines and conducted an advanced forensic investigation into tenure-based temporal data leakage, introducing 14-day and 21-day fixed observation horizons to measure true pre-cancellation predictive signals.
* **Interactive Streamlit Web Dashboard**: Built a decision-support dashboard evaluating saved PySpark pipelines strictly against held-out test partitions to prevent evaluation leakage.
* **Academic Integrity & GDPR Compliance**: Enforced zero customer Personally Identifiable Information (PII) in analytical stores, cryptographic SHA-256 verification of raw datasets, and strict environment-variable credential isolation.

---

## 2. Team Members & Roles

| Name | Role | Responsibilities |
|---|---|---|
| **Divya** | Team Leader | Project coordination, task delegation, requirements tracking, and milestone reviews |
| **Atchyuth Karri** | Team Member | Pipeline architecture, S3 ingestion, PySpark MLlib modeling, temporal leakage auditing, and Streamlit engineering |
| **Lukshan** | Team Member | Exploratory data analysis, Scikit-learn comparative modeling, and feature exploration |
| **Koteswarao** | Team Member | Application deployment, dashboard prototyping, and delivery packaging |
| **Deekshitha** | Team Member | Data quality validation, CRM log processing, and documentation |
| **Srinivas** | Team Member | Web server log processing, schema mapping, and testing |

---

## 3. System Architecture & End-to-End Workflow

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                HETEROGENEOUS DATA SOURCES                              │
│                                                                                        │
│  ┌───────────────────────┐   ┌────────────────────────┐   ┌─────────────────────────┐  │
│  │  Operational SQLite   │   │     CRM Report CSV     │   │   Web Server Logs CSV   │  │
│  │   PerfectPetClub.db   │   │  7,585 support tickets │   │  67,249 access sessions │  │
│  │ (8 relational tables) │   │  (Email, Chat, Phone)  │   │   (Logins, IP, Agents)  │  │
│  └───────────┬───────────┘   └───────────┬────────────┘   └────────────┬────────────┘  │
│              │                           │                             │               │
│              │               ┌───────────┴────────────┐                │               │
│              │               │     Amazon S3 Bucket   │                │               │
│              │               │ Customer Survey (CSV)  │                │               │
│              │               │  221 responses via SDK │                │               │
│              │               └───────────┬────────────┘                │               │
└──────────────┼───────────────────────────┼─────────────────────────────┼───────────────┘
               ▼                           ▼                             ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                             STAGE 1: ETL & DATA RECONCILIATION                         │
│                           Notebook: 01_ETL_Pipeline.ipynb                              │
│                                                                                        │
│  * SHA-256 cryptographic verification of raw data sources                              │
│  * Schema standardization, timestamp casting, and categorical normalization            │
│  * Financial Spend Reconciliation: £1,058,642.73 reconciled with £0.00 discrepancy    │
│  * GDPR & Privacy Protection: Stripped customer PII (names, emails, phones, addresses) │
│  * Missing value imputation: Star rating logic (nulled if has_reviews == 0)            │
│                                                                                        │
│  OUTPUT: Canonical Feature Store -> customer_churn_dataset.parquet (221 × 36)          │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        STAGE 2: MACHINE LEARNING & LEAKAGE AUDIT                       │
│                     Distributed PySpark 4.3.0 MLlib (Java 17/21)                       │
│                                                                                        │
│  ┌────────────────────────────────────────┐  ┌──────────────────────────────────────┐  │
│  │ Full-Window Baseline (Notebook 02)     │  │ Fixed-Cutoff Observation Horizons    │  │
│  │ * Logistic Regression (CV AUC: 0.776)  │  │ * Notebook 03: 14-day Feature Store  │  │
│  │ * Random Forest (CV AUC: 0.717)        │  │ * Notebook 04: 14-day ML Evaluation  │  │
│  │ * GBT Classifier (CV AUC: 0.707)       │  │ * Notebook 05: 21-day Feature Store  │  │
│  │ * Finding: High AUC inflated by        │  │ * Notebook 06: 21-day ML Evaluation  │  │
│  │   variable subscription tenure length. │  │ * Finding: Early-signal AUC 0.48-0.60│  │
│  └────────────────────────────────────────┘  └──────────────────────────────────────┘  │
│                                                                                        │
│  OUTPUT: Serialized MLlib Pipelines -> models/best_pipeline_A_operational/             │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        STAGE 3: OPERATIONAL INFERENCE & DASHBOARD                      │
│                             Streamlit App: gata/app.py                                 │
│                                                                                        │
│  * Live prediction dashboard loading trained PySpark pipeline artifacts                │
│  * Evaluates strictly on held-out test customers (zero training/test leakage)          │
│  * Dynamic classification threshold tuning, confusion matrix, ROC / PR curves         │
│  * High-risk subscriber segmentation and actionable customer retention insights        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Heterogeneous Data Sources

The raw data sources are housed in `PerfectPetClub/data/raw/`:

| Source System | Storage Format | Size / Records | Key Entities / Columns | Primary Join Key |
|---|---|---|---|---|
| **Operational Database** | SQLite (`PerfectPetClub.db`) | 8 Tables, 94,158 total rows | `customer`, `subscription`, `subscription_type`, `sale`, `payment`, `sale_product`, `product`, `product_review` | `customer.id` / `customer_id` |
| **CRM System** | CSV (`PerfectPetClub_CRM_report.csv`) | 7,585 records | `recieved_datatime`, `issue_type`, `channel` (`e-mail`, `live-chat`, `telephone`) | `customer_id` |
| **Web Server Logs** | CSV (`PerfectPetClub_server_log.csv`) | 67,249 records | `datetime`, `ip`, `login` (email), `method`, `url`, `status_code`, `user_agent` | `customer.email` |
| **Customer Survey** | Cloud S3 / CSV (`PerfectPetClub_survey.csv`) | 221 responses | `email`, `satisfaction_score`, `nps`, `household_pets`, `income_bracket`, `price_sensitivity` | `customer.email` |

---

## 5. Data Quality, Privacy & Financial Reconciliation

A full technical audit is maintained in `PerfectPetClub/data/processed/data_quality_report.md`. Key verified metrics:

### 1. Financial Reconciliation
* Total transactional orders spend across 23,188 sales: **105,864,273 pence (£1,058,642.73)**.
* Total payment ledger records across 23,188 payments: **105,864,273 pence (£1,058,642.73)**.
* **Financial Discrepancy: £0.00 (0.00%)**.

### 2. Analytical Feature Store Dimensions & Quality
* **Dimensions**: Exactly **221 rows × 36 columns** (one row per subscriber).
* **Missing Values**: **0 nulls** across the entire canonical Parquet dataset.
* **Duplicate Primary Keys**: **0 duplicate customer IDs**.
* **Target Balance (`auto_renew`)**:
  * **Renewed (`auto_renew = 1`)**: 135 customers (**61.09%**)
  * **Churned (`auto_renew = 0`)**: 86 customers (**38.91%**)

### 3. GDPR Compliance & Privacy Protection
To strictly comply with data protection regulations (GDPR / Data Protection Act 2018), customer PII was completely purged prior to generating the analytical dataset:
* Dropped fields: `first_name`, `last_name`, `email`, `phone`, `street_address`, `postcode`.
* Aggregated behavioral summaries (e.g. `total_spend_gbp`, `total_orders`, `total_crm_tickets`, `web_sessions_count`) are maintained using pseudonymized integer `customer_id` keys.

---

## 6. Machine Learning & PySpark MLlib Workflow

All modeling is implemented natively using **Apache Spark MLlib (`pyspark.ml`)** on Python 3.12.

### 1. ML Pipeline Architecture
The end-to-end preprocessing and model training pipeline is encapsulated inside a PySpark `Pipeline`:
1. **Categorical Encoding**: `StringIndexer` and `OneHotEncoder` handle categorical features (`subscription_type_name`, `preferred_channel`, `income_bracket`, etc.).
2. **Missing Value Imputation**: `MedianImputer` estimates median replacements for numeric fields. Star ratings for customers with zero reviews are set to `null` to avoid artificial 3.0 bias.
3. **Feature Assembly**: `VectorAssembler` packages all transformed features into dense feature vectors.
4. **Feature Scaling**: `StandardScaler` normalizes numeric features for distance and gradient-sensitive models.
5. **Class Assignment**: Churn is mapped as the positive class (`churn_label = 1 - auto_renew` where `1 = Churned`, `0 = Renewed`).
6. **Cross-Validation**: 3-fold stratified cross-validation (`CrossValidator`) tuned using `BinaryClassificationEvaluator` (`areaUnderROC` and `areaUnderPR`).

### 2. Full-Window Baseline Models (`02_Machine_Learning.ipynb`)

| Model | Hyperparameters Tuned | CV AUC-ROC | Test AUC-ROC | Test Accuracy | Test Churn Recall |
|---|---|---|---|---|---|
| **Logistic Regression** | `regParam` [0.01, 0.1, 1.0], `elasticNetParam` [0.0, 0.5] | **0.776** | **0.785** | **77.3%** | **70.6%** |
| **Random Forest** | `maxDepth` [4, 6, 8], `numTrees` [20, 50, 100] | 0.717 | 0.730 | 72.7% | 58.8% |
| **Gradient-Boosted Trees** | `maxDepth` [3, 5], `maxIter` [20, 50] | 0.707 | 0.715 | 70.5% | 64.7% |

---

## 7. Forensic Data Leakage Investigation: Fixed-Cutoff Horizons

A key data engineering discovery in this project is **prediction-time temporal leakage**:

### The Problem
In the initial full-window dataset (`02_Machine_Learning.ipynb`), features are aggregated across the entire subscription lifespan. For churned customers, `end_date` is the date of cancellation (median lifespan of 90 days vs. 180 days for renewed subscribers). As a consequence, churned customers naturally have fewer lifetime orders, web visits, and CRM tickets. Full-window models learn to identify churn using activity volume that was truncated *by the churn event itself*, rather than identifying pre-churn indicators.

### The Engineering Solution
We developed a rigorous fixed-cutoff feature architecture (documented in `PerfectPetClub/documentation/fixed_cutoff_feature_design.md`):
* **14-Day Fixed Cutoff (`03_` and `04_` Notebooks)**: Features are re-aggregated strictly using customer events occurring within `start_date + 14 days`.
* **21-Day Fixed Cutoff (`05_` and `06_` Notebooks)**: Features are re-aggregated within `start_date + 21 days`.
* **Findings**:
  * On a strictly leakage-free 14-day window, models achieve an AUC of **0.48 – 0.60**, demonstrating that 14 days of behavioral interactions contain weak early signals for a 6-month renewal.
  * This technical finding provides invaluable business insight: **early churn cannot be predicted solely on early transactional volume**, demonstrating the absolute necessity of ongoing engagement campaigns, automated check-ins, and proactive onboarding interventions.

---

## 8. Interactive Streamlit Dashboard (`gata/app.py`)

An operational dashboard was developed in Streamlit to bridge data engineering, model deployment, and business decision-making.

```bash
# Run the Streamlit dashboard
streamlit run gata/app.py
```

### Dashboard Capabilities:
* **Leakage-Free Validation**: Ingests canonical Parquet test datasets and loads saved PySpark MLlib pipeline artifacts (`PerfectPetClub/outputs/ml/pipeline_best_A_operational`).
* **Interactive Probability Thresholds**: Allows business stakeholders to slide classification thresholds between 0.10 and 0.90 to balance precision vs. recall depending on customer acquisition costs and retention incentives.
* **Risk Stratification**: Categorizes subscribers into Low Risk, Medium Risk, and High Churn Risk cohorts with immediate export capabilities.
* **Model Diagnostic Suite**: Renders live Confusion Matrices, ROC Curves, and Precision-Recall Curves based on held-out test subscribers.

---

## 9. Repository Structure

```text
7COM2005-Data-Engineering/
├── README.md                                          <- Root project overview and documentation
├── .env.example                                       <- Template for required environment variables (S3 credentials)
├── .gitignore                                         <- Git exclusion rules (ignores .env, .venv*, cache files)
│
├── PerfectPetClub/                                    <- Core data engineering and machine learning package
│   ├── README.md                                      <- Package-level guide and execution steps
│   ├── requirements.txt                               <- Pinned dependencies for Python 3.12 & PySpark 4.x
│   │
│   ├── data/
│   │   ├── raw/                                       <- Read-only original source datasets (verified by SHA-256)
│   │   │   ├── PerfectPetClub.db                      <- SQLite operational database (8 transactional tables)
│   │   │   ├── PerfectPetClub_CRM_report.csv          <- CRM support ticket logs (7,585 records)
│   │   │   ├── PerfectPetClub_server_log.csv          <- Web access logs (67,249 records)
│   │   │   └── PerfectPetClub_survey.csv              <- Ingested customer survey (221 responses)
│   │   │
│   │   ├── processed/                                 <- Canonical analytical feature store
│   │   │   ├── customer_churn_dataset.parquet         <- Primary feature dataset (221 rows × 36 columns)
│   │   │   ├── customer_churn_dataset.csv             <- CSV copy for inspection
│   │   │   └── data_quality_report.md                 <- Full data quality and financial audit report
│   │   │
│   │   ├── cutoff14/                                  <- 14-day temporal cutoff datasets
│   │   └── cutoff21/                                  <- 21-day temporal cutoff datasets
│   │
│   ├── notebooks/                                     <- Reproducible Jupyter workflows (01 through 06)
│   │   ├── 01_ETL_Pipeline.ipynb                      <- Multi-source ETL & financial reconciliation
│   │   ├── 02_Machine_Learning.ipynb                  <- PySpark MLlib classification (full-window baseline)
│   │   ├── 03_Fixed_Cutoff_Features.ipynb             <- 14-day observation horizon feature engineering
│   │   ├── 04_Cutoff_Machine_Learning.ipynb           <- PySpark MLlib modeling on 14-day cutoff features
│   │   ├── 05_Cutoff21_Features.ipynb                 <- 21-day observation horizon feature engineering
│   │   └── 06_Cutoff21_Machine_Learning.ipynb         <- PySpark MLlib modeling on 21-day cutoff features
│   │
│   ├── models/                                        <- Serialized PySpark MLlib pipelines and model artifacts
│   │   ├── best_pipeline_A_operational_LogisticRegression/
│   │   ├── cutoff14/
│   │   └── cutoff21/
│   │
│   ├── outputs/                                       <- Evaluation metrics, confusion matrices, and ROC plots
│   │   ├── ml/                                        <- Full-window model metrics and plots
│   │   ├── ml_cutoff14/                               <- 14-day cutoff evaluation outputs
│   │   └── ml_cutoff21/                               <- 21-day cutoff evaluation outputs
│   │
│   ├── scripts/
│   │   └── download_survey_from_s3.py                 <- Secure Amazon S3 automated survey ingestion script
│   │
│   └── documentation/
│       └── fixed_cutoff_feature_design.md             <- Technical design document for temporal leakage mitigation
│
├── gata/                                              <- Interactive Streamlit prediction dashboard
│   ├── app.py                                         <- Operational Streamlit application
│   ├── requirements.txt                               <- Dashboard runtime dependencies
│   └── customer_churn_dataset.csv                     <- Benchmark evaluation dataset
│
├── Updated_ML_Models/                                 <- Team Scikit-learn comparative model suite (.pkl)
│   ├── Machine Learning model updated.ipynb
│   ├── logistic_regression_model.pkl
│   ├── random_forest_model.pkl
│   ├── gradient_boosting_model.pkl
│   ├── svm_model.pkl
│   ├── ridge_classifier_model.pkl
│   └── linear_regression_model.pkl
│
├── ML_Notebook/                                       <- Initial exploratory ML notebook & prototype
│   ├── data_eng.ipynb
│   └── customer_churn_dataset.csv
│
├── presentation/                                      <- Presentation slides and coursework visual assets
├── reports/                                           <- Academic project reports and technical submissions
└── research/                                          <- Background research, literature review, and references
```

---

## 10. Technology Stack

* **Programming Language**: Python 3.12 (Tested on 3.12.7)
* **Distributed Big Data Computing**: Apache Spark 4.3.0 (`pyspark.sql`, `pyspark.ml`)
* **Java Virtual Machine**: OpenJDK 17 / 21 (Required for PySpark driver & workers)
* **Data Processing & Formats**: Pandas 2.2+, NumPy, PyArrow 15+, DuckDB, Apache Parquet
* **Cloud & Storage**: Amazon AWS S3, Boto3, SQLite3
* **Machine Learning**: PySpark MLlib, Scikit-learn 1.4+
* **Data Visualization**: Matplotlib 3.8+, Seaborn 0.13+, Streamlit
* **Security & Environment**: Python-dotenv, OS-level environment variables
* **Version Control**: Git & GitHub

---

## 11. Step-by-Step Installation & Execution Guide

### Prerequisites
* **Python 3.12** installed.
* **Java 17 or Java 21** installed (e.g. via Homebrew: `brew install openjdk@17`).

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/atchyuth-karri/7COM2005-Data-Engineering.git
cd 7COM2005-Data-Engineering

# Create and activate Python 3.12 virtual environment
python3.12 -m venv .venv312
source .venv312/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r PerfectPetClub/requirements.txt
```

### 2. Configure PySpark Worker & Driver Environments (macOS / Linux)

To ensure PySpark workers match the active virtual environment Python binary:

```bash
export PYSPARK_PYTHON=$(which python)
export PYSPARK_DRIVER_PYTHON=$(which python)
export JAVA_HOME=$(/usr/libexec/java_home -v 17) # macOS with OpenJDK 17
```

### 3. Configure Cloud S3 Credentials (Optional Ingestion)

Copy `.env.example` to `.env` and supply the University S3 credentials:

```bash
cp .env.example .env
# Edit .env with your credentials:
# UH_S3_ACCESS_KEY=your_key
# UH_S3_SECRET_KEY=your_secret
# UH_S3_ENDPOINT_URL=https://s3.example.com
# UH_S3_BUCKET_NAME=your_bucket
```

Execute automated S3 survey retrieval:
```bash
python PerfectPetClub/scripts/download_survey_from_s3.py
```

### 4. Running the Complete Analytical Pipeline

Launch Jupyter Notebook:
```bash
jupyter notebook
```

Execute notebooks sequentially:
1. **`PerfectPetClub/notebooks/01_ETL_Pipeline.ipynb`**: Ingests raw data, validates financial spend, strips PII, and outputs `customer_churn_dataset.parquet`.
2. **`PerfectPetClub/notebooks/02_Machine_Learning.ipynb`**: Trains PySpark MLlib full-window baseline models and saves cross-validation metrics.
3. **`PerfectPetClub/notebooks/03_Fixed_Cutoff_Features.ipynb`**: Generates the 14-day fixed-cutoff leakage-free feature store.
4. **`PerfectPetClub/notebooks/04_Cutoff_Machine_Learning.ipynb`**: Evaluates PySpark MLlib models under the 14-day observation horizon.
5. **`PerfectPetClub/notebooks/05_Cutoff21_Features.ipynb`**: Generates the 21-day fixed-cutoff feature store.
6. **`PerfectPetClub/notebooks/06_Cutoff21_Machine_Learning.ipynb`**: Evaluates PySpark MLlib models under the 21-day observation horizon.

### 5. Launching the Interactive Streamlit Web Application

```bash
streamlit run gata/app.py
```
Open your browser at `http://localhost:8501` to explore the operational dashboard.

---

## 12. Academic Integrity & Version Control Standards

* **Author Attribution**: All commits across this repository are authored exclusively by registered University of Hertfordshire student team members (`ak26adx@herts.ac.uk`, `cn26aav@herts.ac.uk`, etc.).
* **Security & Secret Hygiene**: No live AWS/S3 access keys, passwords, or personal credentials have been committed. `.env` is strictly ignored by `.gitignore`.
* **Data Privacy**: Customer Personally Identifiable Information (PII) is completely excluded from all committed processed datasets and reports in compliance with GDPR.
* **Raw Data Immutability**: All raw datasets in `PerfectPetClub/data/raw/` are treated as read-only assets with cryptographic SHA-256 baseline verification.

---

### Module & Coursework Credits
* **Module Code:** 7COM2005 – Data Engineering
* **Degree Programme:** MSc Data Science / Advanced Computer Science
* **Institution:** University of Hertfordshire
* **Submission Date:** November 2026
