# PerfectPetClub – Data Engineering & Analytics Package

**Module:** 7COM2005 – Data Engineering  
**Institution:** University of Hertfordshire  
**Project Type:** Collaborative Team Project  

---

## 1. Package Overview

The `PerfectPetClub` directory is the core data engineering and machine learning package of the project. It encapsulates:
* Ingestion scripts for multi-source operational data (SQLite, CSV, and Amazon S3).
* Verified end-to-end ETL transformations producing a GDPR-compliant canonical feature store in Apache Parquet.
* Distributed **Apache Spark MLlib (`pyspark.ml`)** machine learning pipelines for customer churn classification.
* Advanced research notebooks resolving prediction-time temporal data leakage using fixed-cutoff observation horizons (14-day and 21-day windows).
* Serialized model artifacts, evaluation metrics, and visual diagnostic outputs.

---

## 2. Directory Structure

```text
PerfectPetClub/
│
├── data/
│   ├── raw/                                           <- Read-only original source datasets (SHA-256 verified)
│   │   ├── PerfectPetClub.db                          <- SQLite transactional database (9 tables)
│   │   ├── PerfectPetClub_CRM_report.csv              <- CRM support logs (7,585 records)
│   │   ├── PerfectPetClub_server_log.csv              <- Web server access logs (67,249 records)
│   │   └── PerfectPetClub_survey.csv                  <- Survey responses (221 rows, S3 ingested)
│   │
│   ├── processed/                                     <- Canonical analytical feature store
│   │   ├── customer_churn_dataset.parquet             <- Single source of truth (221 rows × 36 columns)
│   │   ├── customer_churn_dataset.csv                 <- CSV export for inspection
│   │   └── data_quality_report.md                     <- Detailed data quality and financial audit report
│   │
│   ├── cutoff14/                                      <- 14-day observation cutoff feature store
│   └── cutoff21/                                      <- 21-day observation cutoff feature store
│
├── notebooks/                                         <- Step-by-step reproducible Jupyter workflows
│   ├── 01_ETL_Pipeline.ipynb                          <- Multi-source ETL & financial reconciliation
│   ├── 02_Machine_Learning.ipynb                      <- PySpark MLlib classification (full-window baseline)
│   ├── 03_Fixed_Cutoff_Features.ipynb                 <- 14-day observation horizon feature engineering
│   ├── 04_Cutoff_Machine_Learning.ipynb               <- PySpark MLlib modeling on 14-day cutoff features
│   ├── 05_Cutoff21_Features.ipynb                     <- 21-day observation horizon feature engineering
│   └── 06_Cutoff21_Machine_Learning.ipynb             <- PySpark MLlib modeling on 21-day cutoff features
│
├── models/                                            <- Serialized PySpark MLlib pipeline artifacts
│   ├── best_pipeline_A_operational_LogisticRegression/
│   ├── cutoff14/
│   └── cutoff21/
│
├── outputs/                                           <- Model performance tables, predictions, and plots
│   ├── ml/                                            <- Full-window model metrics and ROC/PR plots
│   ├── ml_cutoff14/                                   <- 14-day cutoff evaluation tables and charts
│   └── ml_cutoff21/                                   <- 21-day cutoff evaluation tables and charts
│
├── scripts/
│   └── download_survey_from_s3.py                     <- Cloud S3 survey ingestion script
│
├── documentation/
│   └── fixed_cutoff_feature_design.md                 <- Technical specification for temporal leakage fix
│
├── requirements.txt                                   <- Pinned package dependencies
└── README.md                                          <- This documentation file
```

---

## 3. Data Sources

The raw datasets in `data/raw/` are immutable and cryptographically audited:

1. **`PerfectPetClub.db` (SQLite)**:
   - Relational operational database with 9 tables: `customer`, `subscription`, `subscription_type`, `sale`, `payment`, `sale_product`, `product`, `product_category` and `product_review`. Totaling 117,790 rows.
2. **`PerfectPetClub_CRM_report.csv`**:
   - Customer support ticket logs capturing queries, channels (`e-mail`, `live-chat`, `telephone`), and issue types.
3. **`PerfectPetClub_server_log.csv`**:
   - Web server access logs detailing user browsing behavior, IP addresses, HTTP status codes, and sessions.
4. **`PerfectPetClub_survey.csv`**:
   - Customer satisfaction feedback, Net Promoter Score (NPS), household pet counts, income brackets, and price sensitivity. Ingested securely from an authenticated AWS S3 bucket.

---

## 4. Notebooks Workflow Summary

Execute the notebooks in sequential order:

* **`notebooks/01_ETL_Pipeline.ipynb`**:
  * Harmonizes heterogeneous schemas and validates cryptographic SHA-256 hashes.
  * Reconciles £1,058,642.73 in transaction revenue against payment records with **£0.00 discrepancy**.
  * Eliminates customer PII in compliance with GDPR.
  * Writes the canonical feature dataset to `data/processed/customer_churn_dataset.parquet` (221 rows × 36 columns, 0 nulls, 0 duplicate keys).
* **`notebooks/02_Machine_Learning.ipynb`**:
  * Builds a modular **PySpark MLlib (`pyspark.ml`)** pipeline with `StringIndexer`, `OneHotEncoder`, `MedianImputer`, `VectorAssembler`, and `StandardScaler`.
  * Trains 3-fold cross-validated Logistic Regression (CV AUC 0.776), Random Forest (CV AUC 0.717), and GBT (CV AUC 0.707).
  * Documents the diagnostic finding: high AUC is inflated by subscription tenure leakage in cumulative count features.
* **`notebooks/03_Fixed_Cutoff_Features.ipynb`**:
  * Re-engineers customer features strictly utilizing events observed within `subscription_start + 14 days`.
  * Generates `data/cutoff14/customer_churn_dataset_cutoff14.parquet`.
* **`notebooks/04_Cutoff_Machine_Learning.ipynb`**:
  * Evaluates PySpark MLlib models under the leakage-free 14-day temporal horizon.
  * Yields realistic early-signal performance (CV AUC 0.48–0.60), providing clear business rationale for proactive post-onboarding engagement.
* **`notebooks/05_Cutoff21_Features.ipynb` & `notebooks/06_Cutoff21_Machine_Learning.ipynb`**:
  * Evaluates an extended 21-day observation horizon to examine feature signal maturation over subscription life.

---

## 5. Execution & Runtime Setup

### Environment Setup

```bash
# Create and activate Python 3.12 virtual environment from repo root
python3.12 -m venv .venv312
source .venv312/bin/activate

# Install package dependencies
pip install -r PerfectPetClub/requirements.txt
```

### PySpark Worker Configuration

```bash
# Set Python runtime paths for PySpark
export PYSPARK_PYTHON=$(which python)
export PYSPARK_DRIVER_PYTHON=$(which python)
export JAVA_HOME=$(/usr/libexec/java_home -v 17) # macOS
```

### Running Jupyter Notebooks

```bash
jupyter notebook PerfectPetClub/notebooks/
```
