# PerfectPetClub ETL Data Quality & Transformation Report

**Project:** PerfectPetClub Data Engineering Pipeline  
**Module:** 7COM2005 Data Engineering Team Project (University of Hertfordshire)  
**Date:** 2026-11-13  
**Target Dataset:** `PerfectPetClub/data/processed/customer_churn_dataset.parquet`  
**Dimensions:** Exactly 221 unique customer rows × 36 engineered features  
**Execution Status:** Verified reproducible end-to-end execution (0 errors, 0 missing values, 0 duplicate IDs)  

---

## 1. Executive Summary & Source Inventory

The ETL pipeline ingests all five required raw datasets from heterogeneous operational and analytical systems into a unified, clean, privacy-preserving customer-level analytical feature store.

| Source System | File / Entity | Ingestion Mode | Records Ingested | Join Key to Customer | Role in Output Dataset |
|---|---|---|---|---|---|
| **SQLite Database** | `customer` | Read-only SQL | 221 | `id` (PK) | Base entity spine (`customer_id`) |
| **SQLite Database** | `subscription` + `subscription_type` | Read-only SQL | 221 | `subscription_id` | Churn target (`auto_renew`), plan tier, tenure |
| **SQLite Database** | `sale` + `payment` | Read-only SQL | 23,188 | `customer_id` | Order volume, monetary spend, delivery delay metrics |
| **SQLite Database** | `sale_product` + `product` | Read-only SQL | 59,759 | `sale_id` → `customer_id` | Quantity of items, unique products purchased |
| **SQLite Database** | `product_review` | Read-only SQL | 10,990 | `customer_id` | Review volume, polarity, satisfaction proxy |
| **CRM System** | `PerfectPetClub_CRM_report.csv` | Chunked CSV read | 7,585 | `customer_id` | Customer service tickets, complaints, channels |
| **Web Server Logs** | `PerfectPetClub_server_log.csv` | Chunked CSV read | 67,249 | `login` → `customer.email` | Total sessions, browsing recency |
| **Customer Survey** | `PerfectPetClub_survey.csv` | Read-only CSV / S3 | 221 | `email` → `customer.email` | Customer sentiment, NPS, income bracket, price sensitivity |

All source files were verified for read-only integrity with SHA-256 cryptographic hashes before and after processing.

---

## 2. Credential Security & Ingestion Architecture

In accordance with enterprise data engineering standards and university information security policies:

1. **Zero Hardcoded Secrets**:
   - Access keys, secret keys, and internal endpoints are loaded dynamically at runtime via `python-dotenv` and system environment variables (`UH_S3_ACCESS_KEY`, `UH_S3_SECRET_KEY`, `UH_S3_ENDPOINT_URL`, `UH_S3_BUCKET_NAME`).
2. **Git Exclusion Verification**:
   - The `.env` file containing live credentials is explicitly matched and excluded by `.gitignore` (line 226: `*.env`).
   - A sanitized template, `.env.example`, is tracked in the repository root to document required environment variables without leaking credentials.
3. **Repository History Audit**:
   - Full Git commit history was audited (`git log -S`, `git grep`, `git log -p`), confirming zero credentials have ever been committed or staged.
4. **Resilient S3 Automated Downloader**:
   - `PerfectPetClub/scripts/download_survey_from_s3.py` implements secure parameterization, socket timeouts (5s connect, 15s read), and descriptive network diagnostics for UH VPN requirements.

---

## 3. Data Cleaning & Transformation Decisions

1. **Survey Validation & PII Elimination**:
   - All 5 survey columns (`email`, `reported_income`, `nps_recommend`, `satisfaction_score`, `online_comparison_freq`) were validated.
   - All 221 respondents matched 1-to-1 with the customer database.
   - Numeric survey scores were validated to strictly adhere to the `[0, 10]` scale.
   - **PII Protection**: The raw `email` field was discarded immediately after mapping to `customer_id`, preventing personal data leakage into the analytical modeling dataset.
2. **CRM Header Typo Correction**:
   - The misspelled column `recieved_datatime` was renamed in-memory to `received_datetime`. The raw source CSV was left completely unmodified.
3. **Platform String Normalization**:
   - Platform values stored as Python string-encoded lists (e.g. `"['e-mail']"`) were cleaned into standardized categories (`"e-mail"`, `"telephone"`, `"live-chat"`).
4. **Delivery Status Discrimination**:
   - Sales records were partitioned into completed orders (`delivered_date.notna()`) and unfulfilled orders (`delivered_date.isna()`).
   - Delivery delay in hours was strictly computed for completed deliveries where `delivered_date > expected_date`.
5. **Memory-Efficient Server Log Ingestion**:
   - The 67,249-row server log CSV (~12 MB) was processed in streaming chunks of 20,000 rows.
   - Logins were translated immediately to integer customer IDs; IP addresses and raw user-agent strings were dropped to prevent memory overhead and sensitive data exposure.

---

## 4. Missing-Value Treatment & Imputation Audit

* **Survey Features (0 missing)**: Exactly 221 out of 221 customers completed the popup survey (100% complete).
* **CRM Interactions (21 customers with 0 support tickets)**:
  - Ticket and channel interaction counts imputed with `0`.
  - `crm_complaint_ratio` safely imputed with `0.0` using a zero-division guard.
* **Product Reviews (8 customers with 0 reviews)**:
  - Review counts imputed with `0`.
  - Binary indicator `has_reviews` set to `0.0`.
  - `avg_star_rating` set to neutral median value `3.0` to preserve mathematical scaling without biasing sentiment upwards or downwards.
* **Final Completeness**: Exactly **0 missing values** across all 36 columns.

---

## 5. Financial Reconciliation & Aggregation Audit

A comprehensive mathematical audit reconciled all aggregated features against primary operational records:

1. **Total Spend Reconciliation**:
   - Sum of customer `total_spend_pence` in final dataset: **£1,058,642.73** (105,864,273 pence).
   - Sum of `amount` in SQLite `payment` table: **£1,058,642.73** (105,864,273 pence).
   - Discrepancy: **£0.00 (100% exact match)** across all 221 subscribers.
2. **Order Volume Reconciliation**:
   - Customer-by-customer order counts match the SQLite `sale` table with 0 discrepancy (23,188 total transactions).
3. **CRM Support Interaction Reconciliation**:
   - Sum of customer `total_crm_interactions` matches the raw CSV row count exactly: **7,585 records (0 discrepancy)**.
4. **Delivery Performance Reconciliation**:
   - Completed deliveries sum to **22,098** and unfulfilled orders sum to **1,090** across all 221 customers, exactly matching the 23,188 rows in the SQLite `sale` table.

---

## 6. Final Feature Catalog (36 Attributes)

### Group 1: Base & Subscription Demographics (5 features)
* `customer_id` (int64) - Unique subscriber identifier (primary key)
* `subscription_tier` (str) - Subscription plan level (`Basic`, `SuperSaver`, `Premium`)
* `delivery_service` (str) - Primary courier provider (`Royal Mail`, `Evri`, `DPD`)
* `duration_days` (int64) - Active subscription duration in days
* `auto_renew` (int64) - **Supervised Target Variable** (1 = Renewed, 0 = Churned)

### Group 2: Purchasing Behaviour (6 features)
* `total_orders` (int64) - Cumulative orders placed
* `total_spend_pence` (int64) - Cumulative spend in pence
* `avg_order_value_pence` (float64) - Mean transaction value
* `total_items_purchased` (int64) - Total unit volume purchased
* `unique_products_purchased` (int64) - Diversity of catalog items purchased
* `avg_items_per_order` (float64) - Mean items per order

### Group 3: Delivery Performance (5 features)
* `completed_deliveries` (int64) - Count of successfully delivered orders
* `unfulfilled_orders` (int64) - Count of unfulfilled or pending deliveries
* `delayed_deliveries` (int64) - Orders delivered after expected delivery date
* `avg_delivery_delay_hours` (float64) - Mean delivery delay duration
* `delivery_delay_ratio` (float64) - Delayed deliveries divided by completed deliveries

### Group 4: Customer Support / CRM (7 features)
* `total_crm_interactions` (float64) - Total customer support tickets
* `crm_complaints_count` (float64) - Support tickets classified as complaints
* `crm_queries_count` (float64) - Support tickets classified as inquiries
* `crm_email_count` (float64) - Support contacts via email channel
* `crm_phone_count` (float64) - Support contacts via telephone channel
* `crm_livechat_count` (float64) - Support contacts via live-chat channel
* `crm_complaint_ratio` (float64) - Proportion of support contacts that were complaints

### Group 5: Product Reviews (5 features)
* `total_reviews` (float64) - Product reviews written by subscriber
* `avg_star_rating` (float64) - Mean review score (1–5 scale, neutral 3.0 if no reviews)
* `low_star_reviews_count` (float64) - Unfavorable reviews (<= 2 stars)
* `high_star_reviews_count` (float64) - Favorable reviews (>= 4 stars)
* `has_reviews` (float64) - Binary indicator (1.0 = has submitted reviews, 0.0 = none)

### Group 6: Web Engagement (4 features)
* `total_web_sessions` (int64) - Total logged website sessions
* `days_since_last_session` (float64) - Recency of latest web visit before renewal point
* `session_frequency_per_month` (float64) - Normalized web engagement velocity
* `order_frequency_per_month` (float64) - Normalized order placement velocity

### Group 7: Customer Survey & Sentiment (4 features)
* `reported_income` (str) - Self-reported household income bracket (e.g. `34k-38k`)
* `nps_recommend` (int64) - Net Promoter Score willingness to recommend (scale 0–10)
* `satisfaction_score` (int64) - Overall customer satisfaction score (scale 0–10)
* `online_comparison_freq` (int64) - Frequency of competitor price comparison (scale 0–10)

---

## 7. Target Variable Distribution

The target variable is `subscription.auto_renew`:
* **Renewed (`1`)**: 135 customers (**61.09%**)
* **Churned (`0`)**: 86 customers (**38.91%**)
* **Class Imbalance Ratio**: Approximately 1.57:1 (mild imbalance, suitable for stratified cross-validation).

---

## 8. Temporal Leakage Limitations & Dual-Model Evaluation Strategy

### Critical Risk: Lack of Survey Timestamps
1. **Unverifiable Collection Point**:
   - `PerfectPetClub_survey.csv` contains customer responses without a timestamp or date-of-completion attribute.
   - Although documented as a "WebApp Popup Survey", it cannot be established whether responses occurred early during the subscription, midway through, or as an exit survey immediately prior to or following non-renewal.
2. **Potential for Data Leakage**:
   - If low satisfaction or low NPS scores were recorded at the moment of cancellation, using these features to predict churn constitutes temporal target leakage (predicting an outcome using data generated as a consequence of the outcome).
   - Conversely, high satisfaction scores might simply reflect active, happy subscribers responding during renewal.

### Methodological Solution: Dual-Model Benchmarking
To critically evaluate this phenomenon in Section 4 (Machine Learning), models must be trained and compared under two distinct configurations:

* **Configuration A — Operational Baseline (31 features + target)**:
  - Uses strictly transactional, operational, CRM, delivery, and web log features.
  - All operational features have verifiable temporal boundaries (100% occurring between `start_date` and `end_date`).
* **Configuration B — Multimodal Full Feature Set (35 features + target)**:
  - Ingests all operational features plus the 4 survey attributes (`reported_income`, `nps_recommend`, `satisfaction_score`, `online_comparison_freq`).
  - Evaluates feature importance and performance gain (ROC-AUC / F1) against Configuration A to isolate and quantify the survey's incremental contribution and potential leakage footprint.

Neither configuration should be prematurely claimed to be "leakage-free" without empirical verification and ablation analysis.

---

## 9. Final Execution Verification

* **Notebook**: `PerfectPetClub/notebooks/01_ETL_Pipeline.ipynb`
* **Python Runtime**: 3.14.2 (`.venv`)
* **Execution Result**: All 24 cells executed sequentially with **0 errors**.
* **Output Format**: Parquet with Snappy compression (`customer_churn_dataset.parquet`).
* **Parquet Roundtrip Verification**: Completed and confirmed identical to in-memory DataFrame.
