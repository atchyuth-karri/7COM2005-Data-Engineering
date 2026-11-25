# PerfectPetClub — Final Technical Audit

Scope: `01_ETL_Pipeline.ipynb` and `02_Machine_Learning.ipynb` (PerfectPetClub/notebooks/). Notebooks 03–06 (14- and 21-day cutoff research) were not audited in depth; they were only checked for output paths that could collide with the main pipeline.

---

## 1. Project Structure

Raw sources (`PerfectPetClub/data/raw/`) → ETL notebook → `PerfectPetClub/data/processed/customer_churn_dataset.parquet` → PySpark MLlib notebook → `outputs/ml/` and `models/`.

| Item | Actual file |
|---|---|
| ETL notebook | `PerfectPetClub/notebooks/01_ETL_Pipeline.ipynb` |
| ML notebook | `PerfectPetClub/notebooks/02_Machine_Learning.ipynb` |
| SQLite source | `PerfectPetClub/data/raw/PerfectPetClub.db` (9 tables) |
| CSV sources | `PerfectPetClub_CRM_report.csv`, `PerfectPetClub_server_log.csv`, `PerfectPetClub_survey.csv` |
| Processed dataset | `PerfectPetClub/data/processed/customer_churn_dataset.parquet` |
| Requirements | `PerfectPetClub/requirements.txt` (pandas `<3`, pyspark `>=4,<5`) |
| ML outputs | `PerfectPetClub/outputs/ml/` (`model_results.csv`, PNG charts, split table) |
| Saved model | `PerfectPetClub/models/best_pipeline_A_operational_LogisticRegression/` |
| Data quality report | `PerfectPetClub/data/processed/data_quality_report.md` (pre-existing, not regenerated) |

Architecture check: the ETL → Parquet → PySpark path is implemented as described. Logistic Regression, Random Forest and GBT are all implemented in notebook 02. The "Decision Tree" in the brief is not implemented; GBT is used in its place.

---

## 2. ETL Audit

- **Data sources:** SQLite (9 tables, 117,790 rows); CRM CSV (7,585 rows); server log CSV (67,249 rows); survey CSV (221 rows).
- **Extraction:** SQLite opened read-only (`mode=ro`). CSVs read with pandas, the server log in 20,000-row chunks. Paths are resolved from the notebook's location, so the notebook runs from `notebooks/` or the project root.
- **Transformation:** one row per customer (`customer.id`), joined from `customer` → `subscription` → `subscription_type`, then left-joined with sales/payments, line items, CRM, reviews, web sessions and survey. Email is mapped to `customer_id` and then dropped (PII removal). Server-log IP and user-agent are dropped.
- **Feature engineering:** purchase (orders, spend in pence, items), delivery (delay counts and ratio), CRM (complaints, channels), reviews, web engagement, survey (4 raw fields). Imputation: zero for CRM counts, zero for review counts, 3.0 for `avg_star_rating` where there are no reviews (see Issues).
- **Data quality checks:** missing-value counts, identifier integrity across sources, timestamp validity, delivery-delay statistics, target distribution, a read-only revenue reconciliation (added in this audit), and a final validation function with assertions (row count, unique IDs, no duplicates, no nulls, target present, survey columns present).
- **Leakage:** the ETL aggregates over each subscription's full window (`start_date` to `end_date`). This is not a leakage control. For churned customers the window ends at cancellation, so features include activity near the decision. See Issues, item 4.
- **Final dataset:** `customer_churn_dataset.parquet`, 221 rows × 36 columns, 0 nulls, unique `customer_id`, target `auto_renew` (135 renewed / 86 not renewed). Read back successfully.

---

## 3. ETL Verification

- **Execution status:** PASS. Executed from a fresh kernel with nbconvert (`.venv312`: Python 3.12, pandas 2.3.3, PySpark 4.3.0). 18/18 code cells executed, 0 errors.
- **Dataset dimensions:** 221 × 36 (verified after the re-run).
- **Schema:** `customer_id` int64; `subscription_tier` and `delivery_service` string; `reported_income` string; counts int64; ratios and averages float64; target `auto_renew` int64. Full list in the notebook output.
- **Parquet round-trip:** passes. The re-run output is content-identical to the pre-audit file (`assert_frame_equal` with exact comparison). The tracked file bytes were restored to the committed version, so the repository's Parquet is unchanged.
- **Quality checks (as printed by the notebook and re-checked here):**
  - CRM: 7,585 rows, 0 duplicates, 0 nulls; 0 tickets dated outside the subscription window.
  - Server log: 67,249 rows, 0 unmatched logins, 0 events outside the subscription window.
  - Orders: 0 before start or after end.
  - Target: 135 renewed (61.1%) / 86 churned (38.9%), 0 missing.
  - Missing values in source: `product.product_category_id` 3; `sale.expected_date` and `delivered_date` 1,090 (unfulfilled orders).
  - Delayed deliveries: 856 of 22,098 fulfilled orders (3.87%).
  - Revenue reconciliation: sale line items £1,058,642.73 = payments £1,058,642.73; discrepancy 0 pence.
  - 21 customers have no CRM tickets; 8 have no reviews.

---

## 4. Machine Learning Audit

- **Input dataset:** the ETL Parquet only. No generated or random data is used in notebook 02 (checked by grep).
- **Target:** `churn_label = 1 − auto_renew`; positive class = churn (did not renew). Target is not in the feature list.
- **Features:**
  - Configuration A (operational): 30 features (2 categorical, 28 numeric).
  - Configuration B (A plus survey): 34 features (adds `reported_income` and three survey scores).
  - Configuration S (sensitivity only): 12 features, rates and flags only, with `duration_days` and raw totals removed.
- **Preprocessing (inside a Pipeline, fitted on the training fold only):** StringIndexer (handleInvalid=keep) → OneHotEncoder → Imputer (median) → VectorAssembler → StandardScaler (withMean=False). Missing `avg_star_rating` is set from `has_reviews`, not from a value.
- **Split:** exact stratified 80/20, seed 42. Train 177 (108 renewed / 69 churned), test 44 (27 / 17). 0 customer overlap.
- **Models implemented:** LogisticRegression, RandomForestClassifier, GBTClassifier. No Decision Tree.
- **Training:** 3-fold CrossValidator on the training set, maximising ROC-AUC. Grids: LR 6 combinations, RF 4, GBT 4. Test data is not used for selection. Final model selected by CV AUC among A and B.
- **Seeds:** SEED = 42 for the split, CV and tree models. The split uses a seeded random order within a single partition, so it is deterministic.
- **Reproducibility:** the re-run produced `outputs/ml/model_results.csv` byte-identical to the pre-audit file.

---

## 5. Model Evaluation

Test set: 44 customers, 17 churned. Metrics are as produced by the notebook. Churn is the positive class.

| Config | Model | CV AUC | Test AUC | Accuracy | Churn precision | Churn recall | Churn F1 | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | LogisticRegression | 0.776 | 0.747 | 0.750 | 0.750 | 0.529 | 0.621 | 24 | 3 | 8 | 9 |
| A | RandomForest | 0.717 | 0.717 | 0.636 | 0.556 | 0.294 | 0.385 | 23 | 4 | 12 | 5 |
| A | GBTClassifier | 0.707 | 0.675 | 0.659 | 0.556 | 0.588 | 0.571 | 19 | 8 | 7 | 10 |
| B | LogisticRegression | 0.767 | 0.739 | 0.727 | 0.692 | 0.529 | 0.600 | 23 | 4 | 8 | 9 |
| B | RandomForest | 0.711 | 0.730 | 0.659 | 0.625 | 0.294 | 0.400 | 24 | 3 | 12 | 5 |
| B | GBTClassifier | 0.696 | 0.734 | 0.659 | 0.556 | 0.588 | 0.571 | 19 | 8 | 7 | 10 |
| S | LogisticRegression | 0.622 | 0.551 | 0.591 | 0.000 | 0.000 | 0.000 | 26 | 1 | 17 | 0 |
| S | RandomForest | 0.643 | 0.564 | 0.636 | 0.600 | 0.176 | 0.273 | 25 | 2 | 14 | 3 |
| S | GBTClassifier | 0.619 | 0.576 | 0.614 | 0.500 | 0.353 | 0.414 | 21 | 6 | 11 | 6 |

**Selected model:** Configuration A, LogisticRegression (CV AUC 0.776, the highest CV AUC among A and B), saved to `models/best_pipeline_A_operational_LogisticRegression/`.

**How to read these numbers:**
- One test customer changes accuracy by 2.3 percentage points. Differences of one or two customers are noise.
- Configuration B vs A: no consistent benefit from survey features (CV AUC lower for all three models).
- Configuration S falls to near chance (test AUC 0.55–0.58). This shows that most of the apparent signal in A comes from tenure and raw totals, which are tied to the outcome.
- These are diagnostics of the full-window features, not a forecast of renewal before the decision. The notebook says this itself.

---

## 6. Issues Identified

1. **Wrong tier names in ETL notebook.** The notebook said tiers were "Standard, Premium, VIP" and "Bronze, Silver, Gold". The data has Basic, Premium, SuperSaver. Fixed (see 7).
2. **Wrong feature counts in ETL notebook.** It said Configuration A has 31 features and B has 35. The ML notebook has 30 and 34. Fixed.
3. **Misleading leakage wording in ETL notebook.** It described the subscription-window check as a "Data Leakage Safeguard". The window ends at cancellation for churned customers, so the features include outcome-proximate activity. Fixed wording.
4. **Prediction-time leakage (not fixed).** Features cover each subscription's full window. For churned customers 17.7% of orders fall in the final 30 days, against 7.4% for renewed customers. Churned median duration is 90 days against 180. Configuration S shows the consequence. Fixing this requires changing the ETL logic, which notebook 02 says needs the team's approval. Notebooks 03–04 (14-day cutoff) implement a fix, with weaker results (CV AUC 0.48–0.60 per the README; not re-verified here).
5. **Survey questionnaire error.** Q2 (`nps_recommend`) and Q3 (`satisfaction_score`) have the same question text. Survey rows have no timestamp, so their timing relative to cancellation is unknown.
6. **Average-rating sentinel.** The ETL fills `avg_star_rating` with 3.0 for customers without reviews. 13 customers with real reviews also have exactly 3.0. The ML notebook masks it using `has_reviews`. The ETL output still contains the 3.0 fill.
7. **README errors.** It said 8 SQLite tables with 94,158 records. The database has 9 tables with 117,790 rows. Fixed.
8. **Reconciliation claim had no code.** README said the ETL reconciles revenue with a £0.00 discrepancy, but no notebook cell did. The reconciliation was true in the data, so a check cell was added to the ETL notebook.
9. **Environment mismatch (fixed in the local venv).** The existing ETL output was produced with pandas 3.1.1 and Python 3.14, which violates the `pandas<3` pin in `requirements.txt`. The `.venv312` environment was set to `pandas>=2.2.0,<3` (2.3.3). All runs in this audit use that environment.
10. **Legacy synthetic-data artefacts (not used, not fixed).** Root-level `Machine Learning model updated.ipynb`, `Updated_ML_Models/*.pkl`, `ML_Notebook/`, root `customer_churn_dataset.csv` and `gata/` are built on `np.random` generated data (for example `CUST_0001` IDs). They are not part of the PerfectPetClub pipeline and must not be presented as results. They were left in place.
11. **Stale outputs (not regenerated).** `PerfectPetClub/outputs/etl/`, `outputs/ml/`, `reports/` and the root `all_results_report.html` pre-date this audit and were not regenerated. They are not used in the submission.

---

## 7. Fixes Applied

| # | What was wrong | Why | Change | Verification |
|---|---|---|---|---|
| 1 | Tier names wrong in ETL notebook (cell output text and Section 3 markdown) | Hand-written text did not match `subscription_type` | Replaced with Basic, Premium, SuperSaver | Checked against `subscription_type` table |
| 2 | Feature counts wrong (31/35) | Hand-written counts | 30 / 34 | Counted in ML notebook cell 16 |
| 3 | Leakage wording overstated | Window treated as a safeguard | Reworded as a window check, not a leakage control; pointed to notebooks 02/03 | Read against ML notebook findings |
| 4 | Section 4 described as future work with a placeholder status | Stale text | Now points to notebook 02 | Read |
| 5 | README: 8 tables, 94,158 records | Miscount | 9 tables, 117,790 rows | Counted from SQLite |
| 6 | Revenue reconciliation claim with no code | Missing check | New cell 2.7 in ETL notebook, asserts zero discrepancy | Executed: 105,864,273 = 105,864,273 pence |
| 7 | pandas 3.1.1 vs `pandas<3` pin | Environment drift | `.venv312` set to pandas 2.3.3 (not a repo file) | `pip show`; both notebooks executed under it |

Not changed, by design: the ETL business logic and the features, the ML models and grids, and the survey handling. Changing these would change results, which the brief prohibits without cause.

---

## 8. Limitations

- **Small dataset:** 221 customers. The test set has 44 customers and 17 churners. Metrics move by several points with one or two customers.
- **Prediction-time leakage:** see Issue 4. Headline numbers (Configurations A and B) are not a valid pre-decision forecast.
- **Label is a proxy:** `auto_renew = 0` is treated as churn. Cancellation was not verified against independent records.
- **Survey:** no timestamps; questionnaire wording error (Issue 5); zeros are kept as valid answers, which was not tested.
- **Imbalance:** 39% churn. Mild, and handled by stratified splitting. No oversampling was used.
- **Threshold and calibration:** default 0.5 threshold, not tuned or calibrated.
- **CV folds:** random, not stratified (Spark `CrossValidator` has no stratified option).
- **Generalisability:** a single company's data. No time-based holdout.
- **Model choice:** GBT and Random Forest recall trade off against precision. The notebook does not decide the business cost trade-off.

---

## 9. Submission Deliverables

Folder: `PerfectPetClub/submission/` (created in this audit).

| File | Size | Source |
|---|---|---|
| `PerfectPetClub/submission/ETL_Pipeline.html` | 465,998 bytes | executed copy of `01_ETL_Pipeline.ipynb` |
| `PerfectPetClub/submission/Machine_Learning.html` | 715,685 bytes | executed copy of `02_Machine_Learning.ipynb` |
| `PerfectPetClub/submission/ETL_Pipeline.ipynb` | 77,399 bytes | executed notebook (optional, for re-running) |
| `PerfectPetClub/submission/Machine_Learning.ipynb` | 349,489 bytes | executed notebook (optional, for re-running) |

The original notebooks in `PerfectPetClub/notebooks/` are unchanged by the export, except for the ETL text fixes and the added reconciliation cell (see Section 7).

---

## 10. Final Status

- **ETL:** PASS (after fixes listed in Section 7)
- **ML:** PASS for execution and reproducibility. The metrics are diagnostic only, because of prediction-time leakage (Issue 4).
- **Notebook execution:** PASS. 01: 18/18 code cells, 0 errors. 02: 15/15 code cells, 0 errors. Both run in fresh kernels via nbconvert.
- **HTML exports:** PASS. Both files are non-zero, contain code and outputs, have 0 tracebacks, and embed the figures.
- **Submission readiness:** READY, conditional on presenting the leakage limitation alongside the Configuration A/B numbers, and on the team deciding whether to add the cutoff results from notebooks 03–04 to the submission.
