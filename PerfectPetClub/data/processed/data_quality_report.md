# PerfectPetClub ETL Data Quality & Transformation Report

**Generated:** 2026-11-13  
**Target Output:** `PerfectPetClub/data/processed/customer_churn_dataset.parquet`  
**Records Produced:** 221 unique customer rows  

---

## 1. Source Row Counts & Ingestion Summary
| Source | Entity / Filename | Records Ingested | Join Key to Customer | Role in Output |
|---|---|---|---|---|
| **SQLite `customer`** | `customer` | 221 | `id` (PK) | Base entity spine |
| **SQLite `subscription`** | `subscription`, `subscription_type` | 221 | `subscription_id` | Churn target (`auto_renew`), plan tier, tenure |
| **SQLite `sale` & `payment`** | `sale`, `payment` | 23,188 | `customer_id` | Orders, monetary spend, delivery delay metrics |
| **SQLite `sale_product`** | `sale_product` | 59,759 | `sale_id` → `customer_id` | Quantity of items, unique products purchased |
| **SQLite `product_review`** | `product_review` | 10,990 | `customer_id` | Average star rating, review volume, polarity |
| **CRM Report** | `PerfectPetClub_CRM_report.csv` | 7,585 | `customer_id` | Customer service tickets, complaints, channels |
| **Server Log** | `PerfectPetClub_server_log.csv` | 67,249 | `login` → `customer.email` | Total sessions, browsing recency |
| **Survey Data** | `PerfectPetClub_survey.csv` | 0 | None (Empty 0 bytes) | Noted as missing; no data fabricated |

---

## 2. Cleaning Operations & Methodological Justifications

1. **CRM Header Typo Correction**:
   - The original CSV header `recieved_datatime` was renamed in-memory to `received_datetime`. The raw file was preserved unchanged.
2. **CRM Platform String Normalization**:
   - Platform values stored as string-encoded lists (e.g., `"['e-mail']"`) were cleaned into standard categorical strings (`"e-mail"`, `"telephone"`, `"live-chat"`).
3. **Delivery Status Discrimination**:
   - `sale.delivered_date` was evaluated to distinguish completed orders (`is_completed = True`) from unfulfilled/in-flight orders (`is_unfulfilled = True`).
   - Delivery delay in hours was strictly computed when `delivered_date > expected_date` on completed deliveries.
4. **Memory-Efficient Server Log Ingestion**:
   - Read in chunks of 20,000 rows.
   - User email logins were immediately mapped to integer `customer_id`, discarding raw IP addresses and user agents to eliminate memory bloat and PII leakage.

---

## 3. Missing-Value Treatment
- **CRM Interactions (21 customers with 0 support tickets)**: Imputed with `0` for ticket/complaint counts and `0.0` for `crm_complaint_ratio`.
- **Product Reviews (8 customers with 0 reviews)**: Imputed with `0` for review counts, `has_reviews = 0`, and filled `avg_star_rating` with neutral `3.0`.
- **Final Dataset Completeness**: Exactly **0 missing values** across all 32 engineered features.

---

## 4. Aggregation Methods
- All transactional and interaction events were aggregated to the customer level using `GROUP BY customer_id`.
- Monetary values were summed across sales (`total_spend_pence`).
- Ratios were safely computed with zero-division guards (e.g. `delivery_delay_ratio`, `crm_complaint_ratio`).
- Normalized rates were computed relative to subscription duration (e.g. `order_frequency_per_month`, `session_frequency_per_month`).

---

## 5. Join Validation & Entity Integrity
- **Customer Spine**: 221 base customers.
- **Post-Join Row Count**: Exactly 221 rows (1-to-1 customer mapping).
- **Duplicate Customer IDs**: 0.
- **Target Variable Completeness**: 0 missing values in `auto_renew`.

---

## 6. Final Feature Catalog (32 Attributes)
1. `customer_id` (int64) - Unique subscriber ID
2. `subscription_tier` (object) - Basic, SuperSaver, Premium
3. `delivery_service` (object) - Delivery provider (Royal Mail, Evri, DPD)
4. `duration_days` (int64) - Length of subscription period
5. `auto_renew` (int64) - **Target variable** (1 = Renewed, 0 = Churned)
6. `total_orders` (int64) - Total purchases placed
7. `total_spend_pence` (int64) - Total monetary spend in pence
8. `avg_order_value_pence` (float64) - Average spend per transaction
9. `total_items_purchased` (int64) - Total units purchased
10. `unique_products_purchased` (int64) - Diversity of catalog items bought
11. `avg_items_per_order` (float64) - Basket item density
12. `completed_deliveries` (int64) - Orders successfully delivered
13. `unfulfilled_orders` (int64) - Orders awaiting delivery or cancelled
14. `delayed_deliveries` (int64) - Orders delivered past expected date
15. `avg_delivery_delay_hours` (float64) - Average delay duration
16. `delivery_delay_ratio` (float64) - Proportion of fulfilled orders delayed
17. `total_crm_interactions` (int64) - Total customer support tickets
18. `crm_complaints_count` (int64) - Support tickets flagged as complaint
19. `crm_queries_count` (int64) - Support tickets flagged as query
20. `crm_email_count` (int64) - Support interactions via email
21. `crm_phone_count` (int64) - Support interactions via telephone
22. `crm_livechat_count` (int64) - Support interactions via live-chat
23. `crm_complaint_ratio` (float64) - Ratio of complaints to total interactions
24. `total_reviews` (int64) - Product reviews submitted
25. `avg_star_rating` (float64) - Average customer review score (1-5)
26. `low_star_reviews_count` (int64) - Unfavorable reviews (<= 2 stars)
27. `high_star_reviews_count` (int64) - Favorable reviews (>= 4 stars)
28. `has_reviews` (int64) - Binary review participation indicator
29. `total_web_sessions` (int64) - Total browsing visits logged
30. `days_since_last_session` (float64) - Days between subscription end and last web visit
31. `session_frequency_per_month` (float64) - Web engagement velocity
32. `order_frequency_per_month` (float64) - Purchase order velocity

---

## 7. Target Variable Distribution (`auto_renew`)
- **Renewed (`1`)**: 135 customers (**61.09%**)
- **Churned (`0`)**: 86 customers (**38.91%**)

---

## 8. Data Leakage Assessment & Limitations
1. **Temporal Boundaries**:
   - Analysis verified that 100% of recorded sales, CRM tickets, and web sessions occur strictly between `start_date` and `end_date`.
   - There are zero post-cancellation transactions or web logs present in the sources.
2. **Exclusion of Personal Identifiers**:
   - Customer names, email addresses, password hashes, and street/postal addresses were strictly omitted from feature tables.
3. **Cohort Window Assumption**:
   - The data represents a closed cohort of subscription lifecycles. Features represent cumulative behavior up to the renewal decision.

