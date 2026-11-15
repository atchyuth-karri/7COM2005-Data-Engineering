# Fixed-Cutoff Feature Design (implemented in notebooks 03 and 04)

**Status:** implemented in `notebooks/03_Fixed_Cutoff_Features.ipynb` (features) and `notebooks/04_Cutoff_Machine_Learning.ipynb` (models). The ETL notebook and canonical Parquet are unchanged: the cutoff features reuse the ETL functions on filtered copies of the raw inputs, and a reuse check confirms those functions reproduce the canonical dataset exactly. Results: CV AUC 0.48 to 0.60, close to chance.
**Problem addressed:** prediction-time leakage in the current features (see the warning at the top of `notebooks/02_Machine_Learning.ipynb`).

## 1. The problem, verified

- Each feature is aggregated over the whole subscription window (`start_date` to `end_date`).
- For churned customers (`auto_renew = 0`) `end_date` is the cancellation date. Their features therefore include activity up to the decision.
- Churned windows are shorter (median 90 days against 180), so count and total features are smaller for churned customers by construction.
- 17.7% of churned customers' orders fall in their final 30 days, against 7.4% for renewed customers.
- Removing tenure and raw totals drops the models to near-chance test AUC (0.55 to 0.58).

## 2. What a valid prediction needs

Features must use only events that happened **before a cutoff** `T`, where `T` is before the renewal decision. The label is `auto_renew` as observed at `end_date`.

## 3. Event timestamps available in the source data

| Source | Timestamp | Usable for a cutoff? |
|---|---|---|
| `sale.order_date` | Yes | Yes |
| `sale.expected_date`, `sale.delivered_date` | Yes | Yes, but a delivery counts only if `delivered_date <= T` |
| `CRM_report.recieved_datatime` | Yes (typo in source header) | Yes |
| `server_log.datetime` | Yes | Yes |
| `product_review` | **No date column** (`id, star_rating, product_id, customer_id, sale_id`) | **No**. Review features cannot be placed before `T`. |
| `payment` | **No date column** | **No**. Use `sale.order_date` instead. |
| Survey CSV | **No timestamp** | **Unknown.** Could be before or after `T`. |
| `subscription.start_date`, `subscription_type` | Yes | Yes, fixed at the start |

**Consequence:** `avg_star_rating`, `total_reviews`, `has_reviews`, `low/high_star_reviews_count` cannot be made cutoff-safe with the current data. Either exclude them, or record review dates in the source system first.

## 4. Choosing the cutoff

Three options:

| Option | Definition | Pros | Cons |
|---|---|---|---|
| **A. Relative to end (`T = end_date - K`)** | Drop the final K days (e.g. K = 30) | Keeps most history | Still predicts from a window that ends near the decision for short subscriptions. Renewed customers get the same cutoff, so it is consistent. |
| **B. Fixed early horizon (`T = start_date + N`)** (recommended) | Use only the first N days of each subscription, e.g. N = 14 or 21 | Predicts renewal before the decision, which is what the business needs. Consistent across churned and renewed customers. | Less history. Some features will be sparse. |
| C. Single calendar date | One global cutoff date | Simple | Subscriptions start at different times (Jan to Nov 2026), so early and late customers get very different windows. Not recommended. |

**Recommendation: option B with N = 14 days.** The shortest subscription in the data is 30 days, so every customer has at least 16 days after the cutoff before the decision. The result is cutoff-safe for all 221 customers.

Before committing to N, check that the first 14 days contain enough orders, CRM and web sessions to be informative. If not, try N = 21 (minimum subscription is still 30 days, so a 9-day gap remains).

## 5. Feature rules under the cutoff

For each customer, with cutoff `T`:

- Include an event only if its timestamp is `< T`.
- Rates use the observed window length `T - start_date`, not `duration_days`. This avoids using the full duration.
- `duration_days` is **excluded**: it is set by the outcome.
- Delivery delay counts a delivery only if `delivered_date <= T`. Orders still undelivered at `T` are counted as in-flight, not as unfulfilled.
- Subscription tier and delivery service are fixed at start and can stay.
- Survey features stay in Configuration B only, with the timing caveat stated. They are not cutoff-verifiable.
- Review features are excluded until review dates exist.

## 6. Implementation outline (for a later ETL change)

1. Add a function `build_features_as_of(cutoff_days=14)` in the ETL notebook.
2. For each customer compute `T = start_date + cutoff_days`.
3. Filter each event table on its timestamp `< T`, then aggregate per customer.
4. Write a new Parquet, e.g. `customer_churn_dataset_cutoff14.parquet`. Keep the existing file unchanged.
5. In the ML notebook, point `PARQUET_PATH` at the new file and re-run. Compare with the current results.

## 7. Validation plan

- **Expectation:** performance will fall. Report that honestly. A drop is the correct result: it shows how much of the current score came from outcome-proximate features.
- **Leakage test:** check that no feature changes when events after `T` are removed.
- **Sensitivity:** repeat for N = 14 and N = 21 and report both.
- **Time-based check:** train on customers whose `start_date` is earlier, test on later ones.
- **Labels:** verify `auto_renew` against actual cancellation records if they exist. `auto_renew = 0` is a proxy.

## 8. Decisions (updated)

The ETL change in section 6 was not needed: the features were built by reusing the unchanged ETL functions on filtered inputs. The remaining open items are:

1. Option B with N = 14 was used. A 21-day sensitivity run is not done.
2. Confirm whether review dates exist in the source system. If not, exclude review features.
3. Confirm how the survey was timed, or accept that survey features are a sensitivity only.
4. No ETL change was made. The cutoff Parquet (`customer_churn_dataset_cutoff14.parquet`) sits alongside the canonical file.
