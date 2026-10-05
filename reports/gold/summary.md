# Phase 4 Gold Aggregation & Multi-Table Assembly Summary

> **Project:** Loan Default Prediction Using Big Data Analytics  
> **Generated:** 2026-10-05 14:40:51 UTC  
> **Training Gold Rows:** 307,511 (100% unique `SK_ID_CURR`)  
> **Training Gold Features:** 241 features  
> **Test Inference Rows:** 48,744 (100% unique `SK_ID_CURR`)  
> **Test Inference Features:** 240 features  
> **Total Execution Time:** 52.23 seconds  

---

## 1. Engineering Architecture & Aggregation Rules

Phase 4 adheres strictly to the **Aggregate First $\rightarrow$ Join Second** pattern:
- Historical tables (`bureau_balance`, `bureau`, `previous_application`, `installments_payments`, `credit_card_balance`, `POS_CASH_balance`) were fully aggregated to `SK_ID_CURR` grain **before** joining to the application spine.
- Left joins were used exclusively from the application spine to preserve all 307,511 applicants.
- Row-explosion assertions verified **zero row increase** after every join (`output_rows == spine_rows`).
- All engineered ratios employed safe division logic to prevent `Infinity`, `-Infinity`, and `NaN`.
- No target leakage: `TARGET` was preserved as the final ground-truth label in training and strictly excluded from test.

---

## 2. Pipeline Stage Performance

| Pipeline Stage | Grain / Output | Duration (s) | Status |
| :--- | :--- | ---: | :---: |
| Bureau Balance Aggregation | Intermediate Parquet | 10.54s | **PASS** |
| Bureau Aggregation | Intermediate Parquet | 4.05s | **PASS** |
| Previous Application Aggregation | Intermediate Parquet | 2.21s | **PASS** |
| Installments Aggregation | Intermediate Parquet | 3.15s | **PASS** |
| Credit Card Aggregation | Intermediate Parquet | 4.89s | **PASS** |
| Pos Cash Aggregation | Intermediate Parquet | 4.81s | **PASS** |
| Final Application Joins | Intermediate Parquet | 21.79s | **PASS** |
| Complete Gold Pipeline | Intermediate Parquet | 52.23s | **PASS** |

---

## 3. Join Audit Summary (Zero Row-Explosion Verification)

| Join Step | Spine | Feature Table | Left Rows | Right Rows | Output Rows | Row Increase | Status |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | :---: |
| `application_train + bureau_features` | `application_train` | `bureau_features` | 307,511 | 305,811 | 307,511 | 0 | **PASS** |
| `application_train + previous_application_features` | `application_train` | `previous_application_features` | 307,511 | 338,857 | 307,511 | 0 | **PASS** |
| `application_train + installments_features` | `application_train` | `installments_features` | 307,511 | 339,587 | 307,511 | 0 | **PASS** |
| `application_train + credit_card_features` | `application_train` | `credit_card_features` | 307,511 | 103,558 | 307,511 | 0 | **PASS** |
| `application_train + pos_cash_features` | `application_train` | `pos_cash_features` | 307,511 | 337,252 | 307,511 | 0 | **PASS** |
| `application_test + bureau_features` | `application_test` | `bureau_features` | 48,744 | 305,811 | 48,744 | 0 | **PASS** |
| `application_test + previous_application_features` | `application_test` | `previous_application_features` | 48,744 | 338,857 | 48,744 | 0 | **PASS** |
| `application_test + installments_features` | `application_test` | `installments_features` | 48,744 | 339,587 | 48,744 | 0 | **PASS** |
| `application_test + credit_card_features` | `application_test` | `credit_card_features` | 48,744 | 103,558 | 48,744 | 0 | **PASS** |
| `application_test + pos_cash_features` | `application_test` | `pos_cash_features` | 48,744 | 337,252 | 48,744 | 0 | **PASS** |

---

## 4. Final Gold Datasets

- **Training Model Input (`data/gold/model_input/`):**
  - Rows: 307,511 (1 row per `SK_ID_CURR`)
  - Columns: 241 (including `TARGET`)
  - Storage Size: 118.66 MB (Snappy Parquet)
- **Inference Model Input (`data/gold/model_input_test/`):**
  - Rows: 48,744 (1 row per `SK_ID_CURR`)
  - Columns: 240 (strictly NO `TARGET`)
  - Storage Size: 24.59 MB (Snappy Parquet)

---

## 5. Artifacts & Documentation

- **Feature Registry:** [`configs/features.yaml`](../../configs/features.yaml)
- **Join Audit:** [`reports/gold/join_audit.json`](join_audit.json)
- **Physical Plan Evidence:** [`reports/gold/physical_plan.txt`](physical_plan.txt)
