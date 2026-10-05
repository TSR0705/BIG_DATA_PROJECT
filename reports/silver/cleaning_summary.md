# Phase 3 Silver Cleaning & Data Quality Summary

> **Project:** Loan Default Prediction Using Big Data Analytics  
> **Generated:** 2026-10-05 14:25:57 UTC  
> **Total Rows Processed:** 58,489,893  
> **Total Silver Size:** 804.39 MB  
> **Total Transformations:** 22 documented anomaly fixes  
> **Total Runtime:** 47.10 seconds  

---

## 1. Silver Objective & Quality Principles

The Silver layer reads strictly from `data/bronze/` and produces deterministic, auditable Parquet datasets under `data/silver/`. Every transformation addresses documented data quality anomalies identified during Phase 1 profiling without data loss or target leakage:

1. **Sentinel Replacement:** Converted `365243` (~1000 years) to `NULL` while creating explicit indicator flags (`FLAG_*_SENTINEL = 1`).
2. **XNA Normalization:** Converted documented categorical `'XNA'` values to `NULL` while preserving information via indicator flags (`FLAG_*_XNA = 1`).
3. **Negative Offset Preservation:** Valid historical negative `DAYS_*` values were preserved unmodified.
4. **Missingness Preservation:** Zero generic imputation (no arbitrary mean/median/zero fills) — null integrity preserved.
5. **Multi-Record Grain Preservation:** 1,061 `POS_CASH_balance` and 58,542 `installments_payments` multi-contract grain records preserved.
6. **Row Count Parity:** 100.0% row-count equality verified between Bronze and Silver across all 8 tables.
7. **Raw Dataset Integrity:** Source CSVs in `dataset/` remained untouched and outside the execution pipeline.

---

## 2. Table-by-Table Execution Results

| Table Name | Bronze Rows | Silver Rows | Row Match | Bronze Cols | Silver Cols | Added Flags | Size (MB) | Runtime (s) | Status |
| :--- | ---: | ---: | :---: | ---: | ---: | :--- | ---: | ---: | :---: |
| `application_train` | 307,511 | 307,511 | PASS | 122 | 125 | `+3` | 22.76 | 10.68s | **PASS** |
| `application_test` | 48,744 | 48,744 | PASS | 121 | 124 | `+3` | 3.92 | 1.35s | **PASS** |
| `bureau` | 1,716,428 | 1,716,428 | PASS | 17 | 17 | 0 | 36.73 | 2.59s | **PASS** |
| `bureau_balance` | 27,299,925 | 27,299,925 | PASS | 3 | 3 | 0 | 137.09 | 9.70s | **PASS** |
| `previous_application` | 1,670,214 | 1,670,214 | PASS | 37 | 52 | `+15` | 60.38 | 10.87s | **PASS** |
| `installments_payments` | 13,605,401 | 13,605,401 | PASS | 8 | 8 | 0 | 306.84 | 4.13s | **PASS** |
| `credit_card_balance` | 3,840,312 | 3,840,312 | PASS | 23 | 23 | 0 | 132.01 | 4.09s | **PASS** |
| `POS_CASH_balance` | 10,001,358 | 10,001,358 | PASS | 8 | 9 | `+1` | 104.67 | 3.70s | **PASS** |
| **TOTAL** | **58,489,893** | **58,489,893** | **PASS** | **—** | **—** | **+18 flags** | **804.39 MB** | **47.10s** | **PASS** |

---

## 3. Data Quality Before / After Audit

| Table | Column | Anomaly Type | Before Anomaly Count | After Anomaly Count | Flag Column | Flag Sum | Status |
| :--- | :--- | :--- | ---: | ---: | :--- | ---: | :---: |
| `application_train` | `DAYS_EMPLOYED` | Sentinel (365243) | 55,374 | 0 | `FLAG_DAYS_EMPLOYED_SENTINEL` | 55,374 | **PASS** |
| `application_train` | `CODE_GENDER` | Categorical ('XNA') | 4 | 0 | `FLAG_CODE_GENDER_XNA` | 4 | **PASS** |
| `application_train` | `ORGANIZATION_TYPE` | Categorical ('XNA') | 55,374 | 0 | `FLAG_ORGANIZATION_TYPE_XNA` | 55,374 | **PASS** |
| `application_test` | `DAYS_EMPLOYED` | Sentinel (365243) | 9,274 | 0 | `FLAG_DAYS_EMPLOYED_SENTINEL` | 9,274 | **PASS** |
| `application_test` | `CODE_GENDER` | Categorical ('XNA') | 0 | 0 | `FLAG_CODE_GENDER_XNA` | 0 | **PASS** |
| `application_test` | `ORGANIZATION_TYPE` | Categorical ('XNA') | 9,274 | 0 | `FLAG_ORGANIZATION_TYPE_XNA` | 9,274 | **PASS** |
| `previous_application` | `DAYS_FIRST_DRAWING` | Sentinel (365243) | 934,444 | 0 | `FLAG_DAYS_FIRST_DRAWING_SENTINEL` | 934,444 | **PASS** |
| `previous_application` | `DAYS_FIRST_DUE` | Sentinel (365243) | 40,645 | 0 | `FLAG_DAYS_FIRST_DUE_SENTINEL` | 40,645 | **PASS** |
| `previous_application` | `DAYS_LAST_DUE_1ST_VERSION` | Sentinel (365243) | 93,864 | 0 | `FLAG_DAYS_LAST_DUE_1ST_VERSION_SENTINEL` | 93,864 | **PASS** |
| `previous_application` | `DAYS_LAST_DUE` | Sentinel (365243) | 211,221 | 0 | `FLAG_DAYS_LAST_DUE_SENTINEL` | 211,221 | **PASS** |
| `previous_application` | `DAYS_TERMINATION` | Sentinel (365243) | 225,913 | 0 | `FLAG_DAYS_TERMINATION_SENTINEL` | 225,913 | **PASS** |
| `previous_application` | `NAME_CONTRACT_TYPE` | Categorical ('XNA') | 346 | 0 | `FLAG_NAME_CONTRACT_TYPE_XNA` | 346 | **PASS** |
| `previous_application` | `NAME_CASH_LOAN_PURPOSE` | Categorical ('XNA') | 677,918 | 0 | `FLAG_NAME_CASH_LOAN_PURPOSE_XNA` | 677,918 | **PASS** |
| `previous_application` | `NAME_PAYMENT_TYPE` | Categorical ('XNA') | 627,384 | 0 | `FLAG_NAME_PAYMENT_TYPE_XNA` | 627,384 | **PASS** |
| `previous_application` | `CODE_REJECT_REASON` | Categorical ('XNA') | 5,244 | 0 | `FLAG_CODE_REJECT_REASON_XNA` | 5,244 | **PASS** |
| `previous_application` | `NAME_CLIENT_TYPE` | Categorical ('XNA') | 1,941 | 0 | `FLAG_NAME_CLIENT_TYPE_XNA` | 1,941 | **PASS** |
| `previous_application` | `NAME_GOODS_CATEGORY` | Categorical ('XNA') | 950,809 | 0 | `FLAG_NAME_GOODS_CATEGORY_XNA` | 950,809 | **PASS** |
| `previous_application` | `NAME_PORTFOLIO` | Categorical ('XNA') | 372,230 | 0 | `FLAG_NAME_PORTFOLIO_XNA` | 372,230 | **PASS** |
| `previous_application` | `NAME_PRODUCT_TYPE` | Categorical ('XNA') | 1,063,666 | 0 | `FLAG_NAME_PRODUCT_TYPE_XNA` | 1,063,666 | **PASS** |
| `previous_application` | `NAME_SELLER_INDUSTRY` | Categorical ('XNA') | 855,720 | 0 | `FLAG_NAME_SELLER_INDUSTRY_XNA` | 855,720 | **PASS** |
| `previous_application` | `NAME_YIELD_GROUP` | Categorical ('XNA') | 517,215 | 0 | `FLAG_NAME_YIELD_GROUP_XNA` | 517,215 | **PASS** |
| `POS_CASH_balance` | `NAME_CONTRACT_STATUS` | Categorical ('XNA') | 2 | 0 | `FLAG_NAME_CONTRACT_STATUS_XNA` | 2 | **PASS** |

---

## 4. Legitimate Multi-Record Grain Audit

- `POS_CASH_balance`: 1,061 composite grain records `(SK_ID_PREV, MONTHS_BALANCE)` preserved as legitimate multiple active contracts in the same reporting month.
- `installments_payments`: 58,542 composite grain records `(SK_ID_PREV, NUM_INSTALMENT_NUMBER, NUM_INSTALMENT_VERSION)` preserved as legitimate split payment installments.
- No rows dropped across any of the 8 tables (`Bronze Rows == Silver Rows` for all tables).

---

## 5. Idempotency & Rerunnability

All Silver Parquet tables are written using `.mode('overwrite')`. Re-executing `scripts/run_phase3_silver.py` is guaranteed deterministic and idempotent.
