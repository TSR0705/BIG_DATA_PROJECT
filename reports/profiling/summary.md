# Phase 1 Dataset Profiling Summary

> **Project:** Loan Default Prediction Using Big Data Analytics  
> **Generated:** 2026-10-05 13:56:28 UTC  
> **Total Execution Time:** 397.23 seconds  

---

## 1. Dataset Overview

This report provides the empirical data profile and frozen schema contract for all 8 tables in the **Home Credit Default Risk** dataset. All measurements were computed using Apache Spark (local mode) on the raw, read-only CSV files without data loss or premature transformation.

---

## 2. Table Scale

| Table | Rows | Columns | Approx Expected Scale | Status |
| :--- | ---: | ---: | ---: | :--- |
| `application_train` | 307,511 | 122 | 307,511 | MATCH (100%) |
| `application_test` | 48,744 | 121 | 48,744 | MATCH (100%) |
| `bureau` | 1,716,428 | 17 | 1,716,428 | MATCH (100%) |
| `bureau_balance` | 27,299,925 | 3 | 27,299,925 | MATCH (100%) |
| `previous_application` | 1,670,214 | 37 | 1,670,214 | MATCH (100%) |
| `installments_payments` | 13,605,401 | 8 | 13,605,401 | MATCH (100%) |
| `credit_card_balance` | 3,840,312 | 23 | 3,840,312 | MATCH (100%) |
| `POS_CASH_balance` | 10,001,358 | 8 | 10,001,358 | MATCH (100%) |
| **TOTAL** | **58,489,893** | **—** | **~58,400,000** | **VERIFIED** |

---

## 3. Primary Key Audit

| Table | Key | Rows | Distinct | Nulls | Duplicates | Unique |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- |
| `application_train` | `SK_ID_CURR` | 307,511 | 307,511 | 0 | 0 | PASS (Unique) |
| `application_test` | `SK_ID_CURR` | 48,744 | 48,744 | 0 | 0 | PASS (Unique) |
| `bureau` | `SK_ID_BUREAU` | 1,716,428 | 1,716,428 | 0 | 0 | PASS (Unique) |
| `previous_application` | `SK_ID_PREV` | 1,670,214 | 1,670,214 | 0 | 0 | PASS (Unique) |

---

## 4. Table Grain (Monthly / History Tables)

| Table | Candidate Grain | Measured Duplicate Count | Notes |
| :--- | :--- | ---: | :--- |
| `bureau_balance` | `SK_ID_BUREAU+MONTHS_BALANCE` | 0 | Unique (Valid Grain) |
| `POS_CASH_balance` | `SK_ID_PREV+MONTHS_BALANCE` | 0 | Unique (Valid Grain) |
| `credit_card_balance` | `SK_ID_PREV+MONTHS_BALANCE` | 0 | Unique (Valid Grain) |
| `installments_payments` | `SK_ID_PREV+NUM_INSTALMENT_NUMBER+NUM_INSTALMENT_VERSION` | 653,483 | 653,483 duplicate composite keys |

---

## 5. Foreign Key Coverage

| Child Table | Child Key | Parent Table | Parent Key | Key Coverage (%) | Row Coverage (%) | Unmatched Keys | Unmatched Rows |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| `bureau` | `SK_ID_CURR` | `application_train + application_test` | `SK_ID_CURR` | 100.00% | 100.00% | 0 | 0 |
| `previous_application` | `SK_ID_CURR` | `application_train + application_test` | `SK_ID_CURR` | 100.00% | 100.00% | 0 | 0 |
| `bureau_balance` | `SK_ID_BUREAU` | `bureau` | `SK_ID_BUREAU` | 94.73% | 88.57% | 43,041 | 3,120,184 |
| `installments_payments` | `SK_ID_PREV` | `previous_application` | `SK_ID_PREV` | 96.11% | 90.81% | 38,847 | 1,250,826 |
| `credit_card_balance` | `SK_ID_PREV` | `previous_application` | `SK_ID_PREV` | 89.10% | 71.80% | 11,372 | 1,082,816 |
| `POS_CASH_balance` | `SK_ID_PREV` | `previous_application` | `SK_ID_PREV` | 96.00% | 96.59% | 37,422 | 340,561 |

> **Note on Foreign Key Coverage:** Child records in `bureau` and `previous_application` cover the full applicant population (train + test). A small percentage of child records link to historical accounts not present in the active loan sample, which is standard in real-world credit bureaus.

---

## 6. Null / Missingness Summary

Top columns across the dataset exhibiting significant missingness (> 50% nulls):

| Table | Column | Type | Null Count | Null % |
| :--- | :--- | :--- | ---: | ---: |
| `previous_application` | `RATE_INTEREST_PRIMARY` | `DoubleType()` | 1,664,263 | 99.64% |
| `previous_application` | `RATE_INTEREST_PRIVILEGED` | `DoubleType()` | 1,664,263 | 99.64% |
| `bureau` | `AMT_ANNUITY` | `DoubleType()` | 1,226,791 | 71.47% |
| `application_train` | `COMMONAREA_AVG` | `DoubleType()` | 214,865 | 69.87% |
| `application_train` | `COMMONAREA_MODE` | `DoubleType()` | 214,865 | 69.87% |
| `application_train` | `COMMONAREA_MEDI` | `DoubleType()` | 214,865 | 69.87% |
| `application_train` | `NONLIVINGAPARTMENTS_AVG` | `DoubleType()` | 213,514 | 69.43% |
| `application_train` | `NONLIVINGAPARTMENTS_MODE` | `DoubleType()` | 213,514 | 69.43% |
| `application_train` | `NONLIVINGAPARTMENTS_MEDI` | `DoubleType()` | 213,514 | 69.43% |
| `application_test` | `COMMONAREA_AVG` | `DoubleType()` | 33,495 | 68.72% |
| `application_test` | `COMMONAREA_MODE` | `DoubleType()` | 33,495 | 68.72% |
| `application_test` | `COMMONAREA_MEDI` | `DoubleType()` | 33,495 | 68.72% |
| `application_test` | `NONLIVINGAPARTMENTS_AVG` | `DoubleType()` | 33,347 | 68.41% |
| `application_test` | `NONLIVINGAPARTMENTS_MODE` | `DoubleType()` | 33,347 | 68.41% |
| `application_test` | `NONLIVINGAPARTMENTS_MEDI` | `DoubleType()` | 33,347 | 68.41% |

*Total columns across all tables with >= 50% missingness: 76*

---

## 7. Anomaly Inventory

### 7.1 DAYS_EMPLOYED Sentinel (365243)

The value `365243` (~1000 years) is used as an undocumented sentinel value representing pensioners or unemployed individuals with no employment history.

| Table | Column | Sentinel Value | Count | Percentage |
| :--- | :--- | ---: | ---: | ---: |
| `application_train` | `DAYS_EMPLOYED` | `365243` | 55,374 | 18.01% |
| `application_test` | `DAYS_EMPLOYED` | `365243` | 9,274 | 19.03% |
| `previous_application` | `DAYS_FIRST_DRAWING` | `365243` | 934,444 | 55.95% |
| `previous_application` | `DAYS_FIRST_DUE` | `365243` | 40,645 | 2.43% |
| `previous_application` | `DAYS_LAST_DUE_1ST_VERSION` | `365243` | 93,864 | 5.62% |
| `previous_application` | `DAYS_LAST_DUE` | `365243` | 211,221 | 12.65% |
| `previous_application` | `DAYS_TERMINATION` | `365243` | 225,913 | 13.53% |

### 7.2 XNA Value Occurrences

The string `'XNA'` appears as a categorical missingness/unspecified placeholder:

| Table | Column | XNA Count | Percentage |
| :--- | :--- | ---: | ---: |
| `application_train` | `CODE_GENDER` | 4 | 0.00% |
| `application_train` | `ORGANIZATION_TYPE` | 55,374 | 18.01% |
| `application_test` | `ORGANIZATION_TYPE` | 9,274 | 19.03% |
| `previous_application` | `NAME_CONTRACT_TYPE` | 346 | 0.02% |
| `previous_application` | `NAME_CASH_LOAN_PURPOSE` | 677,918 | 40.59% |
| `previous_application` | `NAME_PAYMENT_TYPE` | 627,384 | 37.56% |
| `previous_application` | `CODE_REJECT_REASON` | 5,244 | 0.31% |
| `previous_application` | `NAME_CLIENT_TYPE` | 1,941 | 0.12% |
| `previous_application` | `NAME_GOODS_CATEGORY` | 950,809 | 56.93% |
| `previous_application` | `NAME_PORTFOLIO` | 372,230 | 22.29% |
| `previous_application` | `NAME_PRODUCT_TYPE` | 1,063,666 | 63.68% |
| `previous_application` | `NAME_SELLER_INDUSTRY` | 855,720 | 51.23% |
| `previous_application` | `NAME_YIELD_GROUP` | 517,215 | 30.97% |
| `POS_CASH_balance` | `NAME_CONTRACT_STATUS` | 2 | 0.00% |

### 7.3 DAYS_* Columns Behavior

In the Home Credit data model, all `DAYS_*` attributes measure time backward relative to current application date, meaning negative values are **semantically expected**.

| Table | Column | Min | Max | Negative Rows | Positive Rows |
| :--- | :--- | ---: | ---: | ---: | ---: |
| `application_train` | `DAYS_BIRTH` | -25,229 | -7,489 | 307,511 | 0 |
| `application_train` | `DAYS_EMPLOYED` | -17,912 | 365,243 | 252,135 | 55,374 |
| `application_train` | `DAYS_REGISTRATION` | -24,672.0 | 0.0 | 307,431 | 0 |
| `application_train` | `DAYS_ID_PUBLISH` | -7,197 | 0 | 307,495 | 0 |
| `application_train` | `DAYS_LAST_PHONE_CHANGE` | -4,292.0 | 0.0 | 269,838 | 0 |
| `application_test` | `DAYS_BIRTH` | -25,195 | -7,338 | 48,744 | 0 |
| `application_test` | `DAYS_EMPLOYED` | -17,463 | 365,243 | 39,470 | 9,274 |
| `application_test` | `DAYS_REGISTRATION` | -23,722.0 | 0.0 | 48,731 | 0 |
| `application_test` | `DAYS_ID_PUBLISH` | -6,348 | 0 | 48,739 | 0 |
| `application_test` | `DAYS_LAST_PHONE_CHANGE` | -4,361.0 | 0.0 | 42,943 | 0 |
| `bureau` | `DAYS_CREDIT` | -2,922 | 0 | 1,716,403 | 0 |
| `bureau` | `DAYS_CREDIT_ENDDATE` | -42,060.0 | 31,199.0 | 1,007,389 | 602,603 |

### 7.4 Income Distribution & Outliers (application_train)

- **Minimum:** 25,650.00
- **Median (P50):** 144,000.00
- **Mean:** 168,797.92
- **P99:** 450,000.00
- **P99.9:** 117,000,000.00
- **Maximum:** 117,000,000.00 *(Extreme outlier: 117,000,000)*
- **Interquartile Range (IQR):** 90,000.00
- **Upper Whisker (Q3 + 1.5*IQR):** 337,500.00
- **Outlier Count (> Upper Whisker):** 14,035 (4.56%)

---

## 8. Relationship / ER Structure

The empirical entity-relationship model exhibits a **two-branch snowflake architecture** centered on the core loan application:

```
                     APPLICATION (train / test)
                        SK_ID_CURR
                            │
        ┌───────────────────┴───────────────────┐
        ▼                                       ▼
     BUREAU                           PREVIOUS_APPLICATION
   SK_ID_CURR                              SK_ID_CURR
   SK_ID_BUREAU (PK)                       SK_ID_PREV (PK)
        │                                       │
        ▼                     ┌─────────────────┼─────────────────┐
  BUREAU_BALANCE              ▼                 ▼                 ▼
   SK_ID_BUREAU          INSTALLMENTS      CREDIT_CARD        POS_CASH
   MONTHS_BALANCE         SK_ID_PREV        SK_ID_PREV       SK_ID_PREV
                          NUM_INSTALMENT    MONTHS_BALANCE   MONTHS_BALANCE
```

---

## 9. Column Dictionary

The official Home Credit column description file [`dataset/HomeCredit_columns_description.csv`](../../dataset/HomeCredit_columns_description.csv) has been parsed into [`reports/profiling/column_dictionary.json`](column_dictionary.json).

---

## 10. Schema Contract

Explicit PySpark `StructType` schemas for all 8 tables have been generated and frozen in [`configs/schemas.py`](../../configs/schemas.py). Downstream ETL phases MUST import these schemas directly and avoid runtime schema inference (`inferSchema=False`).

---

## 11. Profiling Runtimes

| Table | Profiling Duration (s) |
| :--- | ---: |
| `application_train` | 99.60s |
| `application_test` | 49.31s |
| `bureau` | 8.71s |
| `bureau_balance` | 15.72s |
| `previous_application` | 22.07s |
| `installments_payments` | 16.37s |
| `credit_card_balance` | 38.64s |
| `POS_CASH_balance` | 21.67s |
| **TOTAL** | **397.23s** |
