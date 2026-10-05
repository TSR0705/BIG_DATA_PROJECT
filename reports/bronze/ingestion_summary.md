# Phase 2 Bronze Ingestion Summary

> **Project:** Loan Default Prediction Using Big Data Analytics  
> **Generated:** 2026-10-05 14:04:55 UTC  
> **Total Rows Ingested:** 58,489,893  
> **Total Bronze Size:** 797.47 MB  
> **Total Runtime:** 41.71 seconds  

---

## 1. Bronze Objective & Scope

The Bronze layer converts all 8 raw Home Credit CSV files into typed, validated Parquet files under `data/bronze/`. Strict Phase 1 `StructType` schemas are enforced without runtime schema inference (`inferSchema=False`). Raw data is preserved semantically: no cleaning, imputation, sentinel replacement, deduplication, or filtering is performed in Phase 2.

---

## 2. Ingestion & Validation Results

| Table Name | Source Rows | Bronze Rows | Row Count Match | Schema Match | Files | Size (MB) | Runtime (s) | Status |
| :--- | ---: | ---: | :---: | :---: | ---: | ---: | ---: | :---: |
| `application_train` | 307,511 | 307,511 | PASS | PASS | 2 | 22.73 | 12.77s | **PASS** |
| `application_test` | 48,744 | 48,744 | PASS | PASS | 1 | 3.91 | 1.61s | **PASS** |
| `bureau` | 1,716,428 | 1,716,428 | PASS | PASS | 4 | 36.68 | 2.35s | **PASS** |
| `bureau_balance` | 27,299,925 | 27,299,925 | PASS | PASS | 32 | 137.09 | 8.42s | **PASS** |
| `previous_application` | 1,670,214 | 1,670,214 | PASS | PASS | 4 | 59.98 | 3.65s | **PASS** |
| `installments_payments` | 13,605,401 | 13,605,401 | PASS | PASS | 16 | 300.86 | 5.62s | **PASS** |
| `credit_card_balance` | 3,840,312 | 3,840,312 | PASS | PASS | 8 | 131.86 | 3.56s | **PASS** |
| `POS_CASH_balance` | 10,001,358 | 10,001,358 | PASS | PASS | 12 | 104.37 | 3.73s | **PASS** |
| **TOTAL** | **58,489,893** | **58,489,893** | **PASS** | **PASS** | **—** | **797.47 MB** | **41.71s** | **PASS** |

---

## 3. Partitioning & File Strategy

Partition counts were configured to prevent thousands of small files while ensuring balanced parallelism on a 16 GB laptop:

- `bureau_balance` (27.3M rows): 32 partitions
- `installments_payments` (13.6M rows): 16 partitions
- `POS_CASH_balance` (10.0M rows): 12 partitions
- `credit_card_balance` (3.8M rows): 8 partitions
- `bureau` (1.7M rows): 4 partitions
- `previous_application` (1.67M rows): 4 partitions
- `application_train` (307k rows): 2 partitions
- `application_test` (48k rows): 1 partition

---

## 4. Idempotency & Rerunnability

All Bronze tables are written with `.mode('overwrite')`. Rerunning `scripts/run_phase2_bronze.py` cleanly replaces existing Parquet partitions without accumulating stale or duplicated files.

---

## 5. Dataset Integrity

All source CSV files in `dataset/` were accessed in strict read-only mode. No files were modified, renamed, moved, or deleted.
