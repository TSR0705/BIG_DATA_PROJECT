# Phase 6 Dataset Splitting & Stratification Report

> **Project:** Loan Default Prediction Using Big Data Analytics  
> **Source Dataset:** `C:\Users\ACER\Desktop\BIG_DATA_PROJECT\data\model_input\model_input.parquet`  
> **Random State:** `42`  
> **Stratification Column:** `TARGET`  
> **Generated:** 2026-10-05 15:38:25 UTC  
> **Status:** **PASS** (Zero Overlap, Row Conservation Verified)

---

## 1. Executive Summary & Split Invariants

- **Total Applicants:** 307,511  
- **Split Proportions:** 70.0% Train / 15.0% Validation / 15.0% Test  
- **Split Disjointness:** Verified (0 overlap between any partition pair)  
- **Population Conservation:** 100% of applicants assigned, 0 dropped, 0 duplicated  
- **Persistent Index Files:**
  - `data/splits/train_indices.parquet`
  - `data/splits/val_indices.parquet`
  - `data/splits/test_indices.parquet`

---

## 2. Split Size and Target Distribution

| Split Partition | Row Count | Proportion | Positive (Default) | Negative (Repaid) | Positive Rate (%) | Negative Rate (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TRAIN** | 215,257 | 69.9998% | 17,377 | 197,880 | 8.073% | 91.927% |
| **VALIDATION** | 46,127 | 15.0001% | 3,724 | 42,403 | 8.073% | 91.927% |
| **TEST** | 46,127 | 15.0001% | 3,724 | 42,403 | 8.073% | 91.927% |
| **TOTAL** | **307,511** | **100.00%** | **24,825** | **282,686** | **8.073%** | **91.927%** |

---

## 3. Stratification & Class Imbalance Audit

- **Baseline Class Imbalance:** ~8.07% positive default rate across the population.
- **Maximum Positive Rate Delta:** `0.000005` across all partitions.
- **Resampling Policy:** No oversampling or SMOTE applied in Phase 6. Unaltered ground truth distributions preserved.

---

## 4. Verification Check

- [x] Train / Validation / Test are mutually exclusive (0 overlap).
- [x] Union of splits equals exactly 307,511 rows.
- [x] TARGET is present in all supervised partitions.
- [x] application_test is strictly excluded from supervised splits (reserved for blind inference).
