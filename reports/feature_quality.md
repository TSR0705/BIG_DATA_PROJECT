# Phase 5 Feature Quality & Data Sanity Report

> **Project:** Loan Default Prediction Using Big Data Analytics  
> **Generated:** 2026-10-05 14:59:05 UTC  
> **Training Rows:** 307,511  
> **Inference Rows:** 48,744  
> **Training Features:** 258  
> **Inference Features:** 257  

---

## 1. Numeric Sanity & Safe Division

- **Infinity / -Infinity Count:** 0 (Zero occurrences)  
- **NaN Count:** 0 (Zero occurrences)  
- **Safe Division Policy:** All ratio calculations return `NULL` when denominator is 0, NULL, or NaN.  

---

## 2. Dataset Outputs

- **Final Training Parquet:** `data/model_input/model_input.parquet` (141.39 MB)  
- **Final Inference Parquet:** `data/model_input_test/model_input_test.parquet` (28.66 MB)  
- **Feature Registry:** [`configs/features.yaml`](../../configs/features.yaml)  
