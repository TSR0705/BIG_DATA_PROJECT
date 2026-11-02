# Phase 10 Preflight Report — Gate A Verification

**Project:** Loan Default Prediction Using Big Data Analytics  
**Phase:** Phase 10 — Research Experiments (E1: Feature Ablation, E3: Class Imbalance × Calibration, E4: Decision Analysis)  
**Execution Timestamp:** 2026-11-02 17:05:00 UTC  
**Base Git Commit:** `4c406b656710e5291e7aea10f523749c906131a9` (`develop` branch)  
**Preflight Status:** **GREEN (PASSED)**

---

## 1. Environment & Runtime Verification

| Component | Target Version | Actual Measured Version | Verification Command / Path | Status |
| :--- | :--- | :--- | :--- | :---: |
| **Python** | 3.11.x | 3.11.9 | `venv\Scripts\python.exe` | PASS |
| **Java / JVM** | 17 / 21 LTS | 21.0.10+8-LTS-217 (pinned to JDK 17) | `C:\Java\jdk-17.0.20.1+1` | PASS |
| **PySpark** | 3.5.x | 3.5.9 | `import pyspark; pyspark.__version__` | PASS |
| **Scikit-Learn** | 1.6.x | 1.6.1 | `import sklearn; sklearn.__version__` | PASS |
| **XGBoost** | 3.x | 3.2.0 | `import xgboost; xgboost.__version__` | PASS |
| **LightGBM** | 4.x | 4.7.0 | `import lightgbm; lightgbm.__version__` | PASS |
| **Imbalanced-Learn** | 0.14.x | 0.14.2 | `import imblearn; imblearn.__version__` | PASS |
| **PyArrow** | 25.x | 25.0.1 | `import pyarrow; pyarrow.__version__` | PASS |
| **Pytest** | 9.x | 9.1.1 | `pytest -q` (126/126 tests passing) | PASS |

---

## 2. Frozen Split & Dataset Verification

| Partition | Frozen Path | Expected Rows | Verified Rows | Target Invariant | Leakage Status |
| :--- | :--- | ---:| ---:| :---: | :---: |
| **Train** | `data/splits/train_indices.parquet` | 215,257 | 215,257 | Pos: 17,377 (8.0727%) | Zero Leakage |
| **Validation** | `data/splits/val_indices.parquet` | 46,127 | 46,127 | Pos: 3,724 (8.0734%) | Zero Leakage |
| **Test** | `data/splits/test_indices.parquet` | 46,127 | 46,127 | Pos: 3,724 (8.0734%) | **EXCLUDED** from Phase 10 |
| **Model Input** | `data/model_input/model_input.parquet` | 307,511 | 307,511 | Columns: 258 (256 features + ID + Label) | Verified 1:1 Match |

*Invariants verified:*
1. Split disjointness: `train ∩ val = 0`, `train ∩ test = 0`, `val ∩ test = 0`.
2. Total union: `215,257 + 46,127 + 46,127 = 307,511` (100% completeness).
3. Test partition isolation: Test indices and `application_test.csv` are strictly excluded from Phase 10 experiment pipelines.

---

## 3. Model Finalist Configuration Verification

The primary model for E1 and E4 is the frozen Phase 8 tuned XGBoost finalist:
* **Source Artifact:** `artifacts/models/phase8_best_xgboost.joblib`
* **Metadata Reference:** `artifacts/models/phase8_model_metadata.json`
* **Hyperparameters (frozen):**
  * `objective`: `binary:logistic`
  * `eval_metric`: `logloss`
  * `max_depth`: `3`
  * `learning_rate`: `0.1`
  * `n_estimators`: `500`
  * `min_child_weight`: `10`
  * `subsample`: `0.8`
  * `colsample_bytree`: `1.0`
  * `random_state`: `42`
  * `scale_pos_weight`: `1.0` (unweighted baseline)

---

## 4. Feature Lineage & Family Partitioning Manifest

Analysis of `configs/features.yaml` (258 total entries):
- **Excluded ID:** `SK_ID_CURR` (1 column)
- **Excluded Label:** `TARGET` (1 column)
- **Model Features:** 256 columns categorized into 7 source feature groups:

| Feature Group | Family Code | Source Table / Domain | Feature Count | Sample Features |
| :--- | :--- | :--- | ---:| :--- |
| **Application** | `APP` | `application_train.csv` | 147 | `AMT_CREDIT`, `AMT_INCOME_TOTAL`, `CODE_GENDER`, `EXT_SOURCE_1/2/3`, `APP_AGE_YEARS` |
| **Bureau** | `BURO` + `BURO_BAL` | `bureau.csv` + `bureau_balance.csv` | 26 (20 + 6) | `BURO_AMT_CREDIT_SUM_SUM`, `BURO_DAYS_CREDIT_MAX`, `BURO_BAL_STATUS_0_SUM` |
| **Previous Applications** | `PREV` | `previous_application.csv` | 25 | `PREV_AMT_APPLICATION_MEAN`, `PREV_APPROVAL_RATE`, `PREV_CNT_PAYMENT_MEAN` |
| **Installments** | `INST` | `installments_payments.csv` | 16 | `INST_AMT_PAYMENT_SUM`, `INST_LATE_RATE`, `INST_DELAY_MEAN`, `INST_PAYMENT_RATIO` |
| **Credit Card** | `CC` | `credit_card_balance.csv` | 21 | `CC_AMT_BALANCE_MEAN`, `CC_CNT_DRAWINGS_CURRENT_SUM`, `CC_UTILIZATION_MEAN` |
| **POS / CASH** | `POS` | `POS_CASH_balance.csv` | 14 | `POS_COMPLETION_RATE_MEAN`, `POS_MONTHS_COUNT`, `POS_SK_DPD_MAX` |
| **Cross-Domain Derived** | `DERIVED` | Multi-table cross-domain ratios | 7 | `DERIVED_PAYMENT_DISCIPLINE_SCORE`, `DERIVED_TOTAL_DEBT_TO_INCOME_RATIO`, etc. |
| **Total Model Features** | | | **256** | |

### Resolution of 8 Feature Sets vs. 7 Named Configurations:
1. In the raw Kaggle dataset, there are 8 tables (`application_train`, `application_test`, `bureau`, `bureau_balance`, `previous_application`, `installments_payments`, `credit_card_balance`, `POS_CASH_balance`).
2. In the Medallion feature engineering, `bureau_balance` is rolled into the Bureau feature family (`BURO_BAL`).
3. The 7 explicit named incremental comparisons test Application alone vs. Application + each historical data source:
   - 1. `app_only` (147 features)
   - 2. `app_bureau` (147 + 26 = 173 features)
   - 3. `app_prev` (147 + 25 = 172 features)
   - 4. `app_installments` (147 + 16 = 163 features)
   - 5. `app_credit_card` (147 + 21 = 168 features)
   - 6. `app_pos_cash` (147 + 14 = 161 features)
   - 7. `full_all_tables` (All 256 features)
4. The leave-one-group-out (LOGO) comparisons test the removal of each of the 6 distinct historical groups:
   - `logo_bureau` (Full without Bureau: 230 features)
   - `logo_prev` (Full without Previous Applications: 231 features)
   - `logo_installments` (Full without Installments: 240 features)
   - `logo_credit_card` (Full without Credit Card: 235 features)
   - `logo_pos_cash` (Full without POS/CASH: 242 features)
   - `logo_derived` (Full without Cross-Domain Derived: 249 features)
This provides a total of 13 ablation runs (7 incremental + 6 LOGO), completely covering all feature families with zero double-counting.

---

## 5. Preflight Gate Verdict

**STATUS: GREEN — ALL PREREQUISITES VERIFIED.** Proceeding directly to Gate B (Experiment E1).
