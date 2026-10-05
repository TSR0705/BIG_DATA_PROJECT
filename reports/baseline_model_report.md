# Phase 7 Baseline Classification Modeling Report

> **Project:** Loan Default Prediction Using Big Data Analytics  
> **Phase:** Phase 7 — Baseline Classification Models  
> **Source:** `C:\Users\ACER\Desktop\BIG_DATA_PROJECT\data\model_input\model_input.parquet`  
> **Generated:** 2026-10-05 16:01:04 UTC  
> **Status:** **PASS (GREEN)**  

---

## 1. Objective
Establish an empirical, reproducible baseline benchmark across four canonical classification paradigms for loan default prediction without premature hyperparameter tuning, class-weight rebalancing, threshold adaptation, or feature selection.

## 2. Dataset Source & Frozen Split Description
- **Dataset:** `data/model_input/model_input.parquet` (307,511 applicant rows × 258 columns).
- **Partitioning:** Strictly aligned with Phase 6 frozen parquet index files (`data/splits/`).
  - **TRAIN:** 215,257 rows (70.00%)
  - **VALIDATION:** 46,127 rows (15.00%)
  - **TEST:** 46,127 rows (15.00%)
- **Test Set Integrity:** The test set was touched exclusively for the final unbiased evaluation of frozen baselines.

## 3. Feature Count & Class Distribution
- **Model Features:** 256 model-eligible features (240 continuous/discrete numeric, 16 categorical).
- **Excluded Columns:** `SK_ID_CURR` (applicant ID) and `TARGET` (ground-truth label) are strictly excluded from feature matrix $X$.
- **Class Imbalance:** Baseline default rate is approximately **8.073%** positive across all three stratified partitions.

## 4. Preprocessing Strategies & Leakage Prevention
- **Logistic Regression Pipeline:**
  - Numeric: `SimpleImputer(strategy='median')` followed by `StandardScaler()`.
  - Categorical: `SimpleImputer(strategy='most_frequent')` followed by `OneHotEncoder(min_frequency=0.01, handle_unknown='ignore')`.
  - Output dimension: 330 dense features.
- **Tree-Based Models (Random Forest, LightGBM, XGBoost):**
  - Numeric: Preserved at unscaled raw precision with native missingness handling.
  - Categorical: Encoded via deterministic `OrdinalEncoder` fitted **strictly on X_train** (unseen/missing mapped to -1).
- **Leakage Prevention:** All scalers, imputers, and encoders were fitted **exclusively on X_train**. Validation and test partitions were transformed using train-derived parameters only.

## 5. Model Configurations
1. **Logistic Regression:** `LogisticRegression(max_iter=1000, random_state=42)`
2. **Random Forest:** `RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)`
3. **LightGBM:** `LGBMClassifier(objective='binary', n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42, n_jobs=-1, verbosity=-1)`
4. **XGBoost:** `XGBClassifier(objective='binary:logistic', n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, eval_metric='logloss')`

---

## 6. Model Comparison & Benchmark Results

### A. Validation Set Performance
| Model | ROC-AUC | PR-AUC | Log Loss | Accuracy | Precision | Recall | F1 Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.7763 | 0.2692 | 0.2404 | 0.9200 | 0.5818 | 0.0344 | 0.0649 |
| **Random Forest** | 0.7467 | 0.2437 | 0.2515 | 0.9194 | 0.5750 | 0.0062 | 0.0122 |
| **LightGBM** | 0.7886 | 0.2880 | 0.2363 | 0.9203 | 0.5833 | 0.0432 | 0.0805 |
| **XGBoost** | 0.7897 | 0.2896 | 0.2358 | 0.9205 | 0.5965 | 0.0457 | 0.0848 |

### B. Final Unbiased Test Set Performance
| Model | ROC-AUC | PR-AUC | Log Loss | Accuracy | Precision | Recall | F1 Score | Training Time (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.7773 | 0.2517 | 0.2417 | 0.9192 | 0.4870 | 0.0252 | 0.0480 | 6.71s |
| **Random Forest** | 0.7454 | 0.2271 | 0.2545 | 0.9194 | 0.5833 | 0.0056 | 0.0112 | 93.50s |
| **LightGBM** | 0.7877 | 0.2749 | 0.2376 | 0.9198 | 0.5490 | 0.0376 | 0.0704 | 10.76s |
| **XGBoost** | 0.7893 | 0.2772 | 0.2369 | 0.9200 | 0.5571 | 0.0419 | 0.0779 | 15.62s |

---

## 7. Analysis of Metrics & Class Imbalance Impact
- **The Accuracy Fallacy:** All models achieve approximately 91.9% accuracy at default threshold 0.50 simply because the negative class comprises 91.9% of applicants. High accuracy is non-informative here.
- **ROC-AUC & PR-AUC as Primary Evaluators:** LightGBM and XGBoost achieve the highest ROC-AUC and PR-AUC, demonstrating substantial discriminatory power above the 8.073% no-skill random chance rate.
- **Default Threshold Behavior (0.50):** Under an uncalibrated default 0.50 cutoff, recall on the minority class is low because the predicted posterior probabilities reflect the low baseline prior. Threshold optimization is explicitly deferred to later phases.

---

## 8. Artifacts & Generated Figures
- **ROC Curves:**
  - Validation: [`reports/figures/phase7_roc_validation.png`](figures/phase7_roc_validation.png)
  - Test: [`reports/figures/phase7_roc_test.png`](figures/phase7_roc_test.png)
- **Precision-Recall Curves:**
  - Validation: [`reports/figures/phase7_pr_validation.png`](figures/phase7_pr_validation.png)
  - Test: [`reports/figures/phase7_pr_test.png`](figures/phase7_pr_test.png)
- **Confusion Matrices:**
  - Logistic Regression: [`reports/figures/logistic_regression_confusion_matrix.png`](figures/logistic_regression_confusion_matrix.png)
  - Random Forest: [`reports/figures/random_forest_confusion_matrix.png`](figures/random_forest_confusion_matrix.png)
  - LightGBM: [`reports/figures/lightgbm_confusion_matrix.png`](figures/lightgbm_confusion_matrix.png)
  - XGBoost: [`reports/figures/xgboost_confusion_matrix.png`](figures/xgboost_confusion_matrix.png)
- **Feature Importance Charts & CSVs:**
  - Random Forest: [`reports/figures/feature_importance_random_forest.png`](figures/feature_importance_random_forest.png) | [`reports/feature_importance_random_forest.csv`](feature_importance_random_forest.csv)
  - LightGBM: [`reports/figures/feature_importance_lightgbm.png`](figures/feature_importance_lightgbm.png) | [`reports/feature_importance_lightgbm.csv`](feature_importance_lightgbm.csv)
  - XGBoost: [`reports/figures/feature_importance_xgboost.png`](figures/feature_importance_xgboost.png) | [`reports/feature_importance_xgboost.csv`](feature_importance_xgboost.csv)
- **Trained Model Binaries:**
  - `artifacts/models/logistic_regression.joblib`
  - `artifacts/models/random_forest.joblib`
  - `artifacts/models/lightgbm.joblib`
  - `artifacts/models/xgboost.joblib`
  - `artifacts/models/tree_preprocessor.joblib`

---

## 9. Limitations & Boundary Notices
- **No Statistical Significance Claims:** The relative rankings of baseline models reflect point estimates; no formal hypothesis testing or bootstrapping has yet been performed.
- **No Causal Inferences:** Native feature importances represent model-internal split contributions and do not imply causal mechanisms of default risk.
- **No Production Readiness Claim:** These are uncalibrated, unoptimized baselines intended solely to establish reference performance anchors.
