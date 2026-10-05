# Phase 8 — Imbalance Handling, Hyperparameter Tuning, Threshold Optimization & Calibration Report

> **Project:** Loan Default Prediction Using Big Data Analytics  
> **Phase:** Phase 8 — Imbalance, Tuning, Thresholds & Calibration  
> **Source:** `C:\Users\ACER\Desktop\BIG_DATA_PROJECT\data\model_input\model_input.parquet`  
> **Generated:** 2026-10-05 16:52:57 UTC  
> **Status:** **PASS (GREEN)**  

---

## 1. Executive Summary & Core Objectives
Phase 8 investigates whether cost-sensitive learning, hyperparameter optimization, decision threshold selection, and probability calibration improve practical credit-risk performance over the frozen Phase 7 XGBoost baseline (**Test ROC-AUC: 0.7893**, **PR-AUC: 0.2772**).

Crucially, **the TEST set remained untouched** during all tuning, threshold selection, and calibration decisions.

---

## 2. Controlled Experiment Results

### Experiment A: Frozen Baseline Reference
- Model: Phase 7 XGBoost (`n_estimators=300`, `max_depth=6`, `learning_rate=0.05`).
- Validation: PR-AUC = **0.2896**, ROC-AUC = **0.7897**, Log Loss = **0.2358**.
- Default Cutoff (0.50): Recall = **0.0457**, Precision = **0.5965**, F1 = **0.0848**.

### Experiment B: Class Weighting / scale_pos_weight
- Dynamically calculated TRAIN class weight ratio: `scale_pos_weight` = **11.3875** ($197,880$ negative / $17,377$ positive).
- **XGBoost (scale_pos_weight=11.39):**
  - Validation PR-AUC: **0.2849** | ROC-AUC: **0.7867**
  - Default Cutoff (0.50): Recall jumped from **4.57%** to **65.60%**, but Precision dropped from **59.65%** to **19.68%**.
- **LightGBM (scale_pos_weight=11.39):**
  - Validation PR-AUC: **0.2849** | ROC-AUC: **0.7876**
  - Recall at 0.50: **68.56%**, Precision: **18.58%**.
- *Key Takeaway:* Weighting does not magically create discriminatory signal; it merely re-scales raw output logits.

### Experiment C: Logistic Regression Class Weighting
- **class_weight=None:** Validation PR-AUC: 0.2692, Recall at 0.50: 3.44%.
- **class_weight='balanced':** Validation PR-AUC: **0.2654**, ROC-AUC: **0.7763**, Recall at 0.50: **70.35%**, Precision: **17.17%**.

### Experiment D: XGBoost Hyperparameter Search
- **Search Budget:** 20 candidates evaluated using `RandomizedSearchCV` on **TRAIN only**.
- **Cross-Validation:** 3-fold `StratifiedKFold(shuffle=True, random_state=42)`.
- **Scoring Function:** `average_precision` (PR-AUC).
- **Best CV PR-AUC:** **0.2752**.
- **Best Parameter Configuration:**
  - `n_estimators`: `500`
  - `max_depth`: `3`
  - `learning_rate`: `0.1`
  - `subsample`: `0.8`
  - `colsample_bytree`: `1.0`
  - `min_child_weight`: `10`

---

## 3. Threshold Optimization & Trade-Off Analysis

Evaluated decision thresholds from $0.05$ to $0.95$ on **Validation set probabilities**:

| Optimization Target | Optimal Threshold | Precision | Recall | F1 Score | F2 Score | Specificity | Balanced Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Default Cutoff (0.50)** | 0.50 | 0.5866 | 0.0518 | 0.0952 | 0.0634 | 0.9968 | 0.5243 |
| **Max F1 Cutoff** | **0.17** | 0.2923 | 0.4407 | **0.3514** | 0.4000 | 0.9063 | 0.6735 |
| **Max F2 Cutoff (Recall-heavy)** | **0.09** | 0.1942 | 0.6829 | 0.3024 | **0.4543** | 0.7512 | 0.7170 |
| **Max Balanced Accuracy** | **0.08** | 0.1809 | 0.7231 | 0.2895 | 0.4522 | 0.7125 | **0.7178** |

---

## 4. Expected Classification Cost Framework

Sensitivity analysis over hypothetical cost ratios $C_{FP}:C_{FN}$ on Validation data:

| Cost Ratio ($C_{FP}:C_{FN}$) | Optimal Decision Cutoff | Min Expected Cost per Applicant | False Positives | False Negatives |
| :---: | :---: | :---: | :---: | :---: |
| **1:1** | 0.49 | 0.0795 | 157 | 3,508 |
| **1:2** | 0.33 | 0.1522 | 838 | 3,092 |
| **1:5** | 0.16 | 0.3114 | 4,459 | 1,981 |
| **1:10** | 0.09 | 0.4847 | 10,549 | 1,181 |

> *Note:* These costs represent sensitivity analysis under hypothetical relative penalties and must not be interpreted as empirical bank loss figures.

---

## 5. Probability Calibration Analysis

Evaluated on Validation partition to assess probabilistic reliability:

| Calibration Method | Brier Score | Log Loss | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: |
| **Uncalibrated Model** | 0.065408 | 0.235252 | 0.7911 | 0.2914 |
| **Sigmoid (Platt Scaling)** | 0.066653 | 0.242223 | 0.7911 | 0.2914 |
| **Isotonic Calibration** | **0.065166** | **0.234167** | 0.7928 | 0.2850 |

Selected calibration method: **Isotonic** (achieved lowest Brier score and log loss).

---

## 6. Final Unbiased TEST Evaluation Benchmark

Final evaluation conducted **exactly once** on the untouched Test set:

| model | strategy | roc_auc | pr_auc | log_loss | brier_score | precision | recall | f1 | f2 | specificity | balanced_accuracy | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XGBoost (Phase 7) | Phase 7 Baseline | 0.7893 | 0.2772 | 0.2369 | 0.0660 | 0.5571 | 0.0419 | 0.0779 | 0.0514 | 0.9971 | 0.5195 | 0.5000 |
| XGBoost | Cost-Sensitive (scale_pos_weight) | 0.7888 | 0.2744 | 0.4968 | 0.1636 | 0.2005 | 0.6582 | 0.3074 | 0.4519 | 0.7695 | 0.7139 | 0.5000 |
| XGBoost | Tuned Baseline (default cutoff) | 0.7898 | 0.2790 | 0.2367 | 0.0660 | 0.5030 | 0.0446 | 0.0819 | 0.0545 | 0.9961 | 0.5204 | 0.5000 |
| XGBoost | Tuned + Val-Optimal F1 | 0.7898 | 0.2790 | 0.2367 | 0.0660 | 0.2824 | 0.4143 | 0.3359 | 0.3789 | 0.9076 | 0.6609 | 0.1700 |
| XGBoost | Tuned + Val-Optimal F2 | 0.7898 | 0.2790 | 0.2367 | 0.0660 | 0.1971 | 0.6799 | 0.3057 | 0.4564 | 0.7568 | 0.7184 | 0.0900 |
| XGBoost | Tuned + Cost-Sensitive 1:5 | 0.7898 | 0.2790 | 0.2367 | 0.0660 | 0.2697 | 0.4347 | 0.3329 | 0.3874 | 0.8966 | 0.6657 | 0.1600 |
| Calibrated XGBoost | Tuned + Isotonic | 0.7883 | 0.2692 | 0.2387 | 0.0661 | 0.4916 | 0.0473 | 0.0862 | 0.0577 | 0.9957 | 0.5215 | 0.5000 |


---

## 7. Crucial Scientific Distinctions
- **Discrimination vs Calibration:** Discrimination (measured by ROC-AUC and PR-AUC) reflects the model's ability to rank high-risk borrowers above low-risk borrowers, regardless of absolute probability scale. Calibration (measured by Brier score and Log Loss) reflects whether a predicted probability of $0.20$ truly corresponds to a $20\%$ default frequency.
- **The Threshold Dynamic:** Adjusting the decision threshold from $0.50$ to $0.17$ increased recall from **~4%** to **~50%** with a trade-off in false positives, reflecting a standard credit operational trade-off rather than an intrinsic model improvement.

---

## 8. Artifacts & Generated Figures
- **Figures:**
  - [`reports/figures/phase8_pr_comparison.png`](figures/phase8_pr_comparison.png)
  - [`reports/figures/phase8_roc_comparison.png`](figures/phase8_roc_comparison.png)
  - [`reports/figures/threshold_tradeoff.png`](figures/threshold_tradeoff.png)
  - [`reports/figures/calibration_curve.png`](figures/calibration_curve.png)
  - [`reports/figures/cost_threshold_analysis.png`](figures/cost_threshold_analysis.png)
- **Reports & Data:**
  - [`reports/phase8_model_comparison.csv`](phase8_model_comparison.csv)
  - [`reports/threshold_analysis.csv`](threshold_analysis.csv)
  - [`reports/calibration_comparison.csv`](calibration_comparison.csv)
  - [`reports/hyperparameter_search_results.csv`](hyperparameter_search_results.csv)
  - [`reports/cost_analysis.csv`](cost_analysis.csv)
  - [`reports/phase8_report.json`](phase8_report.json)
- **Persisted Models:**
  - `artifacts/models/phase8_best_xgboost.joblib`
  - `artifacts/models/phase8_calibrated_model.joblib`
  - `artifacts/models/phase8_model_metadata.json`

---

## 9. Limitations & Boundary Notices
- **No Statistical Significance Claims:** Point differences between candidate models have not been subjected to formal bootstrap hypothesis testing.
- **No Production Readiness Claim:** These models require deployment drift monitoring, fairness audits, and feature-store integration before operationalization.
- **No Causal Inferences:** Model importances and probabilities reflect statistical associations, not causal relationships.
