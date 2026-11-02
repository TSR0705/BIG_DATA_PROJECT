# Experiment E3: Class Imbalance × Probability Calibration Report

**Project:** Loan Default Prediction Using Big Data Analytics  
**Phase:** Phase 10 — Research Experiments (E3)  
**Execution Timestamp:** 2026-11-02 17:23:28 UTC  
**Evaluator Models:** 13 configurations spanning Logistic Regression, Random Forest, XGBoost, and LightGBM.  
**Imbalance Strategies:** None (A1-A4), Class Weighting (B5-B7), Positive-Class Weighting (C8-C9), and SMOTE (D10-D13).  
**Data Partitions:** Train ($N=215,257$) for fitting; Validation ($N=46,127$) for evaluation. **Zero test data accessed.**  

---

## 1. Executive Summary & Core Research Questions

### Research Questions Addressed:
1. *Does artificial imbalance correction (class weighting or SMOTE) improve actual discrimination (ROC-AUC / PR-AUC)?*
2. *How severely do imbalance correction methods distort probability calibration?*
3. *Can post-hoc Isotonic Calibration restore operational probability alignment without losing discrimination or using validation labels?*

### Key Empirical Findings:
1. **Discrimination Invariance:** Synthetic resampling (SMOTE) and loss weighting do **not** meaningfully improve rank discrimination (ROC-AUC or PR-AUC). Across all models, baseline unweighted models achieve equal or superior PR-AUC to their SMOTE-augmented counterparts (e.g. XGBoost PR-AUC: 0.2834 baseline vs 0.2708 SMOTE; RF PR-AUC: 0.2523 baseline vs 0.1880 SMOTE).
2. **Severe Probability Distortion:** Class weighting (`class_weight='balanced'`) and positive weighting (`scale_pos_weight=11.3875`) inflate predicted probabilities by roughly an order of magnitude. At the default 0.50 cutoff, this gives the *illusion* of high recall ($\approx 65\text{--}70\%$), but drastically inflates the Brier score (from $\approx 0.066$ up to $0.16\text{--}0.19$) and destroys calibration (ECE increases from $<0.003$ to $0.25\text{--}0.33$; Cox intercept shifts to $-1.87$ to $-2.44$).
3. **Isotonic Recalibration Protocol:** Post-hoc isotonic regression fitted strictly on an internal holdout of the training set ($N=43,052$, zero validation labels accessed) realigns empirical probabilities, reducing validation ECE to $<0.0038$, decreasing Brier scores back to $\approx 0.066$, and bringing Cox calibration slopes close to $1.00\text{--}1.02$.
4. **Calibration Caveat:** While isotonic calibration reduces empirical ECE on validation data, it produces piecewise-constant step functions that discretize continuous probability predictions. Low validation ECE does not guarantee resilience to domain shifts or covariate distribution shifts in out-of-time credit environments, nor does it improve the fundamental discriminative capability (ROC-AUC / PR-AUC) of the underlying learner.

---

## 2. Comprehensive Experimental Results Table (13 Configurations)

### Table 2.1: Discrimination and Calibration Metrics (Before vs. After Isotonic Calibration)

| Config ID | Model | Strategy | Uncal ROC-AUC | Cal ROC-AUC | Uncal PR-AUC | Cal PR-AUC | Uncal Log Loss | Cal Log Loss | Uncal Brier | Cal Brier | Uncal ECE | Cal ECE | Uncal Cox Slope | Cal Cox Slope |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `E3_A1_LR_None` | Logistic Regression | None | 0.7757 | 0.7751 | 0.2688 | 0.2578 | 0.2406 | 0.2409 | 0.0666 | 0.0666 | 0.0023 | 0.0032 | 1.00 | 1.01 |
| `E3_A2_RF_None` | Random Forest | None | 0.7590 | 0.7582 | 0.2523 | 0.2429 | 0.2471 | 0.2469 | 0.0679 | 0.0676 | 0.0109 | 0.0015 | 1.29 | 1.00 |
| `E3_A3_XGB_None` | XGBoost | None | 0.7893 | 0.7890 | 0.2834 | 0.2715 | 0.2362 | 0.2366 | 0.0657 | 0.0658 | 0.0025 | 0.0024 | 1.02 | 1.02 |
| `E3_A4_LGBM_None` | LightGBM | None | 0.7859 | 0.7854 | 0.2830 | 0.2740 | 0.2371 | 0.2373 | 0.0658 | 0.0659 | 0.0020 | 0.0025 | 1.04 | 1.02 |
| `E3_B5_LR_ClassWeight` | Logistic Regression | Class Weight | 0.7755 | 0.7750 | 0.2645 | 0.2562 | 0.5731 | 0.2419 | 0.1946 | 0.0668 | 0.3257 | 0.0022 | 0.98 | 1.01 |
| `E3_B6_RF_ClassWeight` | Random Forest | Class Weight | 0.7567 | 0.7559 | 0.2382 | 0.2289 | 0.4515 | 0.2471 | 0.1411 | 0.0681 | 0.2540 | 0.0027 | 1.22 | 1.01 |
| `E3_B7_LGBM_ClassWeight` | LightGBM | Class Weight | 0.7855 | 0.7851 | 0.2807 | 0.2706 | 0.5125 | 0.2388 | 0.1709 | 0.0660 | 0.2888 | 0.0020 | 1.03 | 1.01 |
| `E3_C8_XGB_ScalePosWeight` | XGBoost | Positive-Class Weight | 0.7854 | 0.7851 | 0.2843 | 0.2726 | 0.4909 | 0.2379 | 0.1616 | 0.0659 | 0.2743 | 0.0024 | 1.01 | 1.02 |
| `E3_C9_LGBM_ScalePosWeight` | LightGBM | Positive-Class Weight | 0.7855 | 0.7851 | 0.2803 | 0.2700 | 0.5103 | 0.2382 | 0.1707 | 0.0660 | 0.2869 | 0.0020 | 1.04 | 1.01 |
| `E3_D10_LR_SMOTE` | Logistic Regression | SMOTE | 0.7648 | 0.7642 | 0.2521 | 0.2425 | 0.5628 | 0.2447 | 0.1908 | 0.0675 | 0.3063 | 0.0038 | 0.81 | 1.02 |
| `E3_D11_RF_SMOTE` | Random Forest | SMOTE | 0.7323 | 0.7314 | 0.1880 | 0.1806 | 0.3159 | 0.2565 | 0.0882 | 0.0702 | 0.1218 | 0.0017 | 1.13 | 1.00 |
| `E3_D12_XGB_SMOTE` | XGBoost | SMOTE | 0.7808 | 0.7802 | 0.2708 | 0.2623 | 0.2394 | 0.2396 | 0.0664 | 0.0664 | 0.0057 | 0.0023 | 1.03 | 1.01 |
| `E3_D13_LGBM_SMOTE` | LightGBM | SMOTE | 0.7838 | 0.7832 | 0.2734 | 0.2643 | 0.2385 | 0.2391 | 0.0663 | 0.0663 | 0.0056 | 0.0031 | 1.06 | 1.00 |

---

### Table 2.2: Operational Classification Metrics at Default Decision Cutoff (τ = 0.50)

| Config ID | Model | Strategy | Precision (Uncal) | Recall (Uncal) | F1 (Uncal) | F2 (Uncal) | Specificity (Uncal) | Balanced Acc (Uncal) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `E3_A1_LR_None` | Logistic Regression | None | 0.5662 | 0.0333 | 0.0629 | 0.0410 | 0.9978 | 0.5155 |
| `E3_A2_RF_None` | Random Forest | None | 0.8000 | 0.0011 | 0.0021 | 0.0013 | 1.0000 | 0.5005 |
| `E3_A3_XGB_None` | XGBoost | None | 0.5730 | 0.0432 | 0.0804 | 0.0530 | 0.9972 | 0.5202 |
| `E3_A4_LGBM_None` | LightGBM | None | 0.5572 | 0.0405 | 0.0756 | 0.0498 | 0.9972 | 0.5189 |
| `E3_B5_LR_ClassWeight` | Logistic Regression | Class Weight | 0.1714 | 0.7017 | 0.2755 | 0.4335 | 0.7021 | 0.7019 |
| `E3_B6_RF_ClassWeight` | Random Forest | Class Weight | 0.2187 | 0.4793 | 0.3004 | 0.3871 | 0.8496 | 0.6645 |
| `E3_B7_LGBM_ClassWeight` | LightGBM | Class Weight | 0.1912 | 0.6775 | 0.2982 | 0.4491 | 0.7483 | 0.7129 |
| `E3_C8_XGB_ScalePosWeight` | XGBoost | Positive-Class Weight | 0.2003 | 0.6493 | 0.3061 | 0.4483 | 0.7723 | 0.7108 |
| `E3_C9_LGBM_ScalePosWeight` | LightGBM | Positive-Class Weight | 0.1897 | 0.6842 | 0.2970 | 0.4497 | 0.7432 | 0.7137 |
| `E3_D10_LR_SMOTE` | Logistic Regression | SMOTE | 0.1712 | 0.6799 | 0.2735 | 0.4264 | 0.7109 | 0.6954 |
| `E3_D11_RF_SMOTE` | Random Forest | SMOTE | 0.2966 | 0.0607 | 0.1008 | 0.0722 | 0.9874 | 0.5240 |
| `E3_D12_XGB_SMOTE` | XGBoost | SMOTE | 0.5489 | 0.0346 | 0.0652 | 0.0426 | 0.9975 | 0.5161 |
| `E3_D13_LGBM_SMOTE` | LightGBM | SMOTE | 0.5455 | 0.0338 | 0.0637 | 0.0416 | 0.9975 | 0.5157 |

---

## 3. Methodological Insights & Practical Credit Underwriting Guidelines

1. **The Fallacy of the 0.50 Decision Cutoff:**
   In highly imbalanced credit datasets (8.07% base default rate), uncalibrated models naturally output low risk probabilities. Applying class weights forces predicted probabilities into the range $[0.30, 0.70]$, which makes a naive $\tau=0.50$ cutoff trigger high recall at the cost of catastrophic false positive rates.
2. **Proper Underwriting Architecture:**
   The recommended production workflow is:
   $$\text{Features} \xrightarrow{\text{LightGBM / XGBoost}} \hat{p}_{\text{raw}} \xrightarrow{\text{Isotonic Calibration}} \hat{p}_{\text{calibrated}} \xrightarrow{\text{Cost-Sensitive Threshold / Risk Bands}} \text{Decision}$$
3. **Zero Test-Leakage Guarantee:**
   Calibration models were trained strictly on internal training set holdouts ($N=43,052$). Validation data ($N=46,127$) was evaluated solely on pre-fitted calibrators without observing validation labels.
