# Experiment E7: Applicant Subgroup Performance Audit Report

**Project:** Loan Default Prediction Using Big Data Analytics  
**Phase:** Phase 11 — Research Validation (Leakage-Free Independent Calibration Audit)  
**Evaluator Model:** Independent Calibrated XGBoost (`artifacts/models/phase11_independent_calibrator.joblib`)  
**Base Estimator:** Frozen Phase 8 Tuned XGBoost (`artifacts/models/phase8_best_xgboost.joblib`)  
**Calibration Method:** Isotonic Regression via 5-Fold Stratified Out-of-Fold (OOF) cross-fitting on Training Set ($N=215,257$).  
**Evaluation Partition:** Validation Split ($N=46,127$). **Zero validation rows used during calibration fitting.**  
**Test Set Status:** Strictly reserved ($N=46,127$). **Zero test rows or labels accessed.**  
**Confidence Intervals:** 1,000 Conventional Percentile Bootstrap Resamples (Seed 42).  
**Operating Decision Threshold:** $\tau = 0.1600$ (Cost-Optimal 1:5 ratio).  
**ECE Definition:** 10 uniform frequency/width partitions spanning $[0.0, 1.0]$.  

---

## 1. Executive Summary & Audit Mandate

### Purpose and Academic / Legal Disclaimers
This subgroup evaluation is an **exploratory performance and calibration audit**, designed solely to assess how model discrimination, probability alignment, and classification metrics vary across demographic categories present in the Kaggle Home Credit dataset.

> [!CAUTION]
> **Interpretation Restrictions:**
> 1. This audit does **NOT** constitute a legal fairness certification, disparate impact review, or regulatory compliance filing.
> 2. No causal inferences regarding gender or age can be drawn from these observational correlations.
> 3. No fairness post-processing, demographic parity constraints, or model adjustments have been implemented, preserving frozen model integrity.

---

## 2. Comprehensive Subgroup Audit Results Table

| Dimension | Subgroup Name | Sample Size ($n$) | Share (%) | Defaults | Default Rate (%) | ROC-AUC (95% CI) | PR-AUC (95% CI) | Brier Score [95% CI] | ECE [95% CI] | Precision ($\tau=0.16$) [95% CI] | Recall ($\tau=0.16$) [95% CI] |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Overall | **Full Population** | 46,127 | 100.0% | 3,724 | 8.07% | 0.7909 [0.7828, 0.7984] | 0.2853 [0.2710, 0.2984] | 0.06542 [0.06364, 0.06723] | 0.00417 [0.00232, 0.00680] | 0.2851 [0.2729, 0.2960] | 0.4619 [0.4455, 0.4780] |
| Gender | **Female (F)** | 30,268 | 65.62% | 2,185 | 7.22% | 0.7965 [0.7870, 0.8061] | 0.2757 [0.2578, 0.2953] | 0.05911 [0.05708, 0.06126] | 0.00251 [0.00178, 0.00577] | 0.2916 [0.2760, 0.3083] | 0.4160 [0.3947, 0.4368] |
| Gender | **Male (M)** | 15,858 | 34.38% | 1,539 | 9.7% | 0.7763 [0.7645, 0.7882] | 0.2994 [0.2776, 0.3229] | 0.07747 [0.07430, 0.08060] | 0.00775 [0.00514, 0.01271] | 0.2783 [0.2621, 0.2937] | 0.5270 [0.5019, 0.5532] |
| Gender | **Unknown / Missing** | 1 | 0.0% | 0 | 0.0% | NA | NA | 0.07950 | 0.28195 | 0.0000 | NA |
| Age Band | **Under 25** | 1,838 | 3.98% | 228 | 12.4% | 0.7403 [0.7077, 0.7732] | 0.2984 [0.2491, 0.3551] | 0.09851 [0.08959, 0.10794] | 0.00627 [0.00753, 0.02466] | 0.2594 [0.2189, 0.2971] | 0.5439 [0.4821, 0.6075] |
| Age Band | **25-40** | 17,354 | 37.62% | 1,734 | 9.99% | 0.7840 [0.7732, 0.7954] | 0.3179 [0.2968, 0.3393] | 0.07863 [0.07558, 0.08141] | 0.00430 [0.00288, 0.00891] | 0.2887 [0.2720, 0.3032] | 0.5254 [0.5015, 0.5472] |
| Age Band | **40-55** | 16,636 | 36.07% | 1,223 | 7.35% | 0.7953 [0.7819, 0.8082] | 0.2753 [0.2512, 0.3008] | 0.06016 [0.05718, 0.06294] | 0.00534 [0.00283, 0.00955] | 0.2847 [0.2652, 0.3046] | 0.4383 [0.4101, 0.4645] |
| Age Band | **55 and above** | 10,299 | 22.33% | 539 | 5.23% | 0.7682 [0.7463, 0.7878] | 0.1920 [0.1634, 0.2235] | 0.04577 [0.04240, 0.04935] | 0.00359 [0.00210, 0.00833] | 0.2893 [0.2495, 0.3283] | 0.2764 [0.2367, 0.3154] |

---

## 3. Subgroup Performance Observations

### 3.1 Gender Disaggregation (`CODE_GENDER`)
- **Female Applicants ($N=30,268$, 65.62% share):**
  - Observed default rate is **7.22%**.
  - Model achieves **0.7965 ROC-AUC** [0.7870, and **0.2757 PR-AUC** [0.2578,.
  - At operating threshold $\tau=0.16$, recall is **41.60%** [0.3947, and precision is **29.16%** [0.2760,.
- **Male Applicants ($N=15,858$, 34.38% share):**
  - Observed default rate is higher at **9.70%**.
  - Model achieves **0.7763 ROC-AUC** [0.7645, and **0.2994 PR-AUC** [0.2776,.
  - At operating threshold $\tau=0.16$, recall is **52.70%** [0.5019, and precision is **27.83%** [0.2621,.
- **Unknown / Missing ($N=1$, <0.01% share):**
  - Exactly 1 non-default applicant (TARGET=0; SK_ID_CURR: 196708).
  - Recall and discrimination metrics (ROC-AUC, PR-AUC) are mathematically undefined and strictly reported as **NA**.
  - Calibrated probability $p = 0.281952 \implies$ Brier score = **0.07950**, ECE = **0.28195**, Precision at $\tau=0.16$ = **0.00%** (1 FP, 0 TP).

### 3.2 Age Band Disaggregation (`DAYS_BIRTH` / -365.25)
- **Under 25 ($N=1,838$, 3.98% share):**
  - Highest risk cohort with an observed default rate of **12.40%**.
  - Model achieves **0.7403 ROC-AUC** [0.7077, and **0.2984 PR-AUC** [0.2491,.
  - Operating recall is **54.39%** [0.4821, and precision is **25.94%** [0.2189,.
- **25–40 ($N=17,354$, 37.62% share):**
  - Young workforce cohort with observed default rate of **9.99%**.
  - ROC-AUC is **0.7840** [0.7732,, PR-AUC is **0.3179** [0.2968,.
  - Operating recall is **52.54%** [0.5015,, precision is **28.87%** [0.2720,.
- **40–55 ($N=16,636$, 36.07% share):**
  - Established cohort with observed default rate of **7.35%**.
  - ROC-AUC is **0.7953** [0.7819,, PR-AUC is **0.2753** [0.2512,.
  - Operating recall is **43.83%** [0.4101,, precision is **28.47%** [0.2652,.
- **55 and above ($N=10,299$, 22.33% share):**
  - Mature / retiree cohort with lowest default rate of **5.23%**.
  - Achieves **0.7682 ROC-AUC** [0.7463,, PR-AUC is **0.1920** [0.1634,.
  - Operating recall is **27.64%** [0.2367,, precision is **28.93%** [0.2495,.

---

## 4. Probability Calibration Analysis Across Groups

- **Overall Population Out-of-Sample ECE:** For the full validation cohort ($N=46,127$), the empirical out-of-sample ECE is **0.00417** (exact: $0.004168$), with a conventional 95% percentile bootstrap confidence interval of **[0.00232, 0.00680]**. Because calibration was fitted entirely out-of-fold on the training set with zero validation leakage, this represents a realistic, positive calibration error ($~0.42\%$) on unseen applicant data.
- **Subgroup Expected Calibration Error (ECE):** Across standard demographic slices, the independent calibrator demonstrates low calibration error:
  - Female (F): **0.00251** [0.00178, 0.00577]
  - Male (M): **0.00775** [0.00514, 0.01271]
  - 25–40: **0.00430** [0.00288, 0.00891]
  - 40–55: **0.00534** [0.00283, 0.00955]
  - 55+: **0.00359** [0.00210, 0.00833]
  - Under 25: **0.00627** [0.00753, 0.02466] (Note: the small cohort size $N=1,838$ induces positive folded-normal noise floor in resampled partitions).
- **Brier Score Alignment:** Brier scores closely mirror group base default rates: lowest in the 55+ cohort (**0.04577**) and highest in the Under-25 cohort (**0.09851**), confirming proper mean squared error probability properties.

---

## 5. Calibration Provenance & Leakage Elimination Audit

### 5.1 Identification of Historical In-Sample Calibration Leakage
In Phase 8 (`scripts/run_phase8_optimization.py` lines 276–277), probability calibration was fitted directly on the validation partition:
```python
calib_isotonic, _ = calibrate_model(best_xgb_model, X_val_tree, y_val, method="isotonic")
```
When Experiment E7 originally evaluated this model on `(X_val, y_val)`, the resulting ECE was artificially near zero ($1.35 \times 10^-7 \approx 0.00000$), because the step functions of the Pool Adjacent Violators Algorithm (PAVA) were fitted to empirical validation frequencies. This was an in-sample evaluation that could not support claims of independent calibration performance.

### 5.2 Leakage-Free Architecture: 5-Fold Stratified OOF on Training Set
To eliminate evaluation leakage while preserving frozen Phase 7 and Phase 8 baseline artifacts:
1. **Training Partition Exclusivity:** The training partition (`data/splits/train_indices.parquet`, $N=215,257$) was split into 5 stratified folds (`random_state=42`).
2. **Out-of-Fold Probability Generation:** In each fold, XGBoost was trained on 4/5 folds using the frozen Phase 8 hyperparameters and predicted probabilities on the 1/5 out-of-fold holdout.
3. **Training Isotonic Calibration:** `IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)` was fitted strictly on `(oof_train_probs, y_train)`.
4. **Separate Production Artifact:** Saved as `artifacts/models/phase11_independent_calibrator.joblib`. The Phase 8 artifact `phase8_calibrated_model.joblib` was preserved untouched.
5. **Zero Validation or Test Contact:** Exactly 0 validation rows and 0 test rows were accessed during calibration fitting.

### 5.3 Old (In-Sample) vs. New (Independent) Validation Results Comparison

| Evaluation Metric | Phase 8 In-Sample Validation (Old) | Phase 11 Independent Validation (New) | Absolute Change | Methodological Status |
| :--- | :---: | :---: | :---: | :--- |
| **Calibration Fit Dataset** | Validation Partition ($N=46,127$) | Train 5-Fold OOF ($N=215,257$) | +169,130 rows | **Zero Validation Leakage** |
| **Validation Rows Used in Fitting** | 46,127 (100% overlap) | 0 (0% overlap) | -46,127 rows | **Strictly Disjoint** |
| **Expected Calibration Error (ECE)** | 0.000000 ($1.35 \times 10^-7$) | **0.004168** (~0.42%) | +0.004168 | **Realistic Out-of-Sample ECE** |
| **ECE 95% Bootstrap CI** | [0.00000, 0.00376] (artificially clamped) | **[0.00232, 0.00680]** | Natural Enclosure | **Pure Conventional Percentile CI** |
| **Brier Score** | 0.065166 | **0.065424** | +0.000258 | Expected slight out-of-sample shift |
| **Log Loss** | 0.234167 | **0.235339** | +0.001172 | Expected slight out-of-sample shift |
| **ROC-AUC** | 0.792769 | **0.790945** | -0.001824 | Monotonic step ties effect |
| **PR-AUC** | 0.285016 | **0.285314** | +0.000298 | Stable precision-recall support |
| **Precision ($\tau=0.16$)** | 0.2852 | **0.2851** | -0.0001 | Stable operational precision |
| **Recall ($\tau=0.16$)** | 0.4640 | **0.4619** | -0.0021 | Stable operational recall |
| **F1-Score ($\tau=0.16$)** | 0.3533 | **0.3526** | -0.0007 | Stable F1 performance |

### 5.4 Subgroup Metrics Old vs. New Comparison

| Subgroup | Old In-Sample Precision | New Independent Precision | Old In-Sample Recall | New Independent Recall | New ECE [95% CI] |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Overall** | 28.52% | **28.51%** [0.2729, 0.2960] | 46.40% | **46.19%** [0.4455, 0.4780] | 0.00417 [0.00232, 0.00680] |
| **Female (F)** | 29.16% | **29.16%** [0.2760, 0.3083] | 41.78% | **41.60%** [0.3947, 0.4368] | 0.00251 [0.00178, 0.00577] |
| **Male (M)** | 27.84% | **27.83%** [0.2621, 0.2937] | 52.96% | **52.70%** [0.5019, 0.5532] | 0.00775 [0.00514, 0.01271] |
| **Under 25** | 25.94% | **25.94%** [0.2189, 0.2971] | 54.39% | **54.39%** [0.4821, 0.6075] | 0.00627 [0.00753, 0.02466] |
| **25–40** | 28.86% | **28.87%** [0.2720, 0.3032] | 52.77% | **52.54%** [0.5015, 0.5472] | 0.00430 [0.00288, 0.00891] |
| **40–55** | 28.47% | **28.47%** [0.2652, 0.3046] | 44.07% | **43.83%** [0.4101, 0.4645] | 0.00534 [0.00283, 0.00955] |
| **55+** | 28.96% | **28.93%** [0.2495, 0.3283] | 27.83% | **27.64%** [0.2367, 0.3154] | 0.00359 [0.00210, 0.00833] |

---

## 6. Small Subgroup and Single-Class Policy

- **Unknown / Missing Gender ($N=1$):** Due to the single-observation sample size and single-class outcome (TARGET=0), ROC-AUC, PR-AUC, and Recall are mathematically undefined and strictly reported as **NA**.
- **Bootstrap Handling of Single-Class Resamples:** In bootstrap resamples of small cohorts, any resample lacking binary class representation is omitted from discrimination metric accumulation. Percentile intervals are computed strictly across valid resamples.

---

## 7. Methodological Summary
- **Evaluation Partition:** Validation ($N=46,127$).
- **Calibration Provenance:** 5-Fold Stratified Out-of-Fold on Training Set ($N=215,257$).
- **Test Set Reservation:** The test partition ($N=46,127$) remains strictly untouched for subsequent final evaluation.
- **Artifact Persistence:** Results are saved in `reports/subgroup_analysis.csv` and figures in `reports/figures/`.
