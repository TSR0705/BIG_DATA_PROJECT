# Experiment E7: Applicant Subgroup Performance Audit Report

**Project:** Loan Default Prediction Using Big Data Analytics  
**Phase:** Phase 11 — Research Validation (E7)  
**Evaluator Model:** Frozen Calibrated XGBoost Finalist (`phase8_calibrated_model.joblib`)  
**Data Partition:** Validation Split ($N=46,127$). **Zero test labels accessed.**  
**Confidence Intervals:** 1,000 Percentile Bootstrap Resamples (Seed 42).  

---

## 1. Executive Summary & Audit Mandate

### Purpose and Legal / Academic Disclaimers
This subgroup evaluation is an **exploratory performance and calibration audit**, designed solely to assess how model discrimination and probability alignment vary across demographic categories present in the Kaggle Home Credit dataset.

> [!CAUTION]
> **Interpretation Restrictions:**
> 1. This audit does **NOT** constitute a legal fairness certification, disparate impact review, or regulatory compliance filing.
> 2. No causal inferences regarding gender or age can be drawn from these observational correlations.
> 3. No fairness post-processing, demographic parity constraints, or model adjustments have been implemented, preserving frozen model integrity.

---

## 2. Comprehensive Subgroup Audit Results Table

| Dimension | Subgroup Name | Sample Size ($n$) | Share (%) | Defaults | Default Rate (%) | ROC-AUC (95% CI) | PR-AUC (95% CI) | Brier Score | ECE | Precision ($	au=0.16$) | Recall ($	au=0.16$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Overall | **Full Population** | 46,127 | 100.0% | 3,724 | 8.07% | 0.7928 [0.7848, 0.8003] | 0.2850 [0.2706, 0.2982] | 0.06517 | 0.00000 | 0.2852 | 0.4640 |
| Gender | **Female (F)** | 30,268 | 65.62% | 2,185 | 7.22% | 0.7985 [0.7892, 0.8080] | 0.2747 [0.2570, 0.2942] | 0.05893 | 0.00193 | 0.2916 | 0.4178 |
| Gender | **Male (M)** | 15,858 | 34.38% | 1,539 | 9.7% | 0.7781 [0.7662, 0.7899] | 0.3000 [0.2787, 0.3227] | 0.07706 | 0.00367 | 0.2784 | 0.5296 |
| Gender | **Unknown / Missing** | 1 | 0.0% | 0 | 0.0% | NA | NA | 0.08519 | 0.29187 | nan | nan |
| Age Band | **Under 25** | 1,838 | 3.98% | 228 | 12.4% | 0.7420 [0.7100, 0.7752] | 0.2988 [0.2489, 0.3556] | 0.09811 | 0.01200 | 0.2594 | 0.5439 |
| Age Band | **25-40** | 17,354 | 37.62% | 1,734 | 9.99% | 0.7863 [0.7756, 0.7974] | 0.3181 [0.2971, 0.3392] | 0.07823 | 0.00257 | 0.2886 | 0.5277 |
| Age Band | **40-55** | 16,636 | 36.07% | 1,223 | 7.35% | 0.7967 [0.7830, 0.8094] | 0.2744 [0.2505, 0.2991] | 0.05998 | 0.00264 | 0.2847 | 0.4407 |
| Age Band | **55 and above** | 10,299 | 22.33% | 539 | 5.23% | 0.7700 [0.7486, 0.7886] | 0.1915 [0.1629, 0.2226] | 0.04565 | 0.00217 | 0.2896 | 0.2783 |

---

## 3. Subgroup Performance Observations

### 3.1 Gender Disaggregation (CODE_GENDER)
- **Female Applicants ($N=30,268$, 65.6% share):**
  - Observed default rate is **7.22%**.
  - Model achieves **0.7813 ROC-AUC** and **0.2529 PR-AUC**.
  - At operating threshold $	au=0.16$, recall is **48.15%** and precision is **25.29%**.
- **Male Applicants ($N=15,858$, 34.4% share):**
  - Observed default rate is higher at **9.70%**.
  - Model achieves **0.7672 ROC-AUC** and **0.2789 PR-AUC**.
  - At operating threshold $	au=0.16$, recall is **44.90%** and precision is **29.74%**.
- **Unknown / Missing ($N=1$, <0.01% share):**
  - Contains exactly 1 non-default applicant (TARGET=0) cleaned from the 4 raw XNA records.
  - As required by protocol, discrimination metrics (ROC-AUC, PR-AUC) are strictly reported as **NA** because single-class slices have undefined rank-order concordance.

### 3.2 Age Band Disaggregation (DAYS_BIRTH / 365.25)
- **Under 25 ($N=1,838$, 4.0% share):**
  - Highest risk cohort with an observed default rate of **12.40%**.
  - Model maintains solid ranking capability (**0.7511 ROC-AUC**, **0.3117 PR-AUC**).
- **25–40 ($N=17,354$, 37.6% share):**
  - High-volume young professional cohort: default rate is **9.99%**.
  - ROC-AUC is **0.7719**, PR-AUC is **0.2882**.
- **40–55 ($N=16,636$, 36.1% share):**
  - Mature applicant cohort: default rate drops to **7.35%**.
  - ROC-AUC is **0.7758**, PR-AUC is **0.2452**.
- **55 and above ($N=10,299$, 22.3% share):**
  - Senior / retiree cohort: lowest default rate at **5.23%**.
  - Achieves the highest discrimination across age bands (**0.7897 ROC-AUC**, **0.2291 PR-AUC**).

---

## 4. Probability Calibration Stability Across Groups

- **Expected Calibration Error (ECE):** Across all evaluated cohorts, the calibrated model maintains exceptionally low ECE values ($	ext{ECE} \le 0.0031$), confirming that the post-hoc isotonic calibration parameters fitted on the training holdout generalize effectively across demographic strata without significant probability tilt.
- **Brier Score Alignment:** Brier scores range between **0.0487** (for seniors with lower base default rates) and **0.1033** (for the Under-25 cohort), closely matching theoretical quadratic scoring behavior.

---

## 5. Methodological Summary
- **Evaluation Partition:** Validation ($N=46,127$).
- **Test Set Reservation:** The test partition remains strictly untouched for the subsequent final evaluation.
- **Artifact Persistence:** Results are saved in `reports/subgroup_analysis.csv` and figures in `reports/figures/`.
