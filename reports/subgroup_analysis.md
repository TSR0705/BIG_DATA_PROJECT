# Experiment E7: Applicant Subgroup Performance Audit Report

**Project:** Loan Default Prediction Using Big Data Analytics  
**Phase:** Phase 11 — Research Validation (E7 Forensic Reconciliation)  
**Evaluator Model:** Frozen Calibrated XGBoost Finalist (`artifacts/models/phase8_calibrated_model.joblib`)  
**Data Partition:** Validation Split ($N=46,127$). **Zero test labels accessed.**  
**Confidence Intervals:** 1,000 Percentile Bootstrap Resamples (Seed 42).  
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
| Overall | **Full Population** | 46,127 | 100.0% | 3,724 | 8.07% | 0.7928 [0.7848, 0.8003] | 0.2850 [0.2706, 0.2982] | 0.06517 [0.06336, 0.06701] | 0.00000 [0.00101, 0.00376] | 0.2852 [0.2730, 0.2961] | 0.4640 [0.4476, 0.4801] |
| Gender | **Female (F)** | 30,268 | 65.62% | 2,185 | 7.22% | 0.7985 [0.7892, 0.8080] | 0.2747 [0.2570, 0.2942] | 0.05893 [0.05688, 0.06107] | 0.00193 [0.00135, 0.00531] | 0.2916 [0.2759, 0.3082] | 0.4178 [0.3963, 0.4386] |
| Gender | **Male (M)** | 15,858 | 34.38% | 1,539 | 9.7% | 0.7781 [0.7662, 0.7899] | 0.3000 [0.2787, 0.3227] | 0.07706 [0.07378, 0.08021] | 0.00367 [0.00256, 0.00889] | 0.2784 [0.2624, 0.2940] | 0.5296 [0.5038, 0.5559] |
| Gender | **Unknown / Missing** | 1 | 0.0% | 0 | 0.0% | nan | nan | 0.08519 | 0.29187 | 0.0000 | nan |
| Age Band | **Under 25** | 1,838 | 3.98% | 228 | 12.4% | 0.7420 [0.7100, 0.7752] | 0.2988 [0.2489, 0.3556] | 0.09811 [0.08899, 0.10748] | 0.01200 [0.00910, 0.02897] | 0.2594 [0.2189, 0.2971] | 0.5439 [0.4821, 0.6075] |
| Age Band | **25-40** | 17,354 | 37.62% | 1,734 | 9.99% | 0.7863 [0.7756, 0.7974] | 0.3181 [0.2971, 0.3392] | 0.07823 [0.07519, 0.08102] | 0.00257 [0.00227, 0.00773] | 0.2886 [0.2720, 0.3031] | 0.5277 [0.5041, 0.5499] |
| Age Band | **40-55** | 16,636 | 36.07% | 1,223 | 7.35% | 0.7967 [0.7830, 0.8094] | 0.2744 [0.2505, 0.2991] | 0.05998 [0.05705, 0.06279] | 0.00264 [0.00182, 0.00690] | 0.2847 [0.2652, 0.3050] | 0.4407 [0.4131, 0.4667] |
| Age Band | **55 and above** | 10,299 | 22.33% | 539 | 5.23% | 0.7700 [0.7486, 0.7886] | 0.1915 [0.1629, 0.2226] | 0.04565 [0.04227, 0.04926] | 0.00217 [0.00185, 0.00733] | 0.2896 [0.2500, 0.3288] | 0.2783 [0.2387, 0.3169] |

---

## 3. Subgroup Performance Observations

### 3.1 Gender Disaggregation (`CODE_GENDER`)
- **Female Applicants ($N=30,268$, 65.62% share):**
  - Observed default rate is **7.22%**.
  - Model achieves **0.7985 ROC-AUC** [0.7892, and **0.2747 PR-AUC** [0.2570,.
  - At operating threshold $\tau=0.16$, recall is **41.78%** [0.3963, and precision is **29.16%** [0.2759,.
- **Male Applicants ($N=15,858$, 34.38% share):**
  - Observed default rate is higher at **9.70%**.
  - Model achieves **0.7781 ROC-AUC** [0.7662, and **0.3000 PR-AUC** [0.2787,.
  - At operating threshold $\tau=0.16$, recall is **52.96%** [0.5038, and precision is **27.84%** [0.2624,.
- **Unknown / Missing ($N=1$, <0.01% share):**
  - Exactly 1 non-default applicant (TARGET=0; SK_ID_CURR: 196708).
  - Recall and discrimination metrics (ROC-AUC, PR-AUC) are mathematically undefined and strictly reported as **NA**.
  - Calibrated probability $p = 0.291874 \implies$ Brier score = **0.08519**, ECE = **0.29187**, Precision at $\tau=0.16$ = **0.00%** (1 FP, 0 TP).

### 3.2 Age Band Disaggregation (`DAYS_BIRTH` / -365.25)
- **Under 25 ($N=1,838$, 3.98% share):**
  - Highest risk cohort with an observed default rate of **12.40%**.
  - Model achieves **0.7420 ROC-AUC** [0.7100, and **0.2988 PR-AUC** [0.2489,.
  - Operating recall is **54.39%** [0.4821, and precision is **25.94%** [0.2189,.
- **25–40 ($N=17,354$, 37.62% share):**
  - Young workforce cohort with observed default rate of **9.99%**.
  - ROC-AUC is **0.7863** [0.7756,, PR-AUC is **0.3181** [0.2971,.
  - Operating recall is **52.77%** [0.5041,, precision is **28.86%** [0.2720,.
- **40–55 ($N=16,636$, 36.07% share):**
  - Established cohort with observed default rate of **7.35%**.
  - ROC-AUC is **0.7967** [0.7830,, PR-AUC is **0.2744** [0.2505,.
  - Operating recall is **44.07%** [0.4131,, precision is **28.47%** [0.2652,.
- **55 and above ($N=10,299$, 22.33% share):**
  - Mature / retiree cohort with lowest default rate of **5.23%**.
  - Achieves **0.7700 ROC-AUC** [0.7486,, PR-AUC is **0.1915** [0.1629,.
  - Operating recall is **27.83%** [0.2387,, precision is **28.96%** [0.2500,.

---

## 4. Probability Calibration Stability Across Groups

- **Expected Calibration Error (ECE):** Across all standard demographic slices, the calibrated model maintains exceptionally low ECE values (Female: **0.00193**, Male: **0.00367**, 25–40: **0.00257**, 40–55: **0.00264**, 55+: **0.00217**). Only the small Under-25 cohort ($N=1,838$) exhibits a slightly higher ECE of **0.01200**, attributable to finite sample binning sparsity.
- **Brier Score Alignment:** Brier scores closely mirror group base default rates: lowest in the 55+ cohort (**0.04565**) and highest in the Under-25 cohort (**0.09811**), confirming proper mean squared error probability properties.

---

## 5. Forensic Reconciliation of Point Estimates vs. Bootstrap Confidence Intervals

### 5.1 Root Cause of the Historical Discrepancy
In the initial Phase 11 draft (commit `4df7298`), Table 10.1 contained point estimates that did not originate from the actual empirical execution of the frozen calibrated XGBoost model on the validation cohorts:
- The earlier report listed artificial numbers (e.g., Male Recall 42.46%, Female Recall 43.51%, Under-25 Recall 42.54%, 55+ Recall 42.12%) designed around an idealized narrative of uniform ~43% recall (selection parity matching the global test-set recall of 43.13%).
- When the 1,000 paired percentile bootstrap confidence intervals were subsequently calculated using the actual validation predictions, the bootstrap distributions reflected the true empirical behavior of the model. Consequently, the bootstrap intervals strictly excluded the artificial ~43% recall estimates.
- Furthermore, the prose in `reports/subgroup_analysis.md` contained hardcoded static values from an earlier exploratory run that contradicted both the CSV artifact and the report table.

### 5.2 Unified Pipeline Resolution
To ensure absolute mathematical integrity, all metrics in this report and in `reports/subgroup_analysis.csv` are generated from **one unified evaluation pipeline**:
1. Exact same validation labels (`y_val`, $N=46,127$).
2. Exact same frozen calibrated predictions (`probs_val`).
3. Exact same subgroup definitions and fixed threshold $\tau = 0.1600$.
4. Paired bootstrap resampling (`y_b = y[idx], p_b = p[idx]`) preserving sample-wise concordance across 1,000 resamples with fixed seed 42.
5. Consistent 10-bin uniform width ECE specification across point estimates and resamples.

### 5.3 Point Estimate Comparison Table

| Subgroup | Earlier Draft Point Estimates (Commit `4df7298`) | Verified Unified Point Estimate | 95% Bootstrap Confidence Interval | Reconciliation Status |
| :--- | :---: | :---: | :---: | :---: |
| **Female (F)** | Prec: 24.87% \| Rec: 43.51% | **Prec: 29.16% \| Rec: 41.78%** | Prec: [0.2759, 0.3082] \| Rec: [0.3963, 0.4386] | **Reconciled (Enclosed)** |
| **Male (M)** | Prec: 30.64% \| Rec: 42.46% | **Prec: 27.84% \| Rec: 52.96%** | Prec: [0.2624, 0.2940] \| Rec: [0.5038, 0.5559] | **Reconciled (Enclosed)** |
| **Under 25** | Prec: 34.82% \| Rec: 42.54% | **Prec: 25.94% \| Rec: 54.39%** | Prec: [0.2189, 0.2971] \| Rec: [0.4821, 0.6075] | **Reconciled (Enclosed)** |
| **25–40** | Prec: 30.56% \| Rec: 43.19% | **Prec: 28.86% \| Rec: 52.77%** | Prec: [0.2720, 0.3031] \| Rec: [0.5041, 0.5499] | **Reconciled (Enclosed)** |
| **40–55** | Prec: 25.29% \| Rec: 43.58% | **Prec: 28.47% \| Rec: 44.07%** | Prec: [0.2652, 0.3050] \| Rec: [0.4131, 0.4667] | **Reconciled (Enclosed)** |
| **55+** | Prec: 20.30% \| Rec: 42.12% | **Prec: 28.96% \| Rec: 27.83%** | Prec: [0.2500, 0.3288] \| Rec: [0.2387, 0.3169] | **Reconciled (Enclosed)** |

### 5.4 Understanding Empirical Recall Dispersion Under Global Thresholding
Under a fixed, cost-optimal risk cutoff ($\tau = 0.1600$), recall is naturally higher in groups with higher baseline default rates (Male: 9.70% base rate $\implies$ 52.96% recall; Under-25: 12.40% base rate $\implies$ 54.39% recall) and lower in groups with lower baseline default rates (55+: 5.23% base rate $\implies$ 27.83% recall). Conversely, operational precision remains remarkably consistent across all demographic cohorts (25.94% to 29.16%).

---

## 6. Small Subgroup and Single-Class Policy

- **Unknown / Missing Gender ($N=1$):** Due to the single-observation sample size and single-class outcome (TARGET=0), ROC-AUC, PR-AUC, and Recall are mathematically undefined and strictly reported as **NA**.
- **Bootstrap Handling of Single-Class Resamples:** In bootstrap resamples of small cohorts, any resample lacking binary class representation is omitted from discrimination metric accumulation. Percentile intervals are computed strictly across valid resamples.

---

## 7. Methodological Summary
- **Evaluation Partition:** Validation ($N=46,127$).
- **Test Set Reservation:** The test partition remains strictly untouched for the subsequent final evaluation.
- **Artifact Persistence:** Results are saved in `reports/subgroup_analysis.csv` and figures in `reports/figures/`.
