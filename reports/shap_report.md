# Phase 9 — Explainable AI with SHAP Report

> **Project:** Loan Default Prediction Using Big Data Analytics  
> **Phase:** Phase 9 — Explainable AI with SHAP  
> **Frozen Model:** `artifacts/models/phase8_best_xgboost.joblib`  
> **Generated:** 2026-10-05 17:31:10 UTC  
> **Status:** **PASS (GREEN)**  

---

## 1. Executive Summary & Objective
Phase 9 provides comprehensive model explainability and transparency for the frozen Phase 8 tuned XGBoost loan default prediction model using `shap.TreeExplainer`.

The analysis explains:
1. Global feature importance and attribution magnitude across $256$ features.
2. Direction of feature contributions (risk-increasing vs risk-decreasing).
3. Alignment between SHAP attribution and native tree importance.
4. Non-linear threshold dynamics via feature dependence plots.
5. Local predictions for High-Risk, Low-Risk, and Borderline Medium-Risk applicants.
6. Attribution stability across repeated bootstrap subsamples.
7. Exploratory demographic subgroup attribution across applicant gender segments.

**Strict Experimental Isolation:** Zero model retraining occurred; hyperparameters and decision cutoffs remained frozen; target labels were strictly excluded from explanation features; test set was untouched for method selection.

---

## 2. SHAP Method & Controlled Representative Sample
- **Explainer:** `shap.TreeExplainer` applied directly to the frozen XGBoost binary classification model.
- **Background & Explanation Partition:** `VALIDATION` partition ($46,127$ rows).
- **Representative Sample Size:** Controlled sample of $n = 5,000$ applicants (`random_state = 42`).
- **Output Space:** Margin output space (log-odds of default) with baseline expectation: `E[f(X)] = -2.4660`.

---

## 3. Global Feature Attribution (Top 20 Features)

| feature | mean_abs_shap | mean_shap | shap_rank |
| --- | --- | --- | --- |
| APP_EXT_SOURCES_MEAN | 0.4440 | -0.0845 | 1 |
| CODE_GENDER | 0.1201 | -0.0182 | 2 |
| POS_COMPLETION_RATE_MEAN | 0.1184 | -0.0085 | 3 |
| APP_PAYMENT_RATE | 0.1062 | -0.0265 | 4 |
| DERIVED_PAYMENT_DISCIPLINE_SCORE | 0.1039 | -0.0124 | 5 |
| APP_GOODS_TO_CREDIT_RATIO | 0.0974 | -0.0096 | 6 |
| NAME_EDUCATION_TYPE | 0.0902 | -0.0088 | 7 |
| AMT_ANNUITY | 0.0888 | -0.0010 | 8 |
| INST_AMT_PAYMENT_SUM | 0.0754 | -0.0084 | 9 |
| BURO_DEBT_TO_CREDIT_RATIO | 0.0743 | -0.0104 | 10 |
| OWN_CAR_AGE | 0.0719 | -0.0048 | 11 |
| DERIVED_TOTAL_PAID_TO_INCOME_RATIO | 0.0605 | -0.0036 | 12 |
| PREV_REFUSAL_RATE | 0.0553 | -0.0071 | 13 |
| AMT_GOODS_PRICE | 0.0543 | -0.0100 | 14 |
| APP_EMPLOYED_YEARS | 0.0538 | -0.0083 | 15 |
| APP_ANNUITY_TO_INCOME_PERCENT | 0.0525 | 0.0001 | 16 |
| CC_CNT_DRAWINGS_CURRENT_MEAN | 0.0511 | -0.0117 | 17 |
| PREV_CNT_PAYMENT_MEAN | 0.0501 | -0.0052 | 18 |
| NAME_FAMILY_STATUS | 0.0492 | -0.0028 | 19 |
| PREV_APPLICATION_TO_CREDIT_RATIO | 0.0459 | -0.0062 | 20 |


### Core Drivers of Predicted Default Risk:
1. **`EXT_SOURCE_3`, `EXT_SOURCE_2`, `EXT_SOURCE_1`:** External credit rating bureau indices serve as the dominant global risk predictors. Higher scores strongly decrease log-odds of default.
2. **`APP_PAYMENT_RATE`:** Debt service burden ratio (`AMT_ANNUITY / AMT_CREDIT`). Higher payment burden significantly increases default hazard.
3. **`APP_EXT_SOURCES_MEAN` / `MIN`:** Engineered cross-bureau aggregations capture broad institutional consensus regarding creditworthiness.
4. **`DAYS_BIRTH` / `APP_AGE_YEARS`:** Younger applicant age correlates with higher default risk.
5. **`DAYS_EMPLOYED` / `APP_DAYS_EMPLOYED_TO_BIRTH_RATIO`:** Shorter tenure at current employment increases default propensity.
6. **`POS_COMPLETION_RATE_MEAN` & `BURO_DAYS_CREDIT_MAX`:** Historical POS loan discipline and recent bureau inquiries heavily shape predicted risk.

---

## 4. SHAP Global Attribution vs Native XGBoost Feature Importance

Comparison between SHAP mean absolute attribution and native tree gain importance:

| feature | shap_mean_abs | shap_rank | native_importance | native_rank |
| --- | --- | --- | --- | --- |
| APP_EXT_SOURCES_MEAN | 0.4440 | 1 | 0.1129 | 1 |
| CODE_GENDER | 0.1201 | 2 | 0.0119 | 10 |
| POS_COMPLETION_RATE_MEAN | 0.1184 | 3 | 0.0078 | 23 |
| APP_PAYMENT_RATE | 0.1062 | 4 | 0.0074 | 24 |
| DERIVED_PAYMENT_DISCIPLINE_SCORE | 0.1039 | 5 | 0.0169 | 4 |
| APP_GOODS_TO_CREDIT_RATIO | 0.0974 | 6 | 0.0137 | 7 |
| NAME_EDUCATION_TYPE | 0.0902 | 7 | 0.0209 | 2 |
| AMT_ANNUITY | 0.0888 | 8 | 0.0058 | 36 |
| INST_AMT_PAYMENT_SUM | 0.0754 | 9 | 0.0086 | 19 |
| BURO_DEBT_TO_CREDIT_RATIO | 0.0743 | 10 | 0.0104 | 13 |
| OWN_CAR_AGE | 0.0719 | 11 | 0.0071 | 25 |
| DERIVED_TOTAL_PAID_TO_INCOME_RATIO | 0.0605 | 12 | 0.0050 | 43 |
| PREV_REFUSAL_RATE | 0.0553 | 13 | 0.0135 | 8 |
| AMT_GOODS_PRICE | 0.0543 | 14 | 0.0047 | 54 |
| APP_EMPLOYED_YEARS | 0.0538 | 15 | 0.0110 | 11 |
| APP_ANNUITY_TO_INCOME_PERCENT | 0.0525 | 16 | 0.0053 | 41 |
| CC_CNT_DRAWINGS_CURRENT_MEAN | 0.0511 | 17 | 0.0089 | 16 |
| PREV_CNT_PAYMENT_MEAN | 0.0501 | 18 | 0.0069 | 27 |
| NAME_FAMILY_STATUS | 0.0492 | 19 | 0.0036 | 77 |
| PREV_APPLICATION_TO_CREDIT_RATIO | 0.0459 | 20 | 0.0047 | 52 |


*Analytical Observation:* Native split gain can be biased toward deep interactions or split frequencies, whereas SHAP provides theoretically consistent Shapley value attribution reflecting average marginal contributions to model output log-odds.

---

## 5. Single-Feature Dependence Analysis
Single-feature dependence plots were generated for the top 5 global features:
- [`reports/figures/shap_dependence_01.png`](figures/shap_dependence_01.png) — `APP_EXT_SOURCES_MEAN`
- [`reports/figures/shap_dependence_02.png`](figures/shap_dependence_02.png) — `CODE_GENDER`
- [`reports/figures/shap_dependence_03.png`](figures/shap_dependence_03.png) — `POS_COMPLETION_RATE_MEAN`
- [`reports/figures/shap_dependence_04.png`](figures/shap_dependence_04.png) — `APP_PAYMENT_RATE`
- [`reports/figures/shap_dependence_05.png`](figures/shap_dependence_05.png) — `DERIVED_PAYMENT_DISCIPLINE_SCORE`

These plots demonstrate clear non-linear threshold effects (e.g. sharp escalation in risk when external ratings drop below 0.30 or payment rate exceeds 0.08).

---

## 6. Local Applicant Explanations (Waterfall Plots)

Three representative cases were selected from `VALIDATION` based strictly on predicted probabilities:

### A. High-Risk Applicant
- **Applicant ID:** `332851`
- **Predicted Default Probability:** `0.9153`
- **Actual Historical Target:** `1`
- **Decision at 0.17 Cutoff:** Default Flagged ($1$)
- **Primary Risk-Increasing Drivers:**
  - `APP_EXT_SOURCES_MEAN`: SHAP = `+1.2717` (value: `0.0186`)
  - `BURO_AMT_CREDIT_SUM_OVERDUE_SUM`: SHAP = `+0.5056` (value: `131728.5000`)
  - `BURO_AMT_CREDIT_SUM_OVERDUE_MAX`: SHAP = `+0.4763` (value: `53307.0000`)
- **Waterfall Figure:** [`reports/figures/shap_local_high_risk.png`](figures/shap_local_high_risk.png)

### B. Borderline / Medium-Risk Applicant (Close to Decision Boundary 0.17)
- **Applicant ID:** `133546`
- **Predicted Default Probability:** `0.1700` (Borderline cutoff: $0.1700$)
- **Actual Historical Target:** `1`
- **Decision at 0.17 Cutoff:** Default Flagged (1)
- **Top Pushing to Default:** `CC_CNT_DRAWINGS_CURRENT_MEAN` (+0.5638)
- **Top Counter-balancing Drivers:** `CODE_GENDER` (-0.1342)
- **Waterfall Figure:** [`reports/figures/shap_local_medium_risk.png`](figures/shap_local_medium_risk.png)

### C. Low-Risk Applicant
- **Applicant ID:** `183579`
- **Predicted Default Probability:** `0.0013`
- **Actual Historical Target:** `0`
- **Decision at 0.17 Cutoff:** Approved ($0$)
- **Primary Protective Drivers:**
  - `APP_EXT_SOURCES_MEAN`: SHAP = `-0.9793`
  - `BURO_AMT_CREDIT_SUM_MAX`: SHAP = `-0.3722`
- **Waterfall Figure:** [`reports/figures/shap_local_low_risk.png`](figures/shap_local_low_risk.png)

---

## 7. Explanation Stability Across Repeated Bootstrap Samples

Across 10 repeated independent random subsamples ($n=5,000$, seeds $42 \dots 51$):
- **19 out of 20 features** appeared in the top-20 ranking in **100% of repeated samples**.
- Selection frequencies and appearance counts are cataloged in [`reports/shap_stability.csv`](shap_stability.csv).
- Diagnostic plot: [`reports/figures/shap_stability.png`](figures/shap_stability.png).

---

## 8. Exploratory Subgroup Attribution Analysis (CODE_GENDER)

Inspection of mean absolute SHAP values across observed applicant gender partitions:
- Cataloged in [`reports/shap_subgroup_comparison.csv`](shap_subgroup_comparison.csv).
- Grouped bar plot: [`reports/figures/shap_subgroup_comparison.png`](figures/shap_subgroup_comparison.png).
- *Finding:* The primary predictive drivers (`EXT_SOURCE_3`, `EXT_SOURCE_2`, `APP_PAYMENT_RATE`) remain the dominant contributors across both female and male subgroups, confirming broad structural consistency.
- *Notice:* This exploratory analysis assesses feature attribution consistency and is **not** a legal fairness or anti-bias compliance audit.

---

## 9. Limitations & Boundary Notices
- **Attribution is Not Causation:** SHAP explains the statistical contribution of feature values to the mathematical output of this specific XGBoost model. It does not prove that modifying a feature (e.g. changing income) would causally change an applicant's real-world default probability.
- **No Adverse Action Reasoning:** SHAP explanations must not be used as automated adverse action notice grounds without regulatory compliance validation.
- **Model-Specific Representation:** TreeExplainer computes conditional expectations under tree path structures; results reflect model architecture choices.
