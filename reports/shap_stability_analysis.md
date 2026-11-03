# Experiment E5: SHAP Explanation Stability Across Retraining

**Project:** Loan Default Prediction Using Big Data Analytics  
**Phase:** Phase 11 — Research Validation (E5)  
**Evaluator Architecture:** Tuned XGBoost (`max_depth=3`, `learning_rate=0.1`, `n_estimators=500`, `min_child_weight=10`, `subsample=0.8`, `colsample_bytree=1.0`)  
**Retraining Scope:** 15 Bootstrap Retrains of Standard XGBoost + 15 Bootstrap Retrains of SMOTE XGBoost (30 Total Models).  
**Explanation Panel:** Fixed stratified validation sample of $N=5,000$ applicants (Seed 42).  
**Zero Test Data Leakage:** All retraining models were fitted strictly on the Training partition ($N=215,257$). Test split was strictly excluded.

---

## 1. Executive Summary & Research Question

**Research Question:**  
*"How stable are global feature attributions, local applicant-level SHAP explanations, and raw-score risk tiers when the underlying GBDT model is retrained on different bootstrap samples of the training data?"*

### 1.1 SHAP Sign Convention & Attribution Terminology
- **Model Output Explained:** Exact TreeSHAP (`shap.TreeExplainer`) explains the raw margin log-odds of default (Class 1, $y=1$) under binary logistic loss.
- **Directional Attribution:**
  - **Positive SHAP ($\phi_i > 0$):** Increases the predicted default log-odds (acts as a default risk factor).
  - **Negative SHAP ($\phi_i < 0$):** Decreases the predicted default log-odds (acts as a protective factor).
- **Attribution Ranking Criterion:** Top-$k$ explanations are extracted based on **absolute attribution magnitude ($| \phi_i |$)**, capturing the primary driving features determining the applicant's prediction. The identical sign and magnitude convention is used consistently across all 30 retraining models.
- **Regulatory Clarification:** These feature rankings represent model-level predictive attributions and are **not** legally certified Adverse Action reason codes under FCRA or ECOA regulations.

### 1.2 Core Empirical Findings
1. **High Global Attribution Stability:** Standard XGBoost displays high global ranking consistency across retraining runs, achieving a mean **Kendall's tau rank correlation of 0.7182 ($\pm$0.0157)** and an average **Top-10 feature overlap of 86.29%**.
2. **Empirical Degradation with SMOTE:** Retraining with SMOTE synthesis is empirically associated with lower global explanation consistency (Kendall's tau drops to **0.4845**, and Top-10 overlap declines to **45.33%**).
3. **Local Explanation Consistency:** For individual applicant explanations, top-3 predictive attribution factors achieve a mean Jaccard similarity of **0.4993** across retraining runs under Standard XGBoost. Sensitivity audits show **0.4817 at $k=4$** and **0.4755 at $k=5$**.
4. **Raw-Score Band Sensitivity:** Across $\binom{15}{2} = 105$ model pairs evaluated over $5,000$ panel applicants ($525,000$ total applicant-pair comparisons), applicants experience a raw-score band shift in **10.39%** of comparisons under Standard XGBoost, with **94.75% retention** for LOW raw-score applicants and **74.97% retention** for HIGH raw-score applicants.

---

## 2. Experimental Results Summary Table

| Metric Category | Metric Dimension | Standard XGBoost (15 Retrains) | SMOTE XGBoost (15 Retrains) | Methodological Definition & Notes |
| :--- | :--- | :---: | :---: | :--- |
| **Global Stability** | Mean Kendall's Tau ($\tau$) | **0.7182** ($\pm$0.0157) | 0.4845 ($\pm$0.0294) | Pairwise rank correlation across all 256 model features (105 pairs). |
| | Top-10 Overlap Ratio | **86.29%** | 45.33% | Percentage of identical features in the top-10 global hierarchy. |
| | Median Feature Rank Shift | **20.99** ranks | 34.98 ranks | Median standard deviation of feature rankings across retrains. |
| **Local Stability** | Jaccard@3 (Top-3 Factors) | **0.4993** ($\pm$0.1467) | 0.2352 ($\pm$0.0984) | Applicant-level Jaccard similarity of top-3 magnitude factors ($| \phi_i |$). |
| | Jaccard@4 (Top-4 Factors) | **0.4817** ($\pm$0.1382) | 0.2336 ($\pm$0.0911) | Attribution factor overlap sensitivity at $k=4$. |
| | Jaccard@5 (Top-5 Factors) | **0.4755** ($\pm$0.1310) | 0.2403 ($\pm$0.0865) | Attribution factor overlap sensitivity at $k=5$. |
| **Score Sensitivity** | Band Shift Rate | **10.39%** | 36.79% | Frequency of applicant changing score tiers across 105 model pairs. |
| | LOW Tier Retention | **94.75%** | 49.36% | Symmetrized empirical retention rate in $[0.00, 0.09)$ band. |
| | MEDIUM Tier Retention | **76.51%** | 38.59% | Symmetrized empirical retention rate in $[0.09, 0.25)$ band. |
| | HIGH Tier Retention | **74.97%** | 77.90% | Symmetrized empirical retention rate in $[0.25, 1.00]$ band. |

---

## 3. Global Feature Stability Ranking (Standard XGBoost)

Top 10 most influential features and their rank distribution across all 15 bootstrap retraining runs:

| Hierarchy Rank | Feature Identifier | Mean Rank | Rank Std ($\sigma$) | Rank Range [Min, Max] | Source Subsystem | Feature Description & Significance |
| :---: | :--- | :---: | :---: | :---: | :--- | :--- |
| **1** | `APP_EXT_SOURCES_MEAN` | **1.00** | 0.00 | [1, 1] | Multi-Table Model | Mean of external credit bureau scores (unanimous #1 across all retrains). |
| **2** | `CODE_GENDER` | **2.53** | 0.81 | [2, 4] | Application Demographic | Applicant sex indicator (demographic proxy reflecting historical disparities). |
| **3** | `APP_PAYMENT_RATE` | **3.53** | 0.88 | [2, 5] | Application Engineered | Annuity to total credit ratio (repayment cash-flow burden). |
| **4** | `POS_COMPLETION_RATE_MEAN` | **3.53** | 1.31 | [2, 7] | POS-Cash History | Historical contract completion discipline on POS loans. |
| **5** | `APP_GOODS_TO_CREDIT_RATIO` | **5.80** | 1.11 | [4, 8] | Application Engineered | Goods price divided by credit granted (loan-to-value proxy). |
| **6** | `NAME_EDUCATION_TYPE` | **6.73** | 1.00 | [5, 8] | Application Categorical | Educational attainment level (socioeconomic earning power proxy). |
| **7** | `AMT_ANNUITY` | **7.20** | 1.64 | [5, 11] | Application Financial | Monthly debt service commitment amount. |
| **8** | `DERIVED_PAYMENT_DISCIPLINE_SCORE` | **7.80** | 4.28 | [3, 21] | Cross-Domain Engineered | Multi-table composite ratio of installment & POS arrears history. |
| **9** | `OWN_CAR_AGE` | **9.47** | 1.86 | [6, 14] | Application Asset | Age of vehicle owned (collateral freshness & wealth proxy). |
| **10** | `BURO_DEBT_TO_CREDIT_RATIO` | **11.60** | 3.52 | [6, 19] | Bureau Aggregation | External debt to credit limit ratio across historical bureau loans. |

---

## 4. Local Applicant Explanation Stability Distribution

Distribution of Jaccard@3 top predictive attribution factor agreement across the 5,000 panel applicants:

- **Mean:** 0.4993
- **Standard Deviation:** 0.1467
- **5th Percentile:** 0.2810
- **25th Percentile (Q1):** 0.3962
- **Median (Q2):** 0.4871
- **75th Percentile (Q3):** 0.5886
- **95th Percentile:** 0.7638

---

## 5. Raw-Score Band Sensitivity & Calibration Caveat

### 5.1 Uncalibrated Score Comparison
The 30 retrained models are uncalibrated estimators evaluated directly on raw predicted probabilities (`predict_proba`).
- Standard models are trained on the natural ~8.07% empirical default rate; their raw probabilities naturally center near 0.08.
- SMOTE models are trained on synthetic 1:1 balanced samples; their raw probabilities center near 0.50.
- Applying frozen probability cutoffs $[0.09, 0.25]$ to uncalibrated SMOTE models reflects **raw-score band sensitivity under prior probability shift**, not directly comparable operational risk bands.

### 5.2 Transition Matrix Mathematics
- **Model Pairs:** $\binom{15}{2} = 105$ pairwise comparisons per strategy.
- **Applicant Denominator:** 5,000 fixed validation panel applicants.
- **Total Comparisons:** $105 \times 5,000 = 525,000$ applicant-pair evaluations.
- **Transition Pooling:** Transition counts are pooled symmetrically across all 105 pairs and row-normalized to estimate empirical tier retention:
  $$P(\text{Tier}_B = k \mid \text{Tier}_A = k) = \frac{\sum_{p=1}^{105} \sum_{i=1}^{5000} \mathbb{I}(b_{p1, i} = k \land b_{p2, i} = k)}{\sum_{p=1}^{105} \sum_{i=1}^{5000} \mathbb{I}(b_{p1, i} = k)}$$

### 5.3 Hypothesized Mechanisms for SMOTE Attribution Variance
The empirical association between SMOTE training and reduced attribution stability (Kendall's tau 0.4845 vs 0.7182) can be explained by several plausible hypotheses:
1. **Manifold Boundary Distortion:** SMOTE synthesizes minority points by linear interpolation in 256-dimensional feature space, placing artificial samples in sparse, unpopulated regions between clusters.
2. **Tree Splitting Instability:** In gradient boosted trees, artificial data points near the boundary create high-variance early splits, shifting downstream feature selections across bootstrap samples.
3. **Prior Probability Shift:** The uncalibrated margin inflation shifts the base leaf value, altering relative gradient directions across feature subsets.
