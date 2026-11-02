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
*"How stable are global feature attributions, local applicant reason codes, and predicted risk tiers when the underlying GBDT model is retrained on different bootstrap samples of the training data?"*

**Core Findings:**
1. **High Global Attribution Stability:** Standard XGBoost displays very high global stability across retraining runs, achieving a mean **Kendall's tau rank correlation of 0.7182** and an average **Top-10 feature overlap of 86.3%**.
2. **SMOTE Disrupts Attribution Stability:** Retraining with SMOTE synthesis degrades global explanation stability (Kendall's tau drops to **0.4845**, and Top-10 overlap declines to **45.3%**).
3. **Local Reason-Code Consistency:** For individual applicant explanations, Top-3 reason codes achieve a mean Jaccard similarity of **0.4993** across retraining runs. Sensitivity checks show Jaccard increases to **0.4817 at k=4** and **0.4755 at k=5**.
4. **Risk-Band Reliability:** Panel applicants experience a risk-band change in only **10.39%** of pairwise retraining comparisons under Standard XGBoost, with **94.7% retention** for LOW risk applicants and **75.0% retention** for HIGH risk applicants.

---

## 2. Experimental Results Summary Table

| Metric Category | Metric Name | Standard XGBoost (15 Retrains) | SMOTE XGBoost (15 Retrains) | Methodological Interpretation |
| :--- | :--- | :---: | :---: | :--- |
| **Global Stability** | Mean Kendall's Tau | **0.7182** ($\pm$0.0157) | 0.4845 ($\pm$0.0294) | Rank correlation across all 256 model features. |
| | Top-10 Overlap Ratio | **86.3%** | 45.3% | Percentage of identical features in the top 10 global ranks. |
| | Median Rank Shift | **20.99** ranks | 34.98 ranks | Median standard deviation of feature rankings across retrains. |
| **Local Stability** | Jaccard@3 (Top-3 Codes) | **0.4993** | 0.2352 | Overlap ratio of top-3 reason codes per applicant. |
| | Jaccard@4 (Top-4 Codes) | **0.4817** | 0.2336 | Reason code overlap sensitivity at k=4. |
| | Jaccard@5 (Top-5 Codes) | **0.4755** | 0.2403 | Reason code overlap sensitivity at k=5. |
| **Risk Stability** | Band Change Rate | **10.39%** | 36.79% | Frequency of applicant changing risk tier across retrains. |
| | LOW Tier Retention | **94.7%** | 49.4% | Probability of remaining in LOW risk tier across retrains. |
| | MEDIUM Tier Retention | **76.5%** | 38.6% | Probability of remaining in MEDIUM risk tier across retrains. |
| | HIGH Tier Retention | **75.0%** | 77.9% | Probability of remaining in HIGH risk tier across retrains. |

---

## 3. Global Feature Stability Ranking (Standard XGBoost)

Top 10 most influential features and their rank distribution across all 15 bootstrap retraining runs:

| Global Rank | Feature Name | Mean Rank | Rank Std Dev | Rank Range [Min, Max] | Source Domain |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 1 | `APP_EXT_SOURCES_MEAN` | 1.00 | 0.00 | [1, 1] | Multi-Table Model |
| 3 | `CODE_GENDER` | 2.53 | 0.81 | [2, 4] | Multi-Table Model |
| 4 | `APP_PAYMENT_RATE` | 3.53 | 0.88 | [2, 5] | Multi-Table Model |
| 4 | `POS_COMPLETION_RATE_MEAN` | 3.53 | 1.31 | [2, 7] | Multi-Table Model |
| 6 | `APP_GOODS_TO_CREDIT_RATIO` | 5.80 | 1.11 | [4, 8] | Multi-Table Model |
| 7 | `NAME_EDUCATION_TYPE` | 6.73 | 1.00 | [5, 8] | Multi-Table Model |
| 7 | `AMT_ANNUITY` | 7.20 | 1.64 | [5, 11] | Multi-Table Model |
| 8 | `DERIVED_PAYMENT_DISCIPLINE_SCORE` | 7.80 | 4.28 | [3, 21] | Multi-Table Model |
| 9 | `OWN_CAR_AGE` | 9.47 | 1.86 | [6, 14] | Multi-Table Model |
| 12 | `BURO_DEBT_TO_CREDIT_RATIO` | 11.60 | 3.52 | [6, 19] | Multi-Table Model |

---

## 4. Local Applicant Reason Code Stability Distribution

Distribution of Jaccard@3 reason-code agreement across the 5,000 panel applicants:

- **Mean:** 0.4993
- **Standard Deviation:** 0.1467
- **5th Percentile:** 0.2810
- **25th Percentile (Q1):** 0.3962
- **Median (Q2):** 0.4871
- **75th Percentile (Q3):** 0.5886
- **95th Percentile:** 0.7638

---

## 5. Methodological & Computational Notes

1. **Full Protocol Execution:** All 30 planned models (15 standard + 15 SMOTE) were trained and evaluated on the full 5,000-applicant panel without requiring runtime fallbacks.
2. **Zero Test Data Access:** Models were fitted strictly on bootstrap resamples of the training set. The panel was selected strictly from the validation set.
3. **Artifact Persistence:** Summary results are persisted to `reports/shap_stability.csv`.
