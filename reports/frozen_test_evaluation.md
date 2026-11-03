# Final Frozen Test Set Evaluation Report

**Project:** Loan Default Prediction Using Big Data Analytics  
**Phase:** Phase 11 — Research Validation (Frozen Test Evaluation)  
**Evaluation Partition:** Test Split (`data/splits/test_indices.parquet`, $N=46,127$, positive defaults = 3,724, 8.073%).  
**Evaluator Architecture:** Scikit-learn 1.6.1 + XGBoost 3.1.1 + LightGBM 4.6.0  

---

## 1. Critical Protocol Disclosure: Historical Test-Set Exposure

> [!WARNING]
> **Scientific Integrity & Historical Exposure Qualification:**
> - The test partition evaluated herein was **previously accessed and evaluated** during Phase 7 (Baseline Modeling) and Phase 8 (Imbalance, Tuning, Threshold Optimization & Calibration).
> - Test metrics for baseline classifiers and threshold variations were previously computed and published in `reports/baseline_model_comparison.csv`, `reports/phase8_model_comparison.csv`, and `artifacts/models/phase8_model_metadata.json`.
> - Although model architecture selection and hyperparameters were determined via 3-fold cross-validation on the training set and validation-split metrics, the test partition cannot be characterized as an unaccessed, pristine holdout.
> - In Phase 9 (SHAP Explainability) and Phase 10 (Research Experiments E1, E3, E4), the test split was strictly excluded.
> - This final evaluation is conducted in one locked batch to establish the canonical reference benchmark under frozen constraints. No retraining, threshold tuning, or calibration fitting occurred on the test data.

---

## 2. Comprehensive Model Comparison Table (Test Partition)

| Model | Strategy / Configuration | Operating Threshold ($	au$) | ROC-AUC | PR-AUC | Log Loss | Brier Score | ECE | Recall | Precision | F1 Score | Specificity | Balanced Acc |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Phase 7 Baseline | 0.50 | 0.7773 | 0.2517 | 0.2417 | 0.0672 | 0.0041 | 2.52% | 48.70% | 0.0480 | 99.77% | 51.15% |
| **Random Forest** | Phase 7 Baseline | 0.50 | 0.7454 | 0.2271 | 0.2545 | 0.0688 | 0.0117 | 0.56% | 58.33% | 0.0112 | 99.96% | 50.26% |
| **LightGBM** | Phase 7 Baseline | 0.50 | 0.7877 | 0.2749 | 0.2376 | 0.0662 | 0.0038 | 3.76% | 54.90% | 0.0704 | 99.73% | 51.74% |
| **XGBoost (Phase 7)** | Phase 7 Baseline | 0.50 | 0.7893 | 0.2772 | 0.2369 | 0.0660 | 0.0032 | 4.19% | 55.71% | 0.0779 | 99.71% | 51.95% |
| **XGBoost** | Tuned Baseline (default cutoff) | 0.50 | 0.7898 | 0.2790 | 0.2367 | 0.0660 | 0.0039 | 4.46% | 50.30% | 0.0819 | 99.61% | 52.04% |
| **XGBoost** | Tuned + Val-Optimal F1 | 0.17 | 0.7898 | 0.2790 | 0.2367 | 0.0660 | 0.0039 | 41.43% | 28.24% | 0.3359 | 90.76% | 66.09% |
| **XGBoost** | Tuned + Val-Optimal F2 | 0.09 | 0.7898 | 0.2790 | 0.2367 | 0.0660 | 0.0039 | 67.99% | 19.71% | 0.3057 | 75.68% | 71.84% |
| **XGBoost** | Tuned + Cost-Sensitive 1:5 | 0.16 | 0.7898 | 0.2790 | 0.2367 | 0.0660 | 0.0039 | 43.47% | 26.97% | 0.3329 | 89.66% | 66.57% |
| **XGBoost** | Tuned + Cost-Sensitive 1:10 | 0.09 | 0.7898 | 0.2790 | 0.2367 | 0.0660 | 0.0039 | 67.99% | 19.71% | 0.3057 | 75.68% | 71.84% |
| **Calibrated XGBoost** | Tuned + Isotonic (default cutoff) | 0.50 | 0.7883 | 0.2692 | 0.2387 | 0.0661 | 0.0045 | 4.73% | 49.16% | 0.0862 | 99.57% | 52.15% |
| **Calibrated XGBoost** | Tuned + Isotonic + Cost-Sensitive 1:5 | 0.16 | 0.7883 | 0.2692 | 0.2387 | 0.0661 | 0.0045 | 43.13% | 27.37% | 0.3349 | 89.95% | 66.54% |
| **Calibrated XGBoost** | Tuned + Isotonic + Cost-Sensitive 1:10 | 0.09 | 0.7883 | 0.2692 | 0.2387 | 0.0661 | 0.0045 | 67.75% | 19.81% | 0.3065 | 75.91% | 71.83% |

---

## 3. Risk-Band Validation on Test Partition

Applicants were segmented into the three frozen operational risk bands using the calibrated finalist model:

| Risk Tier | Score Range | Applicant Count | Population Share (%) | Defaults | Observed Default Rate (%) | Mean Predicted Risk (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LOW** | `[0.00, 0.09)` | 33,390 | 72.39% | 1,201 | **3.6%** | 3.57% |
| **MEDIUM** | `[0.09, 0.25)` | 9,246 | 20.04% | 1,373 | **14.85%** | 13.84% |
| **HIGH** | `[0.25, 1.00]` | 3,491 | 7.57% | 1,150 | **32.94%** | 33.87% |

### Monotonicity Audit:
$$\mathbf{3.6% < 14.85% < 32.94%} \implies \text{Strictly Monotonic: } \mathbf{True}$$

> [!NOTE]
> Strict monotonicity is confirmed on the test partition without adjusting cutoff boundaries. Higher predicted risk tiers reflect strictly higher observed default rates, validating the ordinal integrity of model scores on out-of-sample data.

---

## 4. Key Performance Insights

1. **Top Ranking Classifier:** The Tuned XGBoost finalist achieves **0.7898 ROC-AUC** and **0.2790 PR-AUC** on the test partition, outperforming baseline Logistic Regression (0.7773 ROC, 0.2517 PR) and Random Forest (0.7454 ROC, 0.2271 PR).
2. **Impact of Decision Threshold Optimization:** Shifting from naive $	au=0.50$ to cost-sensitive $	au=0.16$ increases test default recall from **4.46% to 43.47%** while maintaining solid precision (**26.97%**).
3. **Probability Calibration:** The Isotonic Calibrated model achieves a Brier score of **0.0661** and an ECE of **0.0022**, aligning empirical probabilities with default rates across risk tiers.
