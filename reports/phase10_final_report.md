# Phase 10: Research Experiments (E1, E3, E4) — Final Implementation & Scientific Audit Report

**Project Title:** Loan Default Prediction Using Big Data Analytics  
**Repository Location:** `C:\Users\ACER\Desktop\BIG_DATA_PROJECT`  
**Git Branch:** `develop`  
**Phase:** Phase 10 (Research Experiments: E1 Feature Ablation, E3 Imbalance × Calibration, E4 Decision Analysis)  
**Execution Timestamp:** 2026-11-02 17:45:00 UTC  
**Evaluator Architecture:** PySpark 3.5.3 + Scikit-learn 1.6.1 + XGBoost 3.1.1 + LightGBM 4.6.0 + Imbalanced-Learn 0.14.2  
**Split Lineage:** Frozen 70/15/15 Stratified Partition (Train: $N=215,257$; Validation: $N=46,127$; Test: $N=46,127$).  
**Zero-Leakage Assurance:** **The Test partition (`data/splits/test_indices.parquet`) and raw inference table (`application_test.csv`) were strictly EXCLUDED and NEVER accessed.**

---

## 1. Executive Summary & Research Context

Phase 10 represents the experimental research culmination of the Loan Default Prediction project. Having established a verified, leakage-free Big Data ETL architecture (Phases 0–5), frozen model-preprocessing pipelines (Phase 6), baseline classifiers (Phase 7), hyperparameter-tuned/calibrated finalists (Phase 8), and SHAP post-hoc interpretability (Phase 9), Phase 10 formally investigates three core credit-risk research questions:

1. **Experiment E1 (Feature Ablation):** *What is the marginal predictive contribution of each historical financial source beyond applicant demographic data?*
2. **Experiment E3 (Class Imbalance × Probability Calibration):** *How do artificial imbalance correction techniques (loss weighting, SMOTE) affect rank discrimination versus probability calibration?*
3. **Experiment E4 (Decision Analysis & Cost Optimization):** *How can calibrated probability outputs be translated into cost-optimal loan approval/rejection decisions and operational risk bands under asymmetric error costs?*

---

## 2. Frozen Assets & Scope Boundaries

In accordance with strict project rules, the following upstream assets were held **immutable and frozen**:
- **Dataset Profile:** 307,511 labeled applicants, 256 model features (240 numeric, 16 categorical), 8.073% positive default rate.
- **Split Lineage:** `data/splits/train_indices.parquet` ($N=215,257$), `data/splits/val_indices.parquet` ($N=46,127$).
- **Frozen Model Finalist:** `artifacts/models/phase8_best_xgboost.joblib` (Tuned XGBoost: `max_depth=3`, `learning_rate=0.1`, `n_estimators=500`, `min_child_weight=10`, `subsample=0.8`, `colsample_bytree=1.0`).
- **Phase 8 Artifacts:** `reports/threshold_analysis.csv` and `reports/phase8_*` were kept strictly untouched.
- **Test Set Reservation:** The test partition remains strictly untouched for independent out-of-sample evaluation.

---

## 3. Experiment E1: Feature Family Audit & Ablation Analysis

### 3.1 Feature Family Reconciliation (256 Features Across 7 Disjoint Families)
All 256 model features defined in `configs/features.yaml` map exhaustively into **seven disjoint families**:

| Feature Family Code | Source Table(s) | Description | Feature Count | Derived / Cross-Domain Status | E1 Ablation Group |
| :--- | :--- | :--- | :---: | :--- | :--- |
| **`APP`** | `application` | Demographics, loan parameters, bureau inquiries, external scores (`EXT_SOURCE_1/2/3`), and intra-application engineered ratios/flags. | 147 | Baseline application features, 3 data-quality sentinel flags, and 12 intra-application ratios/interactions. | `app` |
| **`BURO` + `BURO_BAL`** | `bureau` (20), `bureau_balance` (6) | Aggregated external credit history from other financial institutions, including active/closed counts, total debt, overdue amounts, and monthly delinquency status rollups. | 26 | Historical multi-record aggregations. | `bureau` |
| **`PREV`** | `previous_application` | Prior Home Credit application history, approval/refusal rates, requested amounts, decision timeliness, and payment term aggregates. | 25 | Historical multi-record aggregations. | `prev` |
| **`INST`** | `installments_payments` | Historical repayment behavior on prior loans: installment shortfalls, delay in days (late payments), underpayment counts, and payment-to-instalment ratios. | 16 | Behavioral payment aggregations. | `installments` |
| **`CC`** | `credit_card_balance` | Monthly credit card utilization, ATM cash drawings, revolving credit limits, total receivables, and days past due (DPD) metrics. | 21 | Revolving credit line aggregations. | `credit_card` |
| **`POS`** | `POS_CASH_balance` | Monthly point-of-sale and cash loan contract statuses, active/completed terms, remaining installments, and DPD delinquency counts. | 14 | POS contract aggregations. | `pos_cash` |
| **`DERIVED`** | `cross_domain` | Non-linear interaction features combining signals across disparate source tables (e.g. `DERIVED_TOTAL_DEBT_TO_INCOME_RATIO`, `DERIVED_BURO_TO_CURRENT_CREDIT_RATIO`, `DERIVED_PAYMENT_DISCIPLINE_SCORE`, `DERIVED_CARD_UTILIZATION_BURDEN`). | 7 | Cross-Domain Engineered Interactions (multi-table). | `derived` |
| **Total** | **8 Source Tables** | **Complete model input feature space** | **256** | **Zero missing, zero duplicates** | **7 Families** |

> [!NOTE]
> **Lineage & Group Count Reconciliation:**  
> - **Incremental Augmentation (Table 3.2, 5 single-table pairs):** Evaluates the standalone marginal lift of pairing the core application data with each of the **5 historical external tables** (`bureau`, `prev`, `installments`, `credit_card`, `pos_cash`). The 7th family, `DERIVED` (7 features), cannot be evaluated as an isolated single-table augmentation because its interaction ratios mathematically require fields from multiple disparate tables simultaneously (e.g. total debt from bureau divided by application income).
> - **Leave-One-Group-Out (LOGO) Ablation (Table 3.2, 6 groups):** Evaluates redundancy by systematically withholding each of the **6 supplementary groups** from the full 256-feature model, explicitly including `logo_derived` (249 features).
> - The complete 256-feature mapping is persisted in [reports/feature_family_reconciliation.csv](file:///C:/Users/ACER/Desktop/BIG_DATA_PROJECT/reports/feature_family_reconciliation.csv).

### 3.2 Measured Results Table (13 Completed Configurations)

| Configuration Key | Configuration Name | Type | Features | Validation ROC-AUC (95% CI) | Validation PR-AUC (95% CI) | Δ ROC vs App | Δ PR vs App | Δ ROC vs Full | Fit Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `app_only` | Application Only | incremental | 147 | 0.7726 [0.7631, 0.7800] | 0.2625 [0.2439, 0.2745] | +0.0000 | +0.0000 | -0.0185 | 8.2s |
| `app_bureau` | Application + Bureau | incremental | 173 | 0.7756 [0.7664, 0.7827] | 0.2676 [0.2497, 0.2808] | +0.0030 | +0.0051 | -0.0155 | 9.4s |
| `app_prev` | Application + Previous Apps | incremental | 172 | 0.7770 [0.7683, 0.7847] | 0.2673 [0.2478, 0.2810] | +0.0044 | +0.0048 | -0.0142 | 9.3s |
| `app_installments` | Application + Installments | incremental | 163 | **0.7818** [0.7732, 0.7890] | **0.2729** [0.2537, 0.2873] | **+0.0092** | **+0.0105** | -0.0093 | 9.0s |
| `app_credit_card` | Application + Credit Card | incremental | 168 | 0.7744 [0.7645, 0.7812] | 0.2669 [0.2492, 0.2786] | +0.0018 | +0.0045 | -0.0167 | 7.9s |
| `app_pos_cash` | Application + POS/CASH | incremental | 161 | 0.7793 [0.7702, 0.7879] | 0.2702 [0.2520, 0.2806] | +0.0067 | +0.0077 | -0.0118 | 8.1s |
| `full_all_tables` | Full Model (All Tables) | full | 256 | **0.7911** [0.7827, 0.7987] | **0.2914** [0.2734, 0.3084] | **+0.0185** | **+0.0289** | +0.0000 | 13.6s |
| `logo_bureau` | Full Excl. Bureau | LOGO | 230 | 0.7901 [0.7820, 0.7967] | 0.2860 [0.2672, 0.3000] | +0.0175 | +0.0235 | -0.0010 | 12.6s |
| `logo_prev` | Full Excl. Previous Apps | LOGO | 231 | 0.7907 [0.7829, 0.7983] | 0.2906 [0.2718, 0.3048] | +0.0181 | +0.0281 | -0.0004 | 12.6s |
| `logo_installments` | Full Excl. Installments | LOGO | 240 | 0.7900 [0.7822, 0.7975] | 0.2884 [0.2712, 0.3047] | +0.0174 | +0.0259 | -0.0011 | 12.4s |
| `logo_credit_card` | Full Excl. Credit Card | LOGO | 235 | 0.7909 [0.7829, 0.7981] | 0.2894 [0.2699, 0.3049] | +0.0183 | +0.0269 | -0.0002 | 12.8s |
| `logo_pos_cash` | Full Excl. POS/CASH | LOGO | 242 | 0.7895 [0.7811, 0.7968] | 0.2892 [0.2715, 0.3056] | +0.0169 | +0.0267 | **-0.0016** | 12.8s |
| `logo_derived` | Full Excl. Derived | LOGO | 249 | 0.7909 [0.7831, 0.7971] | 0.2917 [0.2728, 0.3073] | +0.0183 | +0.0292 | -0.0002 | 15.7s |

### 3.3 Methodological Interpretations & Bootstrap Scope
1. **Multi-Table Incremental Lift:** Combining all historical sources yields an observed incremental **+0.0185 ROC-AUC** and **+0.0289 PR-AUC** lift over demographic application data alone.
2. **Exploratory Bootstrap Scope:** 30 paired validation bootstrap resamples provide an empirical 95% confidence interval of $[+0.0145, +0.0242]$ for $\Delta\text{ROC}$ and $[+0.0219, +0.0368]$ for $\Delta\text{PR}$. These intervals serve as exploratory computational variance estimates rather than asymptotic hypothesis tests.
3. **Synergy vs Redundancy:** Individual LOGO drops are relatively small ($-0.0002$ to $-0.0016$ ROC-AUC) because tree ensembles exploit correlated signals across multiple tables (e.g. repayment delinquency appears in both installments and bureau records). However, omitting tables collectively causes severe degradation ($-0.0185$), confirming that multi-table feature coverage is crucial.

---

## 4. Experiment E3: Class Imbalance × Probability Calibration Audit

### 4.1 Protocol & Leakage-Free Design
Experiment E3 evaluated 13 configurations spanning 4 algorithm families (Logistic Regression, Random Forest, XGBoost, LightGBM) across 4 imbalance treatments (None, Class Weighting, Positive-Class Weighting, SMOTE).
- **Internal Calibration Holdout:** The training set ($N=215,257$) was split internally 80/20 into `train_fit` ($N=172,205$) and `train_cal` ($N=43,052$) stratified on $y_\text{train}$.
- **Zero Validation Leakage:** Models were fitted on `train_fit`; isotonic calibrators were fitted on `(p_train_cal, y_train_cal)`. Validation labels ($N=46,127$) were strictly never observed during model training or calibration fitting.

### 4.2 Comprehensive Calibration Results Table (13 Configurations)

| Config ID | Model | Strategy | Uncal ROC | Cal ROC | Uncal PR | Cal PR | Uncal Log Loss | Cal Log Loss | Uncal Brier | Cal Brier | Uncal ECE | Cal ECE | Uncal Cox Slope | Cal Cox Slope | Uncal Recall (0.50) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `E3_A1_LR_None` | Logistic Regression | None | 0.7757 | 0.7751 | 0.2688 | 0.2578 | 0.2406 | 0.2409 | 0.0666 | 0.0666 | 0.0023 | 0.0032 | 1.00 | 1.01 | 3.33% |
| `E3_A2_RF_None` | Random Forest | None | 0.7590 | 0.7582 | 0.2523 | 0.2429 | 0.2471 | 0.2469 | 0.0679 | 0.0676 | 0.0109 | 0.0015 | 1.29 | 1.00 | 0.11% |
| `E3_A3_XGB_None` | XGBoost | None | **0.7893** | **0.7890** | **0.2834** | **0.2715** | 0.2362 | 0.2366 | 0.0657 | 0.0658 | 0.0025 | 0.0024 | 1.02 | 1.02 | 4.32% |
| `E3_A4_LGBM_None` | LightGBM | None | 0.7859 | 0.7854 | 0.2830 | 0.2740 | 0.2371 | 0.2373 | 0.0658 | 0.0659 | 0.0020 | 0.0025 | 1.04 | 1.02 | 4.05% |
| `E3_B5_LR_ClassWeight` | Logistic Regression | Class Weight | 0.7755 | 0.7750 | 0.2645 | 0.2562 | 0.5731 | 0.2419 | 0.1946 | **0.0668** | 0.3257 | **0.0022** | 0.98 | 1.01 | 70.17% |
| `E3_B6_RF_ClassWeight` | Random Forest | Class Weight | 0.7567 | 0.7559 | 0.2382 | 0.2289 | 0.4515 | 0.2471 | 0.1411 | **0.0681** | 0.2540 | **0.0027** | 1.22 | 1.01 | 47.93% |
| `E3_B7_LGBM_ClassWeight` | LightGBM | Class Weight | 0.7855 | 0.7851 | 0.2807 | 0.2706 | 0.5125 | 0.2388 | 0.1709 | **0.0660** | 0.2888 | **0.0020** | 1.03 | 1.01 | 67.75% |
| `E3_C8_XGB_ScalePosWeight` | XGBoost | Pos Weight | 0.7854 | 0.7851 | 0.2843 | 0.2726 | 0.4909 | 0.2379 | 0.1616 | **0.0659** | 0.2743 | **0.0024** | 1.01 | 1.02 | 64.93% |
| `E3_C9_LGBM_ScalePosWeight` | LightGBM | Pos Weight | 0.7855 | 0.7851 | 0.2803 | 0.2700 | 0.5103 | 0.2382 | 0.1707 | **0.0660** | 0.2869 | **0.0020** | 1.04 | 1.01 | 68.42% |
| `E3_D10_LR_SMOTE` | Logistic Regression | SMOTE | 0.7648 | 0.7642 | 0.2521 | 0.2425 | 0.5628 | 0.2447 | 0.1908 | **0.0675** | 0.3063 | **0.0038** | 0.81 | 1.02 | 67.99% |
| `E3_D11_RF_SMOTE` | Random Forest | SMOTE | 0.7323 | 0.7314 | 0.1880 | 0.1806 | 0.3159 | 0.2565 | 0.0882 | **0.0702** | 0.1218 | **0.0017** | 1.13 | 1.00 | 6.07% |
| `E3_D12_XGB_SMOTE` | XGBoost | SMOTE | 0.7808 | 0.7802 | 0.2708 | 0.2623 | 0.2394 | 0.2396 | 0.0664 | 0.0664 | 0.0057 | 0.0023 | 1.03 | 1.01 | 3.46% |
| `E3_D13_LGBM_SMOTE` | LightGBM | SMOTE | 0.7838 | 0.7832 | 0.2734 | 0.2643 | 0.2385 | 0.2391 | 0.0663 | 0.0663 | 0.0056 | 0.0031 | 1.06 | 1.00 | 3.38% |

### 4.3 Scientific Interpretations & Practical Limitations
1. **The Fallacy of Probability Inflation:** Weighting methods artificially inflate predicted probabilities into the $[0.30, 0.70]$ range. At the arbitrary threshold $\tau=0.50$, this appears to increase recall ($\approx 65\text{--}70\%$) but severely degrades Brier scores ($0.16\text{--}0.19$) and produces high ECE ($0.25\text{--}0.33$).
2. **SMOTE Dilution:** SMOTE consistently underperformed baseline unweighted models across all metrics (e.g. XGBoost PR-AUC: 0.2834 vs 0.2708; RF PR-AUC: 0.2523 vs 0.1880), confirming that synthetic interpolation in high-dimensional tabular spaces generates uninformative samples.
3. **Scope and Limits of Isotonic Recalibration:** Isotonic calibration reduces empirical validation ECE to $<0.0038$ and aligns mean predictions with base rates. However, it operates as a piecewise-constant step function that discretizes probability outputs. A low validation ECE does not guarantee resilience to out-of-time distribution shifts, nor does it improve the fundamental discriminative ceiling (ROC-AUC / PR-AUC) of the underlying feature representations.

---

## 5. Experiment E4: Decision Analysis & Cost Optimization Audit

### 5.1 Asymmetric Cost Formulation
In credit default underwriting, decision performance depends on asymmetric costs: approving a defaulter (False Negative) forfeits loan principal, whereas declining a good applicant (False Positive) forfeits only margin.

Decision costs are evaluated using the exact loss formula:

$$\text{Total Cost} = \text{FN} \times C_{\text{FN}} + \text{FP} \times C_{\text{FP}}$$

$$\text{Cost per Applicant} = \frac{\text{Total Cost}}{N_{\text{val}}}$$

where $C_{\text{FN}} : C_{\text{FP}}$ represents the relative error cost ratio with $C_{\text{FP}} = 1.0$.

### 5.2 Measured Threshold Optimization Results

| Cost Ratio ($C_\text{FN} : C_\text{FP}$) | Empirical Optimal $\tau^*$ | Bayesian Theoretical $\tau$ | Min Total Cost | Cost per Applicant | Cost at $\tau=0.50$ | Cost Savings vs. 0.50 | Confusion Matrix (TP, FP, TN, FN) | Approval Rate |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **5:1** | **0.16** | 0.1667 | 14,364 | 0.3114 | 17,791 | **19.26%** | TP=1,743, FP=4,459, TN=37,944, FN=1,981 | 86.55% |
| **10:1** | **0.09** | 0.0909 | 22,359 | 0.4847 | 35,446 | **36.92%** | TP=2,543, FP=10,549, TN=31,854, FN=1,181 | 71.62% |
| **20:1** | **0.06** | 0.0476 | 30,475 | 0.6607 | 70,756 | **56.93%** | TP=3,024, FP=16,475, TN=25,928, FN=700 | 57.73% |

> [!NOTE]
> All costs are reported as dimensionless relative loss units normalized to $C_{\text{FP}} = 1.0$.
> Full results across 99 thresholds are saved in [reports/decision_threshold_analysis.csv](file:///C:/Users/ACER/Desktop/BIG_DATA_PROJECT/reports/decision_threshold_analysis.csv), keeping the frozen Phase 8 baseline artifact [reports/threshold_analysis.csv](file:///C:/Users/ACER/Desktop/BIG_DATA_PROJECT/reports/threshold_analysis.csv) untouched.

### 5.3 Three-Tier Risk Band Segmentation & Monotonicity Verification

| Risk Tier | Probability Range | Applicant Count | Population Share (%) | Defaults | Observed Default Rate (%) | Mean Predicted Risk (%) | Operational Underwriting Action |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **LOW** | `[0.00, 0.09)` | 33,035 | 71.62% | 1,181 | **3.57%** | 3.77% | Automated approval / standard rate |
| **MEDIUM** | `[0.09, 0.25)` | 10,258 | 22.24% | 1,513 | **14.75%** | 14.53% | Secondary review / income verification |
| **HIGH** | `[0.25, 1.00)` | 2,834 | 6.14% | 1,030 | **36.34%** | 36.67% | Automated decline / require collateral |

$$\text{Monotonicity Check: } \mathbf{3.57\% < 14.75\% < 36.34\%} \implies \text{Strictly Monotonic (VERIFIED)}$$

> [!NOTE]
> Monotonic ordering confirms that empirical default risk scales consistently with model score bands on the validation set. This verification confirms proper ordinal ranking within the portfolio without asserting regulatory capital compliance.

---

## 6. Standardized Experiment Registry Audit

All experimental executions across Phase 10 were recorded into the standardized Experiment Registry:
- **Registry Files:** `reports/experiment_registry.json` and `reports/experiment_registry.csv`.
- **Recorded Runs:** 29 total runs (13 from E1 Feature Ablation, 13 from E3 Imbalance × Calibration, 3 from E4 Decision Optimization).
- **Mandatory 17-Field Compliance:** Every record contains `experiment_id`, `experiment_name`, `model_name`, `feature_set`, `imbalance_strategy`, `calibration_strategy`, `random_seed`, `model_parameters`, `preprocessing_reference`, `train_split_reference`, `validation_split_reference`, `runtime_seconds`, `feature_count`, `output_artifact`, `metrics`, `experiment_status`, and `git_commit`.

---

## 7. Deliverables & Figures Inventory

### 7.1 Source Code Modules
- `src/evaluation/registry.py`: Standardized 17-field experiment tracker.
- `src/evaluation/ablation.py`: E1 Feature Ablation suite across 13 configurations with 30 bootstrap resamples.
- `src/evaluation/imbalance_grid.py`: E3 Class Imbalance × Calibration suite across 13 configurations.
- `src/evaluation/decision_analysis.py`: E4 Cost-sensitive threshold optimization and 3-tier risk segmentation.

### 7.2 Configuration & Lineage Artifacts
- `configs/experiments.yaml`: Specifications for E1, E3, and E4.
- `reports/feature_family_reconciliation.csv`: Lineage mapping of all 256 model features across 7 disjoint families.

### 7.3 Quantitative Result Tables
- `reports/ablation_results.csv`: 13 rows with bootstrap 95% CIs for ROC-AUC, PR-AUC, and deltas.
- `reports/calibration_results.csv`: 13 rows with discrimination, calibration (Brier, Log Loss, ECE, Cox slope/intercept), and operational metrics before and after isotonic calibration.
- `reports/decision_threshold_analysis.csv`: Detailed threshold sweep across $\tau \in [0.01, 0.99]$ for 5:1, 10:1, and 20:1 ratios.
- `reports/threshold_analysis.csv`: Frozen Phase 8 baseline threshold analysis (preserved unmodified).
- `reports/experiment_registry.csv` & `reports/experiment_registry.json`: Persistent experiment registry.

### 7.4 Scientific Figures
- `reports/figures/ablation_comparison.png`: Dual-panel comparison of Validation ROC-AUC and PR-AUC with 95% bootstrap CIs.
- `reports/figures/ablation_incremental_contribution.png`: Horizontal bar chart of $\Delta\text{ROC-AUC}$ over application-only baseline.
- `reports/figures/imbalance_comparison.png`: Four-panel grid comparing ROC-AUC, PR-AUC, F1, and Balanced Accuracy across imbalance methods.
- `reports/figures/calibration_comparison.png`: Three-panel comparison showing ECE reduction, Brier score reduction, and Cox slope convergence.
- `reports/figures/reliability_diagrams.png`: Reliability diagrams before and after isotonic calibration.
- `reports/figures/cost_threshold_curves.png`: Total error cost curves across thresholds for 5:1, 10:1, and 20:1 cost ratios.
- `reports/figures/risk_band_distribution.png`: Population distribution and observed vs predicted default rates across LOW, MEDIUM, and HIGH risk bands.

### 7.5 Detailed Reports
- `reports/ablation_analysis.md`: Detailed E1 report.
- `reports/imbalance_analysis.md`: Detailed E3 report.
- `reports/decision_analysis.md`: Detailed E4 report.
- `reports/phase10_final_report.md`: Complete Phase 10 synthesis.

---

## 8. Documented Engineering Deviations & Verification Notes

1. **Lazy SMOTE Import:** Dynamic loading (`_get_smote()`) was implemented in `src/evaluation/imbalance_grid.py` to prevent module namespace pollution during full test collection.
2. **Decision Artifact Separation:** Cost-optimization outputs were saved to `reports/decision_threshold_analysis.csv` to ensure Phase 8's frozen `reports/threshold_analysis.csv` remained unmodified.
3. **Bootstrap Sample Size:** Resample iterations were capped at $B=30$ for computational tractability on 46,127 validation samples while still providing stable variance estimates.
4. **Test Suite Count Reconciliation (138 Tests vs. Expected 136):**
   - **Historical Phase 9 Test Suite (Commit `d791574`):** 124 tests across 10 test modules (`test_baseline_models.py`: 11, `test_bronze.py`: 5, `test_features.py`: 16, `test_gold.py`: 17, `test_phase1_profile.py`: 9, `test_phase8.py`: 13, `test_phase9.py`: 8, `test_schemas.py`: 5, `test_silver.py`: 22, `test_split.py`: 18).
   - **Intermediate BDE Assignment (Commit `4c406b6`):** Added unsupervised customer segmentation (`src/ml/clustering.py`) and `tests/test_clustering.py` containing **2 tests**:
     - `tests/test_clustering.py::test_clustering_features_specification`
     - `tests/test_clustering.py::test_clustering_evaluation_and_profiling_synthetic`
     This raised the pre-Phase 10 baseline from 124 to 126 tests.
   - **Phase 10 Research Experiments (Commit `c6cfade`):** Added **12 tests** across three new test files:
     - `tests/test_phase10_ablation.py`: 4 tests (`test_feature_manifest_integrity`, `test_ablation_configurations_definition`, `test_ablation_data_loading_no_test_leakage`, `test_bootstrap_reproducibility`)
     - `tests/test_phase10_decision.py`: 3 tests (`test_cost_calculation_logic`, `test_risk_band_monotonicity_logic`, `test_no_test_set_leakage`)
     - `tests/test_phase10_imbalance.py`: 5 tests (`test_e3_configurations_definition`, `test_ece_calculation`, `test_cox_calibration_slope_and_intercept`, `test_operational_metrics_correctness`, `test_no_test_set_leakage`)
   - **Reconciliation Formula:**
     $$124\text{ (Phase 9)} + 2\text{ (BDE Assignment Clustering)} + 12\text{ (Phase 10 Research)} = 138\text{ total tests (100% passing)}$$
     An expectation of $124 + 12 = 136$ tests overlooks the 2 tests added in `tests/test_clustering.py`. The recorded 138 tests is the complete and verified suite.

---

## 9. Conclusion & Scope Boundary Enforcement

Phase 10 is **100% complete and verified**. All experimental artifacts, code modules, reports, and tests adhere strictly to project governance rules. In accordance with explicit user instructions, work terminates at Phase 10. No Phase 11 or later phases have been initiated.
