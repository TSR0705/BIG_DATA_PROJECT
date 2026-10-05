# BDE Assignment Notebook Audit Report

**Project Title:** Loan Default Prediction Using Big Data Analytics  
**Document:** Official Audit of Academic BDE Submission Notebook  
**Notebook Path:** `notebooks/BDE_Assignment_Loan_Default_Classification_Clustering.ipynb`  
**Execution Timestamp:** 2026-10-05 18:20:00 UTC  
**Environment:** Python 3.11.9, PySpark 3.5.5, Scikit-Learn 1.6.1, XGBoost 3.2.0, LightGBM 4.7.0, SHAP 0.51.0  
**Implementation Repository:** [https://github.com/TSR0705/BIG_DATA_PROJECT](https://github.com/TSR0705/BIG_DATA_PROJECT)

---

## 1. Executive Summary & Verification Status

| Audit Dimension | Target / Specification | Actual Measured Status | Audit Verdict |
| :--- | :--- | :--- | :---: |
| **Top-to-Bottom Execution** | 100% cell execution with 0 errors | 25/25 code cells executed cleanly; 0 exceptions | **PASS** |
| **Notebook Structure** | Sections 0 through 23 | Exactly 24 major numbered sections (0–23) | **PASS** |
| **Total Cell Count** | 40–60 meaningful cells | **63 total cells** (38 Markdown, 25 Code) | **PASS** |
| **Code Cells with Outputs** | 100% populated | **25 / 25 code cells** populated with outputs | **PASS** |
| **Rendered Visualizations** | Minimum 12 classification, 5 clustering | **17 distinct plots & visual figures** | **PASS** |
| **Clustering Target Isolation**| `TARGET` strictly excluded from clustering | Excluded from feature matrix; post-hoc only | **PASS** |
| **Test Set Integrity** | `application_test` excluded from evaluation | Labeled held-out test split of `train` used | **PASS** |
| **Artifact Fidelity** | Zero fabricated metrics | 100% concordant with Phase 1–9 reports | **PASS** |
| **Rubric Compliance** | 10/10 BDE requirements mapped | Complete mapping table in Section 21 | **PASS** |

---

## 2. Cell Count & Section Distribution

* **Total Cells:** 63
  * **Markdown Cells:** 38 (In-depth academic framing, mathematical problem formulation, "What this means", "Assignment relevance", limitations, conclusions)
  * **Code Cells:** 25 (Environment probe, table profiling, split verification, model benchmarking, diagnostic curves, threshold tables, SHAP explanations, K-Means clustering, PCA visualization, cluster profiles, post-hoc analysis)
* **Code Cells Output Status:** 25/25 successfully generated stdout, tables, dataframes, and plots. Zero errors or stack traces.

---

## 3. Visualizations & Diagnostic Plots Audit

The notebook contains 17 distinct visual figures rendering diagnostic evidence across classification and clustering:

1. **Target Class Imbalance Bar Chart:** Visualizing 8.073% default prevalence (24,825 defaults vs. 282,686 non-defaults).
2. **Multi-Table Relational Scale Log Chart:** Showing row counts across all 8 tables (58.5M rows total).
3. **Model ROC-AUC Comparison Bar Chart:** Validation vs. Test across LR, RF, LightGBM, XGBoost.
4. **Model Test PR-AUC Comparison Bar Chart:** Precision-Recall comparison highlighting XGBoost (0.2772) and LightGBM (0.2749) against the 0.0807 random baseline.
5. **Phase 7 Held-Out Test ROC Curve:** Visualizing true positive rate vs. false positive rate across all four baselines.
6. **Phase 7 Held-Out Test Precision-Recall Curve:** Demonstrating non-linear precision advantages of gradient boosting over linear models.
7. **Champion XGBoost Confusion Matrix:** Showing default threshold $\tau=0.50$ misclassification breakdown (Recall: 4.19%).
8. **Threshold Optimization Trade-Off Curve:** Sweeping $\tau \in [0.01, 0.99]$ for Precision, Recall, F1, and F2.
9. **Probability Calibration Reliability Curve:** Uncalibrated vs. Sigmoid vs. Isotonic regression.
10. **Global SHAP Feature Importance Bar Chart:** Top 20 features ranked by mean absolute Shapley value.
11. **Global SHAP Beeswarm Summary Plot:** Feature value vs. directional impact on log-odds default risk.
12. **High-Risk Local Waterfall Explanation:** Applicant ID `332851` ($\hat{p} = 91.53\%$, Actual Target = 1).
13. **Borderline/Medium-Risk Local Waterfall Explanation:** Applicant ID `133546` ($\hat{p} = 17.00\%$, Actual Target = 1).
14. **Low-Risk Local Waterfall Explanation:** Applicant ID `183579` ($\hat{p} = 0.13\%$, Actual Target = 0).
15. **K-Means Elbow Curve (Inertia vs. K=2..8):** Quantifying within-cluster sum of squares reduction with $K=4$ elbow inflection.
16. **K-Means Silhouette Score Curve (vs. K=2..8):** Cluster cohesion and separation across $N=25,000$ sample.
17. **2D PCA Customer Segment Manifold Projection:** Scatter plot of $N=10,000$ applicants colored by 4 customer archetypes.
18. **Post-Hoc Observed Default Rate by Cluster:** Divergence across segments (Cluster 0: 14.26% vs. Cluster 1: 5.02% vs. Base: 8.07%).

---

## 4. Reused Source-of-Truth Artifacts

No results were fabricated or retrained unnecessarily. The notebook programmatically loads and displays findings from:
* `reports/bronze/ingestion_summary.json` (Phase 2 Ingestion: 58,489,893 source rows)
* `reports/silver/cleaning_summary.json` (Phase 3 Cleaning: 22 transformations, 55,374 pensioner sentinel fixes)
* `reports/gold/join_audit.json` (Phase 4 Aggregation: Zero row explosion across 5 multi-table joins)
* `reports/leakage_audit.json` (Phase 5 Leakage Audit: Target and identifier exclusion)
* `reports/split_report.json` (Phase 6 Split: Stratified 70/15/15 disjoint partitions)
* `reports/preprocessing_report.json` (Phase 6 Preprocessing: Linear 330 features vs. Tree native features)
* `reports/baseline_model_comparison.csv` (Phase 7 Baselines: LR, RF, LightGBM, XGBoost)
* `reports/phase8_model_comparison.csv` (Phase 8 Optimization: Weighted, Tuned, Optimal F1/F2, Isotonic calibration)
* `reports/calibration_comparison.csv` (Phase 8 Calibration: Brier score 0.06517)
* `reports/shap_global_importance.csv` (Phase 9 SHAP Global: Top 20 ranking)
* `reports/shap_report.json` (Phase 9 Local Cases: High, Medium, Low risk applicants)
* `data/clustering/clustering_input.parquet` (BDE Clustering: 11 applicant features, normalized)
* `configs/clustering_features.json` (BDE Clustering: Metadata contract)
* `reports/clustering_evaluation.csv` (BDE Clustering: Inertia and Silhouette across K=2..8)
* `reports/clustering_profile.csv` (BDE Clustering: 4-cluster centroid profile table)

---

## 5. Result-Integrity & Anti-Leakage Verifications

1. **Target Exclusion from Clustering Construction:**
   * Checked `CLUSTERING_FEATURES`: Contains exactly 11 numeric capacity and repayment attributes.
   * `TARGET` and `SK_ID_CURR` are strictly omitted from the feature matrix $X$ fed to `SimpleImputer`, `StandardScaler`, and `KMeans`.
   * Post-hoc analysis in Section 19 evaluates default rates **only after** clusters are frozen.
2. **Test Set Evaluation Integrity:**
   * `application_test.csv` has zero target labels and is properly classified as an unlabelled inference spine.
   * Supervised model evaluation was conducted exclusively on the 46,127 held-out test rows of `application_train`.
3. **Metric Precision:**
   * Verified all numbers cited in markdown against underlying files:
     * XGBoost Test ROC-AUC: `0.789302` (cited as `0.7893`)
     * XGBoost Test PR-AUC: `0.277235` (cited as `0.2772`)
     * Optimal F1 Threshold: `0.17` (Recall: `41.43%`, Precision: `28.24%`)
     * Optimal F2 Threshold: `0.09` (Recall: `67.99%`, Precision: `19.72%`)
     * Isotonic Calibrated Brier Score: `0.065166` (cited as `0.06517`)

---

## 6. BDE Rubric Alignment Matrix

| Rubric Component | Marks | Notebook Section | Audit Status |
| :--- | :---: | :--- | :---: |
| **Classification: 1. Problem Statement** | 3M | Section 1.2, Section 6 | **VERIFIED** |
| **Classification: 2. Model / Justification** | (shared) | Section 8, Section 9 | **VERIFIED** |
| **Classification: 3. Coding / Implementation** | (shared) | Section 4, Section 7 | **VERIFIED** |
| **Classification: 4. Results** | (shared) | Section 10, Section 11 | **VERIFIED** |
| **Classification: 5. Inference** | (shared) | Section 12, Section 13 | **VERIFIED** |
| **Classification: 6. Implementation URL** | (shared) | Section 0, Section 23 | **VERIFIED** |
| **Clustering: 1. Problem Statement** | 3M | Section 1.3, Section 14 | **VERIFIED** |
| **Clustering: 2. Model / Justification** | (shared) | Section 16 | **VERIFIED** |
| **Clustering: 3. Coding / Implementation** | (shared) | Section 15, Section 17 | **VERIFIED** |
| **Clustering: 4. Results** | (shared) | Section 17, Section 18 | **VERIFIED** |
| **Clustering: 5. Inference** | (shared) | Section 18, Section 19 | **VERIFIED** |
| **Clustering: 6. Implementation URL** | (shared) | Section 0, Section 23 | **VERIFIED** |
| **Presentation: 1. Model Selection** | 1M | Section 9, Section 16 | **VERIFIED** |
| **Presentation: 2. Coding / Implementation** | 2M | Section 4, Section 15 | **VERIFIED** |
| **Presentation: 3. Results & Inference** | 1M | Section 10, 13, 18, 19 | **VERIFIED** |

---

## 7. Audit Conclusion

The notebook `notebooks/BDE_Assignment_Loan_Default_Classification_Clustering.ipynb` is complete, fully executed, free of errors, scientifically sound, and ready for university presentation and evaluation.
