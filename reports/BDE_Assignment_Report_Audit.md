# BDE Assignment Report — Build Audit

Report: `reports/BDE_Assignment_Report.docx`
Generated: 5 October 2026 from the repository artifacts at `C:\Users\ACER\Desktop\BIG_DATA_PROJECT`
Builder: `python-docx` (two-pass build; the second pass fills the static Table of Contents, List of Figures and List of Tables with page numbers measured from a LibreOffice rendering of the first pass)

## 1. Document statistics

| Item | Value |
|---|---|
| Page size | A4, margins 2.3 cm left/right, 2.3 cm top, 2.0 cm bottom |
| Pages (LibreOffice rendering) | **24** (requirement: 15–25) |
| Front matter | cover, Table of Contents, List of Figures + List of Tables (pages 1–3) |
| Body | pages 4–22 (Sections 1–8 and References) |
| Appendix — Selected Code | pages 23–24 |
| Figures | **14** (10 reused from `reports/figures/`, 4 generated from verified CSV/JSON artifacts) |
| Tables | **17** |
| Code listings | **15** (11 in the body: 7 classification, 3 clustering, 1 reproduction; 4 in the Appendix) |
| Header / footer | running title in the header, PAGE field in the footer |
| Schema validation | passes the OOXML validator shipped with the docx skill; opens and renders in LibreOffice without repair |

Section structure matches the required outline exactly: 1 Introduction; 2 Dataset Overview; 3 Classification (3.1–3.10); 4 Clustering (4.1–4.10); 5 Overall Analysis and Discussion; 6 Assignment Requirement Mapping; 7 Conclusion; 8 Implementation and Repository; References; Appendix — Selected Code.

## 2. Figures

| # | Caption | Source | Status |
|---|---|---|---|
| 1 | Class distribution of TARGET | generated from `reports/split_report.json` (totals) | new |
| 2 | ROC-AUC and PR-AUC by model (two panels) | generated from `reports/baseline_model_comparison.csv` | new |
| 3 | ROC curves, test set | `reports/figures/phase7_roc_test.png` | reused |
| 4 | Precision–Recall curves, test set | `reports/figures/phase7_pr_test.png` | reused |
| 5 | Confusion matrix, baseline XGBoost @0.50 | `reports/figures/xgboost_confusion_matrix.png` | reused (y-axis labels are slightly clipped in the original PNG) |
| 6 | Precision/recall/F-scores vs threshold | `reports/figures/threshold_tradeoff.png` | reused |
| 7 | Reliability diagram before/after calibration | `reports/figures/calibration_curve.png` | reused |
| 8 | SHAP global importance | `reports/figures/shap_global_bar.png` | reused |
| 9 | SHAP waterfall, applicant 332851 | `reports/figures/shap_local_high_risk.png` | reused |
| 10 | Elbow curve and silhouette, K = 2–8 | `reports/figures/clustering_elbow_silhouette.png` | reused |
| 11 | Cluster sizes, K = 4 | generated from `reports/clustering_profile.csv` | new |
| 12 | PCA projection of clusters | `reports/figures/clustering_pca_scatter.png` | reused |
| 13 | Cluster profile heatmap (z-scored across clusters, raw means in cells) | generated from `reports/clustering_profile.csv` | new |
| 14 | Post-hoc default rate by cluster | `reports/figures/clustering_default_rates.png` | reused |

The generation script for the four new figures (`make_figures.py`) reads only the CSV/JSON files named above; no value in any figure was typed in by hand.

## 3. Tables and the artifact each was read from

| # | Table | Artifact |
|---|---|---|
| 1 | Source tables (rows, columns) | `reports/profiling/*.json`, `reports/profiling/summary.md` |
| 2 | Stratified split | `reports/split_report.json` |
| 3 | Preprocessing per model family | `reports/preprocessing_report.json` |
| 4 | Models evaluated | `src/ml/baseline_models.py` |
| 5 | Phase 7 model comparison | `reports/baseline_model_comparison.csv` |
| 6 | Phase 8 results, test set | `reports/phase8_model_comparison.csv`, `reports/phase8_report.json` |
| 7 | Operating points on test | `reports/phase8_model_comparison.csv` |
| 8 | Calibration comparison (validation) | `reports/calibration_comparison.csv` |
| 9 | Top-10 SHAP features with gain rank | `reports/shap_global_importance.csv`, `reports/shap_vs_native_importance.csv` |
| 10 | Local explanations, three applicants | `reports/shap_report.json` (`local_cases`) |
| 11 | Clustering features | `configs/clustering_features.json`, `src/ml/clustering.py` |
| 12 | Inertia and silhouette, K = 2–8 | `reports/clustering_evaluation.csv` |
| 13 | Cluster sizes | `reports/clustering_profile.csv` |
| 14 | Cluster profiles (feature means) | `reports/clustering_profile.csv` |
| 15 | Post-hoc default rate by cluster | `reports/clustering_profile.csv` |
| 16 | Classification vs clustering | derived from the above |
| 17 | Requirement coverage | document structure |

## 4. Code listings

All listings are excerpts of files in the repository (`src/` and `scripts/`), trimmed of logging statements and docstrings; none were rewritten.

Body: (1) `src/common/spark_session.py` Spark session; (2) `src/ingestion/bronze.py` explicit-schema CSV→Parquet with row-count assertion; (3) `src/features/gold.py` `aggregate_bureau_balance`; (4) `src/features/gold.py` join loop with row-explosion audit; (5) `src/ml/split.py` two-stage stratified split; (6) `src/ml/baseline_models.py` XGBClassifier; (7) `src/ml/model_utils.py` `evaluate_predictions`; (8) `src/ml/clustering.py` feature selection / TARGET exclusion / impute / clip / scale; (9) `src/ml/clustering.py` K evaluation loop; (10) `src/ml/clustering.py` fit, PCA, profile, post-hoc TARGET; (11) `scripts/` reproduction commands (script names verified against the `scripts/` directory).
Appendix: (12) `src/ingestion/bronze.py` fuller ingestion; (13) `src/features/gold.py` `safe_divide` + bureau-balance aggregation; (14) `src/features/gold.py` `join_features_to_spine`; (15) `src/ml/clustering.py` full preparation / evaluation / profiling.

## 5. Numerical results verified against artifacts

Every number in the report was read from, or recomputed from, the files below; values were cross-checked against the figures the user supplied in the brief, and all matched.

- Dataset: 8 tables, 58,489,893 rows; application_train 307,511 × 122; TARGET positives 24,825 (8.07%) — `reports/profiling/*`, `reports/split_report.json`.
- Gold matrix 307,511 × 241; 10 joins with zero row increase — `reports/gold/summary.md`, `reports/gold/join_audit.json`.
- Feature matrix 258 columns (256 model features + SK_ID_CURR + TARGET); leakage audit PASS; test set has no label — `reports/leakage_audit.md`, `reports/preprocessing_report.json`.
- Split 215,257 / 46,127 / 46,127; random_state 42; partitions disjoint; max default-rate delta 4.807e-06 — `reports/split_report.json`.
- Phase 7 (validation / test): LR 0.7763/0.7773 ROC-AUC, 0.2692/0.2517 PR-AUC; RF 0.7467/0.7454, 0.2437/0.2271; LightGBM 0.7886/0.7877, 0.2880/0.2749; XGBoost 0.7897/0.7893, 0.2896/0.2772 — `reports/baseline_model_comparison.csv`. Accuracy for all four models lies between 0.919 and 0.921 (same file) and was not used for selection.
- Phase 8 best params (max_depth 3, n_estimators 500, min_child_weight 10, learning_rate 0.1, subsample 0.8, colsample_bytree 1.0); tuned test ROC-AUC 0.7898, PR-AUC 0.2790, log loss 0.2367, Brier 0.0660; recall 4.5% → 41.4% and precision 50.3% → 28.2% when the threshold moves 0.50 → 0.17; F2-optimal threshold 0.09 (recall 68.0%); cost-sensitive 1:5 threshold 0.16; scale_pos_weight variant: recall 65.8%, precision 20.1%, log loss 0.4968, ROC-AUC 0.7888 — `reports/phase8_model_comparison.csv`, `reports/phase8_report.json`.
- Confusion matrix (baseline, test, 0.50): TN 42,279 / FP 124 / FN 3,568 / TP 156 — `reports/figures/xgboost_confusion_matrix.png`, consistent with precision 0.5571 and recall 0.0419 in the CSV.
- Calibration (validation): uncalibrated Brier 0.0654, ROC-AUC 0.7911, PR-AUC 0.2914; isotonic Brier 0.0652, PR-AUC 0.2850; isotonic selected — `reports/calibration_comparison.csv`, `reports/phase8_report.json`.
- SHAP: TreeExplainer, 5,000-row validation sample, seed 42; top features APP_EXT_SOURCES_MEAN (0.4440), CODE_GENDER (0.1201), POS_COMPLETION_RATE_MEAN (0.1184); AMT_ANNUITY SHAP rank 8 vs gain rank 36; local cases 332851 (p = 0.9153, defaulted), 183579 (p = 0.0013, repaid), 133546 (p = 0.1700, defaulted) — `reports/shap_global_importance.csv`, `reports/shap_vs_native_importance.csv`, `reports/shap_report.json`.
- Clustering: 11 features; SK_ID_CURR and TARGET excluded (`configs/clustering_features.json`: "target_usage: Strictly post-hoc only"); median imputation, 99.5th-percentile clipping of four monetary/ratio columns, StandardScaler; K = 2–8 inertia and silhouette (silhouette on a 25,000-row sample) — `reports/clustering_evaluation.csv`, `src/ml/clustering.py`.
- Final K = 4: sizes 91,837 (29.9%) / 98,997 (32.2%) / 44,555 (14.5%) / 72,122 (23.5%); default rates 14.26% / 5.02% / 6.09% / 5.61%; all profile means — `reports/clustering_profile.csv`.
- Environment: Python 3.11, scikit-learn 1.6.1, XGBoost 3.2.0, LightGBM 4.7.0 — `reports/phase8_report.json` (`environment`); pyspark 3.5.x — `requirements.lock.txt`.

## 6. Constraint checks

| Constraint | Status |
|---|---|
| TARGET not used to construct clusters | Verified in `src/ml/clustering.py` (feature matrix built from the 11 features only; TARGET read solely for the post-hoc profile) and stated in the report (4.1, 4.3, 4.6, 4.10). |
| SK_ID_CURR not used in clustering | Verified (same file / `configs/clustering_features.json`). |
| application_test not used for evaluation | Verified: all evaluation artifacts use the stratified validation/test partitions of application_train; the report states application_test is inference-only (Section 2, 3.3). |
| No multi-node / cluster claim | Report states Spark ran in local mode on a single machine (Section 5, Section 8). |
| Model not selected on accuracy | Section 3.7 selects on ROC-AUC, PR-AUC and log loss and explains why accuracy is uninformative. |
| SHAP not presented as causal | Section 3.10 and 4.10 state the attributions are descriptive, not causal. |
| K = 4 justified honestly | Section 4.7 reports that silhouette peaks at K = 2 (0.1505), that K = 4 (0.1138) sits in a flat band, that inertia has no sharp elbow, and that K = 4 was chosen on interpretability with the post-hoc default-rate spread as supporting evidence; the full K = 2–8 table is printed. |
| No fabricated results | Every figure and number traces to an artifact listed above. |

## 7. Clustering validation status

Implemented and evaluated in the repository (`scripts/run_clustering.py`, `reports/clustering_evaluation.csv`, `reports/clustering_profile.csv`, three figures). The evaluation computed inertia on the full 307,511 rows and silhouette on a 25,000-row sample for K = 2–8. K = 4 is hard-coded in `scripts/run_clustering.py` ("# Fit best K=4"); no automatic selection rule is implemented, so the report documents the choice as an interpretability decision and acknowledges that the strict silhouette criterion would favour K = 2. This is the one place a reviewer is likely to push back; the report gives the data to defend it.

## 8. Requirement coverage

| Requirement | Section(s) | Evidence in the report |
|---|---|---|
| Classification — Problem Statement | 3.1, 3.2 | problem, inputs, target, output, objective |
| Classification — Model and justification | 3.4, 3.5 | four models, dataset-specific rationale |
| Classification — Coding | 3.6, Appendix | Listings 1–7, 12–14 |
| Classification — Results | 3.8 | Tables 5–8, Figures 2–7 |
| Classification — Inference | 3.9, 3.10 | threshold / calibration analysis; SHAP global and local |
| Classification — Implementation URL | 8 | GitHub URL |
| Clustering — Problem Statement | 4.1, 4.2 | |
| Clustering — Model and justification | 4.4, 4.5 | |
| Clustering — Coding | 4.6, Appendix | Listings 8–10, 15 |
| Clustering — Results | 4.7, 4.8 | Tables 12–13, Figures 10–12 |
| Clustering — Inference | 4.9, 4.10 | Tables 14–15, Figures 13–14 |
| Clustering — Implementation URL | 8 | GitHub URL |
| Presentation — model selection / coding / results | 3.7, 4.7, 3.6, 4.6, 3.8–3.10, 4.8–4.10 | |

## 9. Missing information and caveats

- The two reference documents named in the brief ("big data research data.docx" and "Phase_Wise_Implementation_Plan.docx") were not found in the repository or on the Desktop; the report was written from the implementation and its artifacts only.
- The cover reads "Submitted by: [Student Name(s)]" — fill in names, roll numbers and the course code before submission.
- Figure 5 (confusion matrix) is reused as-is from `reports/figures/`; its y-axis labels are partially clipped in the source PNG. Regenerating it in the repository would fix this.
- Page count was measured with LibreOffice; Microsoft Word may differ by about one page because of font metrics (Calibri/Consolas on the laptop versus their metric-compatible substitutes here). The document is 24 pages here, so a one-page shift still satisfies 15–25.
- The TOC, List of Figures and List of Tables are static text with measured page numbers rather than Word fields, so they do not require "Update field" and will not change if the document is edited. If sections are added or removed, re-run the build script to refresh them.
- No runtime figures (training or clustering durations) are quoted except the Phase 8 search duration, because the artifacts do not record them for the other stages.

## 10. Validation

- `BDE_Assignment_Report.docx` reopened with python-docx (973 paragraphs) and converted to PDF with LibreOffice without errors — not corrupted.
- OOXML schema validation: passed (one python-docx template defect, a `w:zoom` element missing its `percent` attribute, was corrected during the build).
- Visual check of every page at reduced resolution and of pages 7–12, 16–17, 19 and 22–24 at reading resolution: no table overflow, no image distortion, no orphaned captions; tables are kept on one page; listings are kept together.
- All 61 TOC / LOF / LOT page references resolved on the second pass (no unresolved entries).
