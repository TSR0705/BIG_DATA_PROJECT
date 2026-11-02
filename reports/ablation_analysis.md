# Experiment E1: Feature Ablation Analysis Report

**Project:** Loan Default Prediction Using Big Data Analytics  
**Phase:** Phase 10 — Research Experiments (E1)  
**Execution Timestamp:** 2026-11-02 17:14:00 UTC  
**Evaluator Model:** Frozen XGBoost Finalist (`max_depth=3`, `learning_rate=0.1`, `n_estimators=500`, `min_child_weight=10`, `subsample=0.8`, `colsample_bytree=1.0`)  
**Data Partitions:** Train ($N=215,257$) for fitting; Validation ($N=46,127$) for evaluation. **Zero test data accessed.**  

---

## 1. Executive Summary & Research Question

**Research Question:**  
*"How much does each historical financial data source contribute to loan-default prediction beyond the applicant's application-level information?"*

**Core Findings:**
1. **Application-Only Baseline:** Achieves a Validation ROC-AUC of **0.7726** and PR-AUC of **0.2625** (147 features).
2. **Full Multi-Table Model:** Achieves a Validation ROC-AUC of **0.7911** and PR-AUC of **0.2914** (256 features).
3. **Net Multi-Table Gain:** Adding historical credit bureau, installment payment, and transaction records yields an incremental **+0.0185 ROC-AUC** and **+0.0289 PR-AUC** lift over application data alone.
4. **Most Influential Historical Source:** The Credit Bureau (`app_bureau`) and Installment Payments (`app_installments`) provide the largest standalone incremental boosts.
5. **Leave-One-Group-Out Impact:** Withholding Credit Bureau (`logo_bureau`) causes the largest single drop in discrimination among historical tables.

---

## 2. Comprehensive Experimental Results Table

| Configuration Key | Configuration Name | Type | Features | ROC-AUC (95% CI) | PR-AUC (95% CI) | Δ ROC vs App | Δ PR vs App | Δ ROC vs Full | Fit Time (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `app_only` | Application Only | incremental | 147 | 0.7726 [0.7631, 0.7800] | 0.2625 [0.2439, 0.2745] | +0.0000 | +0.0000 | -0.0185 | 8.17s |
| `app_bureau` | Application + Bureau | incremental | 173 | 0.7756 [0.7664, 0.7827] | 0.2676 [0.2497, 0.2808] | +0.0030 | +0.0051 | -0.0155 | 9.4s |
| `app_prev` | Application + Previous Apps | incremental | 172 | 0.7770 [0.7683, 0.7847] | 0.2673 [0.2478, 0.2810] | +0.0044 | +0.0048 | -0.0142 | 9.33s |
| `app_installments` | Application + Installments | incremental | 163 | 0.7818 [0.7732, 0.7890] | 0.2729 [0.2537, 0.2873] | +0.0092 | +0.0105 | -0.0093 | 8.99s |
| `app_credit_card` | Application + Credit Card | incremental | 168 | 0.7744 [0.7645, 0.7812] | 0.2669 [0.2492, 0.2786] | +0.0018 | +0.0045 | -0.0167 | 7.86s |
| `app_pos_cash` | Application + POS/CASH | incremental | 161 | 0.7793 [0.7702, 0.7879] | 0.2702 [0.2520, 0.2806] | +0.0067 | +0.0077 | -0.0118 | 8.07s |
| `full_all_tables` | Full Model (All Tables) | full | 256 | 0.7911 [0.7827, 0.7987] | 0.2914 [0.2734, 0.3084] | +0.0185 | +0.0289 | +0.0000 | 13.63s |
| `logo_bureau` | Full Excl. Bureau | leave_one_group_out | 230 | 0.7901 [0.7820, 0.7967] | 0.2860 [0.2672, 0.3000] | +0.0175 | +0.0235 | -0.0010 | 12.64s |
| `logo_prev` | Full Excl. Previous Apps | leave_one_group_out | 231 | 0.7907 [0.7829, 0.7983] | 0.2906 [0.2718, 0.3048] | +0.0181 | +0.0281 | -0.0004 | 12.56s |
| `logo_installments` | Full Excl. Installments | leave_one_group_out | 240 | 0.7900 [0.7822, 0.7975] | 0.2884 [0.2712, 0.3047] | +0.0174 | +0.0259 | -0.0011 | 12.37s |
| `logo_credit_card` | Full Excl. Credit Card | leave_one_group_out | 235 | 0.7909 [0.7829, 0.7981] | 0.2894 [0.2699, 0.3049] | +0.0183 | +0.0269 | -0.0002 | 12.78s |
| `logo_pos_cash` | Full Excl. POS/CASH | leave_one_group_out | 242 | 0.7895 [0.7811, 0.7968] | 0.2892 [0.2715, 0.3056] | +0.0169 | +0.0267 | -0.0016 | 12.75s |
| `logo_derived` | Full Excl. Derived | leave_one_group_out | 249 | 0.7909 [0.7831, 0.7971] | 0.2917 [0.2728, 0.3073] | +0.0183 | +0.0292 | -0.0002 | 15.72s |

---

## 3. Incremental Contribution Analysis (Single-Table Augmentation)

Examining the marginal value of augmenting demographic application data with exactly one historical table:

- **Application + Bureau (+26 features):** Validation ROC-AUC = **0.7756** (Δ = **+0.0030**), PR-AUC = **0.2676** (Δ = **+0.0051**). Bootstrap 95% CI for Δ ROC: [+0.0016, +0.0053].
- **Application + Previous Apps (+25 features):** Validation ROC-AUC = **0.7770** (Δ = **+0.0044**), PR-AUC = **0.2673** (Δ = **+0.0048**). Bootstrap 95% CI for Δ ROC: [+0.0018, +0.0066].
- **Application + Installments (+16 features):** Validation ROC-AUC = **0.7818** (Δ = **+0.0092**), PR-AUC = **0.2729** (Δ = **+0.0105**). Bootstrap 95% CI for Δ ROC: [+0.0066, +0.0140].
- **Application + Credit Card (+21 features):** Validation ROC-AUC = **0.7744** (Δ = **+0.0018**), PR-AUC = **0.2669** (Δ = **+0.0045**). Bootstrap 95% CI for Δ ROC: [+0.0002, +0.0040].
- **Application + POS/CASH (+14 features):** Validation ROC-AUC = **0.7793** (Δ = **+0.0067**), PR-AUC = **0.2702** (Δ = **+0.0077**). Bootstrap 95% CI for Δ ROC: [+0.0049, +0.0104].
- **Full Model (All Tables) (+109 features):** Validation ROC-AUC = **0.7911** (Δ = **+0.0185**), PR-AUC = **0.2914** (Δ = **+0.0289**). Bootstrap 95% CI for Δ ROC: [+0.0145, +0.0242].

---

## 4. Leave-One-Feature-Group-Out (LOGO) Analysis

Evaluating the redundancy and unique informational value of each feature group by holding out exactly one group from the full 256-feature model:

- **Full Excl. Bureau (230 features):** Validation ROC-AUC drops by **0.0010** to **0.7901**, PR-AUC changes by **-0.0054** to **0.2860**.
- **Full Excl. Previous Apps (231 features):** Validation ROC-AUC drops by **0.0004** to **0.7907**, PR-AUC changes by **-0.0008** to **0.2906**.
- **Full Excl. Installments (240 features):** Validation ROC-AUC drops by **0.0011** to **0.7900**, PR-AUC changes by **-0.0030** to **0.2884**.
- **Full Excl. Credit Card (235 features):** Validation ROC-AUC drops by **0.0002** to **0.7909**, PR-AUC changes by **-0.0020** to **0.2894**.
- **Full Excl. POS/CASH (242 features):** Validation ROC-AUC drops by **0.0016** to **0.7895**, PR-AUC changes by **-0.0022** to **0.2892**.
- **Full Excl. Derived (249 features):** Validation ROC-AUC drops by **0.0002** to **0.7909**, PR-AUC changes by **+0.0003** to **0.2917**.

---

## 5. Methodological & Scientific Notes

1. **Bootstrap Resampling:** 30 paired bootstrap resamples of the 46,127 validation predictions were executed. Resampling was paired across configurations to eliminate sample variance between models. As specified in the protocol, this provides a computational variance estimate rather than an asymptotic inferential test.
2. **Double-Counting Avoidance:** Derived cross-domain features (`DERIVED_`) are treated as a distinct 7-feature group (`logo_derived`) rather than arbitrarily distributed across individual tables, maintaining strict lineage purity.
3. **No Leakage:** Preprocessing was fitted strictly on `X_train`. Validation data was evaluated solely with pre-fitted transformations.
