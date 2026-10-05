# Phase 6 Preprocessing & Feature Pipeline Report

> **Project:** Loan Default Prediction Using Big Data Analytics  
> **Fitted Dataset:** `TRAIN ONLY` (215,257 samples)  
> **Generated:** 2026-10-05 15:38:25 UTC  
> **Status:** **PASS** (Zero Preprocessing Leakage Verified)

---

## 1. Executive Summary

This report specifies the model-specific preprocessing architectures designed for downstream modeling:
- **Logistic Regression**: Linear models require complete missingness imputation, numeric scaling, and one-hot encoding.
- **Tree-Based Models (Random Forest, LightGBM, XGBoost)**: Tree ensembles operate on monotonic feature splits and do NOT receive blind standardization.
- **Leakage Prevention**: All transformations, imputers, scalers, and encoders are fit **STRICTLY ON TRAIN DATA**. Validation and Test sets are transformed using train parameters only.

---

## 2. Feature Type Segregation

| Category | Column Count | Description |
| :--- | :--- | :--- |
| **Numeric Features** | 240 | Financial amounts, age, bureau metrics, repayment counts, derived ratios |
| **Categorical Features** | 16 | Contract types, gender, housing status, education, occupation |
| **Excluded Identifiers** | 2 | `SK_ID_CURR` (applicant key) and `TARGET` (target label) |
| **Total Input Columns** | **258** | Model input dataset schema |

---

## 3. Logistic Regression Pipeline Architecture

```
X_train (256 Features)
   ├── [240 Numeric]    ──> SimpleImputer(median) ──────> StandardScaler() ──────────> (240 Columns)
   └── [ 16 Categorical] ─> SimpleImputer(most_frequent) ─> OneHotEncoder(min_freq=0.01) -> (90 Columns)
                                                                                            ↓
                                                                             Transformed Matrix (330 Features)
```

- **One-Hot Encoding Guardrails:**
  - `handle_unknown="ignore"`: Unseen categories encountered in validation or inference are encoded as all zeros.
  - `min_frequency=0.01`: Rare categorical levels (<1% prevalence in training) are grouped to prevent extreme sparsity.
- **Total Input Features:** 256  
- **Transformed Feature Dimension:** 330  

---

## 4. Tree Models Preprocessing Specification

- **Scaling Policy:** `StandardScaler` is **NOT applied** to tree models.
- **Missing Value Policy:** Preserved for native tree splitting algorithms (LightGBM/XGBoost native missingness support).
- **Categorical Handling:** Documented for native integer category mapping or target encoding in modeling phases.

---

## 5. Artifact Verification

- `artifacts/preprocessing/fitted_logistic_preprocessor.joblib`: Serialized ColumnTransformer.
- `artifacts/preprocessing/preprocessing_metadata.json`: Complete transformation parameters and feature names.
- `artifacts/preprocessing/numeric_features.txt`: List of 240 numeric features.
- `artifacts/preprocessing/categorical_features.txt`: List of 16 categorical features.
- `artifacts/preprocessing/transformed_feature_names.txt`: List of 330 post-transformation feature names.
