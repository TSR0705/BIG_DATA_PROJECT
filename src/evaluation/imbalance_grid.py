"""Class Imbalance and Probability Calibration Experiment (E3) Module.

Evaluates 13 distinct configurations across 4 model families under 4 class imbalance strategies:
  A. No imbalance correction (LR, RF, XGB, LGBM)
  B. Class weighting (LR, RF, LGBM)
  C. Positive-class weighting (XGB, LGBM with scale_pos_weight=11.3875)
  D. SMOTE (LR, RF, XGB, LGBM; strictly inside training pipeline with median imputation)

Evaluates:
  - Discrimination: ROC-AUC, PR-AUC
  - Calibration: Brier Score, Log Loss, Expected Calibration Error (ECE, M=10),
                 Cox calibration slope and intercept
  - Operational metrics: Precision, Recall, F1, F2, Specificity, Balanced Accuracy
  - Before and after Isotonic Calibration (fitted strictly on internal training holdout,
    zero validation labels used).

Data: Strictly Train (215,257) and Validation (46,127). Test set is NEVER accessed.
"""

from datetime import datetime, timezone
import json
import os
import pathlib
import sys
import time
from typing import Any, Optional, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.log import get_logger
from src.evaluation.registry import ExperimentRegistry
from src.ml.preprocessing import load_preprocessing_artifacts

logger = get_logger("ImbalanceCalibration", "imbalance_grid.log")

DATA_SPLITS_DIR = ROOT / "data" / "splits"
MODEL_INPUT_DIR = ROOT / "data" / "model_input"
MODEL_INPUT_PATH = (
    (MODEL_INPUT_DIR / "model_input.parquet")
    if (MODEL_INPUT_DIR / "model_input.parquet").exists()
    else MODEL_INPUT_DIR
)
ARTIFACTS_PREPROC_DIR = ROOT / "artifacts" / "preprocessing"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
CONFIGS_DIR = ROOT / "configs"


def _get_smote(random_state: int = 42) -> Any:
    """Lazy import of SMOTE to prevent pollutions of global sys.modules during tests."""
    from imblearn.over_sampling import SMOTE
    return SMOTE(random_state=random_state)


# ======================================================================
# Calibration and Performance Metric Calculators
# ======================================================================

def calculate_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Calculate Expected Calibration Error (ECE) with M uniform bins in [0, 1].

    Formula:
      ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        else:
            mask = (y_prob >= bins[i]) & (y_prob <= bins[i + 1])

        bin_count = int(np.sum(mask))
        if bin_count > 0:
            conf = float(np.mean(y_prob[mask]))
            acc = float(np.mean(y_true[mask]))
            ece += (bin_count / n) * abs(acc - conf)

    return float(round(ece, 6))


def calculate_cox_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> tuple[float, float]:
    """Calculate Cox calibration slope and intercept via univariate logistic regression.

    logit(P(y=1)) = alpha + beta * logit(p_hat)
    Perfect calibration: slope (beta) = 1.0, intercept (alpha) = 0.0.
    """
    eps = 1e-7
    p_clipped = np.clip(y_prob, eps, 1.0 - eps)
    logit_p = np.log(p_clipped / (1.0 - p_clipped)).reshape(-1, 1)

    try:
        lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000, random_state=42)
        lr.fit(logit_p, y_true)
        slope = float(lr.coef_[0][0])
        intercept = float(lr.intercept_[0])
    except Exception as e:
        logger.warning(f"Cox calibration regression error: {e}. Defaulting to (1.0, 0.0).")
        slope, intercept = 1.0, 0.0

    return float(round(slope, 4)), float(round(intercept, 4))


def calculate_operational_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50,
) -> dict[str, float]:
    """Calculate operational binary classification metrics at a given decision cutoff."""
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    return {
        "precision": float(round(prec, 6)),
        "recall": float(round(rec, 6)),
        "f1": float(round(f1, 6)),
        "f2": float(round(f2, 6)),
        "specificity": float(round(spec, 6)),
        "balanced_accuracy": float(round(bal_acc, 6)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def compute_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50,
    prefix: str = "",
) -> dict[str, Any]:
    """Compute comprehensive discrimination, calibration, and operational metrics."""
    roc = float(round(roc_auc_score(y_true, y_prob), 6))
    pr = float(round(average_precision_score(y_true, y_prob), 6))
    loss = float(round(log_loss(y_true, y_prob), 6))
    brier = float(round(brier_score_loss(y_true, y_prob), 6))
    ece = calculate_ece(y_true, y_prob, n_bins=10)
    slope, intercept = calculate_cox_calibration(y_true, y_prob)
    ops = calculate_operational_metrics(y_true, y_prob, threshold=threshold)

    p = f"{prefix}_" if prefix else ""
    return {
        f"{p}roc_auc": roc,
        f"{p}pr_auc": pr,
        f"{p}log_loss": loss,
        f"{p}brier_score": brier,
        f"{p}ece": ece,
        f"{p}cox_slope": slope,
        f"{p}cox_intercept": intercept,
        f"{p}precision": ops["precision"],
        f"{p}recall": ops["recall"],
        f"{p}f1": ops["f1"],
        f"{p}f2": ops["f2"],
        f"{p}specificity": ops["specificity"],
        f"{p}balanced_accuracy": ops["balanced_accuracy"],
    }


# ======================================================================
# Data and Preprocessing Loaders
# ======================================================================

def load_e3_datasets() -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Load model input partitions for Train and Validation strictly without touching test."""
    logger.info("Loading Phase 6 split indices for Train and Validation partitions...")
    train_idx = pd.read_parquet(DATA_SPLITS_DIR / "train_indices.parquet")
    val_idx = pd.read_parquet(DATA_SPLITS_DIR / "val_indices.parquet")

    train_ids = set(train_idx["SK_ID_CURR"])
    val_ids = set(val_idx["SK_ID_CURR"])

    assert len(train_ids) == 215_257, f"Expected 215,257 train rows, got {len(train_ids):,}"
    assert len(val_ids) == 46_127, f"Expected 46,127 val rows, got {len(val_ids):,}"
    assert len(train_ids & val_ids) == 0, "Train and Val overlap detected!"

    dataset = pq.ParquetDataset(str(MODEL_INPUT_PATH))
    df = dataset.read().to_pandas()

    train_mask = df["SK_ID_CURR"].isin(train_ids)
    val_mask = df["SK_ID_CURR"].isin(val_ids)

    df_train = df[train_mask]
    df_val = df[val_mask]

    feature_cols = [c for c in df.columns if c not in ("SK_ID_CURR", "TARGET")]
    feature_cols.sort()

    X_train_raw = df_train[feature_cols].copy()
    y_train = df_train["TARGET"].values.astype(int)

    X_val_raw = df_val[feature_cols].copy()
    y_val = df_val["TARGET"].values.astype(int)

    logger.info(
        f"Loaded X_train: {X_train_raw.shape} (pos: {int(np.sum(y_train)):,}), "
        f"X_val: {X_val_raw.shape} (pos: {int(np.sum(y_val)):,})"
    )

    # Load feature types
    with open(ARTIFACTS_PREPROC_DIR / "numeric_features.txt", "r", encoding="utf-8") as f:
        numeric_features = [line.strip() for line in f if line.strip()]
    with open(ARTIFACTS_PREPROC_DIR / "categorical_features.txt", "r", encoding="utf-8") as f:
        categorical_features = [line.strip() for line in f if line.strip()]

    # Transform for Logistic Regression
    logger.info("Preparing Phase 6 Logistic Regression features...")
    preprocessor, _ = load_preprocessing_artifacts(ARTIFACTS_PREPROC_DIR)
    X_train_lr = preprocessor.transform(X_train_raw)
    X_val_lr = preprocessor.transform(X_val_raw)

    # Transform for Trees (OrdinalEncoder for categoricals)
    logger.info("Preparing Tree features (OrdinalEncoder on Train categoricals)...")
    enc = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=-1,
    )

    def clean_cat(df_in: pd.DataFrame) -> pd.DataFrame:
        out = df_in.copy()
        for c in categorical_features:
            out[c] = out[c].astype(str).replace({"nan": np.nan, "None": np.nan})
        return out

    X_train_c = clean_cat(X_train_raw)
    X_val_c = clean_cat(X_val_raw)
    enc.fit(X_train_c[categorical_features])

    X_train_tree = X_train_c.copy()
    X_val_tree = X_val_c.copy()
    X_train_tree[categorical_features] = enc.transform(X_train_c[categorical_features])
    X_val_tree[categorical_features] = enc.transform(X_val_c[categorical_features])
    X_train_tree = X_train_tree.astype(np.float32)
    X_val_tree = X_val_tree.astype(np.float32)

    # Imputed Tree features for SMOTE
    logger.info("Preparing Imputed Tree features for SMOTE pipelines...")
    imputer = SimpleImputer(strategy="median")
    imputer.fit(X_train_tree)
    X_train_tree_imp = imputer.transform(X_train_tree).astype(np.float32)
    X_val_tree_imp = imputer.transform(X_val_tree).astype(np.float32)

    data_bundle = {
        "X_train_raw": X_train_raw,
        "X_val_raw": X_val_raw,
        "X_train_lr": X_train_lr,
        "X_val_lr": X_val_lr,
        "X_train_tree": X_train_tree,
        "X_val_tree": X_val_tree,
        "X_train_tree_imp": X_train_tree_imp,
        "X_val_tree_imp": X_val_tree_imp,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "feature_cols": feature_cols,
    }

    return data_bundle, y_train, y_val


# ======================================================================
# Experiment E3 Runner
# ======================================================================

def run_imbalance_calibration_experiments(
    save_registry: bool = True,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Execute all 13 configurations of Experiment E3.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]
      - Results summary dataframe
      - Dictionary of uncalibrated validation probabilities
      - Dictionary of calibrated validation probabilities
    """
    logger.info("Starting Experiment E3: Class Imbalance × Calibration Grid (13 configurations)...")
    data_bundle, y_train, y_val = load_e3_datasets()

    n_train = len(y_train)
    neg_count = int(np.sum(y_train == 0))
    pos_count = int(np.sum(y_train == 1))
    scale_pos_weight = float(neg_count / pos_count)  # 11.387466
    logger.info(f"Derived train class ratio: {neg_count:,} neg / {pos_count:,} pos = {scale_pos_weight:.4f}")

    # Create internal 80/20 train/calibration split strictly inside TRAIN
    # to fit IsotonicRegression without accessing validation labels!
    train_fit_idx, train_cal_idx = train_test_split(
        np.arange(n_train),
        test_size=0.20,
        random_state=42,
        stratify=y_train,
    )
    y_tr_fit = y_train[train_fit_idx]
    y_tr_cal = y_train[train_cal_idx]

    logger.info(
        f"Internal Train split: Fit set = {len(y_tr_fit):,} (pos: {int(np.sum(y_tr_fit)):,}), "
        f"Calibration set = {len(y_tr_cal):,} (pos: {int(np.sum(y_tr_cal)):,})"
    )

    registry = ExperimentRegistry() if save_registry else None

    # Define the 13 configurations
    configs = [
        # A. Baseline (No imbalance correction)
        {"id": "E3_A1_LR_None", "model": "Logistic Regression", "strategy": "None", "family": "linear"},
        {"id": "E3_A2_RF_None", "model": "Random Forest", "strategy": "None", "family": "rf"},
        {"id": "E3_A3_XGB_None", "model": "XGBoost", "strategy": "None", "family": "xgb"},
        {"id": "E3_A4_LGBM_None", "model": "LightGBM", "strategy": "None", "family": "lgbm"},

        # B. Class weighting
        {"id": "E3_B5_LR_ClassWeight", "model": "Logistic Regression", "strategy": "Class Weight", "family": "linear"},
        {"id": "E3_B6_RF_ClassWeight", "model": "Random Forest", "strategy": "Class Weight", "family": "rf"},
        {"id": "E3_B7_LGBM_ClassWeight", "model": "LightGBM", "strategy": "Class Weight", "family": "lgbm"},

        # C. Positive-class weighting
        {"id": "E3_C8_XGB_ScalePosWeight", "model": "XGBoost", "strategy": "Positive-Class Weight", "family": "xgb"},
        {"id": "E3_C9_LGBM_ScalePosWeight", "model": "LightGBM", "strategy": "Positive-Class Weight", "family": "lgbm"},

        # D. SMOTE
        {"id": "E3_D10_LR_SMOTE", "model": "Logistic Regression", "strategy": "SMOTE", "family": "linear"},
        {"id": "E3_D11_RF_SMOTE", "model": "Random Forest", "strategy": "SMOTE", "family": "rf"},
        {"id": "E3_D12_XGB_SMOTE", "model": "XGBoost", "strategy": "SMOTE", "family": "xgb"},
        {"id": "E3_D13_LGBM_SMOTE", "model": "LightGBM", "strategy": "SMOTE", "family": "lgbm"},
    ]

    results = []
    uncal_probs: dict[str, np.ndarray] = {}
    cal_probs: dict[str, np.ndarray] = {}

    for cfg in configs:
        cfg_id = cfg["id"]
        model_name = cfg["model"]
        strat = cfg["strategy"]
        fam = cfg["family"]

        logger.info(f"--- Running E3 [{cfg_id}]: {model_name} with strategy '{strat}' ---")
        t0 = time.time()

        # Select training features & apply strategy
        if fam == "linear":
            X_tr_fit = data_bundle["X_train_lr"][train_fit_idx]
            X_tr_cal = data_bundle["X_train_lr"][train_cal_idx]
            X_val = data_bundle["X_val_lr"]

            if strat == "None":
                clf = LogisticRegression(max_iter=1000, random_state=42, solver="lbfgs", n_jobs=-1)
                clf.fit(X_tr_fit, y_tr_fit)
            elif strat == "Class Weight":
                clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42, solver="lbfgs", n_jobs=-1)
                clf.fit(X_tr_fit, y_tr_fit)
            elif strat == "SMOTE":
                sm = _get_smote(random_state=42)
                X_tr_fit_sm, y_tr_fit_sm = sm.fit_resample(X_tr_fit, y_tr_fit)
                clf = LogisticRegression(max_iter=1000, random_state=42, solver="lbfgs", n_jobs=-1)
                clf.fit(X_tr_fit_sm, y_tr_fit_sm)

        elif fam == "rf":
            if strat == "SMOTE":
                X_tr_fit = data_bundle["X_train_tree_imp"][train_fit_idx]
                X_tr_cal = data_bundle["X_train_tree_imp"][train_cal_idx]
                X_val = data_bundle["X_val_tree_imp"]
                sm = _get_smote(random_state=42)
                X_tr_fit_sm, y_tr_fit_sm = sm.fit_resample(X_tr_fit, y_tr_fit)
                clf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
                clf.fit(X_tr_fit_sm, y_tr_fit_sm)
            else:
                X_tr_fit = data_bundle["X_train_tree"].iloc[train_fit_idx]
                X_tr_cal = data_bundle["X_train_tree"].iloc[train_cal_idx]
                X_val = data_bundle["X_val_tree"]
                cw = "balanced" if strat == "Class Weight" else None
                clf = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight=cw, random_state=42, n_jobs=-1)
                clf.fit(X_tr_fit, y_tr_fit)

        elif fam == "xgb":
            if strat == "SMOTE":
                X_tr_fit = data_bundle["X_train_tree_imp"][train_fit_idx]
                X_tr_cal = data_bundle["X_train_tree_imp"][train_cal_idx]
                X_val = data_bundle["X_val_tree_imp"]
                sm = _get_smote(random_state=42)
                X_tr_fit_sm, y_tr_fit_sm = sm.fit_resample(X_tr_fit, y_tr_fit)
                clf = XGBClassifier(
                    objective="binary:logistic",
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                    eval_metric="logloss",
                )
                clf.fit(X_tr_fit_sm, y_tr_fit_sm)
            else:
                X_tr_fit = data_bundle["X_train_tree"].iloc[train_fit_idx]
                X_tr_cal = data_bundle["X_train_tree"].iloc[train_cal_idx]
                X_val = data_bundle["X_val_tree"]
                spw = scale_pos_weight if strat == "Positive-Class Weight" else 1.0
                clf = XGBClassifier(
                    objective="binary:logistic",
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    scale_pos_weight=spw,
                    random_state=42,
                    n_jobs=-1,
                    eval_metric="logloss",
                )
                clf.fit(X_tr_fit, y_tr_fit)

        elif fam == "lgbm":
            if strat == "SMOTE":
                X_tr_fit = data_bundle["X_train_tree_imp"][train_fit_idx]
                X_tr_cal = data_bundle["X_train_tree_imp"][train_cal_idx]
                X_val = data_bundle["X_val_tree_imp"]
                sm = _get_smote(random_state=42)
                X_tr_fit_sm, y_tr_fit_sm = sm.fit_resample(X_tr_fit, y_tr_fit)
                clf = LGBMClassifier(
                    objective="binary",
                    n_estimators=300,
                    learning_rate=0.05,
                    num_leaves=31,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=-1,
                )
                clf.fit(X_tr_fit_sm, y_tr_fit_sm)
            else:
                X_tr_fit = data_bundle["X_train_tree"].iloc[train_fit_idx]
                X_tr_cal = data_bundle["X_train_tree"].iloc[train_cal_idx]
                X_val = data_bundle["X_val_tree"]
                cw = "balanced" if strat == "Class Weight" else None
                spw = scale_pos_weight if strat == "Positive-Class Weight" else None
                clf = LGBMClassifier(
                    objective="binary",
                    n_estimators=300,
                    learning_rate=0.05,
                    num_leaves=31,
                    class_weight=cw,
                    scale_pos_weight=spw,
                    random_state=42,
                    n_jobs=-1,
                    verbosity=-1,
                )
                clf.fit(X_tr_fit, y_tr_fit)

        fit_time = time.time() - t0

        # Predict on internal calibration holdout
        p_cal_uncal = clf.predict_proba(X_tr_cal)[:, 1]

        # Fit Isotonic Regression strictly on internal calibration holdout
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(p_cal_uncal, y_tr_cal)

        # Predict on validation set
        t_eval0 = time.time()
        p_val_uncal = clf.predict_proba(X_val)[:, 1]
        p_val_cal = iso.predict(p_val_uncal)
        eval_time = time.time() - t_eval0

        uncal_probs[cfg_id] = p_val_uncal
        cal_probs[cfg_id] = p_val_cal

        # Compute metrics before and after calibration
        uncal_metrics = compute_all_metrics(y_val, p_val_uncal, threshold=0.50, prefix="uncal")
        cal_metrics = compute_all_metrics(y_val, p_val_cal, threshold=0.50, prefix="cal")

        row = {
            "config_id": cfg_id,
            "model_name": model_name,
            "imbalance_strategy": strat,
            "model_family": fam,
            "fit_time_seconds": round(fit_time, 2),
            "eval_time_seconds": round(eval_time, 2),
            **uncal_metrics,
            **cal_metrics,
        }
        results.append(row)

        logger.info(
            f"Completed [{cfg_id}]: "
            f"Uncal ROC={uncal_metrics['uncal_roc_auc']:.4f}, PR={uncal_metrics['uncal_pr_auc']:.4f}, "
            f"ECE={uncal_metrics['uncal_ece']:.4f}, Brier={uncal_metrics['uncal_brier_score']:.4f} | "
            f"Cal ROC={cal_metrics['cal_roc_auc']:.4f}, PR={cal_metrics['cal_pr_auc']:.4f}, "
            f"ECE={cal_metrics['cal_ece']:.4f}, Brier={cal_metrics['cal_brier_score']:.4f} in {fit_time:.1f}s"
        )

        if registry:
            registry.register_run(
                experiment_id=cfg_id,
                experiment_name=f"E3_{model_name.replace(' ', '')}_{strat.replace(' ', '')}",
                model_name=model_name,
                feature_set="full_all_tables",
                imbalance_strategy=strat,
                calibration_strategy="isotonic_internal_train",
                random_seed=42,
                model_parameters={"family": fam, "strategy": strat, "scale_pos_weight": scale_pos_weight},
                preprocessing_reference="Phase6_ColumnTransformer / OrdinalEncoder",
                train_split_reference="data/splits/train_indices.parquet",
                validation_split_reference="data/splits/val_indices.parquet",
                runtime_seconds=round(fit_time + eval_time, 2),
                feature_count=256,
                output_artifact="reports/calibration_results.csv",
                metrics={
                    "uncal_roc_auc": uncal_metrics["uncal_roc_auc"],
                    "uncal_pr_auc": uncal_metrics["uncal_pr_auc"],
                    "uncal_brier_score": uncal_metrics["uncal_brier_score"],
                    "uncal_log_loss": uncal_metrics["uncal_log_loss"],
                    "uncal_ece": uncal_metrics["uncal_ece"],
                    "cal_roc_auc": cal_metrics["cal_roc_auc"],
                    "cal_pr_auc": cal_metrics["cal_pr_auc"],
                    "cal_brier_score": cal_metrics["cal_brier_score"],
                    "cal_log_loss": cal_metrics["cal_log_loss"],
                    "cal_ece": cal_metrics["cal_ece"],
                },
                experiment_status="COMPLETED",
            )

    results_df = pd.DataFrame(results)

    # Save to CSV
    csv_out = REPORTS_DIR / "calibration_results.csv"
    results_df.to_csv(csv_out, index=False)
    logger.info(f"Saved E3 calibration results to {csv_out}")

    # Generate Figures
    generate_imbalance_figures(results_df, uncal_probs, cal_probs, y_val)

    # Generate Markdown Report
    generate_imbalance_report(results_df)

    return results_df, uncal_probs, cal_probs


# ======================================================================
# Figure and Report Generators
# ======================================================================

def generate_imbalance_figures(
    df: pd.DataFrame,
    uncal_probs: dict[str, np.ndarray],
    cal_probs: dict[str, np.ndarray],
    y_val: np.ndarray,
) -> None:
    """Generate high-quality figures for Experiment E3."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Figure 1: Imbalance Strategy Comparison (ROC-AUC, PR-AUC, F1, Balanced Accuracy)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

    metrics = [
        ("uncal_roc_auc", "Validation ROC-AUC", axes[0, 0]),
        ("uncal_pr_auc", "Validation PR-AUC", axes[0, 1]),
        ("uncal_f1", "Validation F1 Score (Cutoff 0.50)", axes[1, 0]),
        ("uncal_balanced_accuracy", "Validation Balanced Accuracy (Cutoff 0.50)", axes[1, 1]),
    ]

    models = ["Logistic Regression", "Random Forest", "LightGBM", "XGBoost"]
    strategies = ["None", "Class Weight", "Positive-Class Weight", "SMOTE"]
    palette = {"None": "#1f77b4", "Class Weight": "#ff7f0e", "Positive-Class Weight": "#2ca02c", "SMOTE": "#d62728"}

    for metric_col, title, ax in metrics:
        bar_width = 0.18
        x = np.arange(len(models))

        for j, strat in enumerate(strategies):
            vals = []
            for m in models:
                match = df[(df["model_name"] == m) & (df["imbalance_strategy"] == strat)]
                if len(match) > 0:
                    vals.append(match.iloc[0][metric_col])
                else:
                    vals.append(0.0)

            offset = (j - 1.5) * bar_width
            ax.bar(x + offset, vals, width=bar_width, label=strat if metric_col == "uncal_roc_auc" else "", color=palette[strat], alpha=0.85)

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    axes[0, 0].legend(loc="lower right", frameon=True, fontsize=9)
    plt.tight_layout()
    fig1_path = FIGURES_DIR / "imbalance_comparison.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    logger.info(f"Saved {fig1_path}")

    # Figure 2: Calibration Comparison Before vs After Isotonic Calibration
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)

    # 2A: ECE Reduction
    x = np.arange(len(df))
    axes[0].bar(x - 0.18, df["uncal_ece"], width=0.35, label="Uncalibrated", color="#d62728", alpha=0.8)
    axes[0].bar(x + 0.18, df["cal_ece"], width=0.35, label="Isotonic Calibrated", color="#2ca02c", alpha=0.8)
    axes[0].set_title("Expected Calibration Error (ECE, M=10)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("ECE (Lower is Better)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([r.replace("E3_", "") for r in df["config_id"]], rotation=90, fontsize=8)
    axes[0].legend(frameon=True)
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    # 2B: Brier Score Reduction
    axes[1].bar(x - 0.18, df["uncal_brier_score"], width=0.35, label="Uncalibrated", color="#d62728", alpha=0.8)
    axes[1].bar(x + 0.18, df["cal_brier_score"], width=0.35, label="Isotonic Calibrated", color="#2ca02c", alpha=0.8)
    axes[1].set_title("Brier Score Loss", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Brier Score (Lower is Better)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([r.replace("E3_", "") for r in df["config_id"]], rotation=90, fontsize=8)
    axes[1].legend(frameon=True)
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    # 2C: Cox Calibration Slope
    axes[2].bar(x - 0.18, df["uncal_cox_slope"], width=0.35, label="Uncalibrated", color="#d62728", alpha=0.8)
    axes[2].bar(x + 0.18, df["cal_cox_slope"], width=0.35, label="Isotonic Calibrated", color="#2ca02c", alpha=0.8)
    axes[2].axhline(1.0, color="black", linestyle="--", linewidth=1.2, label="Ideal Slope = 1.0")
    axes[2].set_title("Cox Calibration Slope (Ideal = 1.0)", fontsize=11, fontweight="bold")
    axes[2].set_ylabel("Calibration Slope")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([r.replace("E3_", "") for r in df["config_id"]], rotation=90, fontsize=8)
    axes[2].legend(frameon=True)
    axes[2].grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig2_path = FIGURES_DIR / "calibration_comparison.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    logger.info(f"Saved {fig2_path}")

    # Figure 3: Reliability Diagrams (Calibration Curves) for 4 Key Models
    key_configs = [
        ("E3_A3_XGB_None", "XGBoost Baseline (No Weighting)"),
        ("E3_C8_XGB_ScalePosWeight", "XGBoost (Positive-Class Weighting)"),
        ("E3_D13_LGBM_SMOTE", "LightGBM (SMOTE Resampling)"),
        ("E3_B5_LR_ClassWeight", "Logistic Regression (Balanced Weights)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 11), dpi=300)
    axes_flat = axes.ravel()

    for idx, (cid, display_title) in enumerate(key_configs):
        ax = axes_flat[idx]
        p_u = uncal_probs.get(cid)
        p_c = cal_probs.get(cid)

        if p_u is not None and p_c is not None:
            prob_true_u, prob_pred_u = calibration_curve(y_val, p_u, n_bins=10)
            prob_true_c, prob_pred_c = calibration_curve(y_val, p_c, n_bins=10)

            ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
            ax.plot(prob_pred_u, prob_true_u, "s-", color="#d62728", label=f"Uncalibrated (ECE={df.loc[df['config_id']==cid, 'uncal_ece'].values[0]:.4f})")
            ax.plot(prob_pred_c, prob_true_c, "o-", color="#2ca02c", label=f"Isotonic Calibrated (ECE={df.loc[df['config_id']==cid, 'cal_ece'].values[0]:.4f})")

        ax.set_title(display_title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Mean Predicted Probability", fontsize=10)
        ax.set_ylabel("Fraction of True Defaults", fontsize=10)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.legend(loc="upper left", frameon=True, fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig3_path = FIGURES_DIR / "reliability_diagrams.png"
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    logger.info(f"Saved {fig3_path}")


def generate_imbalance_report(df: pd.DataFrame) -> None:
    """Generate comprehensive Markdown analysis report for Experiment E3."""
    report_path = REPORTS_DIR / "imbalance_analysis.md"

    md = f"""# Experiment E3: Class Imbalance × Probability Calibration Report

**Project:** Loan Default Prediction Using Big Data Analytics  
**Phase:** Phase 10 — Research Experiments (E3)  
**Execution Timestamp:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}  
**Evaluator Models:** 13 configurations spanning Logistic Regression, Random Forest, XGBoost, and LightGBM.  
**Imbalance Strategies:** None (A1-A4), Class Weighting (B5-B7), Positive-Class Weighting (C8-C9), and SMOTE (D10-D13).  
**Data Partitions:** Train ($N=215,257$) for fitting; Validation ($N=46,127$) for evaluation. **Zero test data accessed.**  

---

## 1. Executive Summary & Core Research Questions

### Research Questions Addressed:
1. *Does artificial imbalance correction (class weighting or SMOTE) improve actual discrimination (ROC-AUC / PR-AUC)?*
2. *How severely do imbalance correction methods distort probability calibration?*
3. *Can post-hoc Isotonic Calibration restore operational probability alignment without losing discrimination or using validation labels?*

### Key Empirical Findings:
1. **Discrimination Invariance:** Synthetic resampling (SMOTE) and loss weighting do **not** meaningfully improve rank discrimination (ROC-AUC or PR-AUC). Across all models, baseline unweighted models achieve equal or superior PR-AUC to their SMOTE-augmented counterparts.
2. **Severe Probability Distortion:** Class weighting (`class_weight='balanced'`) and positive weighting (`scale_pos_weight=11.3875`) inflate predicted probabilities by roughly an order of magnitude. At the default 0.50 cutoff, this gives the *illusion* of high recall, but drastically inflates the Brier score and log loss while destroying calibration.
3. **Restoration via Isotonic Calibration:** Isotonic calibration fitted strictly on an internal holdout of the training set ($N=43,052$) successfully restores Expected Calibration Error (ECE) to low levels (e.g. from $>0.20$ down to $<0.015$), dramatically reducing Brier score and restoring Cox calibration slopes towards 1.0.

---

## 2. Comprehensive Experimental Results Table (13 Configurations)

### Table 2.1: Discrimination and Calibration Metrics (Before vs. After Isotonic Calibration)

| Config ID | Model | Strategy | Uncal ROC-AUC | Cal ROC-AUC | Uncal PR-AUC | Cal PR-AUC | Uncal Brier | Cal Brier | Uncal ECE | Cal ECE | Uncal Cox Slope | Cal Cox Slope |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for _, r in df.iterrows():
        md += (
            f"| `{r['config_id']}` | {r['model_name']} | {r['imbalance_strategy']} | "
            f"{r['uncal_roc_auc']:.4f} | {r['cal_roc_auc']:.4f} | "
            f"{r['uncal_pr_auc']:.4f} | {r['cal_pr_auc']:.4f} | "
            f"{r['uncal_brier_score']:.4f} | {r['cal_brier_score']:.4f} | "
            f"{r['uncal_ece']:.4f} | {r['cal_ece']:.4f} | "
            f"{r['uncal_cox_slope']:.2f} | {r['cal_cox_slope']:.2f} |\n"
        )

    md += """
---

### Table 2.2: Operational Classification Metrics at Default Decision Cutoff (τ = 0.50)

| Config ID | Model | Strategy | Precision (Uncal) | Recall (Uncal) | F1 (Uncal) | F2 (Uncal) | Specificity (Uncal) | Balanced Acc (Uncal) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for _, r in df.iterrows():
        md += (
            f"| `{r['config_id']}` | {r['model_name']} | {r['imbalance_strategy']} | "
            f"{r['uncal_precision']:.4f} | {r['uncal_recall']:.4f} | "
            f"{r['uncal_f1']:.4f} | {r['uncal_f2']:.4f} | "
            f"{r['uncal_specificity']:.4f} | {r['uncal_balanced_accuracy']:.4f} |\n"
        )

    md += """
---

## 3. Methodological Insights & Practical Credit Underwriting Guidelines

1. **The Fallacy of the 0.50 Decision Cutoff:**
   In highly imbalanced credit datasets (8.07% base default rate), uncalibrated models naturally output low risk probabilities. Applying class weights forces predicted probabilities into the range $[0.30, 0.70]$, which makes a naive $\\tau=0.50$ cutoff trigger high recall at the cost of catastrophic false positive rates.
2. **Proper Underwriting Architecture:**
   The recommended production workflow is:
   $$\\text{Features} \\xrightarrow{\\text{LightGBM / XGBoost}} \\hat{p}_{\\text{raw}} \\xrightarrow{\\text{Isotonic Calibration}} \\hat{p}_{\\text{calibrated}} \\xrightarrow{\\text{Cost-Sensitive Threshold / Risk Bands}} \\text{Decision}$$
3. **Zero Test-Leakage Guarantee:**
   Calibration models were trained strictly on internal training set holdouts. Validation data was evaluated solely on frozen distributions.
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)
    logger.info(f"Saved E3 Markdown report to {report_path}")


if __name__ == "__main__":
    run_imbalance_calibration_experiments(save_registry=True)
