"""Phase 8 Controlled Experiment Routines.

Implements:
1. Exact class imbalance ratio calculation (negative_count / positive_count)
2. Cost-sensitive training for XGBoost, LightGBM, and Logistic Regression
3. Reproducible RandomizedSearchCV hyperparameter search for XGBoost on TRAIN only
4. Model selection strictly via Validation PR-AUC
5. Probability calibration (Sigmoid/Platt & Isotonic)
6. Artifact persistence for Phase 8 models and metadata
"""

import json
import os
import pathlib
import sys
import time
from typing import Any, Optional, Union

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
try:
    from sklearn.frozen import FrozenEstimator
except ImportError:
    FrozenEstimator = None
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.log import get_logger

logger = get_logger("Phase8Experiments", "phase8_experiments.log")
ARTIFACTS_MODELS_DIR = ROOT / "artifacts" / "models"


def calculate_scale_pos_weight(y_train: np.ndarray) -> float:
    """Calculate exact negative to positive sample ratio from TRAIN partition.

    Never hard-coded; dynamically derived: negative_count / positive_count.
    """
    y_arr = np.asarray(y_train).ravel()
    neg_count = int(np.sum(y_arr == 0))
    pos_count = int(np.sum(y_arr == 1))
    if pos_count == 0:
        raise ValueError("No positive samples in training set.")
    ratio = float(neg_count / pos_count)
    logger.info(
        f"Derived TRAIN class weight ratio: {neg_count:,} neg / {pos_count:,} pos = {ratio:.4f}"
    )
    return ratio


def train_cost_sensitive_xgboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    scale_pos_weight: float = 1.0,
    random_state: int = 42,
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    min_child_weight: int = 1,
    n_jobs: int = -1,
) -> tuple[XGBClassifier, float]:
    """Train XGBoost with specified scale_pos_weight."""
    logger.info(
        f"Training XGBoost (scale_pos_weight={scale_pos_weight:.4f}, n_est={n_estimators}, depth={max_depth})..."
    )
    model = XGBClassifier(
        objective="binary:logistic",
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=n_jobs,
        eval_metric="logloss",
    )
    t0 = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - t0
    logger.info(f"XGBoost trained in {fit_time:.2f}s")
    return model, fit_time


def train_cost_sensitive_lightgbm(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    scale_pos_weight: float = 1.0,
    random_state: int = 42,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    n_jobs: int = -1,
) -> tuple[LGBMClassifier, float]:
    """Train LightGBM with specified scale_pos_weight."""
    logger.info(
        f"Training LightGBM (scale_pos_weight={scale_pos_weight:.4f}, n_est={n_estimators}, num_leaves={num_leaves})..."
    )
    model = LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=n_jobs,
        verbosity=-1,
    )
    t0 = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - t0
    logger.info(f"LightGBM trained in {fit_time:.2f}s")
    return model, fit_time


def train_cost_sensitive_logistic(
    X_train_trans: np.ndarray,
    y_train: np.ndarray,
    class_weight: Optional[str] = "balanced",
    random_state: int = 42,
    max_iter: int = 1000,
) -> tuple[LogisticRegression, float]:
    """Train Logistic Regression with class_weight ('balanced' or None)."""
    logger.info(f"Training Logistic Regression (class_weight='{class_weight}')...")
    model = LogisticRegression(
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=random_state,
        solver="lbfgs",
        n_jobs=-1,
    )
    t0 = time.time()
    model.fit(X_train_trans, y_train)
    fit_time = time.time() - t0
    logger.info(f"Logistic Regression trained in {fit_time:.2f}s")
    return model, fit_time


def run_xgboost_hyperparameter_search(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    n_iter: int = 20,
    random_state: int = 42,
    n_jobs_search: int = 2,
    n_jobs_xgb: int = 4,
) -> tuple[RandomizedSearchCV, pd.DataFrame, float]:
    """Run controlled RandomizedSearchCV for XGBoost strictly on TRAIN.

    Cross-validation: StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    Scoring: average_precision (PR-AUC)
    Max search budget: 20 candidate configurations
    """
    logger.info(
        f"Initiating XGBoost hyperparameter search (n_iter={n_iter}, CV=StratifiedKFold(3), scoring='average_precision')..."
    )

    param_distributions = {
        "n_estimators": [200, 300, 500],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.02, 0.05, 0.1],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "min_child_weight": [1, 5, 10],
    }

    base_xgb = XGBClassifier(
        objective="binary:logistic",
        random_state=random_state,
        n_jobs=n_jobs_xgb,
        eval_metric="logloss",
    )

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

    search = RandomizedSearchCV(
        estimator=base_xgb,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="average_precision",
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs_search,
        refit=True,
        verbose=1,
    )

    t0 = time.time()
    search.fit(X_train, y_train)
    search_duration = time.time() - t0

    cv_results = pd.DataFrame(search.cv_results_)
    # Sort by mean test PR-AUC descending
    cv_results = cv_results.sort_values("mean_test_score", ascending=False).reset_index(drop=True)

    logger.info(
        f"Hyperparameter search completed in {search_duration:.2f}s. "
        f"Best CV PR-AUC = {search.best_score_:.4f} with params: {search.best_params_}"
    )

    return search, cv_results, search_duration


def calibrate_model(
    base_model: Any,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    method: str = "sigmoid",
) -> tuple[CalibratedClassifierCV, float]:
    """Fit probability calibration model on Validation partition.

    Uses CalibratedClassifierCV with FrozenEstimator to preserve base model weights.
    """
    logger.info(f"Fitting {method} calibration model on Validation set...")
    if FrozenEstimator is not None:
        calibrator = CalibratedClassifierCV(FrozenEstimator(base_model), method=method)
    else:
        calibrator = CalibratedClassifierCV(base_model, cv="prefit", method=method)

    t0 = time.time()
    calibrator.fit(X_val, y_val)
    calib_duration = time.time() - t0
    logger.info(f"{method.capitalize()} calibration fitted in {calib_duration:.2f}s")
    return calibrator, calib_duration


def persist_phase8_artifacts(
    best_tuned_model: Any,
    calibrated_model: Any,
    metadata: dict[str, Any],
    output_dir: Union[str, pathlib.Path] = ARTIFACTS_MODELS_DIR,
) -> None:
    """Persist Phase 8 models and metadata to artifacts/models/.

    Guarantees Phase 7 artifacts are not overwritten.
    """
    out_dir = pathlib.Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_xgb_path = out_dir / "phase8_best_xgboost.joblib"
    calib_path = out_dir / "phase8_calibrated_model.joblib"
    meta_path = out_dir / "phase8_model_metadata.json"

    joblib.dump(best_tuned_model, best_xgb_path)
    joblib.dump(calibrated_model, calib_path)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Persisted Phase 8 artifacts: {best_xgb_path.name}, {calib_path.name}, {meta_path.name}")
