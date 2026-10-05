"""Baseline Modeling Layer for Loan Default Prediction.

Implements four clean, reproducible classification baselines:
1. Logistic Regression (Linear baseline with Phase 6 preprocessing)
2. Random Forest (Bagging ensemble baseline on raw numeric features)
3. LightGBM (Gradient boosted decision trees baseline)
4. XGBoost (Extreme gradient boosting baseline)

Enforces strict boundaries:
- Fitted exclusively on TRAIN (215,257 rows)
- Evaluated on VALIDATION (46,127 rows)
- Final unbiased baseline benchmark on TEST (46,127 rows)
- Zero hyperparameter tuning, zero threshold optimization, zero SMOTE
- ID ('SK_ID_CURR') and Label ('TARGET') strictly excluded from X
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
import pyarrow.parquet as pq
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.log import get_logger
from src.ml.model_utils import evaluate_predictions
from src.ml.preprocessing import load_preprocessing_artifacts

logger = get_logger("MLBaselineModels", "baseline_models.log")

DATA_SPLITS_DIR = ROOT / "data" / "splits"
MODEL_INPUT_DIR = ROOT / "data" / "model_input"
MODEL_INPUT_PATH = (
    (MODEL_INPUT_DIR / "model_input.parquet")
    if (MODEL_INPUT_DIR / "model_input.parquet").exists()
    else MODEL_INPUT_DIR
)
ARTIFACTS_PREPROC_DIR = ROOT / "artifacts" / "preprocessing"
ARTIFACTS_MODELS_DIR = ROOT / "artifacts" / "models"


def load_modeling_data(
    model_input_path: Union[str, pathlib.Path] = MODEL_INPUT_PATH,
    split_indices_dir: Union[str, pathlib.Path] = DATA_SPLITS_DIR,
) -> dict[str, Any]:
    """Load model input data and partition strictly according to persisted Phase 6 split indices.

    Guarantees:
    - Exactly 215,257 train rows, 46,127 val rows, 46,127 test rows
    - No SK_ID_CURR or TARGET inside feature matrices X
    - TARGET retained only in ground truth vectors y
    """
    logger.info("Loading Phase 6 split indices...")
    sdir = pathlib.Path(split_indices_dir)
    train_idx_df = pd.read_parquet(sdir / "train_indices.parquet")
    val_idx_df = pd.read_parquet(sdir / "val_indices.parquet")
    test_idx_df = pd.read_parquet(sdir / "test_indices.parquet")

    train_ids = set(train_idx_df["SK_ID_CURR"])
    val_ids = set(val_idx_df["SK_ID_CURR"])
    test_ids = set(test_idx_df["SK_ID_CURR"])

    # Split assertions
    assert len(train_ids) == 215_257, f"Expected 215,257 train rows, got {len(train_ids):,}"
    assert len(val_ids) == 46_127, f"Expected 46,127 val rows, got {len(val_ids):,}"
    assert len(test_ids) == 46_127, f"Expected 46,127 test rows, got {len(test_ids):,}"
    assert len(train_ids & val_ids) == 0, "Train and Val overlap detected!"
    assert len(train_ids & test_ids) == 0, "Train and Test overlap detected!"
    assert len(val_ids & test_ids) == 0, "Val and Test overlap detected!"

    logger.info(f"Loading full model input dataset from {model_input_path}...")
    dataset = pq.ParquetDataset(str(model_input_path))
    table = dataset.read()
    df = table.to_pandas()

    assert len(df) == 307_511, f"Expected 307,511 total rows, got {len(df):,}"
    assert "SK_ID_CURR" in df.columns, "Missing SK_ID_CURR"
    assert "TARGET" in df.columns, "Missing TARGET"

    # Identify model features
    excluded = {"SK_ID_CURR", "TARGET"}
    feature_cols = [c for c in df.columns if c not in excluded]
    feature_cols.sort()

    logger.info(f"Identified {len(feature_cols)} model features (strictly excluding ID & TARGET).")

    # Load categorical & numeric feature lists from Phase 6 artifacts
    num_txt = ARTIFACTS_PREPROC_DIR / "numeric_features.txt"
    cat_txt = ARTIFACTS_PREPROC_DIR / "categorical_features.txt"
    with open(num_txt, "r", encoding="utf-8") as f:
        numeric_features = [line.strip() for line in f if line.strip()]
    with open(cat_txt, "r", encoding="utf-8") as f:
        categorical_features = [line.strip() for line in f if line.strip()]

    # Index partitioning
    train_mask = df["SK_ID_CURR"].isin(train_ids)
    val_mask = df["SK_ID_CURR"].isin(val_ids)
    test_mask = df["SK_ID_CURR"].isin(test_ids)

    df_train = df[train_mask]
    df_val = df[val_mask]
    df_test = df[test_mask]

    X_train = df_train[feature_cols].copy()
    y_train = df_train["TARGET"].values

    X_val = df_val[feature_cols].copy()
    y_val = df_val["TARGET"].values

    X_test = df_test[feature_cols].copy()
    y_test = df_test["TARGET"].values

    # Programmatic feature safety verification
    assert "SK_ID_CURR" not in X_train.columns
    assert "TARGET" not in X_train.columns
    assert "SK_ID_CURR" not in X_val.columns
    assert "TARGET" not in X_val.columns
    assert "SK_ID_CURR" not in X_test.columns
    assert "TARGET" not in X_test.columns

    logger.info(
        f"Data partitioned: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}"
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "feature_cols": feature_cols,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
    }


def prepare_logistic_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    preprocessor_dir: Union[str, pathlib.Path] = ARTIFACTS_PREPROC_DIR,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
    """Transform features for Logistic Regression using Phase 6 fitted ColumnTransformer."""
    logger.info("Loading Phase 6 fitted Logistic Regression ColumnTransformer...")
    preprocessor, _ = load_preprocessing_artifacts(preprocessor_dir)

    t0 = time.time()
    X_train_trans = preprocessor.transform(X_train)
    X_val_trans = preprocessor.transform(X_val)
    X_test_trans = preprocessor.transform(X_test)
    logger.info(
        f"Transformed features for Logistic Regression: "
        f"Train={X_train_trans.shape}, Val={X_val_trans.shape}, Test={X_test_trans.shape} "
        f"in {time.time()-t0:.2f}s"
    )

    return X_train_trans, X_val_trans, X_test_trans, preprocessor


def prepare_tree_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, OrdinalEncoder]:
    """Prepare features for Tree models (Random Forest, LightGBM, XGBoost).

    Encoding strategy:
    - Numeric features: Kept unscaled at raw native precision (native NaN handling).
    - Categorical features: Encoded via OrdinalEncoder fitted STRICTLY on X_train.
      Unknown or missing values are deterministically assigned -1.
    """
    logger.info(
        f"Fitting OrdinalEncoder strictly on X_train for {len(categorical_features)} categorical features..."
    )
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=-1,
    )

    # Format categoricals to clean string representation
    def clean_cat(df_in: pd.DataFrame) -> pd.DataFrame:
        out = df_in.copy()
        for c in categorical_features:
            out[c] = out[c].astype(str).replace({"nan": np.nan, "None": np.nan})
        return out

    X_train_clean = clean_cat(X_train)
    X_val_clean = clean_cat(X_val)
    X_test_clean = clean_cat(X_test)

    # Fit encoder strictly on train
    encoder.fit(X_train_clean[categorical_features])

    X_train_tree = X_train_clean.copy()
    X_val_tree = X_val_clean.copy()
    X_test_tree = X_test_clean.copy()

    X_train_tree[categorical_features] = encoder.transform(X_train_clean[categorical_features])
    X_val_tree[categorical_features] = encoder.transform(X_val_clean[categorical_features])
    X_test_tree[categorical_features] = encoder.transform(X_test_clean[categorical_features])

    # Convert all columns to float32 for memory efficiency and fast tree building
    X_train_tree = X_train_tree.astype(np.float32)
    X_val_tree = X_val_tree.astype(np.float32)
    X_test_tree = X_test_tree.astype(np.float32)

    logger.info(
        f"Tree features prepared: {X_train_tree.shape[1]} columns (240 raw numeric + 16 ordinal encoded)"
    )
    return X_train_tree, X_val_tree, X_test_tree, encoder


# ======================================================================
# Model Training Routines
# ======================================================================

def train_logistic_regression(
    X_train_trans: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
) -> tuple[LogisticRegression, float]:
    """Train baseline Logistic Regression classifier."""
    logger.info("Training Baseline Model 1: Logistic Regression...")
    model = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
    )
    t0 = time.time()
    model.fit(X_train_trans, y_train)
    duration = time.time() - t0
    logger.info(f"Logistic Regression trained in {duration:.2f}s")
    return model, duration


def train_random_forest(
    X_train_tree: pd.DataFrame,
    y_train: np.ndarray,
    random_state: int = 42,
) -> tuple[RandomForestClassifier, float]:
    """Train baseline Random Forest classifier."""
    logger.info("Training Baseline Model 2: Random Forest (300 trees)...")
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1,
    )
    t0 = time.time()
    model.fit(X_train_tree, y_train)
    duration = time.time() - t0
    logger.info(f"Random Forest trained in {duration:.2f}s")
    return model, duration


def train_lightgbm(
    X_train_tree: pd.DataFrame,
    y_train: np.ndarray,
    random_state: int = 42,
) -> tuple[LGBMClassifier, float]:
    """Train baseline LightGBM classifier."""
    logger.info("Training Baseline Model 3: LightGBM (300 trees)...")
    model = LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
    )
    t0 = time.time()
    model.fit(X_train_tree, y_train)
    duration = time.time() - t0
    logger.info(f"LightGBM trained in {duration:.2f}s")
    return model, duration


def train_xgboost(
    X_train_tree: pd.DataFrame,
    y_train: np.ndarray,
    random_state: int = 42,
) -> tuple[XGBClassifier, float]:
    """Train baseline XGBoost classifier."""
    logger.info("Training Baseline Model 4: XGBoost (300 trees)...")
    model = XGBClassifier(
        objective="binary:logistic",
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1,
        eval_metric="logloss",
    )
    t0 = time.time()
    model.fit(X_train_tree, y_train)
    duration = time.time() - t0
    logger.info(f"XGBoost trained in {duration:.2f}s")
    return model, duration


def extract_feature_importances(
    model: Any,
    feature_names: list[str],
) -> pd.DataFrame:
    """Extract native feature importances from fitted tree model."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        raise AttributeError("Model does not expose feature_importances_ or coef_")

    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)

    return imp_df


def save_model_artifact(
    model: Any,
    filename: str,
    output_dir: Union[str, pathlib.Path] = ARTIFACTS_MODELS_DIR,
) -> pathlib.Path:
    """Persist trained model artifact using joblib."""
    out_dir = pathlib.Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    joblib.dump(model, out_path)
    logger.info(f"Saved model artifact to {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")
    return out_path
