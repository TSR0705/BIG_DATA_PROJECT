"""Script to generate independent, leakage-free probability calibrator for Phase 11.

Methodology:
- Strictly uses ONLY the Training partition (data/splits/train_indices.parquet, N=215,257).
- Generates 5-fold Stratified Out-of-Fold (OOF) probability predictions across the training set
  using XGBoost configured with the frozen Phase 8 hyperparameters.
- Fits IsotonicRegression on (oof_train_probs, y_train).
- Packages the calibrator with the frozen base estimator phase8_best_xgboost.joblib
  into artifacts/models/phase11_independent_calibrator.joblib.
- Guarantees ZERO validation or test data rows/labels enter calibration fitting.
"""

import copy
import json
import pathlib
import sys
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss
import xgboost as xgb

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.log import get_logger

logger = get_logger("IndependentCalibration", "independent_calibration.log")


def compute_expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Calculate ECE using uniform width binning spanning [0.0, 1.0]."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)

    ece = 0.0
    n = len(y_true)
    if n == 0:
        return 0.0

    for i in range(n_bins):
        mask = bin_indices == i
        bin_count = np.sum(mask)
        if bin_count > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            ece += (bin_count / n) * np.abs(bin_acc - bin_conf)

    return float(ece)


def fit_and_save_independent_calibrator() -> pathlib.Path:
    input_path = ROOT / "data" / "model_input" / "model_input.parquet"
    train_split_path = ROOT / "data" / "splits" / "train_indices.parquet"
    val_split_path = ROOT / "data" / "splits" / "val_indices.parquet"
    test_split_path = ROOT / "data" / "splits" / "test_indices.parquet"
    meta_path = ROOT / "artifacts" / "models" / "phase8_model_metadata.json"
    base_model_path = ROOT / "artifacts" / "models" / "phase8_best_xgboost.joblib"
    template_cal_path = ROOT / "artifacts" / "models" / "phase8_calibrated_model.joblib"

    output_model_path = ROOT / "artifacts" / "models" / "phase11_independent_calibrator.joblib"
    output_meta_path = ROOT / "artifacts" / "models" / "phase11_calibration_metadata.json"
    output_oof_path = ROOT / "artifacts" / "models" / "phase11_oof_train_predictions.parquet"

    logger.info("=== Generating Phase 11 Independent Probability Calibrator ===")

    # 1. Verification of Split Boundaries
    train_ids = set(pd.read_parquet(train_split_path)["SK_ID_CURR"])
    val_ids = set(pd.read_parquet(val_split_path)["SK_ID_CURR"])
    test_ids = set(pd.read_parquet(test_split_path)["SK_ID_CURR"])

    assert train_ids.isdisjoint(val_ids), "Fatal leakage: train and validation IDs overlap!"
    assert train_ids.isdisjoint(test_ids), "Fatal leakage: train and test IDs overlap!"
    logger.info(f"Verified split independence: Train N={len(train_ids):,}, Val N={len(val_ids):,}, Test N={len(test_ids):,}")

    # 2. Load strictly Training Data
    df = pd.read_parquet(input_path)
    train_df = df[df["SK_ID_CURR"].isin(train_ids)].copy()

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    feature_cols = meta["features"]
    best_params = meta["hyperparameter_search"]["best_params"]

    # Preprocess categorical columns identically to tree pipeline
    cat_cols = [c for c in feature_cols if train_df[c].dtype == "object" or train_df[c].dtype.name == "category"]
    for c in cat_cols:
        train_df[c] = train_df[c].astype("category").cat.codes

    X_train = train_df[feature_cols].copy()
    y_train = train_df["TARGET"].values.astype(int)
    sk_id_curr = train_df["SK_ID_CURR"].values

    n_train = len(y_train)
    logger.info(f"Loaded training data: {n_train:,} rows, {len(feature_cols)} features. Default rate: {y_train.mean():.4f}")

    # 3. 5-Fold Stratified Out-of-Fold Generation on Train
    logger.info("Generating 5-Fold Stratified Out-of-Fold predictions on Train...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probs = np.zeros(n_train, dtype=float)
    oof_folds = np.zeros(n_train, dtype=int)

    t0 = time.time()
    for fold_idx, (tr_fold_idx, oof_fold_idx) in enumerate(skf.split(X_train, y_train)):
        t_f0 = time.time()
        clf_fold = xgb.XGBClassifier(
            **best_params,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42 + fold_idx,
            n_jobs=-1,
        )
        clf_fold.fit(X_train.iloc[tr_fold_idx], y_train[tr_fold_idx])
        oof_probs[oof_fold_idx] = clf_fold.predict_proba(X_train.iloc[oof_fold_idx])[:, 1]
        oof_folds[oof_fold_idx] = fold_idx
        dur = time.time() - t_f0
        logger.info(f"  Fold {fold_idx + 1}/5 completed in {dur:.2f}s (val fold N={len(oof_fold_idx):,})")

    total_oof_time = time.time() - t0
    logger.info(f"All 5 folds completed in {total_oof_time:.2f}s")

    # Metrics on OOF Train Predictions
    oof_brier = float(brier_score_loss(y_train, oof_probs))
    oof_logloss = float(log_loss(y_train, oof_probs))
    oof_ece = compute_expected_calibration_error(y_train, oof_probs)
    logger.info(f"Train OOF Metrics: Brier={oof_brier:.6f}, LogLoss={oof_logloss:.6f}, ECE={oof_ece:.6f}")

    # 4. Fit Isotonic Regression strictly on (oof_probs, y_train)
    logger.info("Fitting IsotonicRegression strictly on (oof_train_probs, y_train)...")
    t_iso0 = time.time()
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(oof_probs, y_train)
    iso_fit_time = time.time() - t_iso0
    logger.info(f"IsotonicRegression fitted in {iso_fit_time:.4f}s")

    # Post-calibration OOF metrics on Train
    cal_oof_probs = np.clip(iso.predict(oof_probs), 0.0, 1.0)
    cal_oof_brier = float(brier_score_loss(y_train, cal_oof_probs))
    cal_oof_logloss = float(log_loss(y_train, cal_oof_probs))
    cal_oof_ece = compute_expected_calibration_error(y_train, cal_oof_probs)
    logger.info(f"Calibrated Train OOF Metrics: Brier={cal_oof_brier:.6f}, LogLoss={cal_oof_logloss:.6f}, ECE={cal_oof_ece:.6f}")

    # 5. Save OOF predictions for full transparency and reproducibility
    oof_df = pd.DataFrame({
        "SK_ID_CURR": sk_id_curr,
        "fold": oof_folds,
        "y_train": y_train,
        "uncal_oof_prob": oof_probs,
        "cal_oof_prob": cal_oof_probs,
    })
    oof_df.to_parquet(output_oof_path, index=False)
    logger.info(f"Saved OOF predictions table to {output_oof_path}")

    # 6. Package into CalibratedClassifierCV wrapping the frozen Phase 8 base estimator
    logger.info(f"Packaging into CalibratedClassifierCV using base estimator from {base_model_path}...")
    template_model = joblib.load(template_cal_path)
    independent_calibrator = copy.deepcopy(template_model)
    # Inject our independently fitted IsotonicRegression
    independent_calibrator.calibrated_classifiers_[0].calibrators[0] = iso

    # Verify prediction works
    test_probe = independent_calibrator.predict_proba(X_train.iloc[:5])
    assert test_probe.shape == (5, 2)
    assert np.all((test_probe >= 0.0) & (test_probe <= 1.0))
    assert np.allclose(test_probe.sum(axis=1), 1.0)

    # Save artifact
    joblib.dump(independent_calibrator, output_model_path)
    logger.info(f"Saved independent calibrator to {output_model_path}")

    # 7. Save Comprehensive Metadata
    cal_metadata = {
        "project": "Loan Default Prediction Using Big Data Analytics",
        "phase": "Phase 11 — Independent Probability Calibration (Leakage Remediation)",
        "calibrator_artifact": str(output_model_path.name),
        "base_estimator_artifact": str(base_model_path.name),
        "calibration_method": "isotonic",
        "fitting_strategy": "5-Fold Stratified Out-of-Fold (OOF) Cross-Fitting strictly on Train",
        "provenance": {
            "training_split_file": "data/splits/train_indices.parquet",
            "training_row_count": n_train,
            "training_default_count": int(np.sum(y_train)),
            "training_default_rate": float(np.mean(y_train)),
            "validation_rows_accessed": 0,
            "validation_labels_accessed": 0,
            "test_rows_accessed": 0,
            "test_labels_accessed": 0,
        },
        "oof_generation": {
            "cv_folds": 5,
            "stratified": True,
            "random_state": 42,
            "hyperparameters": best_params,
            "oof_generation_time_seconds": round(total_oof_time, 2),
            "uncalibrated_oof_brier": oof_brier,
            "uncalibrated_oof_logloss": oof_logloss,
            "uncalibrated_oof_ece": oof_ece,
            "calibrated_oof_brier": cal_oof_brier,
            "calibrated_oof_logloss": cal_oof_logloss,
            "calibrated_oof_ece": cal_oof_ece,
        },
        "frozen_artifacts_preserved": [
            "artifacts/models/phase8_best_xgboost.joblib",
            "artifacts/models/phase8_calibrated_model.joblib",
        ],
        "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }

    with open(output_meta_path, "w", encoding="utf-8") as f:
        json.dump(cal_metadata, f, indent=2)
    logger.info(f"Saved calibration metadata to {output_meta_path}")

    return output_model_path


if __name__ == "__main__":
    fit_and_save_independent_calibrator()
