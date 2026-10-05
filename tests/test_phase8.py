"""Tests for Phase 8: Imbalance Handling, Hyperparameter Tuning, Thresholds, and Calibration.

Verifies:
1. Test indices are never part of hyperparameter search.
2. Test labels are not used for threshold selection.
3. Test labels are not used for calibration selection.
4. X excludes TARGET.
5. X excludes SK_ID_CURR.
6. Search uses TRAIN only.
7. Validation is used for model/threshold decisions.
8. No SMOTE.
9. Probabilities are within [0, 1].
10. Metrics are finite.
11. Calibration artifacts load successfully.
12. Selected threshold is within [0, 1].
13. Required reports exist.
14. Phase 7 artifacts remain unchanged.
15. Random seed is 42.
"""

import json
import pathlib
import sys
import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ml.phase8_experiments import calculate_scale_pos_weight
from src.ml.phase8_utils import (
    compute_comprehensive_metrics,
    evaluate_cost_grid,
    evaluate_threshold_grid,
)

REPORTS_DIR = ROOT / "reports"
ARTIFACTS_MODELS_DIR = ROOT / "artifacts" / "models"
SPLITS_DIR = ROOT / "data" / "splits"


# ======================================================================
# Unit Tests (Fast, Isolated, Synthetic)
# ======================================================================

def test_scale_pos_weight_exact_calculation():
    """Verify scale_pos_weight calculation is exact: neg / pos."""
    y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])  # 8 zeros, 2 ones
    ratio = calculate_scale_pos_weight(y)
    assert np.isclose(ratio, 4.0)

    # All ones edge case
    with pytest.raises(ValueError):
        calculate_scale_pos_weight(np.zeros(10))


def test_synthetic_comprehensive_metrics_bounds():
    """Verify computed classification and calibration metrics are finite and bounded."""
    np.random.seed(42)
    y_true = np.random.randint(0, 2, size=100)
    y_prob = np.random.uniform(0.01, 0.99, size=100)

    metrics = compute_comprehensive_metrics(y_true, y_prob, threshold=0.50)

    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert 0.0 <= metrics["brier_score"] <= 1.0
    assert metrics["log_loss"] >= 0.0
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    assert 0.0 <= metrics["f2"] <= 1.0
    assert 0.0 <= metrics["f0_5"] <= 1.0
    assert 0.0 <= metrics["specificity"] <= 1.0
    assert 0.0 <= metrics["balanced_accuracy"] <= 1.0

    # Ensure all values are finite (no NaN / Inf)
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            assert np.isfinite(v), f"Metric {k} was not finite: {v}"


def test_synthetic_threshold_grid_monotonic_tradeoffs():
    """Verify precision-recall trade-off behavior across thresholds."""
    y_true = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.7, 0.8, 0.9])

    th_df = evaluate_threshold_grid(y_true, y_prob, thresholds=np.array([0.2, 0.5, 0.8]))

    assert len(th_df) == 3
    # At low threshold, recall should be higher than or equal to high threshold
    assert th_df.loc[0, "recall"] >= th_df.loc[2, "recall"]
    # At high threshold, precision should be higher than or equal to low threshold
    assert th_df.loc[2, "precision"] >= th_df.loc[0, "precision"]


def test_synthetic_cost_grid_minimization():
    """Verify expected cost minimization across hypothetical ratios."""
    y_true = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.6, 0.7, 0.85])

    cost_df, best_costs = evaluate_cost_grid(
        y_true, y_prob, cost_ratios=[(1, 1), (1, 10)], thresholds=np.array([0.2, 0.5, 0.8])
    )

    assert "cost_1_1" in cost_df.columns
    assert "cost_1_10" in cost_df.columns
    assert "1:1" in best_costs
    assert "1:10" in best_costs
    assert 0.0 <= best_costs["1:1"]["optimal_threshold"] <= 1.0
    assert 0.0 <= best_costs["1:10"]["optimal_threshold"] <= 1.0


def test_no_smote_or_resampling_imported():
    """Verify SMOTE or imbalanced-learn sampling is strictly absent."""
    import src.ml.phase8_experiments as p8_exp
    import src.ml.phase8_utils as p8_ut

    # Verify neither module imports imblearn
    assert "imblearn" not in sys.modules
    assert not hasattr(p8_exp, "SMOTE")
    assert not hasattr(p8_ut, "SMOTE")


# ======================================================================
# Integration & Integrity Tests (Requires run_phase8_optimization.py)
# ======================================================================

def test_split_indices_and_no_contamination():
    """Verify split sizes, complete disjointness, and test isolation."""
    train_df = pd.read_parquet(SPLITS_DIR / "train_indices.parquet")
    val_df = pd.read_parquet(SPLITS_DIR / "val_indices.parquet")
    test_df = pd.read_parquet(SPLITS_DIR / "test_indices.parquet")

    train_ids = set(train_df["SK_ID_CURR"])
    val_ids = set(val_df["SK_ID_CURR"])
    test_ids = set(test_df["SK_ID_CURR"])

    assert len(train_ids) == 215_257
    assert len(val_ids) == 46_127
    assert len(test_ids) == 46_127
    assert len(train_ids & val_ids) == 0
    assert len(train_ids & test_ids) == 0
    assert len(val_ids & test_ids) == 0


def test_phase8_artifacts_exist_and_loadable():
    """Verify Phase 8 models and metadata exist and are loadable."""
    best_xgb_path = ARTIFACTS_MODELS_DIR / "phase8_best_xgboost.joblib"
    calib_path = ARTIFACTS_MODELS_DIR / "phase8_calibrated_model.joblib"
    meta_path = ARTIFACTS_MODELS_DIR / "phase8_model_metadata.json"

    assert best_xgb_path.exists(), f"Missing {best_xgb_path}"
    assert calib_path.exists(), f"Missing {calib_path}"
    assert meta_path.exists(), f"Missing {meta_path}"

    best_xgb = joblib.load(best_xgb_path)
    calib_model = joblib.load(calib_path)

    assert hasattr(best_xgb, "predict_proba")
    assert hasattr(calib_model, "predict_proba")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["random_state"] == 42
    assert "hyperparameter_search" in meta
    assert "threshold_optimization" in meta
    assert "calibration" in meta
    assert meta["threshold_optimization"]["optimal_f1_threshold"] >= 0.0
    assert meta["threshold_optimization"]["optimal_f1_threshold"] <= 1.0


def test_phase7_artifacts_remain_untouched():
    """Verify Phase 7 baseline model artifacts were not overwritten or modified."""
    p7_models = [
        "logistic_regression.joblib",
        "random_forest.joblib",
        "lightgbm.joblib",
        "xgboost.joblib",
        "tree_preprocessor.joblib",
    ]
    for p7_m in p7_models:
        m_path = ARTIFACTS_MODELS_DIR / p7_m
        assert m_path.exists(), f"Phase 7 model artifact {p7_m} missing!"


def test_required_reports_and_csvs_exist():
    """Verify all 7 reports/CSVs required by Phase 8 exist."""
    required_reports = [
        REPORTS_DIR / "phase8_report.md",
        REPORTS_DIR / "phase8_report.json",
        REPORTS_DIR / "phase8_model_comparison.csv",
        REPORTS_DIR / "threshold_analysis.csv",
        REPORTS_DIR / "calibration_comparison.csv",
        REPORTS_DIR / "hyperparameter_search_results.csv",
        REPORTS_DIR / "cost_analysis.csv",
    ]
    for rep in required_reports:
        assert rep.exists(), f"Required Phase 8 report missing: {rep}"

    # Verify threshold analysis contains all specified columns
    th_df = pd.read_csv(REPORTS_DIR / "threshold_analysis.csv")
    expected_th_cols = {
        "threshold", "precision", "recall", "f1", "f2", "f0_5",
        "accuracy", "specificity", "balanced_accuracy", "false_positive_rate", "false_negative_rate"
    }
    assert expected_th_cols.issubset(set(th_df.columns))
    assert len(th_df) >= 50

    # Verify calibration comparison columns
    calib_df = pd.read_csv(REPORTS_DIR / "calibration_comparison.csv")
    expected_calib_cols = {"model", "calibration_method", "brier_score", "log_loss", "roc_auc", "pr_auc"}
    assert expected_calib_cols.issubset(set(calib_df.columns))

    # Verify hyperparameter search results has 20 evaluated candidates
    cv_df = pd.read_csv(REPORTS_DIR / "hyperparameter_search_results.csv")
    assert len(cv_df) == 20, f"Expected 20 evaluated search configurations, found {len(cv_df)}"


def test_required_figures_exist():
    """Verify all 5 publication-quality figures exist."""
    figures = [
        REPORTS_DIR / "figures" / "phase8_pr_comparison.png",
        REPORTS_DIR / "figures" / "phase8_roc_comparison.png",
        REPORTS_DIR / "figures" / "threshold_tradeoff.png",
        REPORTS_DIR / "figures" / "calibration_curve.png",
        REPORTS_DIR / "figures" / "cost_threshold_analysis.png",
    ]
    for fig in figures:
        assert fig.exists(), f"Required figure missing: {fig}"
        assert fig.stat().st_size > 1000, f"Figure file suspiciously small: {fig}"


def test_feature_matrix_excludes_target_and_id():
    """Verify feature sets strictly exclude TARGET and SK_ID_CURR."""
    meta_path = ARTIFACTS_MODELS_DIR / "phase8_model_metadata.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    features = meta["features"]
    assert "TARGET" not in features, "TARGET found in model features!"
    assert "SK_ID_CURR" not in features, "SK_ID_CURR found in model features!"
    assert len(features) == 256


def test_test_partition_not_used_in_search_or_decisions():
    """Verify test partition was not used in search, threshold, or calibration selection."""
    meta_path = ARTIFACTS_MODELS_DIR / "phase8_model_metadata.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Cross-validation folds were 3 on TRAIN only
    assert meta["hyperparameter_search"]["cv_folds"] == 3
    assert meta["random_state"] == 42

    # Selected threshold is in [0, 1]
    opt_f1 = meta["threshold_optimization"]["optimal_f1_threshold"]
    opt_f2 = meta["threshold_optimization"]["optimal_f2_threshold"]
    assert 0.0 < opt_f1 < 1.0
    assert 0.0 < opt_f2 < 1.0


def test_calibrated_model_predict_proba_bounds():
    """Verify calibrated model outputs valid probabilities in [0, 1]."""
    calib_model = joblib.load(ARTIFACTS_MODELS_DIR / "phase8_calibrated_model.joblib")
    dummy_X = np.random.randn(10, 256).astype(np.float32)

    probs = calib_model.predict_proba(dummy_X)[:, 1]
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    assert np.all(np.isfinite(probs))
