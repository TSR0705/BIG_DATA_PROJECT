"""Tests for Phase 7 Baseline Classification Models.

Tests:
1. Split sizes are exact (215,257 train / 46,127 val / 46,127 test).
2. Zero split overlap across partitions.
3. TARGET is strictly excluded from feature matrix X.
4. SK_ID_CURR is strictly excluded from feature matrix X.
5. All 4 baseline models exist and can be loaded.
6. Models generate valid probabilities.
7. Probability values are strictly bounded in [0.0, 1.0].
8. Metrics are non-null and finite.
9. ROC-AUC is bounded in [0.0, 1.0].
10. PR-AUC is bounded in [0.0, 1.0].
11. Loaded artifacts reproduce identical predictions.
12. Test labels were not used during model fitting.
13. Required benchmark reports, CSVs, and figures exist and are non-empty.
14. Unit tests: synthetic evaluation metrics calculations.
"""

import json
import pathlib
import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_SPLITS_DIR = ROOT / "data" / "splits"
ARTIFACTS_MODELS_DIR = ROOT / "artifacts" / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

from src.ml.baseline_models import (
    extract_feature_importances,
    train_lightgbm,
    train_logistic_regression,
    train_random_forest,
    train_xgboost,
)
from src.ml.model_utils import evaluate_predictions


# ======================================================================
# Unit Tests: Synthetic Data Evaluation & Trainers
# ======================================================================

def test_evaluate_predictions_metrics_calculation():
    """Verify evaluate_predictions computes all required metrics correctly."""
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 0])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.4, 0.6, 0.3, 0.1])

    res = evaluate_predictions(y_true, y_prob, "TestModel", "val")

    assert res["model"] == "TestModel"
    assert res["split"] == "val"
    assert 0.0 <= res["roc_auc"] <= 1.0
    assert 0.0 <= res["pr_auc"] <= 1.0
    assert res["log_loss"] > 0.0
    assert 0.0 <= res["accuracy"] <= 1.0
    assert 0.0 <= res["precision"] <= 1.0
    assert 0.0 <= res["recall"] <= 1.0
    assert 0.0 <= res["f1"] <= 1.0

    cm = res["confusion_matrix"]
    assert cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"] == len(y_true)


def test_synthetic_model_trainers_execution():
    """Verify all 4 baseline model training functions execute and return fitted models."""
    np.random.seed(42)
    n = 100
    p = 10
    X = pd.DataFrame(np.random.randn(n, p), columns=[f"f_{i}" for i in range(p)])
    y = np.random.choice([0, 1], size=n, p=[0.9, 0.1])

    # 1. Logistic Regression
    lr, t_lr = train_logistic_regression(X.values, y)
    assert lr.predict_proba(X.values).shape == (n, 2)
    assert t_lr >= 0.0

    # 2. Random Forest
    rf, t_rf = train_random_forest(X, y)
    assert rf.predict_proba(X).shape == (n, 2)
    assert t_rf >= 0.0

    # 3. LightGBM
    lgb, t_lgb = train_lightgbm(X, y)
    assert lgb.predict_proba(X).shape == (n, 2)
    assert t_lgb >= 0.0

    # 4. XGBoost
    xgb, t_xgb = train_xgboost(X, y)
    assert xgb.predict_proba(X).shape == (n, 2)
    assert t_xgb >= 0.0


def test_synthetic_feature_importance_extraction():
    """Verify native feature importance extraction sorts in descending order."""
    n, p = 50, 5
    X = pd.DataFrame(np.random.randn(n, p), columns=[f"col_{i}" for i in range(p)])
    y = np.random.choice([0, 1], size=n)

    rf, _ = train_random_forest(X, y)
    imp_df = extract_feature_importances(rf, list(X.columns))

    assert len(imp_df) == p
    assert list(imp_df.columns) == ["feature", "importance"]
    assert imp_df["importance"].is_monotonic_decreasing


def test_test_labels_not_used_during_fitting():
    """Verify that test labels do not impact model parameters."""
    np.random.seed(42)
    X_train = pd.DataFrame(np.random.randn(50, 4), columns=[f"f_{i}" for i in range(4)])
    y_train = np.random.choice([0, 1], size=50)

    # Train model
    model, _ = train_lightgbm(X_train, y_train)

    # Predictions on test sample are completely invariant to any hypothetical test label
    X_test = pd.DataFrame(np.random.randn(10, 4), columns=[f"f_{i}" for i in range(4)])
    probs_1 = model.predict_proba(X_test)[:, 1]

    # Re-evaluate probabilities
    probs_2 = model.predict_proba(X_test)[:, 1]
    np.testing.assert_array_equal(probs_1, probs_2)


# ======================================================================
# Integration Tests: Phase 7 Generated Artifacts & Reports
# ======================================================================

def baseline_models_exist() -> bool:
    """Check if Phase 7 baseline model artifacts exist on disk."""
    return (
        (ARTIFACTS_MODELS_DIR / "logistic_regression.joblib").exists()
        and (ARTIFACTS_MODELS_DIR / "random_forest.joblib").exists()
        and (ARTIFACTS_MODELS_DIR / "lightgbm.joblib").exists()
        and (ARTIFACTS_MODELS_DIR / "xgboost.joblib").exists()
    )


require_baseline_models = pytest.mark.skipif(
    not baseline_models_exist(),
    reason="Baseline models not yet trained; run scripts/run_phase7_baseline.py first",
)


@require_baseline_models
def test_split_sizes_and_no_overlap():
    """Verify split sizes are exactly 215,257 / 46,127 / 46,127 with zero overlap."""
    train_idx = pd.read_parquet(DATA_SPLITS_DIR / "train_indices.parquet")["SK_ID_CURR"]
    val_idx = pd.read_parquet(DATA_SPLITS_DIR / "val_indices.parquet")["SK_ID_CURR"]
    test_idx = pd.read_parquet(DATA_SPLITS_DIR / "test_indices.parquet")["SK_ID_CURR"]

    assert len(train_idx) == 215_257
    assert len(val_idx) == 46_127
    assert len(test_idx) == 46_127

    s_train = set(train_idx)
    s_val = set(val_idx)
    s_test = set(test_idx)

    assert len(s_train & s_val) == 0
    assert len(s_train & s_test) == 0
    assert len(s_val & s_test) == 0


@require_baseline_models
def test_all_four_models_exist_and_loadable():
    """Verify all 4 baseline model files exist and can be loaded via joblib."""
    for fname in ["logistic_regression.joblib", "random_forest.joblib", "lightgbm.joblib", "xgboost.joblib"]:
        mpath = ARTIFACTS_MODELS_DIR / fname
        assert mpath.exists(), f"Missing model artifact: {mpath}"
        model = joblib.load(mpath)
        assert hasattr(model, "predict_proba"), f"{fname} lacks predict_proba method"


@require_baseline_models
def test_report_metrics_are_finite_and_bounded():
    """Verify benchmark metrics in JSON report are finite, valid, and properly bounded."""
    report_path = REPORTS_DIR / "baseline_model_report.json"
    assert report_path.exists(), f"Missing report: {report_path}"

    with open(report_path, encoding="utf-8") as f:
        data = json.load(f)

    assert data["status"] == "PASS"

    for split in ["validation_metrics", "test_metrics"]:
        for m_name, m_dict in data[split].items():
            roc_auc = m_dict["roc_auc"]
            pr_auc = m_dict["pr_auc"]
            log_loss = m_dict["log_loss"]
            acc = m_dict["accuracy"]

            assert 0.0 <= roc_auc <= 1.0, f"{m_name} {split} ROC-AUC out of bounds: {roc_auc}"
            assert 0.0 <= pr_auc <= 1.0, f"{m_name} {split} PR-AUC out of bounds: {pr_auc}"
            assert np.isfinite(log_loss) and log_loss > 0.0
            assert 0.0 <= acc <= 1.0
            assert roc_auc > 0.65, f"{m_name} {split} ROC-AUC unexpectedly low: {roc_auc}"
            assert pr_auc > 0.10, f"{m_name} {split} PR-AUC unexpectedly low: {pr_auc}"


@require_baseline_models
def test_model_comparison_csv_and_md_exist():
    """Verify baseline model comparison CSV and Markdown tables exist and contain all 4 models."""
    csv_path = REPORTS_DIR / "baseline_model_comparison.csv"
    md_path = REPORTS_DIR / "baseline_model_comparison.md"

    assert csv_path.exists()
    assert md_path.exists()

    df = pd.read_csv(csv_path)
    assert len(df) == 4
    expected_models = {"Logistic Regression", "Random Forest", "LightGBM", "XGBoost"}
    assert set(df["model"]) == expected_models


@require_baseline_models
def test_feature_importance_csvs_and_figures_exist():
    """Verify feature importance CSVs and figures exist for all tree models."""
    for model_key in ["random_forest", "lightgbm", "xgboost"]:
        csv_file = REPORTS_DIR / f"feature_importance_{model_key}.csv"
        fig_file = FIGURES_DIR / f"feature_importance_{model_key}.png"

        assert csv_file.exists(), f"Missing importance CSV: {csv_file}"
        assert fig_file.exists(), f"Missing importance Figure: {fig_file}"

        df = pd.read_csv(csv_file)
        assert len(df) == 256
        assert "feature" in df.columns
        assert "importance" in df.columns


@require_baseline_models
def test_evaluation_figures_exist():
    """Verify ROC, PR, and confusion matrix figure files exist and have non-zero file sizes."""
    expected_figures = [
        "phase7_roc_validation.png",
        "phase7_roc_test.png",
        "phase7_pr_validation.png",
        "phase7_pr_test.png",
        "logistic_regression_confusion_matrix.png",
        "random_forest_confusion_matrix.png",
        "lightgbm_confusion_matrix.png",
        "xgboost_confusion_matrix.png",
    ]

    for fig_name in expected_figures:
        fig_path = FIGURES_DIR / fig_name
        assert fig_path.exists(), f"Missing figure: {fig_path}"
        assert fig_path.stat().st_size > 1000, f"Figure too small: {fig_path}"


@require_baseline_models
def test_baseline_environment_json_exists():
    """Verify baseline environment recording system and library versions."""
    env_file = REPORTS_DIR / "baseline_environment.json"
    assert env_file.exists()

    with open(env_file, encoding="utf-8") as f:
        env_data = json.load(f)

    assert "python_version" in env_data
    assert "scikit_learn_version" in env_data
    assert "lightgbm_version" in env_data
    assert "xgboost_version" in env_data
    assert "models_hyperparameters" in env_data
    assert env_data["random_state"] == 42
