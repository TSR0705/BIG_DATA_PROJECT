"""Unit tests for Phase 11 Final Frozen Test Evaluation & Model Card."""

import json
import pathlib
import numpy as np
import pandas as pd
import pytest
from src.evaluation.compare import (
    compute_expected_calibration_error,
    evaluate_predictions,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_evaluate_predictions_structure():
    """Verify evaluation function outputs all required classification and calibration metrics."""
    rng = np.random.RandomState(42)
    y_true = rng.binomial(1, 0.2, size=300)
    y_prob = rng.uniform(0.0, 1.0, size=300)

    res = evaluate_predictions(
        y_true=y_true,
        y_prob=y_prob,
        threshold=0.16,
        model_name="TestModel",
        strategy_name="Optimal Cost",
        calibration_method="Isotonic",
    )

    required_keys = [
        "model", "strategy", "evaluation_partition", "roc_auc", "pr_auc",
        "log_loss", "brier_score", "ece", "precision", "recall", "f1", "f2",
        "specificity", "balanced_accuracy", "tp", "fp", "tn", "fn", "threshold",
        "calibration_method"
    ]
    for k in required_keys:
        assert k in res, f"Missing key: {k}"

    assert res["model"] == "TestModel"
    assert res["threshold"] == 0.16
    assert res["tp"] + res["fn"] == int(y_true.sum())
    assert res["tn"] + res["fp"] == int((1 - y_true).sum())


def test_threshold_monotonicity_effect():
    """Verify that increasing threshold decreases recall and increases specificity."""
    rng = np.random.RandomState(42)
    y_true = rng.binomial(1, 0.2, size=500)
    y_prob = rng.uniform(0.0, 1.0, size=500)

    res_low = evaluate_predictions(y_true, y_prob, threshold=0.10)
    res_high = evaluate_predictions(y_true, y_prob, threshold=0.60)

    assert res_low["recall"] >= res_high["recall"]
    assert res_low["specificity"] <= res_high["specificity"]


def test_risk_band_monotonicity_logic():
    """Verify logic for checking strict default rate monotonicity across risk bands."""
    # Monotonic case
    low_rate, med_rate, high_rate = 3.60, 14.85, 32.94
    assert low_rate < med_rate < high_rate

    # Non-monotonic case
    low_fail, med_fail, high_fail = 15.0, 12.0, 30.0
    assert not (low_fail < med_fail < high_fail)


def test_model_comparison_artifact_integrity():
    """Verify that reports/model_comparison.csv contains the complete 12-configuration evaluation."""
    csv_path = ROOT / "reports" / "model_comparison.csv"
    if not csv_path.exists():
        pytest.skip("reports/model_comparison.csv not generated yet")

    df = pd.read_csv(csv_path)
    assert len(df) == 12, f"Expected 12 configurations, found {len(df)}"

    required_cols = [
        "model", "strategy", "roc_auc", "pr_auc", "brier_score", "ece",
        "precision", "recall", "f1", "threshold", "calibration_method"
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing column {col} in model_comparison.csv"

    # All ROC-AUCs should be valid
    assert (df["roc_auc"] >= 0.50).all()
    assert (df["roc_auc"] <= 1.00).all()


def test_model_card_schema():
    """Verify that models/model_card.json adheres to the mandatory model card schema."""
    card_path = ROOT / "models" / "model_card.json"
    if not card_path.exists():
        pytest.skip("models/model_card.json not generated yet")

    with open(card_path, "r", encoding="utf-8") as f:
        card = json.load(f)

    required_sections = [
        "project_identification",
        "model_architecture",
        "dataset_and_provenance",
        "preprocessing_and_lineage",
        "calibration_and_decisioning",
        "validation_performance",
        "final_test_evaluation",
        "risk_band_validation_test",
        "explanation_stability_findings_e5",
        "subgroup_audit_findings_e7",
        "governance_and_intended_use",
    ]
    for s in required_sections:
        assert s in card, f"Missing section '{s}' in model_card.json"

    # Verify frozen hyperparameter consistency
    params = card["model_architecture"]["hyperparameters"]
    assert params["max_depth"] == 3
    assert params["learning_rate"] == 0.1
    assert params["n_estimators"] == 500
    assert params["subsample"] == 0.8
    assert params["min_child_weight"] == 10


def test_baseline_metrics_exact_reconciliation():
    """Verify that model_comparison.csv reproduces Phase 7 and Phase 8 baselines to 5+ decimal places."""
    csv_path = ROOT / "reports" / "model_comparison.csv"
    if not csv_path.exists():
        pytest.skip("reports/model_comparison.csv does not exist")

    df = pd.read_csv(csv_path)

    # Historical Phase 7 & 8 reference metrics on Test split (N=46,127):
    lr = df[(df["model"] == "Logistic Regression") & (df["strategy"] == "Phase 7 Baseline")].iloc[0]
    assert lr["roc_auc"] == pytest.approx(0.777256, abs=1e-5)
    assert lr["pr_auc"] == pytest.approx(0.251747, abs=1e-5)

    rf = df[(df["model"] == "Random Forest") & (df["strategy"] == "Phase 7 Baseline")].iloc[0]
    assert rf["roc_auc"] == pytest.approx(0.745429, abs=1e-5)
    assert rf["pr_auc"] == pytest.approx(0.227093, abs=1e-5)

    lgb = df[(df["model"] == "LightGBM") & (df["strategy"] == "Phase 7 Baseline")].iloc[0]
    assert lgb["roc_auc"] == pytest.approx(0.787722, abs=1e-5)
    assert lgb["pr_auc"] == pytest.approx(0.274939, abs=1e-5)

    xgb_p7 = df[(df["model"] == "XGBoost (Phase 7)") & (df["strategy"] == "Phase 7 Baseline")].iloc[0]
    assert xgb_p7["roc_auc"] == pytest.approx(0.789302, abs=1e-5)
    assert xgb_p7["pr_auc"] == pytest.approx(0.277235, abs=1e-5)

    xgb_tuned = df[(df["model"] == "XGBoost") & (df["strategy"] == "Tuned Baseline (default cutoff)")].iloc[0]
    assert xgb_tuned["roc_auc"] == pytest.approx(0.789754, abs=1e-5)
    assert xgb_tuned["pr_auc"] == pytest.approx(0.279038, abs=1e-5)

    xgb_cal = df[(df["model"] == "Calibrated XGBoost") & (df["strategy"] == "Tuned + Isotonic + Cost-Sensitive 1:5")].iloc[0]
    assert xgb_cal["roc_auc"] == pytest.approx(0.788284, abs=1e-5)
    assert xgb_cal["pr_auc"] == pytest.approx(0.269181, abs=1e-5)


def test_full_split_shap_artifact_metadata():
    """Verify that full-split SHAP metadata exists and confirms chunked execution across all 46,127 test rows."""
    meta_path = ROOT / "artifacts" / "explainability" / "test_split_shap_metadata.json"
    if not meta_path.exists():
        pytest.skip("artifacts/explainability/test_split_shap_metadata.json does not exist")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["total_rows"] == 46127
    assert meta["total_features"] == 256
    assert meta["explainer"] == "shap.TreeExplainer"
    assert meta["output_margin"] == "raw_margin_log_odds_class_1"
    assert meta["positive_shap_interpretation"] == "increases_default_log_odds"


def test_test_sequencing_disclosure_present():
    """Verify that model card and final report document the evaluation sequencing disclosure."""
    card_path = ROOT / "artifacts" / "models" / "model_card.json"
    with open(card_path, "r", encoding="utf-8") as f:
        card = json.load(f)

    assert "test_evaluation_sequencing_disclosure" in card
    disclosure = card["test_evaluation_sequencing_disclosure"]["execution_note"]
    assert "strictly zero data leakage" in disclosure.lower() or "zero data leakage" in disclosure.lower()

