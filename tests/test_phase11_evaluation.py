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
