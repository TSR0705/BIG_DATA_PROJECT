"""Unit tests for Phase 10 Experiment E4 (Decision Analysis & Cost-Sensitive Optimization)."""

import pathlib
import numpy as np
import pytest
import yaml

from src.evaluation.decision_analysis import (
    compute_cost_metrics,
    perform_risk_band_segmentation,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_cost_calculation_logic():
    """Verify cost calculation with known TP, FP, TN, FN counts."""
    # 10 defaulters, 90 non-defaulters
    y_true = np.array([1] * 10 + [0] * 90)
    # Model predicts 5 defaulters as default, 5 defaulters as non-default (FN=5)
    # Model predicts 10 non-defaulters as default (FP=10)
    p_pred = np.array([0.9] * 5 + [0.1] * 5 + [0.9] * 10 + [0.1] * 80)

    # Cost ratio 5:1 -> C_FN = 5, C_FP = 1
    # Expected cost = (5 * 5) + (1 * 10) = 25 + 10 = 35
    metrics_5_1 = compute_cost_metrics(y_true, p_pred, c_fn=5.0, c_fp=1.0, threshold=0.50)
    assert metrics_5_1["fn"] == 5
    assert metrics_5_1["fp"] == 10
    assert metrics_5_1["total_cost"] == 35.0
    assert metrics_5_1["cost_per_applicant"] == pytest.approx(35.0 / 100, abs=1e-5)

    # Cost ratio 10:1 -> C_FN = 10, C_FP = 1
    # Expected cost = (10 * 5) + (1 * 10) = 50 + 10 = 60
    metrics_10_1 = compute_cost_metrics(y_true, p_pred, c_fn=10.0, c_fp=1.0, threshold=0.50)
    assert metrics_10_1["total_cost"] == 60.0

    # Cost ratio 20:1 -> C_FN = 20, C_FP = 1
    # Expected cost = (20 * 5) + (1 * 10) = 100 + 10 = 110
    metrics_20_1 = compute_cost_metrics(y_true, p_pred, c_fn=20.0, c_fp=1.0, threshold=0.50)
    assert metrics_20_1["total_cost"] == 110.0


def test_risk_band_monotonicity_logic():
    """Verify risk band segmentation correctly flags monotonic default rates."""
    y_true = np.array([0] * 70 + [1] * 2 + [0] * 18 + [1] * 2 + [0] * 4 + [1] * 4)
    # LOW: < 0.09 (72 applicants, 2 defaults = 2.78%)
    # MEDIUM: 0.09 to 0.25 (20 applicants, 2 defaults = 10.0%)
    # HIGH: >= 0.25 (8 applicants, 4 defaults = 50.0%)
    p_pred = np.array([0.05] * 72 + [0.15] * 20 + [0.40] * 8)

    band_df = perform_risk_band_segmentation(y_true, p_pred)
    assert len(band_df) == 3
    rates = band_df["observed_default_rate_pct"].tolist()
    assert rates[0] < rates[1] < rates[2]
    assert band_df.iloc[0]["risk_band"] == "LOW"
    assert band_df.iloc[1]["risk_band"] == "MEDIUM"
    assert band_df.iloc[2]["risk_band"] == "HIGH"


def test_no_test_set_leakage():
    """Verify test split is never loaded or referenced in decision_analysis.py."""
    code_path = ROOT / "src" / "evaluation" / "decision_analysis.py"
    with open(code_path, "r", encoding="utf-8") as f:
        src = f.read()

    assert "test_indices.parquet" not in src
    assert "application_test.csv" not in src
    assert "model_input_test.parquet" not in src
