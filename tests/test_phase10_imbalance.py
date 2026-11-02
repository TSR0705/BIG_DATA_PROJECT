"""Unit tests for Phase 10 Experiment E3 (Class Imbalance × Calibration)."""

import pathlib
import numpy as np
import pytest
import yaml

from src.evaluation.imbalance_grid import (
    calculate_cox_calibration,
    calculate_ece,
    calculate_operational_metrics,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_e3_configurations_definition():
    """Verify that experiments.yaml and E3 grid specify exactly 13 configurations."""
    cfg_path = ROOT / "configs" / "experiments.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    e3_configs = cfg.get("e3_imbalance_grid", {}).get("configurations", [])
    assert len(e3_configs) == 13, f"Expected exactly 13 E3 configurations, got {len(e3_configs)}"

    # Check distribution of strategies
    strategies = [c["strategy"] for c in e3_configs]
    assert strategies.count("None") == 4
    assert strategies.count("Class Weight") == 3
    assert strategies.count("Positive-Class Weight") == 2
    assert strategies.count("SMOTE") == 4


def test_ece_calculation():
    """Verify Expected Calibration Error calculation on perfect and imperfect predictions."""
    # Perfectly calibrated case: 100 samples with prob 0.1, exactly 10 positives
    y_true_perfect = np.array([1] * 10 + [0] * 90)
    p_pred_perfect = np.array([0.1] * 100)
    ece_perf = calculate_ece(y_true_perfect, p_pred_perfect, n_bins=10)
    assert ece_perf == pytest.approx(0.0, abs=1e-5)

    # Completely uncalibrated case: prob 0.9 for all 100, but only 10 positives
    p_pred_bad = np.array([0.9] * 100)
    ece_bad = calculate_ece(y_true_perfect, p_pred_bad, n_bins=10)
    assert ece_bad == pytest.approx(0.8, abs=1e-5)


def test_cox_calibration_slope_and_intercept():
    """Verify Cox calibration returns valid slope and intercept."""
    np.random.seed(42)
    y_true = np.random.binomial(1, 0.1, size=500)
    p_pred = np.clip(np.random.beta(1, 9, size=500), 0.01, 0.99)

    slope, intercept = calculate_cox_calibration(y_true, p_pred)
    assert isinstance(slope, float)
    assert isinstance(intercept, float)
    assert not np.isnan(slope)
    assert not np.isnan(intercept)


def test_operational_metrics_correctness():
    """Verify classification metrics on a known confusion matrix."""
    # TP=40, FP=10, FN=10, TN=40
    y_true = np.array([1] * 50 + [0] * 50)
    p_pred = np.array([0.9] * 40 + [0.1] * 10 + [0.9] * 10 + [0.1] * 40)

    ops = calculate_operational_metrics(y_true, p_pred, threshold=0.50)
    assert ops["precision"] == pytest.approx(40 / 50, abs=1e-4)  # 0.80
    assert ops["recall"] == pytest.approx(40 / 50, abs=1e-4)     # 0.80
    assert ops["specificity"] == pytest.approx(40 / 50, abs=1e-4)  # 0.80
    assert ops["balanced_accuracy"] == pytest.approx(0.80, abs=1e-4)
    assert ops["f1"] == pytest.approx(0.80, abs=1e-4)
    assert ops["f2"] == pytest.approx(0.80, abs=1e-4)


def test_no_test_set_leakage():
    """Verify test split is never loaded or referenced in imbalance_grid.py."""
    code_path = ROOT / "src" / "evaluation" / "imbalance_grid.py"
    with open(code_path, "r", encoding="utf-8") as f:
        src = f.read()

    assert "test_indices.parquet" not in src
    assert "application_test.csv" not in src
    assert "model_input_test.parquet" not in src
