"""Unit tests for Phase 11 Experiment E7 (Applicant Subgroup Performance Audit)."""

import numpy as np
import pytest
from src.evaluation.subgroup import (
    compute_expected_calibration_error,
    compute_metrics_for_subgroup,
    compute_bootstrap_confidence_intervals,
)


def test_ece_calculation():
    """Verify Expected Calibration Error calculation behaves correctly."""
    # Perfectly calibrated case: 100 samples with prob=0.1 having 10% positive, 100 with prob=0.9 having 90% positive
    y_prob = np.concatenate([np.full(100, 0.1), np.full(100, 0.9)])
    y_true = np.concatenate([np.array([1]*10 + [0]*90), np.array([1]*90 + [0]*10)])
    ece = compute_expected_calibration_error(y_true, y_prob, n_bins=10)
    assert pytest.approx(ece, abs=0.05) == 0.0

    # Inverted calibration: prob=0.9 for all 0s, prob=0.1 for all 1s
    y_prob_inv = np.concatenate([np.full(100, 0.9), np.full(100, 0.1)])
    ece_inv = compute_expected_calibration_error(y_true, y_prob_inv, n_bins=10)
    assert ece_inv > 0.5


def test_subgroup_metrics_multiclass():
    """Verify subgroup metric calculation returns all expected fields for 2-class data."""
    rng = np.random.RandomState(42)
    y_true = rng.binomial(1, 0.2, size=500)
    y_prob = rng.uniform(0, 1, size=500)

    res = compute_metrics_for_subgroup(y_true, y_prob, threshold=0.16)

    assert res["n"] == 500
    assert res["defaults"] == int(y_true.sum())
    assert 0.0 <= res["default_rate"] <= 1.0
    assert 0.0 <= res["roc_auc"] <= 1.0
    assert 0.0 <= res["pr_auc"] <= 1.0
    assert 0.0 <= res["brier_score"] <= 1.0
    assert 0.0 <= res["ece"] <= 1.0
    assert 0.0 <= res["precision"] <= 1.0
    assert 0.0 <= res["recall"] <= 1.0
    assert 0.0 <= res["f1"] <= 1.0
    assert 0.0 <= res["specificity"] <= 1.0
    assert "Standard multi-class" in res["note"]


def test_subgroup_metrics_single_class():
    """Verify single-class slice sets ROC-AUC to None gracefully without exception."""
    y_true = np.zeros(200, dtype=int)
    y_prob = np.random.uniform(0, 0.3, size=200)

    res = compute_metrics_for_subgroup(y_true, y_prob, threshold=0.16)

    assert res["n"] == 200
    assert res["defaults"] == 0
    assert res["default_rate"] == 0.0
    assert res["roc_auc"] is None
    assert res["pr_auc"] is None
    assert res["f1"] is None
    assert res["recall"] is None
    assert "Single-class" in res["note"]


def test_bootstrap_confidence_intervals():
    """Verify reproducible percentile bootstrap CI generation."""
    rng = np.random.RandomState(42)
    y_true = rng.binomial(1, 0.15, size=400)
    y_prob = rng.uniform(0, 0.8, size=400)

    ci_1 = compute_bootstrap_confidence_intervals(y_true, y_prob, threshold=0.16, n_bootstraps=50, seed=42)
    ci_2 = compute_bootstrap_confidence_intervals(y_true, y_prob, threshold=0.16, n_bootstraps=50, seed=42)

    # Determinism
    assert ci_1["roc_auc"] == ci_2["roc_auc"]
    assert ci_1["pr_auc"] == ci_2["pr_auc"]

    # Bounds validity (lower <= upper)
    low, high = ci_1["roc_auc"]
    assert low is not None and high is not None
    assert low <= high


def test_age_band_partitioning():
    """Verify age band definitions are mutually exclusive and collectively exhaustive."""
    ages = np.array([18.0, 24.99, 25.0, 39.99, 40.0, 54.99, 55.0, 70.0])
    bands = np.full(len(ages), "Other", dtype=object)

    bands[ages < 25] = "Under 25"
    bands[(ages >= 25) & (ages < 40)] = "25-40"
    bands[(ages >= 40) & (ages < 55)] = "40-55"
    bands[ages >= 55] = "55+"

    expected = ["Under 25", "Under 25", "25-40", "25-40", "40-55", "40-55", "55+", "55+"]
    assert list(bands) == expected
    assert "Other" not in bands


def test_subgroup_metrics_single_class_n1():
    """Verify single-class subgroup with n=1 and zero positive cases (CODE_GENDER=None cohort).

    Verifies:
    - Recall is undefined (None / NA) because TP + FN = 0.
    - Discrimination metrics (ROC-AUC, PR-AUC) are undefined (None).
    - Precision is 0.0 when threshold predicts positive (0 TP, 1 FP).
    - Brier score: (y - p)^2 = (0 - 0.291874)^2 = 0.08519.
    - ECE: |y - p| = |0 - 0.291874| = 0.29187.
    """
    y_true = np.array([0])
    y_prob = np.array([0.291874])
    threshold = 0.16

    res = compute_metrics_for_subgroup(y_true, y_prob, threshold=threshold)

    assert res["n"] == 1
    assert res["defaults"] == 0
    assert res["default_rate"] == 0.0
    assert res["roc_auc"] is None
    assert res["pr_auc"] is None
    assert res["recall"] is None
    assert res["f1"] is None
    assert res["precision"] == 0.0
    assert res["brier_score"] == pytest.approx(0.08519, abs=1e-4)
    assert res["ece"] == pytest.approx(0.29187, abs=1e-4)


def test_bootstrap_confidence_intervals_includes_all_metrics():
    """Verify bootstrap CI generation includes Brier, ECE, Precision, and Recall."""
    rng = np.random.RandomState(42)
    y_true = rng.binomial(1, 0.15, size=200)
    y_prob = rng.uniform(0, 0.8, size=200)

    cis = compute_bootstrap_confidence_intervals(
        y_true, y_prob, threshold=0.16, n_bootstraps=20, seed=42
    )

    required_ci_keys = ["roc_auc", "pr_auc", "brier_score", "ece", "precision", "recall"]
    for k in required_ci_keys:
        assert k in cis, f"Missing CI key: {k}"
        low, high = cis[k]
        assert low is not None and high is not None
        assert low <= high

