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


def test_point_estimate_contained_in_bootstrap_ci():
    """Verify that the full-subgroup point estimate is strictly enclosed by the bootstrap 95% CI."""
    rng = np.random.RandomState(42)
    n_samples = 1000
    y_true = rng.binomial(1, 0.15, size=n_samples)
    y_prob = rng.beta(2, 8, size=n_samples)
    threshold = 0.16

    pt = compute_metrics_for_subgroup(y_true, y_prob, threshold=threshold)
    cis = compute_bootstrap_confidence_intervals(y_true, y_prob, threshold=threshold, n_bootstraps=200, seed=42)

    for metric in ["roc_auc", "pr_auc", "brier_score", "ece", "precision", "recall"]:
        pt_val = pt[metric]
        low, high = cis[metric]
        assert pt_val is not None, f"Point estimate for {metric} is None"
        assert low is not None and high is not None, f"CI bounds for {metric} are None"
        # Point estimate must be compatible with bootstrap distribution
        assert low <= pt_val <= high or pytest.approx(pt_val, abs=1e-3) in (low, high), (
            f"Point estimate {pt_val:.4f} for {metric} not enclosed by CI [{low:.4f}, {high:.4f}]"
        )


def test_paired_resampling_alignment():
    """Verify that paired bootstrap resampling preserves exact applicant alignment between y_true and y_prob."""
    n_samples = 500
    rng = np.random.RandomState(42)
    y_true = rng.binomial(1, 0.2, size=n_samples)
    # Perfect predictor: y_prob == y_true
    y_prob = y_true.astype(float)
    threshold = 0.50

    cis = compute_bootstrap_confidence_intervals(y_true, y_prob, threshold=threshold, n_bootstraps=50, seed=42)

    # With paired resampling and perfect predictions, precision and recall must be 1.0 across all resamples
    assert cis["precision"] == (1.0, 1.0)
    assert cis["recall"] == (1.0, 1.0)
    assert cis["roc_auc"] == (1.0, 1.0)


def test_ece_consistent_binning_across_pipeline():
    """Verify that ECE binning definition (n_bins=10) is used identically in point estimates and bootstrap."""
    rng = np.random.RandomState(42)
    y_true = rng.binomial(1, 0.1, size=300)
    y_prob = rng.uniform(0, 0.5, size=300)

    # Point estimate ECE
    pt = compute_metrics_for_subgroup(y_true, y_prob)
    ece_direct = compute_expected_calibration_error(y_true, y_prob, n_bins=10)
    assert pt["ece"] == pytest.approx(ece_direct, abs=1e-6)


def test_e7_data_partition_isolation():
    """Verify that E7 subgroup audit strictly loads validation data and isolates test data."""
    import inspect
    from src.evaluation import subgroup

    source = inspect.getsource(subgroup.run_e7_subgroup_audit)
    assert "val_indices.parquet" in source
    assert "test_indices.parquet" not in source




def test_overall_ece_point_estimate_matches_independent_reference():
    """Verify that Overall ECE point estimate matches independent reference calculations.

    Tests against:
    1. Independent manual uniform binning loop.
    2. Scikit-learn calibration_curve(strategy='uniform', n_bins=10).
    3. Confirms that due to isotonic prefit calibration in Phase 8, in-sample ECE < 1e-5 (machine zero).
    """
    import pathlib
    import json
    import joblib
    import pandas as pd
    from sklearn.calibration import calibration_curve

    ROOT = pathlib.Path(__file__).resolve().parents[1]
    input_path = ROOT / "data" / "model_input" / "model_input.parquet"
    val_split_path = ROOT / "data" / "splits" / "val_indices.parquet"
    model_path = ROOT / "artifacts" / "models" / "phase8_calibrated_model.joblib"
    meta_path = ROOT / "artifacts" / "models" / "phase8_model_metadata.json"

    df = pd.read_parquet(input_path)
    val_ids = set(pd.read_parquet(val_split_path)["SK_ID_CURR"])
    val_df = df[df["SK_ID_CURR"].isin(val_ids)].copy()

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    feature_cols = meta["features"]

    cat_cols = [c for c in feature_cols if val_df[c].dtype == "object" or val_df[c].dtype.name == "category"]
    for c in cat_cols:
        val_df[c] = val_df[c].astype("category").cat.codes

    X_val = val_df[feature_cols].copy()
    y_val = val_df["TARGET"].values.astype(int)

    cal_model = joblib.load(model_path)
    probs_val = cal_model.predict_proba(X_val)[:, 1]

    # Pipeline ECE
    ece_pipe = compute_expected_calibration_error(y_val, probs_val, n_bins=10)

    # Reference Calculation 1: Independent manual bin loop
    bins = np.linspace(0.0, 1.0, 11)
    bin_idx = np.clip(np.digitize(probs_val, bins) - 1, 0, 9)
    manual_ece = 0.0
    for b in range(10):
        mask = bin_idx == b
        cnt = np.sum(mask)
        if cnt > 0:
            manual_ece += (cnt / len(y_val)) * abs(np.mean(y_val[mask]) - np.mean(probs_val[mask]))

    assert ece_pipe == pytest.approx(manual_ece, abs=1e-9)

    # Reference Calculation 2: Scikit-learn calibration_curve
    prob_true, prob_pred = calibration_curve(y_val, probs_val, n_bins=10, strategy="uniform")
    bin_counts, _ = np.histogram(probs_val, bins=10, range=(0.0, 1.0))
    non_empty = bin_counts > 0
    sklearn_ece = float(np.sum((bin_counts[non_empty] / len(y_val)) * np.abs(prob_true - prob_pred)))

    assert ece_pipe == pytest.approx(sklearn_ece, abs=1e-9)

    # In-sample isotonic calibration forces ECE to machine zero
    assert ece_pipe < 1e-5
    assert round(ece_pipe, 5) == 0.00000


def test_overall_ece_point_estimate_and_ci_consistency():
    """Verify that Overall ECE point estimate is rigorously enclosed by its bootstrap confidence interval."""
    rng = np.random.RandomState(42)
    # Perfectly calibrated synthetic population (simulating isotonic prefit)
    n = 2000
    p = rng.choice([0.05, 0.15, 0.25, 0.45], size=n)
    y = rng.binomial(1, p)

    pt = compute_metrics_for_subgroup(y, p)
    cis = compute_bootstrap_confidence_intervals(y, p, n_bootstraps=50, seed=42)

    ece_pt = pt["ece"]
    ece_low, ece_high = cis["ece"]

    assert ece_low is not None and ece_high is not None
    assert ece_low >= 0.0
    assert ece_low <= ece_high
    assert ece_low <= ece_pt <= ece_high or pytest.approx(ece_pt, abs=1e-5) in (ece_low, ece_high), (
        f"Point estimate {ece_pt} not enclosed by CI [{ece_low}, {ece_high}]"
    )


def test_all_subgroup_ece_strictly_enclosed_in_reported_csv():
    """Verify that in reports/subgroup_analysis.csv, every single subgroup ECE point estimate is enclosed by its CI."""
    import pathlib
    import re
    import pandas as pd

    ROOT = pathlib.Path(__file__).resolve().parents[1]
    csv_path = ROOT / "reports" / "subgroup_analysis.csv"
    assert csv_path.exists(), "subgroup_analysis.csv does not exist"

    df = pd.read_csv(csv_path)

    for _, row in df.iterrows():
        ece_pt = row["ece"]
        ece_ci_str = str(row["ece_ci"])
        if pd.isna(ece_pt) or ece_ci_str == "NA":
            continue

        match = re.search(r"\[([0-9.]+),\s*([0-9.]+)\]", ece_ci_str)
        if match:
            low = float(match.group(1))
            high = float(match.group(2))
            assert low <= ece_pt <= high or pytest.approx(ece_pt, abs=1e-5) in (low, high), (
                f"Subgroup '{row['subgroup_name']}' ECE point estimate {ece_pt} not in CI [{low}, {high}]"
            )


def test_ece_empty_bins_and_single_observation_handling():
    """Verify compute_expected_calibration_error edge cases: empty array, n=1, and multi-empty bins."""
    # Empty array
    assert compute_expected_calibration_error(np.array([]), np.array([])) == 0.0

    # n=1 single observation
    y_single = np.array([0])
    p_single = np.array([0.25])
    assert compute_expected_calibration_error(y_single, p_single) == pytest.approx(0.25, abs=1e-6)

    # Multi-empty bins: only bins [0.1, 0.2] and [0.8, 0.9] populated
    y_sparse = np.array([0, 0, 1, 1])
    p_sparse = np.array([0.15, 0.15, 0.85, 0.85])
    ece_sparse = compute_expected_calibration_error(y_sparse, p_sparse, n_bins=10)
    # Bin 1 [0.1, 0.2): acc=0.0, conf=0.15, diff=0.15, weight=2/4
    # Bin 8 [0.8, 0.9): acc=1.0, conf=0.85, diff=0.15, weight=2/4
    expected_sparse = 0.5 * 0.15 + 0.5 * 0.15
    assert ece_sparse == pytest.approx(expected_sparse, abs=1e-6)
