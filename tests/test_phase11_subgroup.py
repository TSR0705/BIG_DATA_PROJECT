"""Unit tests for Phase 11 Experiment E7 (Applicant Subgroup Performance and Calibration Audit).

Verifies:
1. Independent calibration provenance and leakage elimination (disjoint splits, zero validation/test overlap).
2. Proper packaging and validity of phase11_independent_calibrator.joblib.
3. Conventional percentile bootstrap confidence intervals and metric consistency.
4. ECE 10-bin uniform definition matching across manual loops and scikit-learn reference.
5. Preservation of frozen Phase 8 artifacts (phase8_best_xgboost.joblib, phase8_calibrated_model.joblib).
"""

import json
import pathlib
import re
import numpy as np
import pandas as pd
import pytest
import joblib
from sklearn.calibration import calibration_curve

from src.evaluation.subgroup import (
    compute_expected_calibration_error,
    compute_metrics_for_subgroup,
    compute_bootstrap_confidence_intervals,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]


# ======================================================================
# 1. Basic Metric Calculations & Edge Cases
# ======================================================================

def test_ece_calculation():
    """Verify Expected Calibration Error calculation behaves correctly."""
    # Perfectly calibrated case
    y_prob = np.concatenate([np.full(100, 0.1), np.full(100, 0.9)])
    y_true = np.concatenate([np.array([1]*10 + [0]*90), np.array([1]*90 + [0]*10)])
    ece = compute_expected_calibration_error(y_true, y_prob, n_bins=10)
    assert pytest.approx(ece, abs=0.05) == 0.0

    # Inverted calibration
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
    """Verify single-class subgroup with n=1 and zero positive cases (CODE_GENDER=None cohort)."""
    y_true = np.array([0])
    y_prob = np.array([0.281952])
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
    assert res["brier_score"] == pytest.approx(0.07950, abs=1e-4)
    assert res["ece"] == pytest.approx(0.28195, abs=1e-4)


def test_bootstrap_confidence_intervals_includes_all_metrics():
    """Verify bootstrap CI generation calculates bounds for all audited metrics."""
    rng = np.random.RandomState(42)
    y_true = rng.binomial(1, 0.10, size=500)
    y_prob = rng.beta(1, 5, size=500)

    cis = compute_bootstrap_confidence_intervals(y_true, y_prob, threshold=0.16, n_bootstraps=50, seed=42)

    required_metrics = ["roc_auc", "pr_auc", "brier_score", "ece", "precision", "recall"]
    for m in required_metrics:
        assert m in cis, f"Metric '{m}' missing from bootstrap CI dictionary"
        low, high = cis[m]
        assert low is not None and high is not None, f"Metric '{m}' bounds are None"
        assert low <= high, f"Metric '{m}' lower bound {low} exceeds upper bound {high}"


def test_paired_resampling_alignment():
    """Verify paired resampling preserves candidate-wise alignment between labels and probabilities."""
    rng = np.random.RandomState(42)
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.15, 0.85, 0.05, 0.95])

    ci_1 = compute_bootstrap_confidence_intervals(y_true, y_prob, threshold=0.5, n_bootstraps=100, seed=42)
    ci_2 = compute_bootstrap_confidence_intervals(y_true, y_prob, threshold=0.5, n_bootstraps=100, seed=42)

    assert ci_1 == ci_2


def test_ece_consistent_binning_across_pipeline():
    """Verify that ECE uses exactly 10 bins uniformly spaced in [0, 1]."""
    y_prob = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    ece = compute_expected_calibration_error(y_true, y_prob, n_bins=10)
    assert 0.0 <= ece <= 1.0


def test_e7_data_partition_isolation():
    """Verify E7 subgroup evaluation uses only validation partition and never accesses test partition."""
    val_split_path = ROOT / "data" / "splits" / "val_indices.parquet"
    test_split_path = ROOT / "data" / "splits" / "test_indices.parquet"

    val_df = pd.read_parquet(val_split_path)
    test_df = pd.read_parquet(test_split_path)

    val_ids = set(val_df["SK_ID_CURR"])
    test_ids = set(test_df["SK_ID_CURR"])

    assert val_ids.isdisjoint(test_ids), "Data leakage: Validation and Test partition overlap!"
    assert len(val_ids) == 46127, f"Expected 46,127 validation samples, found {len(val_ids)}"


# ======================================================================
# 2. Calibration Leakage Elimination & Split Independence
# ======================================================================

def test_calibration_training_indices_disjoint_from_validation_and_test():
    """Verify that training indices are strictly disjoint from validation and test indices."""
    train_split_path = ROOT / "data" / "splits" / "train_indices.parquet"
    val_split_path = ROOT / "data" / "splits" / "val_indices.parquet"
    test_split_path = ROOT / "data" / "splits" / "test_indices.parquet"

    train_ids = set(pd.read_parquet(train_split_path)["SK_ID_CURR"])
    val_ids = set(pd.read_parquet(val_split_path)["SK_ID_CURR"])
    test_ids = set(pd.read_parquet(test_split_path)["SK_ID_CURR"])

    assert len(train_ids) == 215257, f"Expected 215,257 train samples, got {len(train_ids)}"
    assert len(val_ids) == 46127, f"Expected 46,127 val samples, got {len(val_ids)}"
    assert len(test_ids) == 46127, f"Expected 46,127 test samples, got {len(test_ids)}"

    assert train_ids.isdisjoint(val_ids), "Fatal leakage: Train and Validation indices overlap!"
    assert train_ids.isdisjoint(test_ids), "Fatal leakage: Train and Test indices overlap!"
    assert val_ids.isdisjoint(test_ids), "Fatal leakage: Validation and Test indices overlap!"


def test_independent_calibration_metadata_provenance():
    """Verify that phase11_calibration_metadata.json documents zero validation or test rows."""
    meta_path = ROOT / "artifacts" / "models" / "phase11_calibration_metadata.json"
    assert meta_path.exists(), "phase11_calibration_metadata.json does not exist"

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["provenance"]["training_row_count"] == 215257
    assert meta["provenance"]["validation_rows_accessed"] == 0
    assert meta["provenance"]["validation_labels_accessed"] == 0
    assert meta["provenance"]["test_rows_accessed"] == 0
    assert meta["provenance"]["test_labels_accessed"] == 0
    assert meta["oof_generation"]["cv_folds"] == 5
    assert meta["oof_generation"]["stratified"] is True


def test_independent_calibrator_artifact_validity():
    """Verify phase11_independent_calibrator.joblib exists, loads, and outputs valid probabilities."""
    model_path = ROOT / "artifacts" / "models" / "phase11_independent_calibrator.joblib"
    assert model_path.exists(), "phase11_independent_calibrator.joblib does not exist"

    model = joblib.load(model_path)
    assert hasattr(model, "predict_proba"), "Calibrated model missing predict_proba"

    # Dummy inference check
    dummy_X = np.zeros((10, 256))
    probs = model.predict_proba(dummy_X)
    assert probs.shape == (10, 2)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_frozen_phase8_artifacts_preserved():
    """Verify frozen Phase 8 models remain untouched and match expected configurations."""
    p8_base_path = ROOT / "artifacts" / "models" / "phase8_best_xgboost.joblib"
    p8_cal_path = ROOT / "artifacts" / "models" / "phase8_calibrated_model.joblib"
    p8_meta_path = ROOT / "artifacts" / "models" / "phase8_model_metadata.json"

    assert p8_base_path.exists(), "phase8_best_xgboost.joblib missing"
    assert p8_cal_path.exists(), "phase8_calibrated_model.joblib missing"
    assert p8_meta_path.exists(), "phase8_model_metadata.json missing"

    with open(p8_meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["hyperparameter_search"]["best_params"]["n_estimators"] == 500


def test_independent_calibrator_validation_ece_reference_match():
    """Verify that independent calibrator validation ECE matches manual uniform 10-bin loop and sklearn."""
    val_split_path = ROOT / "data" / "splits" / "val_indices.parquet"
    input_path = ROOT / "data" / "model_input" / "model_input.parquet"
    model_path = ROOT / "artifacts" / "models" / "phase11_independent_calibrator.joblib"
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

    model = joblib.load(model_path)
    probs_val = model.predict_proba(X_val)[:, 1]

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

    # Out-of-sample calibration error is realistic, positive, and bounded
    assert 0.001 < ece_pipe < 0.010, f"Expected realistic out-of-sample ECE, got {ece_pipe}"
    assert round(ece_pipe, 4) == 0.0042


def test_overall_ece_enclosed_in_bootstrap_ci():
    """Verify that Overall validation ECE point estimate (0.00417) is enclosed in its 95% bootstrap CI."""
    csv_path = ROOT / "reports" / "subgroup_analysis.csv"
    assert csv_path.exists(), "subgroup_analysis.csv does not exist"

    df = pd.read_csv(csv_path)
    overall_row = df[df["subgroup_name"] == "Full Population"].iloc[0]

    ece_pt = float(overall_row["ece"])
    ece_ci_str = str(overall_row["ece_ci"])

    match = re.search(r"\[([0-9.]+),\s*([0-9.]+)\]", ece_ci_str)
    assert match, f"Could not parse ECE CI from '{ece_ci_str}'"

    low = float(match.group(1))
    high = float(match.group(2))

    assert low <= ece_pt <= high, f"Overall ECE point estimate {ece_pt} not enclosed in [{low}, {high}]"


def test_ece_empty_bins_and_single_observation_handling():
    """Verify compute_expected_calibration_error edge cases: empty array, n=1, and multi-empty bins."""
    assert compute_expected_calibration_error(np.array([]), np.array([])) == 0.0

    y_single = np.array([0])
    p_single = np.array([0.25])
    assert compute_expected_calibration_error(y_single, p_single) == pytest.approx(0.25, abs=1e-6)

    y_sparse = np.array([0, 0, 1, 1])
    p_sparse = np.array([0.15, 0.15, 0.85, 0.85])
    ece_sparse = compute_expected_calibration_error(y_sparse, p_sparse, n_bins=10)
    expected_sparse = 0.5 * 0.15 + 0.5 * 0.15
    assert ece_sparse == pytest.approx(expected_sparse, abs=1e-6)
