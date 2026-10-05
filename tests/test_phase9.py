"""Tests for Phase 9: Explainable AI with SHAP.

Verifies:
1. Phase 8 model loads.
2. Model parameters remain unchanged.
3. SHAP explainer can be created.
4. SHAP values have correct dimensions.
5. SHAP feature names match model features.
6. No TARGET in SHAP features.
7. No SK_ID_CURR in SHAP features.
8. SHAP values are finite.
9. Explanation sample size is correct.
10. Stability outputs exist.
11. Required figures exist.
12. Required reports exist.
"""

import json
import pathlib
import sys
import joblib
import numpy as np
import pandas as pd
import pytest
import shap
from xgboost import XGBClassifier

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ml.shap_analysis import (
    compute_global_importance,
    create_tree_explainer,
    load_frozen_phase8_model,
)

REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
ARTIFACTS_MODELS_DIR = ROOT / "artifacts" / "models"
ARTIFACTS_EXPLAIN_DIR = ROOT / "artifacts" / "explainability"


# ======================================================================
# Unit & Pre-Execution Tests
# ======================================================================

def test_phase8_model_loads_and_is_valid():
    """Verify Phase 8 tuned XGBoost model loads properly."""
    model = load_frozen_phase8_model()
    assert isinstance(model, XGBClassifier)
    assert hasattr(model, "predict_proba")
    assert hasattr(model, "feature_names_in_")
    assert len(model.feature_names_in_) == 256


def test_model_parameters_remain_unchanged():
    """Verify frozen Phase 8 hyperparameters are exactly preserved."""
    model = load_frozen_phase8_model()
    params = model.get_params()

    assert params["max_depth"] == 3
    assert params["learning_rate"] == 0.1
    assert params["n_estimators"] == 500
    assert params["min_child_weight"] == 10
    assert params["subsample"] == 0.8
    assert params["colsample_bytree"] == 1.0
    assert params["random_state"] == 42


def test_shap_explainer_creation_and_synthetic_computation():
    """Verify TreeExplainer initializes and computes finite SHAP values on dummy data."""
    model = load_frozen_phase8_model()
    explainer = create_tree_explainer(model)
    assert isinstance(explainer, shap.TreeExplainer)

    # Synthetic input with matching 256 features
    feature_names = list(model.feature_names_in_)
    dummy_X = pd.DataFrame(
        np.random.randn(5, 256).astype(np.float32),
        columns=feature_names,
    )

    expl = explainer(dummy_X)
    assert expl.values.shape == (5, 256)
    assert np.all(np.isfinite(expl.values))
    assert list(expl.feature_names) == feature_names

    global_df = compute_global_importance(expl)
    assert len(global_df) == 256
    assert list(global_df.columns) == ["feature", "mean_abs_shap", "mean_shap", "shap_rank"]
    assert np.all(global_df["mean_abs_shap"] >= 0.0)


def test_no_target_or_id_in_shap_feature_names():
    """Verify TARGET and SK_ID_CURR are strictly absent from model and SHAP feature names."""
    model = load_frozen_phase8_model()
    feature_names = set(model.feature_names_in_)

    assert "TARGET" not in feature_names
    assert "SK_ID_CURR" not in feature_names


# ======================================================================
# Integration & Integrity Tests (Requires run_phase9_explainability.py)
# ======================================================================

def test_explanation_sample_size_and_metadata():
    """Verify explanation metadata records sample size n=5,000 and seed 42."""
    meta_path = ARTIFACTS_EXPLAIN_DIR / "shap_metadata.json"
    assert meta_path.exists(), f"Missing {meta_path}"

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["explanation_sample_size"] == 5000
    assert meta["random_state"] == 42
    assert meta["explanation_method"] == "shap.TreeExplainer"
    assert meta["stability_repetitions"] == 10
    assert len(meta["top_20_features"]) == 20
    assert "high_risk" in meta["local_cases"]
    assert "low_risk" in meta["local_cases"]
    assert "medium_risk" in meta["local_cases"]


def test_global_importance_and_stability_outputs_exist():
    """Verify all tabular outputs for global importance, stability, and comparison exist."""
    required_csvs = [
        REPORTS_DIR / "shap_global_importance.csv",
        REPORTS_DIR / "shap_vs_native_importance.csv",
        REPORTS_DIR / "shap_stability.csv",
        REPORTS_DIR / "shap_subgroup_comparison.csv",
        ARTIFACTS_EXPLAIN_DIR / "shap_global_importance.csv",
        ARTIFACTS_EXPLAIN_DIR / "shap_stability.csv",
    ]
    for csv_file in required_csvs:
        assert csv_file.exists(), f"Missing required CSV: {csv_file}"

    # Verify global importance structure
    g_df = pd.read_csv(REPORTS_DIR / "shap_global_importance.csv")
    assert len(g_df) == 256
    assert list(g_df.columns) == ["feature", "mean_abs_shap", "mean_shap", "shap_rank"]
    assert np.all(np.isfinite(g_df["mean_abs_shap"]))

    # Verify stability structure
    s_df = pd.read_csv(REPORTS_DIR / "shap_stability.csv")
    assert "selection_frequency" in s_df.columns
    assert np.all(s_df["selection_frequency"] <= 1.0)
    assert np.all(s_df["selection_frequency"] >= 0.0)


def test_required_figures_exist():
    """Verify all 12 publication-quality SHAP figures exist and are non-empty."""
    figures = [
        FIGURES_DIR / "shap_global_bar.png",
        FIGURES_DIR / "shap_beeswarm.png",
        FIGURES_DIR / "shap_dependence_01.png",
        FIGURES_DIR / "shap_dependence_02.png",
        FIGURES_DIR / "shap_dependence_03.png",
        FIGURES_DIR / "shap_dependence_04.png",
        FIGURES_DIR / "shap_dependence_05.png",
        FIGURES_DIR / "shap_local_high_risk.png",
        FIGURES_DIR / "shap_local_medium_risk.png",
        FIGURES_DIR / "shap_local_low_risk.png",
        FIGURES_DIR / "shap_stability.png",
        FIGURES_DIR / "shap_subgroup_comparison.png",
    ]
    for fig in figures:
        assert fig.exists(), f"Required SHAP figure missing: {fig}"
        assert fig.stat().st_size > 1000, f"Figure suspiciously small: {fig}"


def test_required_reports_exist():
    """Verify shap_report.md and shap_report.json exist."""
    rep_md = REPORTS_DIR / "shap_report.md"
    rep_json = REPORTS_DIR / "shap_report.json"

    assert rep_md.exists(), f"Missing {rep_md}"
    assert rep_json.exists(), f"Missing {rep_json}"
    assert rep_md.stat().st_size > 1000
    assert rep_json.stat().st_size > 500
