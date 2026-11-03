"""Unit tests for Phase 11 Experiment E5 (SHAP Stability Across Retraining)."""

import pathlib
import numpy as np
import pandas as pd
import pytest
from scipy.stats import kendalltau
from src.explainability.stability import (
    assign_risk_bands,
    select_fixed_explanation_panel,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_assign_risk_bands():
    """Verify risk band boundaries: LOW [0, 0.09), MEDIUM [0.09, 0.25), HIGH [0.25, 1.0]."""
    probs = np.array([0.00, 0.05, 0.0899, 0.09, 0.15, 0.2499, 0.25, 0.50, 0.99])
    bands = assign_risk_bands(probs)
    expected = ["LOW", "LOW", "LOW", "MEDIUM", "MEDIUM", "MEDIUM", "HIGH", "HIGH", "HIGH"]
    assert list(bands) == expected


def test_select_fixed_explanation_panel_stratification():
    """Verify fixed explanation panel selects exact count and preserves stratification."""
    n_total = 1000
    X_val = pd.DataFrame(np.random.randn(n_total, 10))
    # 10% default rate
    y_val = np.array([1] * 100 + [0] * 900)

    X_panel, y_panel, idx = select_fixed_explanation_panel(
        X_val, y_val, n_panel=200, random_state=42
    )

    assert len(X_panel) == 200
    assert len(y_panel) == 200
    assert len(idx) == 200
    # Stratified: should be exactly 20 ones and 180 zeros
    assert y_panel.sum() == 20
    assert y_panel.mean() == pytest.approx(0.10, abs=0.01)


def test_panel_selection_reproducibility():
    """Verify panel selection is deterministic with fixed seed."""
    X_val = pd.DataFrame(np.random.randn(500, 5))
    y_val = np.array([1] * 50 + [0] * 450)

    _, _, idx1 = select_fixed_explanation_panel(X_val, y_val, n_panel=100, random_state=42)
    _, _, idx2 = select_fixed_explanation_panel(X_val, y_val, n_panel=100, random_state=42)

    assert np.array_equal(idx1, idx2)


def test_kendall_tau_ranking_stability():
    """Verify Kendall's rank correlation metric calculation."""
    rank_a = np.arange(1, 257)
    rank_b = np.arange(1, 257)

    tau_perf, _ = kendalltau(rank_a, rank_b)
    assert pytest.approx(tau_perf, 1e-5) == 1.0

    rank_inv = rank_a[::-1]
    tau_inv, _ = kendalltau(rank_a, rank_inv)
    assert pytest.approx(tau_inv, 1e-5) == -1.0

    # Slight perturbation (swap 2 adjacent features)
    rank_perturbed = rank_a.copy()
    rank_perturbed[0], rank_perturbed[1] = rank_perturbed[1], rank_perturbed[0]
    tau_pert, _ = kendalltau(rank_a, rank_perturbed)
    assert tau_pert > 0.99


def test_jaccard_reason_code_stability():
    """Verify Jaccard index computation for local explanation reason codes."""
    set_a = {"EXT_SOURCE_2", "EXT_SOURCE_3", "EXT_SOURCE_1", "DAYS_BIRTH"}
    set_b = {"EXT_SOURCE_2", "EXT_SOURCE_3", "EXT_SOURCE_1", "AMT_CREDIT"}

    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    jaccard = intersection / union

    # 3 shared out of 5 total unique
    assert pytest.approx(jaccard, 1e-4) == 3 / 5

    # Identical
    assert len(set_a.intersection(set_a)) / len(set_a.union(set_a)) == 1.0

    # Disjoint
    set_c = {"A", "B", "C"}
    assert len(set_a.intersection(set_c)) / len(set_a.union(set_c)) == 0.0


def test_risk_band_transition_matrix_row_stochasticity():
    """Verify that empirical transition matrices satisfy row-stochasticity."""
    # Simulated transition counts
    transition_counts = np.array([
        [3600, 200, 0],
        [150, 800, 50],
        [0, 30, 170],
    ])
    row_sums = transition_counts.sum(axis=1, keepdims=True)
    transition_matrix = transition_counts / row_sums

    # Each row sums to 1.0
    for r in transition_matrix.sum(axis=1):
        assert pytest.approx(r, 1e-5) == 1.0

    # Diagonals represent retention / stability
    retention_rates = np.diag(transition_matrix)
    assert (retention_rates > 0.70).all()


def test_extract_topk_shap_attributions_sign_and_magnitude():
    """Verify that extract_topk_shap_attributions adheres to sign and magnitude conventions.

    Conventions:
    - XGBoost binary:logistic margin output: positive SHAP (phi_i > 0) increases default risk.
    - Negative SHAP (phi_i < 0) decreases default risk.
    - mode='magnitude': ranks by |phi_i| (top predictive attribution factors).
    - mode='risk_increasing': ranks strictly by positive phi_i > 0.
    """
    from src.explainability.stability import extract_topk_shap_attributions

    # Synthetic SHAP matrix for 2 applicants across 4 features:
    # Feature 0: +2.5 (strong risk factor)
    # Feature 1: -4.0 (strong protective factor)
    # Feature 2: +1.0 (moderate risk factor)
    # Feature 3: -0.5 (slight protective factor)
    shap_matrix = np.array([
        [2.5, -4.0, 1.0, -0.5],
        [-1.0, 3.0, -2.0, 0.5],
    ])

    # Magnitude mode: ranks by |phi|:
    # Applicant 0: |-4.0|=4.0 (feat 1), |2.5|=2.5 (feat 0), |1.0|=1.0 (feat 2), |-0.5|=0.5 (feat 3) -> [1, 0, 2]
    top3_mag = extract_topk_shap_attributions(shap_matrix, k=3, ranking_mode="magnitude")
    assert np.array_equal(top3_mag[0], [1, 0, 2])
    # Applicant 1: |3.0|=3.0 (feat 1), |-2.0|=2.0 (feat 2), |-1.0|=1.0 (feat 0) -> [1, 2, 0]
    assert np.array_equal(top3_mag[1], [1, 2, 0])

    # Risk-increasing mode: ranks strictly by positive SHAP
    # Applicant 0: 2.5 (feat 0), 1.0 (feat 2) -> [0, 2]
    top2_risk = extract_topk_shap_attributions(shap_matrix, k=2, ranking_mode="risk_increasing")
    assert np.array_equal(top2_risk[0], [0, 2])
    # Applicant 1: 3.0 (feat 1), 0.5 (feat 3) -> [1, 3]
    assert np.array_equal(top2_risk[1], [1, 3])

    # Invalid mode raises ValueError
    with pytest.raises(ValueError):
        extract_topk_shap_attributions(shap_matrix, k=2, ranking_mode="unsupported_mode")


def test_e5_pairwise_comparisons_count():
    """Verify combinatorial math for E5 model pairs and applicant comparisons."""
    from math import comb

    n_retrains = 15
    panel_size = 5000
    n_pairs = comb(n_retrains, 2)
    assert n_pairs == 105
    total_applicant_comparisons = n_pairs * panel_size
    assert total_applicant_comparisons == 525000

