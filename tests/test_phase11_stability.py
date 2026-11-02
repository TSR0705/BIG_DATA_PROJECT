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
