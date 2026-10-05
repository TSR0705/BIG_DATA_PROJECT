"""Unit tests for unsupervised clustering module."""

import numpy as np
import pandas as pd
import pytest
from src.ml.clustering import (
    CLUSTERING_FEATURES,
    evaluate_k_range,
    fit_and_profile_clusters,
)

def test_clustering_features_specification():
    """Verify clustering features contract strictly excludes identifier and target."""
    assert "TARGET" not in CLUSTERING_FEATURES
    assert "SK_ID_CURR" not in CLUSTERING_FEATURES
    assert len(CLUSTERING_FEATURES) == 11

def test_clustering_evaluation_and_profiling_synthetic():
    """Verify K-Means evaluation, fitting, PCA, and post-hoc profiling on synthetic data."""
    np.random.seed(42)
    n_samples = 200
    n_features = len(CLUSTERING_FEATURES)
    X_synthetic = np.random.randn(n_samples, n_features)

    # Evaluate K=2..3
    eval_df = evaluate_k_range(X_synthetic, k_range=range(2, 4), sample_size=100, random_state=42)
    assert len(eval_df) == 2
    assert "inertia" in eval_df.columns
    assert "silhouette_score" in eval_df.columns
    assert eval_df["inertia"].iloc[1] < eval_df["inertia"].iloc[0]

    # Fit and profile
    df_synthetic = pd.DataFrame(X_synthetic, columns=CLUSTERING_FEATURES)
    df_synthetic["SK_ID_CURR"] = np.arange(1000, 1000 + n_samples)
    df_synthetic["TARGET"] = np.random.binomial(1, 0.08, size=n_samples)

    km, df_assigned, profile = fit_and_profile_clusters(
        df=df_synthetic,
        X_scaled=X_synthetic,
        k=3,
        random_state=42
    )

    assert "cluster" in df_assigned.columns
    assert "pca_x" in df_assigned.columns
    assert "pca_y" in df_assigned.columns
    assert len(profile) == 3
    assert "count" in profile.columns
    assert "pct" in profile.columns
    assert "default_rate_pct" in profile.columns
