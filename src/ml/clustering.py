"""Unsupervised customer segmentation module for BDE Assignment.

Provides data preparation, K-Means clustering across K=2..8,
PCA 2D projection, cluster profiling, and post-hoc target analysis.
Strictly excludes TARGET and SK_ID_CURR from cluster construction.
"""

from pathlib import Path
import json
import logging
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

CLUSTERING_FEATURES = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "APP_EXT_SOURCES_MEAN",
    "APP_PAYMENT_RATE",
    "APP_GOODS_TO_CREDIT_RATIO",
    "APP_AGE_YEARS",
    "APP_EMPLOYED_YEARS",
    "BURO_DEBT_TO_CREDIT_RATIO",
    "POS_COMPLETION_RATE_MEAN",
    "DERIVED_PAYMENT_DISCIPLINE_SCORE",
]


def load_and_preprocess_clustering_data(
    model_input_path: str = "data/model_input/model_input.parquet",
    features: list[str] | None = None,
    clip_percentile: float = 0.995,
) -> tuple[pd.DataFrame, np.ndarray, StandardScaler, SimpleImputer]:
    """Load model input data and prepare feature matrix for unsupervised clustering."""
    if features is None:
        features = CLUSTERING_FEATURES

    table = pq.read_table(model_input_path, columns=["SK_ID_CURR", "TARGET"] + features)
    df = table.to_pandas()

    # Feature matrix strictly excludes SK_ID_CURR and TARGET
    X_raw = df[features].copy()

    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X_raw), columns=features)

    # Handle extreme monetary/ratio outliers to prevent single-observation cluster distortion
    for col in ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "BURO_DEBT_TO_CREDIT_RATIO"]:
        if col in X_imputed.columns:
            upper_bound = X_imputed[col].quantile(clip_percentile)
            X_imputed[col] = X_imputed[col].clip(upper=upper_bound)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    return df, X_scaled, scaler, imputer


def evaluate_k_range(
    X_scaled: np.ndarray,
    k_range: range = range(2, 9),
    sample_size: int = 25000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Evaluate K-Means for a range of K using Inertia and Silhouette score."""
    np.random.seed(random_state)
    sample_idx = np.random.choice(len(X_scaled), size=min(sample_size, len(X_scaled)), replace=False)
    X_sample = X_scaled[sample_idx]

    records = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        km.fit(X_scaled)
        labels_sample = km.predict(X_sample)
        sil = silhouette_score(X_sample, labels_sample)
        records.append({
            "k": k,
            "inertia": float(km.inertia_),
            "silhouette_score": float(sil),
        })

    return pd.DataFrame(records)


def fit_and_profile_clusters(
    df: pd.DataFrame,
    X_scaled: np.ndarray,
    k: int = 4,
    random_state: int = 42,
) -> tuple[KMeans, pd.DataFrame, pd.DataFrame]:
    """Fit selected K-Means, compute PCA, and generate cluster profiles."""
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    cluster_labels = km.fit_predict(X_scaled)
    df["cluster"] = cluster_labels

    # PCA 2D projection for visualization
    pca = PCA(n_components=2, random_state=random_state)
    coords = pca.fit_transform(X_scaled)
    df["pca_x"] = coords[:, 0]
    df["pca_y"] = coords[:, 1]

    # Cluster profile table
    features = [c for c in CLUSTERING_FEATURES if c in df.columns]
    profile = df.groupby("cluster")[features].mean()
    profile["count"] = df.groupby("cluster").size()
    profile["pct"] = (profile["count"] / len(df)) * 100

    # Post-hoc TARGET analysis (strictly NOT used in cluster generation)
    if "TARGET" in df.columns:
        profile["default_rate_pct"] = df.groupby("cluster")["TARGET"].mean() * 100

    return km, df, profile
