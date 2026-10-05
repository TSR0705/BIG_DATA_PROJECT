"""Phase 9 Explainable AI Layer using SHAP.

Provides:
- Model explanation using shap.TreeExplainer on the frozen Phase 8 XGBoost model
- Global feature attribution (mean absolute SHAP) and beeswarm visualization
- Comparison between SHAP attribution and native tree feature importance
- Single-feature dependence plots with automatic interaction feature detection
- Local applicant waterfall explanations (High-risk, Low-risk, Borderline/Medium-risk)
- Bootstrap explanation stability assessment across repeated random samples
- Exploratory subgroup explanation analysis across demographic partitions
- Full runtime isolation: zero target leakage, zero model retraining, strictly frozen inputs
"""

import json
import os
import pathlib
import sys
import time
from typing import Any, Optional, Union

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.log import get_logger

logger = get_logger("SHAPAnalysis", "shap_analysis.log")

ARTIFACTS_EXPLAIN_DIR = ROOT / "artifacts" / "explainability"
ARTIFACTS_MODELS_DIR = ROOT / "artifacts" / "models"


def load_frozen_phase8_model(
    model_path: Union[str, pathlib.Path] = ARTIFACTS_MODELS_DIR / "phase8_best_xgboost.joblib",
) -> XGBClassifier:
    """Load the frozen Phase 8 tuned XGBoost model and verify integrity."""
    p = pathlib.Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"Frozen Phase 8 model missing at: {p}")
    model = joblib.load(p)
    if not isinstance(model, XGBClassifier):
        raise TypeError(f"Expected XGBClassifier, got {type(model)}")
    logger.info(f"Loaded frozen Phase 8 XGBoost model from {p} ({len(model.feature_names_in_)} features).")
    return model


def get_validation_explanation_sample(
    X_val: pd.DataFrame,
    n_samples: int = 5000,
    random_state: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Extract a reproducible controlled representative sample from VALIDATION for SHAP analysis.

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray]
        Sampled feature DataFrame and integer index array in X_val.
    """
    if len(X_val) <= n_samples:
        indices = np.arange(len(X_val))
    else:
        rng = np.random.RandomState(random_state)
        indices = rng.choice(len(X_val), size=n_samples, replace=False)
        indices.sort()

    X_sample = X_val.iloc[indices].copy()
    logger.info(f"Extracted validation explanation sample: {len(X_sample):,} rows (seed={random_state}).")
    return X_sample, indices


def create_tree_explainer(model: XGBClassifier) -> shap.TreeExplainer:
    """Initialize shap.TreeExplainer for the frozen XGBoost model."""
    logger.info("Initializing shap.TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    logger.info(f"TreeExplainer created successfully with base value: {explainer.expected_value}")
    return explainer


def compute_shap_explanation(
    explainer: shap.TreeExplainer,
    X_sample: pd.DataFrame,
) -> shap.Explanation:
    """Compute full SHAP Explanation object for the given feature sample."""
    t0 = time.time()
    logger.info(f"Computing SHAP values for {len(X_sample):,} sample rows...")
    expl = explainer(X_sample)
    duration = time.time() - t0
    logger.info(f"SHAP computation completed in {duration:.2f}s.")
    return expl


def compute_global_importance(expl: shap.Explanation) -> pd.DataFrame:
    """Calculate mean absolute SHAP and mean SHAP for each feature across the sample.

    Returns
    -------
    pd.DataFrame
        Table with columns ['feature', 'mean_abs_shap', 'mean_shap', 'shap_rank'],
        sorted by mean_abs_shap descending.
    """
    feature_names = expl.feature_names
    shap_vals = expl.values

    mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
    mean_shap = np.mean(shap_vals, axis=0)

    df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": np.round(mean_abs_shap, 6),
        "mean_shap": np.round(mean_shap, 6),
    })
    df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    df["shap_rank"] = df.index + 1

    logger.info(f"Global SHAP importance calculated for {len(df)} features.")
    return df


def compare_shap_vs_native_importance(
    shap_df: pd.DataFrame,
    model: XGBClassifier,
    top_n: int = 20,
) -> pd.DataFrame:
    """Compare SHAP global importance with native XGBoost feature importance.

    Returns
    -------
    pd.DataFrame
        Top N SHAP features with both SHAP and native metrics and ranks.
    """
    native_importances = model.feature_importances_
    feature_names = list(model.feature_names_in_)

    native_df = pd.DataFrame({
        "feature": feature_names,
        "native_importance": np.round(native_importances, 6),
    })
    native_df = native_df.sort_values("native_importance", ascending=False).reset_index(drop=True)
    native_df["native_rank"] = native_df.index + 1

    merged = pd.merge(
        shap_df[["feature", "mean_abs_shap", "shap_rank"]],
        native_df[["feature", "native_importance", "native_rank"]],
        on="feature",
        how="left",
    )
    merged = merged.rename(columns={"mean_abs_shap": "shap_mean_abs"})
    top_df = merged.head(top_n).copy()
    logger.info(f"Comparison of top {top_n} features between SHAP and native tree importance compiled.")
    return top_df


def compute_explanation_stability(
    explainer: shap.TreeExplainer,
    X_val: pd.DataFrame,
    n_samples: int = 5000,
    seeds: Optional[list[int]] = None,
    top_k: int = 20,
) -> pd.DataFrame:
    """Assess explanation stability across 10 repeated bootstrap samples from VALIDATION.

    Returns
    -------
    pd.DataFrame
        Features ranked by selection frequency across repeated samples.
    """
    if seeds is None:
        seeds = list(range(42, 52))

    logger.info(f"Evaluating SHAP stability across {len(seeds)} repeated samples (n={n_samples})...")
    feature_counts: dict[str, int] = {}
    feature_names = list(X_val.columns)

    for seed in seeds:
        rng = np.random.RandomState(seed)
        sub_idx = rng.choice(len(X_val), size=n_samples, replace=False)
        X_sub = X_val.iloc[sub_idx]

        vals = explainer.shap_values(X_sub)
        mean_abs = np.mean(np.abs(vals), axis=0)

        # Top K features for this sample
        top_indices = np.argsort(-mean_abs)[:top_k]
        top_feats = [feature_names[i] for i in top_indices]

        for feat in top_feats:
            feature_counts[feat] = feature_counts.get(feat, 0) + 1

    rows = []
    n_repeats = len(seeds)
    for feat, count in feature_counts.items():
        rows.append({
            "feature": feat,
            "number_of_appearances_in_top20": count,
            "selection_frequency": round(count / n_repeats, 4),
        })

    stab_df = pd.DataFrame(rows)
    stab_df = stab_df.sort_values(
        ["selection_frequency", "number_of_appearances_in_top20"], ascending=[False, False]
    ).reset_index(drop=True)

    logger.info(f"Explanation stability evaluated: {len(stab_df)} distinct features appeared in top {top_k}.")
    return stab_df


def compute_subgroup_analysis(
    explainer: shap.TreeExplainer,
    X_sample: pd.DataFrame,
    subgroup_series: pd.Series,
    top_n: int = 10,
) -> pd.DataFrame:
    """Exploratory subgroup SHAP analysis comparing attribution across categorical segments (e.g. CODE_GENDER).

    Note: This is an exploratory inspection of model attribution variation, NOT a fairness or bias audit.
    """
    logger.info("Computing exploratory subgroup SHAP comparison...")
    feature_names = list(X_sample.columns)

    subgroups = [g for g in subgroup_series.unique() if pd.notna(g) and str(g).lower() not in ("none", "nan")]
    subgroups.sort()

    group_top_dfs = {}
    for group in subgroups:
        mask = (subgroup_series == group).values
        if np.sum(mask) < 50:
            continue
        X_grp = X_sample[mask]
        vals_grp = explainer.shap_values(X_grp)
        mean_abs_grp = np.mean(np.abs(vals_grp), axis=0)

        df_grp = pd.DataFrame({
            "feature": feature_names,
            f"mean_abs_shap_{group}": np.round(mean_abs_grp, 6),
        }).sort_values(f"mean_abs_shap_{group}", ascending=False).reset_index(drop=True)
        df_grp[f"rank_{group}"] = df_grp.index + 1
        group_top_dfs[str(group)] = df_grp

    # Combine top features across subgroups
    all_top_features = set()
    for g, df_g in group_top_dfs.items():
        all_top_features.update(df_g["feature"].head(top_n).tolist())

    combined_rows = []
    for feat in sorted(all_top_features):
        row = {"feature": feat}
        for g, df_g in group_top_dfs.items():
            f_row = df_g[df_g["feature"] == feat]
            if not f_row.empty:
                row[f"mean_abs_shap_{g}"] = float(f_row[f"mean_abs_shap_{g}"].values[0])
                row[f"rank_{g}"] = int(f_row[f"rank_{g}"].values[0])
            else:
                row[f"mean_abs_shap_{g}"] = 0.0
                row[f"rank_{g}"] = len(feature_names)
        combined_rows.append(row)

    comp_df = pd.DataFrame(combined_rows)
    # Sort by primary subgroup mean_abs_shap
    first_col = [c for c in comp_df.columns if c.startswith("mean_abs_shap_")][0]
    comp_df = comp_df.sort_values(first_col, ascending=False).reset_index(drop=True)

    logger.info(f"Subgroup comparison compiled across {list(group_top_dfs.keys())}.")
    return comp_df


def select_local_cases(
    model: XGBClassifier,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    val_ids: np.ndarray,
    f1_threshold: float = 0.17,
) -> dict[str, dict[str, Any]]:
    """Select representative High-Risk, Low-Risk, and Borderline/Medium-Risk applicants from VALIDATION."""
    y_prob = model.predict_proba(X_val)[:, 1]

    # High-Risk: maximum predicted probability
    high_idx = int(np.argmax(y_prob))

    # Low-Risk: minimum predicted probability
    low_idx = int(np.argmin(y_prob))

    # Medium-Risk / Borderline: applicant closest to decision threshold
    diffs = np.abs(y_prob - f1_threshold)
    med_idx = int(np.argmin(diffs))

    cases = {
        "high_risk": {
            "category": "High-Risk Applicant",
            "val_index": high_idx,
            "sk_id_curr": int(val_ids[high_idx]),
            "predicted_probability": float(round(y_prob[high_idx], 4)),
            "actual_target": int(y_val[high_idx]),
            "decision_threshold": f1_threshold,
            "prediction_at_threshold": int(y_prob[high_idx] >= f1_threshold),
        },
        "low_risk": {
            "category": "Low-Risk Applicant",
            "val_index": low_idx,
            "sk_id_curr": int(val_ids[low_idx]),
            "predicted_probability": float(round(y_prob[low_idx], 4)),
            "actual_target": int(y_val[low_idx]),
            "decision_threshold": f1_threshold,
            "prediction_at_threshold": int(y_prob[low_idx] >= f1_threshold),
        },
        "medium_risk": {
            "category": "Borderline / Medium-Risk Applicant",
            "val_index": med_idx,
            "sk_id_curr": int(val_ids[med_idx]),
            "predicted_probability": float(round(y_prob[med_idx], 4)),
            "actual_target": int(y_val[med_idx]),
            "decision_threshold": f1_threshold,
            "prediction_at_threshold": int(y_prob[med_idx] >= f1_threshold),
        },
    }

    logger.info(
        f"Selected local explanation cases: "
        f"High: SK_ID_CURR={cases['high_risk']['sk_id_curr']} (p={cases['high_risk']['predicted_probability']:.4f}), "
        f"Low: SK_ID_CURR={cases['low_risk']['sk_id_curr']} (p={cases['low_risk']['predicted_probability']:.4f}), "
        f"Medium: SK_ID_CURR={cases['medium_risk']['sk_id_curr']} (p={cases['medium_risk']['predicted_probability']:.4f})"
    )
    return cases


# ======================================================================
# Publication-Quality Diagnostic Plotting
# ======================================================================

def plot_global_bar(
    shap_df: pd.DataFrame,
    output_path: pathlib.Path,
    top_n: int = 20,
) -> None:
    """Plot horizontal bar chart of top N mean absolute SHAP values."""
    top_df = shap_df.head(top_n).iloc[::-1]

    plt.figure(figsize=(9, 7), dpi=300)
    plt.barh(top_df["feature"], top_df["mean_abs_shap"], color="#1f77b4", edgecolor="black", alpha=0.85)

    plt.title(f"Global SHAP Feature Attribution — Top {top_n} Features (Validation)", fontsize=13, fontweight="bold")
    plt.xlabel("Mean |SHAP Value| (Average Impact on Model Output Magnitude)", fontsize=10)
    plt.ylabel("Feature Name", fontsize=10)
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved global SHAP bar plot to {output_path}")


def plot_beeswarm(
    expl: shap.Explanation,
    output_path: pathlib.Path,
    max_display: int = 20,
) -> None:
    """Plot SHAP beeswarm summary plot showing magnitude and directional contribution."""
    plt.figure(figsize=(10, 8), dpi=300)
    shap.plots.beeswarm(expl, max_display=max_display, show=False)
    plt.title("SHAP Summary Beeswarm Plot — Top Features (Validation)", fontsize=13, fontweight="bold")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved SHAP beeswarm plot to {output_path}")


def plot_dependence(
    expl: shap.Explanation,
    feature_name: str,
    output_path: pathlib.Path,
) -> None:
    """Plot SHAP dependence plot for an individual feature."""
    plt.figure(figsize=(8, 6), dpi=300)
    shap.plots.scatter(expl[:, feature_name], color=expl, show=False)
    plt.title(f"SHAP Dependence Plot — {feature_name}", fontsize=12, fontweight="bold")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved SHAP dependence plot for {feature_name} to {output_path}")


def plot_waterfall(
    expl_row: shap.Explanation,
    output_path: pathlib.Path,
    title_prefix: str = "",
    max_display: int = 15,
) -> None:
    """Plot local waterfall plot for an individual applicant explanation."""
    plt.figure(figsize=(9, 7), dpi=300)
    shap.plots.waterfall(expl_row, max_display=max_display, show=False)
    if title_prefix:
        plt.title(f"SHAP Local Explanation — {title_prefix}", fontsize=12, fontweight="bold")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved local SHAP waterfall plot to {output_path}")


def plot_stability(
    stability_df: pd.DataFrame,
    output_path: pathlib.Path,
    top_n: int = 20,
) -> None:
    """Plot bar chart of feature selection frequency across repeated bootstrap samples."""
    top_df = stability_df.head(top_n).iloc[::-1]

    plt.figure(figsize=(9, 7), dpi=300)
    colors = ["#2ca02c" if f == 1.0 else "#ff7f0e" for f in top_df["selection_frequency"]]
    plt.barh(top_df["feature"], top_df["selection_frequency"] * 100, color=colors, edgecolor="black", alpha=0.85)

    plt.title(f"SHAP Explanation Stability — Top {top_n} Features (10 Bootstrap Repeats)", fontsize=13, fontweight="bold")
    plt.xlabel("Selection Frequency in Top-20 Features (%)", fontsize=10)
    plt.ylabel("Feature Name", fontsize=10)
    plt.xlim([0, 105])
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved SHAP stability plot to {output_path}")


def plot_subgroup_comparison(
    subgroup_df: pd.DataFrame,
    output_path: pathlib.Path,
    top_n: int = 10,
) -> None:
    """Plot grouped horizontal bar chart comparing feature importance across subgroups."""
    top_df = subgroup_df.head(top_n).iloc[::-1]
    features = top_df["feature"]
    y_pos = np.arange(len(features))
    height = 0.35

    val_f = top_df["mean_abs_shap_F"] if "mean_abs_shap_F" in top_df.columns else top_df.iloc[:, 1]
    val_m = top_df["mean_abs_shap_M"] if "mean_abs_shap_M" in top_df.columns else top_df.iloc[:, 3]

    plt.figure(figsize=(10, 7), dpi=300)
    plt.barh(y_pos - height/2, val_f, height, label="Female (F)", color="#1f77b4", edgecolor="black", alpha=0.85)
    plt.barh(y_pos + height/2, val_m, height, label="Male (M)", color="#ff7f0e", edgecolor="black", alpha=0.85)

    plt.yticks(y_pos, features)
    plt.title(f"Exploratory Subgroup SHAP Comparison (CODE_GENDER) — Top {top_n} Features", fontsize=13, fontweight="bold")
    plt.xlabel("Mean |SHAP Value|", fontsize=10)
    plt.ylabel("Feature Name", fontsize=10)
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", fontsize=10, frameon=True)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved subgroup comparison plot to {output_path}")


def persist_explainability_artifacts(
    shap_global_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    metadata: dict[str, Any],
    output_dir: Union[str, pathlib.Path] = ARTIFACTS_EXPLAIN_DIR,
) -> None:
    """Persist explainability artifacts to artifacts/explainability/."""
    out_dir = pathlib.Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    g_path = out_dir / "shap_global_importance.csv"
    s_path = out_dir / "shap_stability.csv"
    m_path = out_dir / "shap_metadata.json"

    shap_global_df.to_csv(g_path, index=False)
    stability_df.to_csv(s_path, index=False)
    with open(m_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Persisted explainability artifacts to {out_dir}")
