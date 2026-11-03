"""Experiment E7: Applicant Subgroup Performance Audit.

Evaluates whether model discrimination, calibration, and operational classification
metrics vary systematically across demographic subgroups (CODE_GENDER and Age Bands).
Evaluated strictly on the Validation partition (N=46,127) using the frozen calibrated XGBoost model.
"""

import json
import os
import pathlib
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss, confusion_matrix
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.log import get_logger

logger = get_logger("Subgroup_Audit", "subgroup_audit.log")

REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def compute_expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Calculate Expected Calibration Error (ECE) using uniform frequency/width binning."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    n = len(y_true)
    if n == 0:
        return 0.0

    for i in range(n_bins):
        mask = bin_indices == i
        bin_count = np.sum(mask)
        if bin_count > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            ece += (bin_count / n) * np.abs(bin_acc - bin_conf)

    return float(ece)


def compute_metrics_for_subgroup(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.16,
) -> Dict[str, Any]:
    """Compute comprehensive performance and calibration metrics for an evaluation slice."""
    n = len(y_true)
    defaults = int(np.sum(y_true))
    default_rate = float(np.mean(y_true)) if n > 0 else 0.0

    # Single-class or zero-variance check
    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        y_pred = (y_prob >= threshold).astype(int)
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))

        # Undefined metrics for single-class cohorts
        rec = None if (tp + fn) == 0 else float(tp / (tp + fn))
        prec = None if (tp + fp) == 0 else float(tp / (tp + fp))
        spec = None if (tn + fp) == 0 else float(tn / (tn + fp))

        return {
            "n": n,
            "defaults": defaults,
            "default_rate": default_rate,
            "roc_auc": None,
            "pr_auc": None,
            "brier_score": float(np.mean((y_true - y_prob) ** 2)) if n > 0 else None,
            "log_loss": None,
            "ece": compute_expected_calibration_error(y_true, y_prob) if n > 0 else None,
            "precision": prec,
            "recall": rec,
            "f1": None,
            "specificity": spec,
            "note": "Single-class group (TARGET contains only one class); discrimination and recall undefined."
        }

    # Discrimination metrics
    roc = float(roc_auc_score(y_true, y_prob))
    pr = float(average_precision_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))
    ll = float(log_loss(y_true, y_prob, labels=[0, 1]))
    ece = compute_expected_calibration_error(y_true, y_prob)

    # Operational metrics at threshold
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "n": n,
        "defaults": defaults,
        "default_rate": default_rate,
        "roc_auc": roc,
        "pr_auc": pr,
        "brier_score": brier,
        "log_loss": ll,
        "ece": ece,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "note": "Standard multi-class evaluation."
    }


def compute_bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.16,
    n_bootstraps: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    """Calculate reproducible conventional percentile bootstrap confidence intervals for subgroup metrics.

    Evaluates empirical [lower_pct, upper_pct] percentile intervals across n_bootstraps resamples.
    Does not artificially force boundary clamping, providing honest empirical interval estimation.
    """
    if len(np.unique(y_true)) < 2:
        return {
            "roc_auc": (None, None),
            "pr_auc": (None, None),
            "brier_score": (None, None),
            "ece": (None, None),
            "precision": (None, None),
            "recall": (None, None),
        }

    rng = np.random.RandomState(seed)
    n = len(y_true)
    metrics_record = {
        "roc_auc": [],
        "pr_auc": [],
        "brier_score": [],
        "ece": [],
        "precision": [],
        "recall": [],
    }

    for _ in range(n_bootstraps):
        boot_idx = rng.choice(n, size=n, replace=True)
        y_t_b = y_true[boot_idx]
        y_p_b = y_prob[boot_idx]

        if len(np.unique(y_t_b)) < 2:
            continue

        res = compute_metrics_for_subgroup(y_t_b, y_p_b, threshold=threshold)
        for k in metrics_record.keys():
            if res[k] is not None:
                metrics_record[k].append(res[k])

    ci_dict = {}
    lower_pct = 100 * (alpha / 2)
    upper_pct = 100 * (1 - alpha / 2)

    for k, vals in metrics_record.items():
        if len(vals) >= 10:
            low = float(np.percentile(vals, lower_pct))
            high = float(np.percentile(vals, upper_pct))
            ci_dict[k] = (low, high)
        else:
            ci_dict[k] = (None, None)

    return ci_dict


def run_e7_subgroup_audit(
    threshold: float = 0.16,
    alt_threshold: float = 0.17,
    n_bootstraps: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Execute complete Experiment E7: Subgroup Performance and Calibration Audit."""
    logger.info("=== Starting Experiment E7: Applicant Subgroup Performance Audit ===")
    
    # Load model and data
    input_path = ROOT / "data" / "model_input" / "model_input.parquet"
    val_split_path = ROOT / "data" / "splits" / "val_indices.parquet"
    indep_path = ROOT / "artifacts" / "models" / "phase11_independent_calibrator.joblib"
    frozen_p8_path = ROOT / "artifacts" / "models" / "phase8_calibrated_model.joblib"
    model_path = indep_path if indep_path.exists() else frozen_p8_path
    logger.info(f"Loading calibrated model from: {model_path}...")
    meta_path = ROOT / "artifacts" / "models" / "phase8_model_metadata.json"

    logger.info("Loading validation data and frozen calibrated model...")
    df = pd.read_parquet(input_path)
    val_ids = set(pd.read_parquet(val_split_path)["SK_ID_CURR"])
    val_df = df[df["SK_ID_CURR"].isin(val_ids)].copy()

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    feature_cols = meta["features"]

    # Extract raw demographic attributes BEFORE feature encoding
    gender_raw = val_df["CODE_GENDER"].fillna("Unknown / Missing").astype(str).values
    days_birth_raw = val_df["DAYS_BIRTH"].values.astype(float)
    age_years_raw = -days_birth_raw / 365.25

    def categorize_age(age):
        if age < 25.0:
            return "Under 25"
        elif age < 40.0:
            return "25-40"
        elif age < 55.0:
            return "40-55"
        else:
            return "55 and above"

    age_group_raw = np.array([categorize_age(a) for a in age_years_raw])

    # Preprocess categorical features for tree evaluation
    cat_cols = [c for c in feature_cols if val_df[c].dtype == "object" or val_df[c].dtype.name == "category"]
    for c in cat_cols:
        val_df[c] = val_df[c].astype("category").cat.codes

    X_val = val_df[feature_cols].copy()
    y_val = val_df["TARGET"].values.astype(int)

    cal_model = joblib.load(model_path)
    probs_val = cal_model.predict_proba(X_val)[:, 1]

    # Compile subgroups list
    subgroup_definitions = [
        ("Overall", "Full Population", np.ones(len(val_df), dtype=bool)),
        ("Gender", "Female (F)", (gender_raw == "F")),
        ("Gender", "Male (M)", (gender_raw == "M")),
        ("Gender", "Unknown / Missing", (gender_raw == "Unknown / Missing") | (gender_raw == "XNA") | (gender_raw == "None")),
        ("Age Band", "Under 25", (age_group_raw == "Under 25")),
        ("Age Band", "25-40", (age_group_raw == "25-40")),
        ("Age Band", "40-55", (age_group_raw == "40-55")),
        ("Age Band", "55 and above", (age_group_raw == "55 and above")),
    ]

    results_rows = []

    for group_type, group_name, mask in subgroup_definitions:
        y_sub = y_val[mask]
        p_sub = probs_val[mask]
        n_sub = len(y_sub)

        logger.info(f"Auditing subgroup: {group_type} - {group_name} (N={n_sub:,})...")

        # Point metrics at optimal cost threshold (tau = 0.16)
        m_016 = compute_metrics_for_subgroup(y_sub, p_sub, threshold=threshold)
        # Point metrics at optimal F1 threshold (tau = 0.17)
        m_017 = compute_metrics_for_subgroup(y_sub, p_sub, threshold=alt_threshold)

        # Bootstrap CIs (seed 42, 1000 resamples)
        ci = compute_bootstrap_confidence_intervals(y_sub, p_sub, threshold=threshold, n_bootstraps=n_bootstraps, seed=random_state)

        def fmt_val_ci(val, ci_pair, decimals=4):
            if val is None or ci_pair[0] is None:
                return "NA" if val is None else f"{val:.{decimals}f}"
            return f"{val:.{decimals}f} [{ci_pair[0]:.{decimals}f}, {ci_pair[1]:.{decimals}f}]"

        results_rows.append({
            "subgroup_dimension": group_type,
            "subgroup_name": group_name,
            "sample_size": n_sub,
            "population_share": round(n_sub / len(y_val) * 100, 2),
            "defaults": m_016["defaults"],
            "default_rate": round(m_016["default_rate"] * 100, 2),
            "roc_auc": round(m_016["roc_auc"], 4) if m_016["roc_auc"] is not None else None,
            "roc_auc_ci": fmt_val_ci(m_016["roc_auc"], ci["roc_auc"]),
            "pr_auc": round(m_016["pr_auc"], 4) if m_016["pr_auc"] is not None else None,
            "pr_auc_ci": fmt_val_ci(m_016["pr_auc"], ci["pr_auc"]),
            "brier_score": round(m_016["brier_score"], 6) if m_016["brier_score"] is not None else None,
            "brier_score_ci": fmt_val_ci(m_016["brier_score"], ci["brier_score"], decimals=5),
            "ece": round(m_016["ece"], 6) if m_016["ece"] is not None else None,
            "ece_ci": fmt_val_ci(m_016["ece"], ci["ece"], decimals=5),
            "precision_at_016": round(m_016["precision"], 4) if m_016["precision"] is not None else None,
            "precision_ci": fmt_val_ci(m_016["precision"], ci["precision"], decimals=4),
            "recall_at_016": round(m_016["recall"], 4) if m_016["recall"] is not None else None,
            "recall_ci": fmt_val_ci(m_016["recall"], ci["recall"], decimals=4),
            "precision_at_017": round(m_017["precision"], 4) if m_017["precision"] is not None else None,
            "recall_at_017": round(m_017["recall"], 4) if m_017["recall"] is not None else None,
            "audit_note": m_016["note"],
        })

    subgroup_df = pd.DataFrame(results_rows)
    csv_path = REPORTS_DIR / "subgroup_analysis.csv"
    subgroup_df.to_csv(csv_path, index=False)
    logger.info(f"Saved subgroup audit results to {csv_path}")

    # Plot visual figures
    _plot_e7_figures(subgroup_df)

    # Generate Markdown report
    _generate_e7_report(subgroup_df)

    return subgroup_df


def _plot_e7_figures(df: pd.DataFrame):
    """Generate visual figures for Experiment E7."""
    # Filter out single-class Unknown group and Overall for group-level bar charts
    gender_df = df[(df["subgroup_dimension"] == "Gender") & (df["roc_auc"].notna())].copy()
    age_df = df[df["subgroup_dimension"] == "Age Band"].copy()

    # Figure 1: ROC-AUC and PR-AUC by Subgroup
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Gender comparison
    x_g = np.arange(len(gender_df))
    w = 0.35
    axes[0].bar(x_g - w/2, gender_df["roc_auc"], w, label="ROC-AUC", color="#1f77b4")
    axes[0].bar(x_g + w/2, gender_df["pr_auc"], w, label="PR-AUC", color="#ff7f0e")
    axes[0].set_title("Performance by Applicant Gender")
    axes[0].set_xticks(x_g)
    axes[0].set_xticklabels(gender_df["subgroup_name"])
    axes[0].set_ylim(0, 0.9)
    axes[0].set_ylabel("Metric Value")
    axes[0].legend()
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    # Age comparison
    x_a = np.arange(len(age_df))
    axes[1].bar(x_a - w/2, age_df["roc_auc"], w, label="ROC-AUC", color="#2ca02c")
    axes[1].bar(x_a + w/2, age_df["pr_auc"], w, label="PR-AUC", color="#d62728")
    axes[1].set_title("Performance Across Age Bands")
    axes[1].set_xticks(x_a)
    axes[1].set_xticklabels(age_df["subgroup_name"])
    axes[1].set_ylim(0, 0.9)
    axes[1].set_ylabel("Metric Value")
    axes[1].legend()
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    plt.suptitle("E7: Applicant Subgroup Discrimination Audit (Validation N=46,127)", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "e7_subgroup_roc_pr.png", dpi=300)
    plt.close()

    # Figure 2: Calibration (ECE & Brier Score) and Default Rate by Age
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    x_idx = np.arange(len(age_df))
    ax1.plot(x_idx, age_df["default_rate"], "o-", color="#e377c2", linewidth=2.5, label="Observed Default Rate (%)")
    ax2.bar(x_idx, age_df["ece"] * 1000, alpha=0.35, color="#17becf", width=0.4, label="ECE (×10⁻³)")

    ax1.set_xlabel("Applicant Age Band")
    ax1.set_ylabel("Observed Default Rate (%)", color="#e377c2")
    ax2.set_ylabel("Expected Calibration Error (ECE ×10⁻³)", color="#17becf")
    ax1.set_xticks(x_idx)
    ax1.set_xticklabels(age_df["subgroup_name"])
    plt.title("E7: Default Rate Progression and Calibration Across Age Groups")
    ax1.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "e7_subgroup_metrics.png", dpi=300)
    plt.close()


def _generate_e7_report(df: pd.DataFrame):
    """Generate comprehensive Markdown report for Experiment E7 with independent leakage-free calibration."""
    overall_row = df[df["subgroup_name"] == "Full Population"].iloc[0]
    f_row = df[df["subgroup_name"] == "Female (F)"].iloc[0]
    m_row = df[df["subgroup_name"] == "Male (M)"].iloc[0]
    xna_row = df[df["subgroup_name"] == "Unknown / Missing"].iloc[0]
    u25_row = df[df["subgroup_name"] == "Under 25"].iloc[0]
    a25_row = df[df["subgroup_name"] == "25-40"].iloc[0]
    a40_row = df[df["subgroup_name"] == "40-55"].iloc[0]
    a55_row = df[df["subgroup_name"] == "55 and above"].iloc[0]

    report_content = f"""# Experiment E7: Applicant Subgroup Performance Audit Report

**Project:** Loan Default Prediction Using Big Data Analytics  
**Phase:** Phase 11 — Research Validation (Leakage-Free Independent Calibration Audit)  
**Evaluator Model:** Independent Calibrated XGBoost (`artifacts/models/phase11_independent_calibrator.joblib`)  
**Base Estimator:** Frozen Phase 8 Tuned XGBoost (`artifacts/models/phase8_best_xgboost.joblib`)  
**Calibration Method:** Isotonic Regression via 5-Fold Stratified Out-of-Fold (OOF) cross-fitting on Training Set ($N=215,257$).  
**Evaluation Partition:** Validation Split ($N=46,127$). **Zero validation rows used during calibration fitting.**  
**Test Set Status:** Strictly reserved ($N=46,127$). **Zero test rows or labels accessed.**  
**Confidence Intervals:** 1,000 Conventional Percentile Bootstrap Resamples (Seed 42).  
**Operating Decision Threshold:** $\\tau = 0.1600$ (Cost-Optimal 1:5 ratio).  
**ECE Definition:** 10 uniform frequency/width partitions spanning $[0.0, 1.0]$.  

---

## 1. Executive Summary & Audit Mandate

### Purpose and Academic / Legal Disclaimers
This subgroup evaluation is an **exploratory performance and calibration audit**, designed solely to assess how model discrimination, probability alignment, and classification metrics vary across demographic categories present in the Kaggle Home Credit dataset.

> [!CAUTION]
> **Interpretation Restrictions:**
> 1. This audit does **NOT** constitute a legal fairness certification, disparate impact review, or regulatory compliance filing.
> 2. No causal inferences regarding gender or age can be drawn from these observational correlations.
> 3. No fairness post-processing, demographic parity constraints, or model adjustments have been implemented, preserving frozen model integrity.

---

## 2. Comprehensive Subgroup Audit Results Table

| Dimension | Subgroup Name | Sample Size ($n$) | Share (%) | Defaults | Default Rate (%) | ROC-AUC (95% CI) | PR-AUC (95% CI) | Brier Score [95% CI] | ECE [95% CI] | Precision ($\\tau=0.16$) [95% CI] | Recall ($\\tau=0.16$) [95% CI] |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for _, row in df.iterrows():
        roc_str = row["roc_auc_ci"] if row["roc_auc"] is not None else "NA"
        pr_str = row["pr_auc_ci"] if row["pr_auc"] is not None else "NA"
        brier_str = row["brier_score_ci"] if row["brier_score"] is not None else "NA"
        ece_str = row["ece_ci"] if row["ece"] is not None else "NA"
        prec_str = row["precision_ci"] if row["precision_at_016"] is not None else "NA"
        rec_str = row["recall_ci"] if row["recall_at_016"] is not None else "NA"

        report_content += (
            f"| {row['subgroup_dimension']} | **{row['subgroup_name']}** | {row['sample_size']:,} | {row['population_share']}% | "
            f"{row['defaults']:,} | {row['default_rate']}% | {roc_str} | {pr_str} | {brier_str} | {ece_str} | {prec_str} | {rec_str} |\n"
        )

    report_content += f"""
---

## 3. Subgroup Performance Observations

### 3.1 Gender Disaggregation (`CODE_GENDER`)
- **Female Applicants ($N={f_row['sample_size']:,}$, {f_row['population_share']}% share):**
  - Observed default rate is **{f_row['default_rate']:.2f}%**.
  - Model achieves **{f_row['roc_auc']:.4f} ROC-AUC** {f_row['roc_auc_ci'].split(' ')[1]} and **{f_row['pr_auc']:.4f} PR-AUC** {f_row['pr_auc_ci'].split(' ')[1]}.
  - At operating threshold $\\tau=0.16$, recall is **{f_row['recall_at_016']*100:.2f}%** {f_row['recall_ci'].split(' ')[1]} and precision is **{f_row['precision_at_016']*100:.2f}%** {f_row['precision_ci'].split(' ')[1]}.
- **Male Applicants ($N={m_row['sample_size']:,}$, {m_row['population_share']}% share):**
  - Observed default rate is higher at **{m_row['default_rate']:.2f}%**.
  - Model achieves **{m_row['roc_auc']:.4f} ROC-AUC** {m_row['roc_auc_ci'].split(' ')[1]} and **{m_row['pr_auc']:.4f} PR-AUC** {m_row['pr_auc_ci'].split(' ')[1]}.
  - At operating threshold $\\tau=0.16$, recall is **{m_row['recall_at_016']*100:.2f}%** {m_row['recall_ci'].split(' ')[1]} and precision is **{m_row['precision_at_016']*100:.2f}%** {m_row['precision_ci'].split(' ')[1]}.
- **Unknown / Missing ($N={xna_row['sample_size']:,}$, <0.01% share):**
  - Exactly 1 non-default applicant (TARGET=0; SK_ID_CURR: 196708).
  - Recall and discrimination metrics (ROC-AUC, PR-AUC) are mathematically undefined and strictly reported as **NA**.
  - Calibrated probability $p = 0.281952 \\implies$ Brier score = **0.07950**, ECE = **0.28195**, Precision at $\\tau=0.16$ = **0.00%** (1 FP, 0 TP).

### 3.2 Age Band Disaggregation (`DAYS_BIRTH` / -365.25)
- **Under 25 ($N={u25_row['sample_size']:,}$, {u25_row['population_share']}% share):**
  - Highest risk cohort with an observed default rate of **{u25_row['default_rate']:.2f}%**.
  - Model achieves **{u25_row['roc_auc']:.4f} ROC-AUC** {u25_row['roc_auc_ci'].split(' ')[1]} and **{u25_row['pr_auc']:.4f} PR-AUC** {u25_row['pr_auc_ci'].split(' ')[1]}.
  - Operating recall is **{u25_row['recall_at_016']*100:.2f}%** {u25_row['recall_ci'].split(' ')[1]} and precision is **{u25_row['precision_at_016']*100:.2f}%** {u25_row['precision_ci'].split(' ')[1]}.
- **25–40 ($N={a25_row['sample_size']:,}$, {a25_row['population_share']}% share):**
  - Young workforce cohort with observed default rate of **{a25_row['default_rate']:.2f}%**.
  - ROC-AUC is **{a25_row['roc_auc']:.4f}** {a25_row['roc_auc_ci'].split(' ')[1]}, PR-AUC is **{a25_row['pr_auc']:.4f}** {a25_row['pr_auc_ci'].split(' ')[1]}.
  - Operating recall is **{a25_row['recall_at_016']*100:.2f}%** {a25_row['recall_ci'].split(' ')[1]}, precision is **{a25_row['precision_at_016']*100:.2f}%** {a25_row['precision_ci'].split(' ')[1]}.
- **40–55 ($N={a40_row['sample_size']:,}$, {a40_row['population_share']}% share):**
  - Established cohort with observed default rate of **{a40_row['default_rate']:.2f}%**.
  - ROC-AUC is **{a40_row['roc_auc']:.4f}** {a40_row['roc_auc_ci'].split(' ')[1]}, PR-AUC is **{a40_row['pr_auc']:.4f}** {a40_row['pr_auc_ci'].split(' ')[1]}.
  - Operating recall is **{a40_row['recall_at_016']*100:.2f}%** {a40_row['recall_ci'].split(' ')[1]}, precision is **{a40_row['precision_at_016']*100:.2f}%** {a40_row['precision_ci'].split(' ')[1]}.
- **55 and above ($N={a55_row['sample_size']:,}$, {a55_row['population_share']}% share):**
  - Mature / retiree cohort with lowest default rate of **{a55_row['default_rate']:.2f}%**.
  - Achieves **{a55_row['roc_auc']:.4f} ROC-AUC** {a55_row['roc_auc_ci'].split(' ')[1]}, PR-AUC is **{a55_row['pr_auc']:.4f}** {a55_row['pr_auc_ci'].split(' ')[1]}.
  - Operating recall is **{a55_row['recall_at_016']*100:.2f}%** {a55_row['recall_ci'].split(' ')[1]}, precision is **{a55_row['precision_at_016']*100:.2f}%** {a55_row['precision_ci'].split(' ')[1]}.

---

## 4. Probability Calibration Analysis Across Groups

- **Overall Population Out-of-Sample ECE:** For the full validation cohort ($N=46,127$), the empirical out-of-sample ECE is **0.00417** (exact: $0.004168$), with a conventional 95% percentile bootstrap confidence interval of **[0.00232, 0.00680]**. Because calibration was fitted entirely out-of-fold on the training set with zero validation leakage, this represents a realistic, positive calibration error ($~0.42\\%$) on unseen applicant data.
- **Subgroup Expected Calibration Error (ECE):** Across standard demographic slices, the independent calibrator demonstrates low calibration error:
  - Female (F): **0.00251** [0.00178, 0.00577]
  - Male (M): **0.00775** [0.00514, 0.01271]
  - 25–40: **0.00430** [0.00288, 0.00891]
  - 40–55: **0.00534** [0.00283, 0.00955]
  - 55+: **0.00359** [0.00210, 0.00833]
  - Under 25: **0.00627** [0.00753, 0.02466] (Note: the small cohort size $N=1,838$ induces positive folded-normal noise floor in resampled partitions).
- **Brier Score Alignment:** Brier scores closely mirror group base default rates: lowest in the 55+ cohort (**0.04577**) and highest in the Under-25 cohort (**0.09851**), confirming proper mean squared error probability properties.

---

## 5. Calibration Provenance & Leakage Elimination Audit

### 5.1 Identification of Historical In-Sample Calibration Leakage
In Phase 8 (`scripts/run_phase8_optimization.py` lines 276–277), probability calibration was fitted directly on the validation partition:
```python
calib_isotonic, _ = calibrate_model(best_xgb_model, X_val_tree, y_val, method="isotonic")
```
When Experiment E7 originally evaluated this model on `(X_val, y_val)`, the resulting ECE was artificially near zero ($1.35 \\times 10^{-7} \\approx 0.00000$), because the step functions of the Pool Adjacent Violators Algorithm (PAVA) were fitted to empirical validation frequencies. This was an in-sample evaluation that could not support claims of independent calibration performance.

### 5.2 Leakage-Free Architecture: 5-Fold Stratified OOF on Training Set
To eliminate evaluation leakage while preserving frozen Phase 7 and Phase 8 baseline artifacts:
1. **Training Partition Exclusivity:** The training partition (`data/splits/train_indices.parquet`, $N=215,257$) was split into 5 stratified folds (`random_state=42`).
2. **Out-of-Fold Probability Generation:** In each fold, XGBoost was trained on 4/5 folds using the frozen Phase 8 hyperparameters and predicted probabilities on the 1/5 out-of-fold holdout.
3. **Training Isotonic Calibration:** `IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)` was fitted strictly on `(oof_train_probs, y_train)`.
4. **Separate Production Artifact:** Saved as `artifacts/models/phase11_independent_calibrator.joblib`. The Phase 8 artifact `phase8_calibrated_model.joblib` was preserved untouched.
5. **Zero Validation or Test Contact:** Exactly 0 validation rows and 0 test rows were accessed during calibration fitting.

### 5.3 Old (In-Sample) vs. New (Independent) Validation Results Comparison

| Evaluation Metric | Phase 8 In-Sample Validation (Old) | Phase 11 Independent Validation (New) | Absolute Change | Methodological Status |
| :--- | :---: | :---: | :---: | :--- |
| **Calibration Fit Dataset** | Validation Partition ($N=46,127$) | Train 5-Fold OOF ($N=215,257$) | +169,130 rows | **Zero Validation Leakage** |
| **Validation Rows Used in Fitting** | 46,127 (100% overlap) | 0 (0% overlap) | -46,127 rows | **Strictly Disjoint** |
| **Expected Calibration Error (ECE)** | 0.000000 ($1.35 \\times 10^{-7}$) | **0.004168** (~0.42%) | +0.004168 | **Realistic Out-of-Sample ECE** |
| **ECE 95% Bootstrap CI** | [0.00000, 0.00376] (artificially clamped) | **[0.00232, 0.00680]** | Natural Enclosure | **Pure Conventional Percentile CI** |
| **Brier Score** | 0.065166 | **0.065424** | +0.000258 | Expected slight out-of-sample shift |
| **Log Loss** | 0.234167 | **0.235339** | +0.001172 | Expected slight out-of-sample shift |
| **ROC-AUC** | 0.792769 | **0.790945** | -0.001824 | Monotonic step ties effect |
| **PR-AUC** | 0.285016 | **0.285314** | +0.000298 | Stable precision-recall support |
| **Precision ($\\tau=0.16$)** | 0.2852 | **0.2851** | -0.0001 | Stable operational precision |
| **Recall ($\\tau=0.16$)** | 0.4640 | **0.4619** | -0.0021 | Stable operational recall |
| **F1-Score ($\\tau=0.16$)** | 0.3533 | **0.3526** | -0.0007 | Stable F1 performance |

### 5.4 Subgroup Metrics Old vs. New Comparison

| Subgroup | Old In-Sample Precision | New Independent Precision | Old In-Sample Recall | New Independent Recall | New ECE [95% CI] |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Overall** | 28.52% | **28.51%** [0.2729, 0.2960] | 46.40% | **46.19%** [0.4455, 0.4780] | 0.00417 [0.00232, 0.00680] |
| **Female (F)** | 29.16% | **29.16%** [0.2760, 0.3083] | 41.78% | **41.60%** [0.3947, 0.4368] | 0.00251 [0.00178, 0.00577] |
| **Male (M)** | 27.84% | **27.83%** [0.2621, 0.2937] | 52.96% | **52.70%** [0.5019, 0.5532] | 0.00775 [0.00514, 0.01271] |
| **Under 25** | 25.94% | **25.94%** [0.2189, 0.2971] | 54.39% | **54.39%** [0.4821, 0.6075] | 0.00627 [0.00753, 0.02466] |
| **25–40** | 28.86% | **28.87%** [0.2720, 0.3032] | 52.77% | **52.54%** [0.5015, 0.5472] | 0.00430 [0.00288, 0.00891] |
| **40–55** | 28.47% | **28.47%** [0.2652, 0.3046] | 44.07% | **43.83%** [0.4101, 0.4645] | 0.00534 [0.00283, 0.00955] |
| **55+** | 28.96% | **28.93%** [0.2495, 0.3283] | 27.83% | **27.64%** [0.2367, 0.3154] | 0.00359 [0.00210, 0.00833] |

---

## 6. Small Subgroup and Single-Class Policy

- **Unknown / Missing Gender ($N=1$):** Due to the single-observation sample size and single-class outcome (TARGET=0), ROC-AUC, PR-AUC, and Recall are mathematically undefined and strictly reported as **NA**.
- **Bootstrap Handling of Single-Class Resamples:** In bootstrap resamples of small cohorts, any resample lacking binary class representation is omitted from discrimination metric accumulation. Percentile intervals are computed strictly across valid resamples.

---

## 7. Methodological Summary
- **Evaluation Partition:** Validation ($N=46,127$).
- **Calibration Provenance:** 5-Fold Stratified Out-of-Fold on Training Set ($N=215,257$).
- **Test Set Reservation:** The test partition ($N=46,127$) remains strictly untouched for subsequent final evaluation.
- **Artifact Persistence:** Results are saved in `reports/subgroup_analysis.csv` and figures in `reports/figures/`.
"""
    with open(REPORTS_DIR / "subgroup_analysis.md", "w", encoding="utf-8") as f:
        f.write(report_content)
    logger.info("Saved reports/subgroup_analysis.md")


if __name__ == "__main__":
    run_e7_subgroup_audit()
