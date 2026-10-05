"""Phase 8 Utility Functions for Evaluation, Threshold Analysis, Cost Analysis, and Calibration.

Provides:
- Comprehensive metric computation including Brier score, F2, F0.5, Specificity, and Balanced Accuracy
- Threshold grid evaluation [0.05, 0.95]
- Expected classification cost optimization across hypothetical FP:FN ratios (1:1, 1:2, 1:5, 1:10)
- Probability calibration evaluation and reliability curves
- Publication-quality diagnostic plotting
"""

import json
import pathlib
import sys
import time
from typing import Any, Optional, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.log import get_logger

logger = get_logger("Phase8Utils", "phase8_utils.log")


def compute_comprehensive_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_prob: np.ndarray,
    threshold: float = 0.50,
    model_name: str = "",
    strategy: str = "",
    split_name: str = "",
) -> dict[str, Any]:
    """Calculate comprehensive evaluation metrics across discrimination, calibration, and classification.

    Parameters
    ----------
    y_true : 1D array of binary ground-truth labels (0 or 1).
    y_prob : 1D array of predicted probabilities for positive class (1).
    threshold : Decision cutoff in [0, 1].
    model_name : Identifier for the model.
    strategy : Imbalance / tuning strategy description.
    split_name : Name of data partition ('train', 'validation', 'test').

    Returns
    -------
    dict[str, Any]
        Dictionary with all evaluated metrics.
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    # 1. Discrimination metrics (threshold-independent)
    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))

    # 2. Calibration / Probabilistic loss metrics (threshold-independent)
    loss = float(log_loss(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))

    # 3. Classification behavior (threshold-dependent)
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    f2 = float(fbeta_score(y_true, y_pred, beta=2.0, zero_division=0))
    f0_5 = float(fbeta_score(y_true, y_pred, beta=0.5, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    balanced_acc = float((rec + specificity) / 2.0)
    fpr = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0
    fnr = float(fn / (tp + fn)) if (tp + fn) > 0 else 0.0

    return {
        "model": model_name,
        "strategy": strategy,
        "split": split_name,
        "threshold": round(threshold, 4),
        "roc_auc": round(roc_auc, 6),
        "pr_auc": round(pr_auc, 6),
        "log_loss": round(loss, 6),
        "brier_score": round(brier, 6),
        "accuracy": round(acc, 6),
        "precision": round(prec, 6),
        "recall": round(rec, 6),
        "f1": round(f1, 6),
        "f2": round(f2, 6),
        "f0_5": round(f0_5, 6),
        "specificity": round(specificity, 6),
        "balanced_accuracy": round(balanced_acc, 6),
        "false_positive_rate": round(fpr, 6),
        "false_negative_rate": round(fnr, 6),
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "matrix": [[tn, fp], [fn, tp]],
        },
        "total_samples": len(y_true),
        "positive_samples": int(np.sum(y_true == 1)),
        "negative_samples": int(np.sum(y_true == 0)),
    }


def evaluate_threshold_grid(
    y_true: Union[pd.Series, np.ndarray],
    y_prob: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Evaluate classification behavior across a grid of decision thresholds on validation data.

    Parameters
    ----------
    y_true : 1D ground-truth labels.
    y_prob : 1D predicted probabilities for class 1.
    thresholds : 1D array of cutoffs in [0.05, 0.95]. Defaults to np.arange(0.05, 0.96, 0.01).

    Returns
    -------
    pd.DataFrame
        Table with metrics evaluated for each threshold.
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.01)

    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()

    rows = []
    for th in thresholds:
        th = round(float(th), 4)
        y_pred = (y_prob >= th).astype(int)

        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        f2 = float(fbeta_score(y_true, y_pred, beta=2.0, zero_division=0))
        f0_5 = float(fbeta_score(y_true, y_pred, beta=0.5, zero_division=0))
        acc = float(accuracy_score(y_true, y_pred))

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = [int(v) for v in cm.ravel()]

        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        balanced_acc = float((rec + specificity) / 2.0)
        fpr = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0
        fnr = float(fn / (tp + fn)) if (tp + fn) > 0 else 0.0

        rows.append({
            "threshold": th,
            "precision": round(prec, 6),
            "recall": round(rec, 6),
            "f1": round(f1, 6),
            "f2": round(f2, 6),
            "f0_5": round(f0_5, 6),
            "accuracy": round(acc, 6),
            "specificity": round(specificity, 6),
            "balanced_accuracy": round(balanced_acc, 6),
            "false_positive_rate": round(fpr, 6),
            "false_negative_rate": round(fnr, 6),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        })

    return pd.DataFrame(rows)


def evaluate_cost_grid(
    y_true: Union[pd.Series, np.ndarray],
    y_prob: np.ndarray,
    cost_ratios: Optional[list[tuple[int, int]]] = None,
    thresholds: Optional[np.ndarray] = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Evaluate hypothetical expected classification cost across thresholds for multiple FP:FN ratios.

    Expected cost per applicant: (cost_fp * FP + cost_fn * FN) / N.

    Parameters
    ----------
    y_true : 1D ground-truth labels.
    y_prob : 1D predicted probabilities for class 1.
    cost_ratios : List of (cost_fp, cost_fn) tuples. Defaults to [(1, 1), (1, 2), (1, 5), (1, 10)].
    thresholds : Array of thresholds. Defaults to np.arange(0.05, 0.96, 0.01).

    Returns
    -------
    tuple[pd.DataFrame, dict[str, dict[str, Any]]]
        - DataFrame containing expected costs for all thresholds and ratios
        - Dictionary summarizing optimal threshold and minimum cost for each ratio
    """
    if cost_ratios is None:
        cost_ratios = [(1, 1), (1, 2), (1, 5), (1, 10)]
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.01)

    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    n_samples = len(y_true)

    rows = []
    for th in thresholds:
        th = round(float(th), 4)
        y_pred = (y_prob >= th).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = [int(v) for v in cm.ravel()]

        row_dict = {"threshold": th, "tn": tn, "fp": fp, "fn": fn, "tp": tp}
        for c_fp, c_fn in cost_ratios:
            ratio_label = f"cost_{c_fp}_{c_fn}"
            total_cost = (c_fp * fp) + (c_fn * fn)
            expected_cost_per_applicant = total_cost / n_samples
            row_dict[ratio_label] = round(expected_cost_per_applicant, 6)
            row_dict[f"total_{ratio_label}"] = total_cost

        rows.append(row_dict)

    cost_df = pd.DataFrame(rows)

    best_costs = {}
    for c_fp, c_fn in cost_ratios:
        ratio_label = f"cost_{c_fp}_{c_fn}"
        min_idx = cost_df[ratio_label].idxmin()
        best_row = cost_df.loc[min_idx]
        best_costs[f"{c_fp}:{c_fn}"] = {
            "cost_fp": c_fp,
            "cost_fn": c_fn,
            "optimal_threshold": float(best_row["threshold"]),
            "min_expected_cost_per_applicant": float(best_row[ratio_label]),
            "fp_at_threshold": int(best_row["fp"]),
            "fn_at_threshold": int(best_row["fn"]),
            "tn_at_threshold": int(best_row["tn"]),
            "tp_at_threshold": int(best_row["tp"]),
        }

    return cost_df, best_costs


# ======================================================================
# Publication-Quality Diagnostic Plots
# ======================================================================

def plot_threshold_tradeoff(
    threshold_df: pd.DataFrame,
    output_path: pathlib.Path,
    best_f1_th: Optional[float] = None,
    best_f2_th: Optional[float] = None,
) -> None:
    """Plot Precision, Recall, F1, and F2 vs Decision Threshold."""
    plt.figure(figsize=(9, 6), dpi=300)

    plt.plot(threshold_df["threshold"], threshold_df["precision"], label="Precision", color="#1f77b4", linewidth=2.0)
    plt.plot(threshold_df["threshold"], threshold_df["recall"], label="Recall", color="#2ca02c", linewidth=2.0)
    plt.plot(threshold_df["threshold"], threshold_df["f1"], label="F1 Score", color="#ff7f0e", linewidth=2.2)
    plt.plot(threshold_df["threshold"], threshold_df["f2"], label="F2 Score", color="#d62728", linewidth=2.2, linestyle="--")

    if best_f1_th is not None:
        f1_val = threshold_df.loc[np.isclose(threshold_df["threshold"], best_f1_th), "f1"].values[0]
        plt.axvline(best_f1_th, color="#ff7f0e", linestyle=":", alpha=0.7, label=f"Max F1 ({f1_val:.4f} @ th={best_f1_th:.2f})")
    if best_f2_th is not None:
        f2_val = threshold_df.loc[np.isclose(threshold_df["threshold"], best_f2_th), "f2"].values[0]
        plt.axvline(best_f2_th, color="#d62728", linestyle=":", alpha=0.7, label=f"Max F2 ({f2_val:.4f} @ th={best_f2_th:.2f})")

    # Default 0.50 cutoff marker
    plt.axvline(0.50, color="gray", linestyle="-.", alpha=0.6, label="Default Cutoff (0.50)")

    plt.title("Classification Metric Trade-offs Across Decision Thresholds (Validation)", fontsize=13, fontweight="bold")
    plt.xlabel("Decision Threshold", fontsize=11)
    plt.ylabel("Score", fontsize=11)
    plt.xlim([0.05, 0.95])
    plt.ylim([0.0, 1.02])
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="best", fontsize=9, frameon=True)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved threshold trade-off figure to {output_path}")


def plot_cost_threshold_analysis(
    cost_df: pd.DataFrame,
    best_costs: dict[str, dict[str, Any]],
    output_path: pathlib.Path,
) -> None:
    """Plot Expected Classification Cost vs Decision Threshold across FP:FN ratios."""
    plt.figure(figsize=(9, 6), dpi=300)

    colors = {"1:1": "#1f77b4", "1:2": "#2ca02c", "1:5": "#ff7f0e", "1:10": "#d62728"}
    for ratio_str, info in best_costs.items():
        c_fp, c_fn = info["cost_fp"], info["cost_fn"]
        col = f"cost_{c_fp}_{c_fn}"
        opt_th = info["optimal_threshold"]
        min_cost = info["min_expected_cost_per_applicant"]
        line_color = colors.get(ratio_str, "purple")

        plt.plot(
            cost_df["threshold"],
            cost_df[col],
            label=f"FP:FN {ratio_str} (Min {min_cost:.3f} @ th={opt_th:.2f})",
            color=line_color,
            linewidth=2.0,
        )
        plt.plot(opt_th, min_cost, "o", color=line_color, markersize=7)

    plt.title("Expected Classification Cost vs Decision Threshold (Sensitivity Analysis)", fontsize=13, fontweight="bold")
    plt.xlabel("Decision Threshold", fontsize=11)
    plt.ylabel("Expected Cost per Applicant (Hypothetical Units)", fontsize=11)
    plt.xlim([0.05, 0.95])
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", fontsize=9, frameon=True)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved cost threshold figure to {output_path}")


def plot_calibration_curves(
    calibration_models: dict[str, dict[str, Any]],
    output_path: pathlib.Path,
    n_bins: int = 10,
) -> None:
    """Plot reliability diagram comparing Uncalibrated, Sigmoid, and Isotonic models."""
    fig, (ax1, ax2) = plt.subplots(
        nrows=2, ncols=1, figsize=(8, 8), dpi=300, gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    colors = {"Uncalibrated": "#1f77b4", "Sigmoid (Platt)": "#2ca02c", "Isotonic": "#d62728"}
    # Diagonal perfectly calibrated line
    ax1.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated", alpha=0.7)

    for name, data in calibration_models.items():
        y_true = data["y_true"]
        y_prob = data["y_prob"]
        brier = data["brier_score"]
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy="uniform"
        )
        color = colors.get(name, "#ff7f0e")
        ax1.plot(
            mean_predicted_value,
            fraction_of_positives,
            "s-",
            label=f"{name} (Brier: {brier:.4f})",
            color=color,
            linewidth=2.0,
            markersize=6,
        )

        ax2.hist(
            y_prob,
            bins=30,
            histtype="step",
            density=True,
            stacked=True,
            label=name,
            color=color,
            linewidth=1.5,
        )

    ax1.set_ylabel("Fraction of Positives (Observed Rate)", fontsize=11)
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right", fontsize=9, frameon=True)
    ax1.set_title("Probability Calibration Reliability Diagram (Validation)", fontsize=13, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.set_xlabel("Mean Predicted Probability", fontsize=11)
    ax2.set_ylabel("Density", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.set_xlim([0.0, 1.0])

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved calibration curve figure to {output_path}")


def plot_phase8_roc_comparison(
    models_dict: dict[str, dict[str, Any]],
    output_path: pathlib.Path,
    split_name: str = "Test",
) -> None:
    """Plot comparative ROC curves for Phase 8 models vs Phase 7 baseline."""
    plt.figure(figsize=(8, 6), dpi=300)
    colors = ["#7f7f7f", "#1f77b4", "#2ca02c", "#d62728"]

    for idx, (m_label, m_data) in enumerate(models_dict.items()):
        y_true = m_data["y_true"]
        y_prob = m_data["y_prob"]
        roc_val = m_data["roc_auc"]
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        col = colors[idx % len(colors)]
        plt.plot(fpr, tpr, label=f"{m_label} (ROC-AUC = {roc_val:.4f})", color=col, linewidth=2.0)

    plt.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random Guess (ROC-AUC = 0.5000)")
    plt.title(f"Phase 8 Models vs Baseline — ROC Curves ({split_name.capitalize()} Set)", fontsize=13, fontweight="bold")
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate (Recall)", fontsize=11)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", fontsize=9, frameon=True)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved Phase 8 ROC comparison curve to {output_path}")


def plot_phase8_pr_comparison(
    models_dict: dict[str, dict[str, Any]],
    output_path: pathlib.Path,
    split_name: str = "Test",
    baseline_rate: float = 0.0807,
) -> None:
    """Plot comparative Precision-Recall curves for Phase 8 models vs Phase 7 baseline."""
    plt.figure(figsize=(8, 6), dpi=300)
    colors = ["#7f7f7f", "#1f77b4", "#2ca02c", "#d62728"]

    for idx, (m_label, m_data) in enumerate(models_dict.items()):
        y_true = m_data["y_true"]
        y_prob = m_data["y_prob"]
        pr_val = m_data["pr_auc"]
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        col = colors[idx % len(colors)]
        plt.plot(rec, prec, label=f"{m_label} (PR-AUC = {pr_val:.4f})", color=col, linewidth=2.0)

    plt.axhline(y=baseline_rate, color="k", linestyle="--", alpha=0.6, label=f"No-Skill Baseline ({baseline_rate:.3%})")
    plt.title(f"Phase 8 Models vs Baseline — Precision-Recall Curves ({split_name.capitalize()} Set)", fontsize=13, fontweight="bold")
    plt.xlabel("Recall", fontsize=11)
    plt.ylabel("Precision", fontsize=11)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", fontsize=9, frameon=True)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved Phase 8 PR comparison curve to {output_path}")
