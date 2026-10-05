"""Model Evaluation Metrics, Visualization, and Reporting Utilities.

Provides comprehensive, reproducible baseline evaluation:
- Imbalanced binary classification metrics (ROC-AUC, PR-AUC, Log Loss, F1, etc.)
- Publication-quality ROC and Precision-Recall curve comparisons
- Detailed 2x2 confusion matrix figures
- Native feature importance extraction and visualization
- Structured Markdown and JSON reporting
"""

import json
import pathlib
import sys
import time
from typing import Any, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
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

logger = get_logger("MLModelUtils", "model_utils.log")


def evaluate_predictions(
    y_true: Union[pd.Series, np.ndarray],
    y_prob: np.ndarray,
    model_name: str,
    split_name: str,
    threshold: float = 0.50,
    training_time: float = 0.0,
    prediction_time: float = 0.0,
) -> dict[str, Any]:
    """Calculate comprehensive evaluation metrics for binary classification.

    Parameters
    ----------
    y_true : 1D array-like of ground truth binary labels (0 or 1).
    y_prob : 1D array-like of predicted probabilities for positive class (1).
    model_name : str, human-readable model identifier.
    split_name : str, 'validation' or 'test'.
    threshold : float, classification decision threshold (default 0.50).
    training_time : float, model training duration in seconds.
    prediction_time : float, inference prediction duration in seconds.

    Returns
    -------
    dict[str, Any]
        Dictionary with all evaluation metrics.
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()

    # Threshold probability for binary predictions
    y_pred = (y_prob >= threshold).astype(int)

    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))
    loss = float(log_loss(y_true, y_prob))
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    metrics = {
        "model": model_name,
        "split": split_name,
        "threshold": threshold,
        "roc_auc": round(roc_auc, 6),
        "pr_auc": round(pr_auc, 6),
        "log_loss": round(loss, 6),
        "accuracy": round(acc, 6),
        "precision": round(prec, 6),
        "recall": round(rec, 6),
        "f1": round(f1, 6),
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "matrix": [[tn, fp], [fn, tp]],
        },
        "training_time_seconds": round(training_time, 4),
        "prediction_time_seconds": round(prediction_time, 4),
        "total_samples": len(y_true),
        "positive_samples": int(np.sum(y_true == 1)),
        "negative_samples": int(np.sum(y_true == 0)),
    }

    logger.info(
        f"Evaluated {model_name} on {split_name}: "
        f"ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}, LogLoss={loss:.4f}, "
        f"F1={f1:.4f}, Precision={prec:.4f}, Recall={rec:.4f}"
    )
    return metrics


def plot_roc_comparison(
    models_data: dict[str, dict[str, Any]],
    output_path: pathlib.Path,
    split_name: str = "Validation",
) -> None:
    """Generate comparative ROC curves for all baseline models."""
    plt.figure(figsize=(8, 6), dpi=300)

    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
    for idx, (m_name, m_info) in enumerate(models_data.items()):
        y_true = m_info["y_true"]
        y_prob = m_info["y_prob"]
        auc = m_info["roc_auc"]
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        color = colors[idx % len(colors)]
        plt.plot(fpr, tpr, label=f"{m_name} (ROC-AUC = {auc:.4f})", color=color, linewidth=2.0)

    # Diagonal chance line
    plt.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random Guess (ROC-AUC = 0.5000)")

    plt.title(f"Baseline Models — ROC Curve ({split_name.capitalize()} Set)", fontsize=13, fontweight="bold")
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", fontsize=10, frameon=True)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved ROC comparison curve to {output_path}")


def plot_pr_comparison(
    models_data: dict[str, dict[str, Any]],
    output_path: pathlib.Path,
    split_name: str = "Validation",
    baseline_rate: float = 0.0807,
) -> None:
    """Generate comparative Precision-Recall curves for all baseline models."""
    plt.figure(figsize=(8, 6), dpi=300)

    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
    for idx, (m_name, m_info) in enumerate(models_data.items()):
        y_true = m_info["y_true"]
        y_prob = m_info["y_prob"]
        pr_auc = m_info["pr_auc"]
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        color = colors[idx % len(colors)]
        plt.plot(rec, prec, label=f"{m_name} (PR-AUC = {pr_auc:.4f})", color=color, linewidth=2.0)

    # Baseline no-skill horizontal line
    plt.axhline(
        y=baseline_rate,
        color="k",
        linestyle="--",
        alpha=0.6,
        label=f"No-Skill Baseline ({baseline_rate:.3%})",
    )

    plt.title(f"Baseline Models — Precision-Recall Curve ({split_name.capitalize()} Set)", fontsize=13, fontweight="bold")
    plt.xlabel("Recall", fontsize=11)
    plt.ylabel("Precision", fontsize=11)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", fontsize=10, frameon=True)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved PR comparison curve to {output_path}")


def plot_confusion_matrix_figure(
    cm: list[list[int]],
    model_name: str,
    output_path: pathlib.Path,
    threshold: float = 0.50,
) -> None:
    """Plot an annotated 2x2 confusion matrix heatmap."""
    tn, fp = cm[0]
    fn, tp = cm[1]
    total = tn + fp + fn + tp

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    data = np.array([[tn, fp], [fn, tp]])
    im = ax.imshow(data, cmap="Blues", interpolation="nearest")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted Repaid (0)", "Predicted Default (1)"], fontsize=10)
    ax.set_yticklabels(["Actual Repaid (0)", "Actual Default (1)"], fontsize=10)

    labels = [
        [f"TN\n{tn:,}\n({tn/total:.1%})", f"FP\n{fp:,}\n({fp/total:.1%})"],
        [f"FN\n{fn:,}\n({fn/total:.1%})", f"TP\n{tp:,}\n({tp/total:.1%})"],
    ]

    for i in range(2):
        for j in range(2):
            color = "white" if data[i, j] > (total * 0.4) else "black"
            ax.text(j, i, labels[i][j], ha="center", va="center", color=color, fontsize=11, fontweight="bold")

    plt.title(f"Confusion Matrix — {model_name}\n(Test Set, Threshold = {threshold:.2f})", fontsize=12, fontweight="bold")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved confusion matrix plot for {model_name} to {output_path}")


def plot_feature_importance_figure(
    importance_df: pd.DataFrame,
    model_name: str,
    output_path: pathlib.Path,
    top_n: int = 20,
) -> None:
    """Plot horizontal bar chart of top N feature importances."""
    top_df = importance_df.head(top_n).iloc[::-1]  # Reverse for top-down bar chart

    plt.figure(figsize=(9, 7), dpi=300)
    plt.barh(top_df["feature"], top_df["importance"], color="#2b5c8f", edgecolor="black", alpha=0.85)

    plt.title(f"Top {top_n} Feature Importances — {model_name}", fontsize=12, fontweight="bold")
    plt.xlabel("Native Feature Importance Score", fontsize=10)
    plt.ylabel("Feature Name", fontsize=10)
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved feature importance figure for {model_name} to {output_path}")
