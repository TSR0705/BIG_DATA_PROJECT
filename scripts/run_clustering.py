import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from src.ml.clustering import (
    CLUSTERING_FEATURES,
    load_and_preprocess_clustering_data,
    evaluate_k_range,
    fit_and_profile_clusters,
)

def main():
    print("Preparing clustering data...")
    df, X_scaled, scaler, imputer = load_and_preprocess_clustering_data()

    # Save clustering input parquet (without target or id)
    out_dir = Path("data/clustering")
    out_dir.mkdir(parents=True, exist_ok=True)
    df_clustering = pd.DataFrame(X_scaled, columns=CLUSTERING_FEATURES)
    df_clustering["SK_ID_CURR"] = df["SK_ID_CURR"].values
    df_clustering.to_parquet(out_dir / "clustering_input.parquet", index=False)
    print("Saved clustering input parquet to data/clustering/clustering_input.parquet")

    # Save feature metadata json
    configs_dir = Path("configs")
    configs_dir.mkdir(parents=True, exist_ok=True)
    with open(configs_dir / "clustering_features.json", "w") as f:
        json.dump({
            "features": CLUSTERING_FEATURES,
            "feature_count": len(CLUSTERING_FEATURES),
            "methodology": "Median imputation, 99.5th percentile upper-bound clipping on monetary attributes, StandardScaler",
            "excluded_columns": ["SK_ID_CURR", "TARGET"],
            "target_usage": "Strictly post-hoc only"
        }, f, indent=2)

    # Evaluate K=2..8
    print("Evaluating K=2 to 8...")
    eval_df = evaluate_k_range(X_scaled, k_range=range(2, 9), sample_size=25000, random_state=42)
    eval_df.to_csv("reports/clustering_evaluation.csv", index=False)
    print("Saved evaluation results to reports/clustering_evaluation.csv")
    print(eval_df)

    # Fit best K=4
    print("Fitting K=4...")
    km, df_assigned, profile = fit_and_profile_clusters(df, X_scaled, k=4, random_state=42)
    profile.to_csv("reports/clustering_profile.csv")
    print("Saved cluster profiles to reports/clustering_profile.csv")

    # Generate Figures
    fig_dir = Path("reports/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Elbow & Silhouette Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(eval_df["k"], eval_df["inertia"], marker="o", color="#1f77b4", linewidth=2)
    ax1.axvline(4, color="crimson", linestyle="--", alpha=0.7, label="Selected K=4")
    ax1.set_title("Elbow Curve (Inertia vs. Number of Clusters K)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Number of Clusters (K)", fontsize=10)
    ax1.set_ylabel("Inertia (Within-Cluster Sum of Squares)", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    ax2.plot(eval_df["k"], eval_df["silhouette_score"], marker="s", color="#2ca02c", linewidth=2)
    ax2.axvline(4, color="crimson", linestyle="--", alpha=0.7, label="Selected K=4")
    ax2.set_title("Silhouette Score vs. Number of Clusters K", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Number of Clusters (K)", fontsize=10)
    ax2.set_ylabel("Silhouette Score (N=25,000 Sample)", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "clustering_elbow_silhouette.png", dpi=150)
    plt.close()
    print("Saved elbow/silhouette figure.")

    # 2. PCA 2D Scatter
    np.random.seed(42)
    plot_sample = df_assigned.sample(n=10000, random_state=42)
    palette = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
    cluster_names = {
        0: "Cluster 0: Young / Subprime / Elevated Risk",
        1: "Cluster 1: Mature / Prime / Established",
        2: "Cluster 2: Short-Term / High Payment Rate",
        3: "Cluster 3: Affluent / High Credit / High Capacity"
    }
    plot_sample["Cluster Name"] = plot_sample["cluster"].map(cluster_names)

    plt.figure(figsize=(10, 7))
    for c_id in sorted(df_assigned["cluster"].unique()):
        sub = plot_sample[plot_sample["cluster"] == c_id]
        plt.scatter(
            sub["pca_x"],
            sub["pca_y"],
            color=palette[c_id],
            label=cluster_names[c_id],
            alpha=0.45,
            s=20,
            edgecolors="none"
        )
    plt.title("2D PCA Projection of Customer Clusters (N=10,000 Sample)", fontsize=13, fontweight="bold")
    plt.xlabel("Principal Component 1 (Size & Capacity Dimension)", fontsize=11)
    plt.ylabel("Principal Component 2 (Tenure & Age Dimension)", fontsize=11)
    plt.legend(title="Customer Segment", loc="upper right")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(fig_dir / "clustering_pca_scatter.png", dpi=150)
    plt.close()
    print("Saved PCA scatter figure.")

    # 3. Post-Hoc Default Rate by Cluster
    plt.figure(figsize=(9, 5))
    bars = plt.bar(
        [cluster_names[i] for i in profile.index],
        profile["default_rate_pct"],
        color=["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"],
        alpha=0.85
    )
    plt.axhline(8.073, color="black", linestyle="--", linewidth=1.5, label="Global Population Default Rate (8.07%)")
    plt.title("Post-Hoc Loan Default Rate by Customer Segment", fontsize=13, fontweight="bold")
    plt.xlabel("Customer Cluster Segment", fontsize=11)
    plt.ylabel("Observed Default Rate (%)", fontsize=11)
    plt.xticks(rotation=20, ha="right", fontsize=9)
    plt.ylim(0, 18)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + 0.4, f"{h:.2f}%", ha="center", fontweight="bold", fontsize=10)
    plt.legend()
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(fig_dir / "clustering_default_rates.png", dpi=150)
    plt.close()
    print("Saved default rate figure.")

if __name__ == "__main__":
    main()
