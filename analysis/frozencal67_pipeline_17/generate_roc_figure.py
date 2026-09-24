#!/usr/bin/env python3
"""
generate_roc_figure.py

Generates a two-panel ROC figure (static + mobile) from the cached
test predictions produced by cluster_bootstrap_ci.py.

Models plotted (top performers + LogReg as contrast):
  - Stacking_Ensemble
  - xgboost
  - catboost
  - randomforest
  - logisticregression

Output:
  results/cluster_bootstrap/roc_curves.pdf
  results/cluster_bootstrap/roc_curves.png

Usage:
  python3 generate_roc_figure.py
"""

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve


MODELS_TO_PLOT = [
    "Stacking_Ensemble",
    "xgboost",
    "catboost",
    "randomforest",
    "logisticregression",
]

DISPLAY_NAMES = {
    "Stacking_Ensemble": "Stacking Ensemble",
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
    "randomforest": "Random Forest",
    "logisticregression": "Logistic Regression",
}

# Consistent colors across panels (same model = same color)
COLORS = {
    "Stacking_Ensemble": "#1f77b4",   # blue
    "xgboost": "#d62728",             # red
    "catboost": "#2ca02c",            # green
    "randomforest": "#ff7f0e",        # orange
    "logisticregression": "#9467bd",  # purple
}

LINESTYLES = {
    "Stacking_Ensemble": "-",
    "xgboost": "-",
    "catboost": "-",
    "randomforest": "-",
    "logisticregression": "--",   # dashed to highlight as baseline
}


def plot_panel(ax, cache, title):
    """Plot ROC curves for the configured models on one axis."""
    y_test = cache["y_test"]
    per_model = cache["per_model"]

    for name in MODELS_TO_PLOT:
        if name not in per_model:
            print(f"  WARNING: model {name} not in cache, skipping")
            continue
        entry = per_model[name]
        scores = entry["scores_test"]
        fpr, tpr, _ = roc_curve(y_test, scores)
        # Use the cached AUC (same computation as cluster_bootstrap_ci.py)
        auc = entry["point_auc"]
        label = f"{DISPLAY_NAMES[name]} (AUC = {auc:.3f})"
        ax.plot(
            fpr, tpr,
            label=label,
            color=COLORS[name],
            linestyle=LINESTYLES[name],
            linewidth=1.5,
        )

    # Diagonal reference (random classifier)
    ax.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=0.8)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8, frameon=True)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        default="./results",
        help="Root of the results/ directory",
    )
    parser.add_argument(
        "--out-dir",
        default="./results/cluster_bootstrap",
        help="Where to write the figure",
    )
    args = parser.parse_args()

    static_cache_path = Path(args.results_root) / "hp_search_final" / "static" / "test_predictions_cache.pkl"
    mobile_cache_path = Path(args.results_root) / "hp_search_final" / "mobile" / "test_predictions_cache.pkl"

    for p in [static_cache_path, mobile_cache_path]:
        if not p.exists():
            raise SystemExit(f"ERROR: missing {p}\nRun cluster_bootstrap_ci.py first.")

    static_cache = joblib.load(static_cache_path)
    mobile_cache = joblib.load(mobile_cache_path)

    # Layout: 2 panels side by side, IEEE single-column friendly
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.2))

    plot_panel(axes[0], static_cache, "(a) Static")
    plot_panel(axes[1], mobile_cache, "(b) Mobile")

    plt.tight_layout()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = out_dir / "roc_curves.pdf"
    png_path = out_dir / "roc_curves.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")

    print(f"Wrote: {pdf_path}")
    print(f"Wrote: {png_path}")


if __name__ == "__main__":
    main()