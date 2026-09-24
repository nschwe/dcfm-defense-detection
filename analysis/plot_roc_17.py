#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""plot_roc_17.py -- R#3.18's Figure 1: five highlighted classifiers over the
envelope of all seventeen.

WHY NOT generate_roc_figure.py
    That script hard-codes five models under their old names (xgboost, catboost,
    randomforest, logisticregression), which do not exist in the 17-observable
    campaign, and it shows nothing of the remaining twelve. The reviewer's
    objection to Figure 1 is precisely that the reader cannot see the spread.
    ⛔ The original script is left untouched; this one reads the same cache.

WHAT IT DRAWS
    Per panel: the twelve non-highlighted classifiers as thin grey curves, then
    the five named ones in colour with their AUC in the legend. The caption line
    printed at the end gives the AUC range over ALL seventeen -- that is the
    number the caption must quote, and it is computed here, not by hand.

USAGE
    cd ~/ns3/nv347/ns-3.47/analysis
    ~/miniconda3/envs/manet/bin/python -u plot_roc_17.py \
        --cache-root arms_r34_17feat_frozencal67/listener/results/roc_cache \
        --out arms_r34_17feat_frozencal67/listener/results/roc_cache/roc_curves.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from sklearn.metrics import roc_curve    # noqa: E402

HERE = Path(__file__).resolve().parent

HIGHLIGHT = ["Stacking_Ensemble", "XGB_conservative", "CatBoost_deep",
             "RandomForest", "LogisticRegression"]
# ⛔ Title Case, as in generate_roc_figure.py (the submitted figure's generator),
# in Table 4 and in the body of Sec. 6.1. Sentence case here contradicted all three.
DISPLAY = {
    "Stacking_Ensemble": "Stacking Ensemble",
    "XGB_conservative": "XGBoost",
    "CatBoost_deep": "CatBoost",
    "RandomForest": "Random Forest",
    "LogisticRegression": "Logistic Regression",
}
# ⛔ The submitted figure's assignment, kept model by model so a reader comparing
# the two versions sees the same model in the same colour (generate_roc_figure.py:48).
COLORS = {
    "Stacking_Ensemble": "#1f77b4",    # blue
    "XGB_conservative": "#d62728",     # red
    "CatBoost_deep": "#2ca02c",        # green
    "RandomForest": "#ff7f0e",         # orange
    "LogisticRegression": "#9467bd",   # purple
}
STYLES = {"LogisticRegression": "--"}


def panel(ax, cache, title):
    y = cache["y_test"]
    per = cache["per_model"]
    aucs = {n: e["point_auc"] for n, e in per.items()}

    for name, entry in sorted(per.items()):
        if name in HIGHLIGHT:
            continue
        fpr, tpr, _ = roc_curve(y, entry["scores_test"])
        ax.plot(fpr, tpr, color="0.75", linewidth=0.7, zorder=1)

    for name in HIGHLIGHT:
        if name not in per:
            print(f"  ⚠️  {title}: {name} not in cache, skipped")
            continue
        fpr, tpr, _ = roc_curve(y, per[name]["scores_test"])
        ax.plot(fpr, tpr, color=COLORS[name], linewidth=1.5, zorder=3,
                linestyle=STYLES.get(name, "-"),
                label=f"{DISPLAY[name]} (AUC = {aucs[name]:.3f})")

    ax.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=0.8, zorder=2)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8, frameon=True)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    lo = min(aucs.values())
    hi = max(aucs.values())
    lo_n = min(aucs, key=aucs.get)
    return len(aucs), lo, lo_n, hi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", required=True,
                    help="the results_root build_roc_cache.py reported")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.cache_root)
    if not root.is_absolute():
        root = HERE / root
    caches = {}
    for mode in ("static", "mobile"):
        p = root / "hp_search_final" / mode / "test_predictions_cache.pkl"
        if not p.is_file():
            sys.exit(f"ABORT: no cache at {p} -- run build_roc_cache.py first")
        caches[mode] = joblib.load(p)
        print(f"  {mode}: cache from arm {caches[mode].get('arm')} "
              f"via {caches[mode].get('pipeline')}")

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.2))
    n_s, lo_s, lon_s, hi_s = panel(axes[0], caches["static"], "(a) Static")
    n_m, lo_m, lon_m, hi_m = panel(axes[1], caches["mobile"], "(b) Mobile")
    fig.tight_layout()

    out = Path(args.out)
    if not out.is_absolute():
        out = HERE / out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\n  wrote {out}")

    print("\n=== caption facts, computed not typed ===")
    print(f"  static: {n_s} classifiers, AUC {lo_s:.4f} ({lon_s}) to {hi_s:.4f}")
    print(f"  mobile: {n_m} classifiers, AUC {lo_m:.4f} ({lon_m}) to {hi_m:.4f}")
    print("  ⛔ the caption must quote these ranges, and name the weakest of each "
          "panel; grey curves are the classifiers not highlighted.")


if __name__ == "__main__":
    main()
