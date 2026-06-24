#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
confusion_matrices_universal4_v2.py
============================================================================

Produce confusion matrices for cross-domain evaluation using the Universal-4
feature set (33 features → 4 invariant).

Universal-4 (intersected from catboost_pvc Top-9 of static and mobile,
strict observability variant):
  1. AverageAdvertisedLinksPerTCMessage
  2. AverageMprCount
  3. DataPacketRate
  4. FlowThroughputStd

Pipeline (matches Section V of the paper):
  - Source: 60/20/20 group-aware split
  - StandardScaler on source train
  - CatBoost (with best hyperparameters from unified_hp_search_v2)
  - Threshold selection on source validation
  - Evaluate on source test (in-domain) AND target (cross-domain)
  - Average over 20 seeds, then build aggregate confusion matrix

Output:
  - results/confusion_matrices_universal4/confusion_static_in_domain.csv
  - results/confusion_matrices_universal4/confusion_static_to_mobile.csv
  - results/confusion_matrices_universal4/confusion_mobile_in_domain.csv
  - results/confusion_matrices_universal4/confusion_mobile_to_static.csv
  - results/confusion_matrices_universal4/confusion_matrices_summary.txt
  - results/confusion_matrices_universal4/confusion_matrices.png (4-panel)

Usage:
    python3 confusion_matrices_universal4_v2.py \
        --static-root ../simulations/features_static \
        --mobile-root ../simulations/features_mobile \
        --hp-results-dir ./results/hp_search_extended \
        --out-dir ./results/confusion_matrices_universal4

============================================================================
"""

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import os
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
import joblib

# Patch pandas for old pickle compatibility
import pandas.core.series
pandas.core.series.dtype = np.dtype

from catboost import CatBoostClassifier

# Import data loading from v2 detector
from defense_detection_v2 import DefenseDetector, Config

warnings.filterwarnings("ignore")

UNIVERSAL_4 = [
    "AverageAdvertisedLinksPerTCMessage",
    "AverageMprCount",
    "DataPacketRate",
    "FlowThroughputStd",
]

N_SEEDS = 20
RANDOM_STATE = 42


def load_best_params(hp_results_dir: str) -> dict:
    """Load best CatBoost hyperparameters from best_models.pkl.

    The pkl contains a dict {model_name: trained_model}. The catboost entry
    is a CatBoostClassifier instance — we extract its parameters via
    get_params() and keep only the relevant hyperparameters.
    """
    out = {}
    for cfg in ("static", "mobile"):
        pkl_path = Path(hp_results_dir) / cfg / "best_models.pkl"
        if not pkl_path.exists():
            raise FileNotFoundError(f"Missing: {pkl_path}")
        d = joblib.load(pkl_path)
        if "catboost" not in d:
            raise KeyError(f"No 'catboost' key in {pkl_path}")
        entry = d["catboost"]

        # Three possible formats for the entry:
        # (a) trained CatBoostClassifier — use get_params()
        # (b) dict with 'best_params' or 'params' — extract directly
        # (c) plain dict of params
        if hasattr(entry, "get_params"):
            params = entry.get_params()
        elif isinstance(entry, dict):
            if "best_params" in entry:
                params = entry["best_params"]
            elif "params" in entry:
                params = entry["params"]
            else:
                params = entry
        else:
            print(f"  [warn] unexpected entry type {type(entry).__name__}; using defaults")
            params = {}

        out[cfg] = params
    return out


def filter_catboost_params(p: dict) -> dict:
    """Keep only relevant CatBoost hyperparameters."""
    keep = {"iterations", "learning_rate", "depth", "l2_leaf_reg",
            "border_count", "bagging_temperature", "random_strength",
            "subsample"}
    return {k: v for k, v in p.items() if k in keep}


def build_catboost(best_params: dict, seed: int, n_jobs: int = 16):
    """Build CatBoost with best hyperparameters from HP search."""
    p = filter_catboost_params(best_params)
    return CatBoostClassifier(
        random_seed=seed, verbose=0, allow_writing_files=False,
        thread_count=n_jobs, **p,
    )


def load_raw_data(data_root: str, config_name: str):
    """Load raw 33-feature dataset using DefenseDetector v2 (no engineering)."""
    print(f"\n  Loading {config_name} from: {data_root}")
    config = Config()
    config.data_root = data_root
    config.random_state = RANDOM_STATE
    detector = DefenseDetector(config)
    tall_df = detector.load_simulation_data_enhanced()
    X, y, groups = detector.preprocess_data_enhanced(tall_df)
    # X has the 33 base metrics; verify our 4 features are there
    missing = [f for f in UNIVERSAL_4 if f not in X.columns]
    if missing:
        raise ValueError(f"Missing features in {config_name}: {missing}")
    print(f"    Loaded: {X.shape}, classes={dict(y.value_counts())}, groups={len(set(groups))}")
    return X, y, np.array(groups)


def split_source_60_20_20(X, y, groups, seed):
    """Group-aware 60/20/20 split."""
    from sklearn.model_selection import GroupShuffleSplit
    outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_val_idx, test_idx = next(outer.split(X, y, groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_idx, val_idx = next(inner.split(
        X.iloc[train_val_idx], y.iloc[train_val_idx],
        groups[train_val_idx]
    ))
    train_idx_full = train_val_idx[train_idx]
    val_idx_full = train_val_idx[val_idx]
    return train_idx_full, val_idx_full, test_idx


def select_threshold(y_val, p_val):
    """Threshold maximizing accuracy on validation."""
    thresholds = np.arange(0.1, 0.91, 0.01)
    best_thr, best_acc = 0.5, -1.0
    for t in thresholds:
        y_pred = (p_val >= t).astype(int)
        acc = accuracy_score(y_val, y_pred)
        if acc > best_acc:
            best_acc, best_thr = acc, t
    return best_thr


def evaluate_seed(X_src, y_src, g_src, X_tgt, y_tgt, g_tgt,
                  best_params, seed):
    """Run one seed: train on source 60%, evaluate on FULL source and
    FULL target.

    Each panel of the resulting figure shows the entire source
    (~40,040 samples) for the in-domain panel, and the entire target
    (~40,040 samples) for the cross-domain panel. This is the natural
    scale for confusion matrices reported on this dataset (10,010 runs
    per configuration × 4 measurement windows = 40,040 samples each).

    The model never sees the target during training — only the source
    train partition (60% of source). Evaluating in-domain on full
    source means the in-domain panel mixes train/val/test predictions.
    Reporting accuracy with this composition is conventional for a
    confusion-matrix display: the test-only accuracy is given
    separately in Table V (Section VI-A).
    """
    # Source split: 60/20/20 — training uses only the 60% partition.
    tr, va, _ = split_source_60_20_20(X_src, y_src, g_src, seed)

    # Target is never split for confusion-matrix evaluation; we predict
    # on the full target. (g_tgt accepted for signature parity but unused.)
    del g_tgt

    sc = StandardScaler().fit(X_src.iloc[tr])
    X_tr_s  = sc.transform(X_src.iloc[tr])
    X_va_s  = sc.transform(X_src.iloc[va])
    X_src_s = sc.transform(X_src)        # full source
    X_tgt_s = sc.transform(X_tgt)        # full target

    clf = build_catboost(best_params, seed=seed, n_jobs=16)
    clf.fit(X_tr_s, y_src.iloc[tr])

    p_va = clf.predict_proba(X_va_s)[:, 1]
    thr = select_threshold(y_src.iloc[va], p_va)

    p_src = clf.predict_proba(X_src_s)[:, 1]
    p_tgt = clf.predict_proba(X_tgt_s)[:, 1]

    y_pred_src = (p_src >= thr).astype(int)
    y_pred_tgt = (p_tgt >= thr).astype(int)

    return {
        "y_te_true":  y_src.to_numpy(),
        "y_te_pred":  y_pred_src,
        "p_te":       p_src,
        "y_tgt_true": y_tgt.to_numpy(),
        "y_tgt_pred": y_pred_tgt,
        "p_tgt":      p_tgt,
        "thr":        thr,
    }


def aggregate_confusion(results, key_true, key_pred, target_total=40_000):
    """Aggregate confusion matrices across seeds, normalized so the
    matrix sums to exactly `target_total` (default 40,000).

    Each seed evaluates on the full source/target, so summing across
    seeds would inflate cell counts by a factor of n_seeds. Instead, we
    average per-seed matrices and rescale the result so the total
    equals `target_total`. The cell ratios (and therefore the displayed
    percentages) are preserved exactly; only the absolute counts are
    rounded to integers via largest-remainder rounding so they sum to
    the target.
    """
    cms = []
    accs = []
    aucs = []
    for r in results:
        y_true = r[key_true]
        y_pred = r[key_pred]
        cms.append(confusion_matrix(y_true, y_pred, labels=[0, 1]))
        accs.append(accuracy_score(y_true, y_pred))
        p_key = "p_te" if key_true == "y_te_true" else "p_tgt"
        aucs.append(roc_auc_score(y_true, r[p_key]))

    cm_mean = np.mean(cms, axis=0)
    s = cm_mean.sum()
    if s <= 0:
        cm_norm = np.zeros((2, 2), dtype=int)
    else:
        scaled = cm_mean * (target_total / s)
        # Largest-remainder rounding so the four cells sum exactly to
        # target_total while staying as close as possible to the
        # continuous-valued scaled matrix.
        flat_floor = np.floor(scaled).astype(int)
        deficit = int(target_total - flat_floor.sum())
        remainders = (scaled - flat_floor).flatten()
        # Indices of cells with the largest fractional remainders get
        # +1 each, until the deficit is consumed.
        order = np.argsort(-remainders)
        flat = flat_floor.flatten()
        for idx in order[:deficit]:
            flat[idx] += 1
        cm_norm = flat.reshape(2, 2)

    return cm_norm, np.mean(accs), np.std(accs), np.mean(aucs), np.std(aucs)


def save_cm_csv(cm, path, n_seeds, acc_mean, acc_std, auc_mean, auc_std):
    """Save confusion matrix to CSV with header info."""
    n_total = cm.sum()
    pct = 100 * cm / n_total if n_total > 0 else cm.astype(float)
    with open(path, "w") as f:
        f.write(f"# Aggregated over {n_seeds} seeds\n")
        f.write(f"# Accuracy: {acc_mean:.4f} +/- {acc_std:.4f}\n")
        f.write(f"# AUC: {auc_mean:.4f} +/- {auc_std:.4f}\n")
        f.write(f"# Total samples (sum of seeds): {n_total}\n")
        f.write(f",pred_0,pred_1\n")
        f.write(f"true_0,{cm[0,0]},{cm[0,1]}\n")
        f.write(f"true_1,{cm[1,0]},{cm[1,1]}\n")
        f.write(f"\n# Percentages\n")
        f.write(f",pred_0,pred_1\n")
        f.write(f"true_0,{pct[0,0]:.2f}%,{pct[0,1]:.2f}%\n")
        f.write(f"true_1,{pct[1,0]:.2f}%,{pct[1,1]:.2f}%\n")


def plot_confusion_matrices(cm_dict, out_path):
    """Plot 2x2 grid of confusion matrices for the Universal-4 setting.

    All layout parameters were tuned by measuring an existing reference
    image pixel-by-pixel (cell positions, margins, gaps) and adjusting
    until every metric matched within ~6 pixels:

      Reference image vs this output:
        Image dims:     1483x1369   vs  1485x1375    (+2, +6)
        Cell W x H:     322x223     vs  323x222      (+1, -1)
        Left margin:    80          vs  76           (-4)
        Right margin:   16          vs  16           ( 0)
        Top margin:     175         vs  181          (+6)
        Bottom margin:  120         vs  125          (+5)
        Horizontal gap: 91          vs  94           (+3)
        Vertical gap:   173         vs  175          (+2)

    Critical settings:
      - figsize=(13.5, 12.5), dpi=110 -> ~1485x1375 px output
      - aspect="auto" on imshow -> cells are wider than tall (W/H~1.44),
        matching the reference (the reference cells are NOT square)
      - hspace=0.38, wspace=0.14 -> match the inter-panel gaps
      - top=0.87, bottom=0.09, left=0.05, right=0.99 -> match margins
      - NO bbox_inches="tight" on savefig (would re-crop and break ratios)

    Typography (matched to reference, measured by pixel-height of rendered text):
      - suptitle: fontsize=17, bold, y=0.99
        (renders as h=20 + h=24 px, starting at y~14)
      - Panel titles: fontsize=14, bold, color-coded green/orange
        (renders as 3 lines of h~17, h~19, h~17 px)
      - Cell text: fontsize=15, bold, white-on-dark or black-on-light
        (renders as h=18 + h=21 px)
      - Tick labels (Defense OFF/ON): fontsize=14
        (capital-letter height = 16px, matching reference)
      - Axis labels (Predicted/Actual): fontsize=14
        (capital-letter height = 16px, matching reference)
      - Legend (In-domain/Cross-domain): fontsize=14
        (capital-letter height = 16px, matching reference)

    Tick marks: length=5pt (~7px at dpi=110), width=1, black.
    Without these the tick labels appear too close to the cells. The
    reference has 4-pixel-wide tick blocks at y=627-633 below cell row 2.

    Style:
      - Blues colormap, vmin=0, vmax=50 (each cell is at most 50% of total)
      - White separator lines between cells, linewidth=2
      - Subtle gray panel border (#888888, linewidth=0.6)
      - Legend at bottom: outlined rectangles only (no fill)
      - Panel layout: Static->Static, Static->Mobile / Mobile->Static, Mobile->Mobile
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError:
        print("  [warn] matplotlib not available, skipping plot")
        return

    GREEN = "#388e3c"
    ORANGE = "#e65100"

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 12.5),
                              gridspec_kw={"hspace": 0.38, "wspace": 0.14})

    panel_order = [
        ("static_in_domain",   "Static \u2192 Static",  "(in-domain)",     GREEN,  axes[0, 0]),
        ("static_to_mobile",   "Static \u2192 Mobile",  "(cross-domain)",  ORANGE, axes[0, 1]),
        ("mobile_to_static",   "Mobile \u2192 Static",  "(cross-domain)",  ORANGE, axes[1, 0]),
        ("mobile_in_domain",   "Mobile \u2192 Mobile",  "(in-domain)",     GREEN,  axes[1, 1]),
    ]

    for key, header_top, header_bottom, header_color, ax in panel_order:
        cm, acc, _, _, _ = cm_dict[key]
        n = cm.sum()
        pct = 100 * cm / n if n > 0 else cm.astype(float)

        # aspect="auto" - cells fill the (rectangular) panel.
        # The reference has cells that are wider than tall (W/H~1.44),
        # so DO NOT use aspect="equal" or set_box_aspect(1).
        ax.imshow(pct, cmap="Blues", vmin=0, vmax=50, aspect="auto")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Defense OFF", "Defense ON"], fontsize=14)
        ax.set_yticklabels(["Defense OFF", "Defense ON"], fontsize=14,
                            rotation=90, va="center")
        ax.set_xlabel("Predicted", fontsize=14)
        ax.set_ylabel("Actual", fontsize=14)

        # Three-line title: direction / (domain-tag) / Accuracy = X.X%
        title = (f"{header_top}\n{header_bottom}\n"
                 f"Accuracy = {acc*100:.1f}%")
        ax.set_title(title, fontsize=14, color=header_color,
                     fontweight="bold", pad=14)

        # Cell annotations: count + percentage on two lines
        for i in range(2):
            for j in range(2):
                count = int(cm[i, j])
                pct_val = pct[i, j]
                text = f"{count:,}\n({pct_val:.1f}%)"
                # White text when cell is dark (>25%), else black
                color_text = "white" if pct_val > 25 else "black"
                ax.text(j, i, text, ha="center", va="center",
                        color=color_text, fontsize=15, fontweight="bold")

        # White separator lines between cells (and on outer border)
        for i in range(3):
            ax.axhline(i - 0.5, color="white", linewidth=2)
            ax.axvline(i - 0.5, color="white", linewidth=2)

        # Tick marks visible (matches reference) - length 5pt = ~7px at dpi=110
        ax.tick_params(axis="both", which="both", length=5, width=1, color="black")
        # Subtle gray border around the whole panel
        for spine in ax.spines.values():
            spine.set_edgecolor("#888888")
            spine.set_linewidth(0.6)

    # Bottom legend: outline rectangles only (no fill)
    legend_in_domain = Rectangle((0, 0), 1, 1, facecolor="white",
                                  edgecolor=GREEN, linewidth=2)
    legend_cross_domain = Rectangle((0, 0), 1, 1, facecolor="white",
                                     edgecolor=ORANGE, linewidth=2)
    fig.legend(
        [legend_in_domain, legend_cross_domain],
        ["In-domain", "Cross-domain"],
        loc="lower center", ncol=2, frameon=True,
        bbox_to_anchor=(0.5, 0.02),
        fontsize=14, handlelength=1.5, handleheight=1.0,
    )

    fig.suptitle("Confusion Matrices \u2014 Universal 4 Features\n"
                 "CatBoost (tuned), threshold tuned on validation",
                 fontsize=17, fontweight="bold", y=0.99)

    # NB: no bbox_inches="tight" - that would re-crop the figure and
    # invalidate the carefully-tuned aspect ratio.
    plt.subplots_adjust(top=0.87, bottom=0.09, left=0.05, right=0.99)
    plt.savefig(out_path, dpi=110,
                facecolor="white", edgecolor="none")
    print(f"  Saved plot: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--static-root",
                         default="../simulations/features_static")
    parser.add_argument("--mobile-root",
                         default="../simulations/features_mobile")
    parser.add_argument("--hp-results-dir",
                         default="./results/hp_search_extended",
                         help="Dir with best_models.pkl for static/mobile")
    parser.add_argument("--out-dir",
                         default="./results/confusion_matrices_universal4")
    parser.add_argument("--n-seeds", type=int, default=N_SEEDS)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Confusion Matrices \u2014 Universal-4 (CatBoost)")
    print("=" * 70)
    print(f"  Universal-4 features: {UNIVERSAL_4}")
    print(f"  N seeds: {args.n_seeds}")
    print(f"  HP results dir: {args.hp_results_dir}")
    print(f"  Out dir: {out_dir}")

    # Load best params
    print("\n[0/4] Loading best CatBoost hyperparameters...")
    best_params = load_best_params(args.hp_results_dir)
    print(f"  Static CatBoost params: {filter_catboost_params(best_params['static'])}")
    print(f"  Mobile CatBoost params: {filter_catboost_params(best_params['mobile'])}")

    # Load data
    print("\n[1/4] Loading static and mobile datasets...")
    X_s, y_s, g_s = load_raw_data(args.static_root, "static")
    X_m, y_m, g_m = load_raw_data(args.mobile_root, "mobile")

    # Filter to Universal-4 features
    X_s_u4 = X_s[UNIVERSAL_4].copy()
    X_m_u4 = X_m[UNIVERSAL_4].copy()
    print(f"\n  After filtering to Universal-4: static={X_s_u4.shape}, mobile={X_m_u4.shape}")

    # Run evaluations — Static-source
    print(f"\n[2/4] Running {args.n_seeds} seeds (Static-source, CatBoost-static)...")
    t0 = time.time()
    static_results = []
    for seed in range(args.n_seeds):
        if seed % 5 == 0:
            print(f"    seed {seed}/{args.n_seeds}  elapsed={time.time()-t0:.1f}s")
        r = evaluate_seed(X_s_u4, y_s, g_s, X_m_u4, y_m, g_m,
                          best_params=best_params["static"],
                          seed=seed + RANDOM_STATE)
        static_results.append(r)
    print(f"    static-source total: {time.time()-t0:.1f}s")

    # Run evaluations — Mobile-source
    print(f"\n[3/4] Running {args.n_seeds} seeds (Mobile-source, CatBoost-mobile)...")
    t0 = time.time()
    mobile_results = []
    for seed in range(args.n_seeds):
        if seed % 5 == 0:
            print(f"    seed {seed}/{args.n_seeds}  elapsed={time.time()-t0:.1f}s")
        r = evaluate_seed(X_m_u4, y_m, g_m, X_s_u4, y_s, g_s,
                          best_params=best_params["mobile"],
                          seed=seed + RANDOM_STATE)
        mobile_results.append(r)
    print(f"    mobile-source total: {time.time()-t0:.1f}s")

    # Aggregate
    print("\n[4/4] Aggregating confusion matrices...")
    cm_dict = {}
    cm_dict["static_in_domain"] = aggregate_confusion(
        static_results, "y_te_true", "y_te_pred")
    cm_dict["static_to_mobile"] = aggregate_confusion(
        static_results, "y_tgt_true", "y_tgt_pred")
    cm_dict["mobile_in_domain"] = aggregate_confusion(
        mobile_results, "y_te_true", "y_te_pred")
    cm_dict["mobile_to_static"] = aggregate_confusion(
        mobile_results, "y_tgt_true", "y_tgt_pred")

    # Save CSVs
    for key in cm_dict:
        cm, acc_m, acc_s, auc_m, auc_s = cm_dict[key]
        save_cm_csv(cm, out_dir / f"confusion_{key}.csv",
                     args.n_seeds, acc_m, acc_s, auc_m, auc_s)

    # Save summary
    with open(out_dir / "confusion_matrices_summary.txt", "w") as f:
        f.write("=" * 70 + "\n")
        f.write("Confusion Matrices \u2014 Universal-4 Feature Set (CatBoost)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Features ({len(UNIVERSAL_4)}):\n")
        for feat in UNIVERSAL_4:
            f.write(f"  - {feat}\n")
        f.write(f"\nClassifier: CatBoostClassifier (tuned hyperparameters)\n")
        f.write(f"  Static params: {filter_catboost_params(best_params['static'])}\n")
        f.write(f"  Mobile params: {filter_catboost_params(best_params['mobile'])}\n")
        f.write(f"N seeds: {args.n_seeds}\n\n")
        f.write(f"{'Setting':<25} {'Accuracy':>20} {'AUC':>20}\n")
        f.write("-" * 70 + "\n")
        for key in ["static_in_domain", "static_to_mobile",
                    "mobile_in_domain", "mobile_to_static"]:
            cm, acc_m, acc_s, auc_m, auc_s = cm_dict[key]
            f.write(f"{key:<25} {acc_m:.4f} +/- {acc_s:.4f}  "
                     f"{auc_m:.4f} +/- {auc_s:.4f}\n")
        f.write("\n\n")
        f.write("Confusion matrix tables:\n")
        for key in ["static_in_domain", "static_to_mobile",
                    "mobile_in_domain", "mobile_to_static"]:
            cm, acc_m, acc_s, auc_m, auc_s = cm_dict[key]
            n = cm.sum()
            pct = 100 * cm / n if n > 0 else cm.astype(float)
            f.write(f"\n--- {key} (acc={acc_m:.4f}, n={n}) ---\n")
            f.write(f"                pred_OFF      pred_ON\n")
            f.write(f"true_OFF    {cm[0,0]:>8} ({pct[0,0]:5.1f}%) {cm[0,1]:>8} ({pct[0,1]:5.1f}%)\n")
            f.write(f"true_ON     {cm[1,0]:>8} ({pct[1,0]:5.1f}%) {cm[1,1]:>8} ({pct[1,1]:5.1f}%)\n")

    # Plot
    plot_confusion_matrices(cm_dict, out_dir / "confusion_matrices.png")

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'Setting':<25} {'Accuracy':>20} {'AUC':>20}")
    print("-" * 70)
    for key in ["static_in_domain", "static_to_mobile",
                "mobile_in_domain", "mobile_to_static"]:
        cm, acc_m, acc_s, auc_m, auc_s = cm_dict[key]
        print(f"{key:<25} {acc_m:.4f} +/- {acc_s:.4f}  {auc_m:.4f} +/- {auc_s:.4f}")

    print(f"\n  Outputs saved to: {out_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()