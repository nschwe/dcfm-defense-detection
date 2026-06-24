#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
flowthroughputstd_analysis_v2.py

Kiril note 1: substantiate the claim that FlowThroughputStd contributes through
interactions with the other three Universal-4 features, despite its univariate
class separation collapsing under mobility (Cohen's d: 1.3006 -> 0.0364).

Two analyses, both using the SAME pipeline as confusion_matrices_universal4_v2.py
and k_sweep_universal4_v2.py (CatBoost tuned, 60/20/20 group-aware split,
StandardScaler on train, threshold tuning on validation, 20 seeds):

  A) Universal-3 vs Universal-4 cross-domain.
     Universal-4 = [AverageAdvertisedLinksPerTCMessage, AverageMprCount,
                    DataPacketRate, FlowThroughputStd]
     Universal-3 = Universal-4 minus FlowThroughputStd.
     Reports in-domain and cross-domain (S->M, M->S) accuracy + AUC for both,
     and the paired delta (U4 - U3) with a paired t-test across seeds.

  B) Conditional (leave-one-in / leave-one-out) importance of each Universal-4
     feature WITHIN the subset:
       - drop-column importance: train on the full Universal-4, then on
         Universal-4 minus each feature; the in-domain accuracy drop is the
         feature's marginal contribution GIVEN the other three.
       - This is the direct test of "depends on interactions": if
         FlowThroughputStd has a non-trivial drop-column contribution in BOTH
         configs even though its univariate d collapses under mobility, the
         interaction claim is supported.

This script does NOT invent any numbers. It prints measured values and writes
them to CSV. Run it on the workstation with the sionna env REPLACED by an env
that has CatBoost (the sionna env lacks CatBoost/XGBoost/LightGBM).

Usage:
  cd ~/ns3/Final_Project_NS3-master/strict_observable_v2

  OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 \
  nohup python3 -u flowthroughputstd_analysis_v2.py \
      --static-root ../simulations/features_static \
      --mobile-root ../simulations/features_mobile \
      --hp-results-dir ./results/hp_search_extended \
      --out-dir ./results/flowthroughputstd_analysis \
      --n-seeds 20 \
      > flowthroughputstd_analysis.log 2>&1 &

  echo $! > flowthroughputstd_analysis.pid
"""

import argparse
import gc
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Universal-4 (order fixed; matches Table VI / VII in the paper)
# ----------------------------------------------------------------------------
UNIVERSAL_4 = [
    "AverageAdvertisedLinksPerTCMessage",
    "AverageMprCount",
    "DataPacketRate",
    "FlowThroughputStd",
]
DROP_FEATURE = "FlowThroughputStd"
UNIVERSAL_3 = [f for f in UNIVERSAL_4 if f != DROP_FEATURE]

# ----------------------------------------------------------------------------
# Import the project pipeline. All data-source / engineering logic lives in
# defense_detection_v2.py; we are a thin client (consistent with the project's
# architecture principle).
#
# The sionna env lacks seaborn, which defense_detection_v2 imports only for
# plotting. Inject a stub module so the import succeeds without graphics
# (documented workaround, README part 25.8).
# ----------------------------------------------------------------------------
import types as _types
if "seaborn" not in sys.modules:
    sys.modules["seaborn"] = _types.ModuleType("seaborn")

try:
    import defense_detection_v2 as dd
except Exception as e:  # pragma: no cover
    print(f"[fatal] cannot import defense_detection_v2: {e}", file=sys.stderr)
    print("        run this from the strict_observable_v2 directory.", file=sys.stderr)
    raise

try:
    from catboost import CatBoostClassifier
except Exception as e:  # pragma: no cover
    print(f"[fatal] CatBoost not available: {e}", file=sys.stderr)
    print("        the sionna env lacks CatBoost; activate an env that has it.", file=sys.stderr)
    raise

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy import stats


# ----------------------------------------------------------------------------
# CatBoost hyperparameters: load the tuned config per configuration from the
# saved best_models.pkl, exactly as confusion_matrices_universal4_v2.py does.
# Falls back to the documented tuned values if loading fails.
# ----------------------------------------------------------------------------
DEFAULT_CB_PARAMS = {
    # documented in README part 0.20 / Tables XII-XIII
    "static": {"iterations": 800, "learning_rate": 0.05, "depth": 7},
    "mobile": {"iterations": 1000, "learning_rate": 0.05, "depth": 4},
}

ALLOWED_CB_KEYS = {
    "iterations", "learning_rate", "depth", "l2_leaf_reg",
    "subsample", "random_strength", "bagging_temperature", "border_count",
}


def filter_catboost_params(params):
    """Keep only real CatBoost hyperparameters; strip GPU/task flags etc."""
    return {k: v for k, v in params.items() if k in ALLOWED_CB_KEYS}


def load_tuned_cb_params(hp_results_dir, cfg):
    """Try to read tuned CatBoost params from best_models.pkl; else default."""
    import joblib
    pkl = Path(hp_results_dir) / cfg / "best_models.pkl"
    try:
        models = joblib.load(pkl)
        cb = models.get("catboost", None)
        if cb is not None and hasattr(cb, "get_params"):
            p = filter_catboost_params(cb.get_params())
            if p:
                print(f"[{cfg}] loaded tuned CatBoost params from {pkl}: {p}")
                return p
    except Exception as e:
        print(f"[{cfg}] could not load tuned params ({e}); using documented defaults")
    return dict(DEFAULT_CB_PARAMS[cfg])


def make_cb(params, seed):
    p = dict(params)
    p.update({
        "random_seed": seed,
        "thread_count": 16,
        "verbose": False,
        "allow_writing_files": False,
    })
    return CatBoostClassifier(**p)


# ----------------------------------------------------------------------------
# Data loading: reuse the project pipeline so the 33 base observable metrics
# are produced identically. We then select the Universal-k columns by name.
# ----------------------------------------------------------------------------
def load_base_observable(root, cfg_label):
    """
    Returns (X_df, y, groups) where X_df contains the 33 base observable
    metrics (NOT the 141 engineered space). Universal subsets are drawn from
    these base metrics, consistent with Section VI-D..VI-F of the paper.

    Uses the DefenseDetector pipeline exactly as variance_decomposition_v2.py.
    NOTE: the 33-feature v2 metric set is DefenseDetector.METRICS, which is the
    default (observable_only=False AND report_only=False). The observable_only
    flag selects the SHORTER OBSERVABLE_METRICS list (12 features) that does NOT
    contain AverageMprCount or FlowThroughputStd, so it must stay False here.
      cfg = Config(data_root=root)            # defaults -> METRICS (33)
      pipe = DefenseDetector(cfg)
      tall = pipe.load_simulation_data_enhanced()
      X, y, groups = pipe.preprocess_data_enhanced(tall)
    """
    print(f"[{cfg_label}] loading base observable metrics from {root} ...")
    cfg = dd.Config()
    cfg.data_root = root
    # leave observable_only / report_only False -> DefenseDetector.METRICS (33)
    pipe = dd.DefenseDetector(cfg)
    tall = pipe.load_simulation_data_enhanced()
    X, y, groups = pipe.preprocess_data_enhanced(tall)
    X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
    # sanity: ensure the Universal-4 columns exist
    missing = [f for f in UNIVERSAL_4 if f not in X.columns]
    if missing:
        raise KeyError(
            f"[{cfg_label}] missing expected columns {missing}. "
            f"Available (first 40): {list(X.columns)[:40]}"
        )
    print(f"[{cfg_label}] rows={len(X)} unique_groups={len(np.unique(groups))} "
          f"cols={X.shape[1]}")
    return X, np.asarray(y), np.asarray(groups)


def split_60_20_20(X, y, groups, seed):
    """Group-aware 60/20/20 (test 20%, then val 25% of remaining 80%)."""
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    trval_idx, test_idx = next(gss1.split(X, y, groups))
    g_trval = groups[trval_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    tr_rel, val_rel = next(gss2.split(X.iloc[trval_idx], y[trval_idx], g_trval))
    train_idx = trval_idx[tr_rel]
    val_idx = trval_idx[val_rel]
    return train_idx, val_idx, test_idx


def tune_threshold(y_val, p_val):
    """Pick threshold in [0.10, 0.90] step 0.01 maximizing validation accuracy."""
    best_t, best_a = 0.5, -1.0
    for t in np.arange(0.10, 0.9001, 0.01):
        a = accuracy_score(y_val, (p_val >= t).astype(int))
        if a > best_a:
            best_a, best_t = a, t
    return best_t


def fit_eval(features, X_src, y_src, g_src, X_tgt, y_tgt, params, seed):
    """
    Train CatBoost on source[features], tune threshold on source-val,
    evaluate in-domain (source-test) and cross-domain (full target).
    Returns dict with acc/auc for in-domain and cross-domain.
    """
    tr, va, te = split_60_20_20(X_src, y_src, g_src, seed)
    Xtr = X_src.iloc[tr][features].to_numpy()
    Xva = X_src.iloc[va][features].to_numpy()
    Xte = X_src.iloc[te][features].to_numpy()
    Xtg = X_tgt[features].to_numpy()

    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xva_s, Xte_s, Xtg_s = (scaler.transform(z) for z in (Xtr, Xva, Xte, Xtg))

    clf = make_cb(params, seed)
    clf.fit(Xtr_s, y_src[tr])

    p_va = clf.predict_proba(Xva_s)[:, 1]
    thr = tune_threshold(y_src[va], p_va)

    p_te = clf.predict_proba(Xte_s)[:, 1]
    p_tg = clf.predict_proba(Xtg_s)[:, 1]

    return {
        "in_acc": accuracy_score(y_src[te], (p_te >= thr).astype(int)),
        "in_auc": roc_auc_score(y_src[te], p_te),
        "cross_acc": accuracy_score(y_tgt, (p_tg >= thr).astype(int)),
        "cross_auc": roc_auc_score(y_tgt, p_tg),
        "threshold": thr,
    }


def agg(vals):
    a = np.asarray(vals, dtype=float)
    return float(a.mean()), float(a.std())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--static-root", required=True)
    ap.add_argument("--mobile-root", required=True)
    ap.add_argument("--hp-results-dir", required=True,
                    help="dir with {static,mobile}/best_models.pkl for tuned CatBoost params")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-seeds", type=int, default=20)
    ap.add_argument("--seed-start", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    print(f"Seeds: {seeds}")
    print(f"Universal-4: {UNIVERSAL_4}")
    print(f"Universal-3: {UNIVERSAL_3} (dropped: {DROP_FEATURE})")

    cb_static = load_tuned_cb_params(args.hp_results_dir, "static")
    cb_mobile = load_tuned_cb_params(args.hp_results_dir, "mobile")

    Xs, ys, gs = load_base_observable(args.static_root, "static")
    Xm, ym, gm = load_base_observable(args.mobile_root, "mobile")

    # =====================================================================
    # ANALYSIS A: Universal-3 vs Universal-4 cross-domain
    # =====================================================================
    print("\n" + "=" * 78)
    print(" ANALYSIS A: Universal-3 (no FlowThroughputStd) vs Universal-4")
    print("=" * 78)

    rows_a = []
    # per-seed storage for paired tests
    paired = {
        ("U4", "SM"): [], ("U3", "SM"): [],
        ("U4", "MS"): [], ("U3", "MS"): [],
        ("U4", "in_static"): [], ("U3", "in_static"): [],
        ("U4", "in_mobile"): [], ("U3", "in_mobile"): [],
    }

    for subset_name, feats in [("U4", UNIVERSAL_4), ("U3", UNIVERSAL_3)]:
        for seed in seeds:
            # S->M : train static, eval mobile
            r_sm = fit_eval(feats, Xs, ys, gs, Xm, ym, cb_static, seed)
            # M->S : train mobile, eval static
            r_ms = fit_eval(feats, Xm, ym, gm, Xs, ys, cb_mobile, seed)

            paired[(subset_name, "SM")].append(r_sm["cross_acc"])
            paired[(subset_name, "MS")].append(r_ms["cross_acc"])
            paired[(subset_name, "in_static")].append(r_sm["in_acc"])
            paired[(subset_name, "in_mobile")].append(r_ms["in_acc"])

            rows_a.append({
                "subset": subset_name, "seed": seed,
                "in_static_acc": r_sm["in_acc"], "in_static_auc": r_sm["in_auc"],
                "in_mobile_acc": r_ms["in_acc"], "in_mobile_auc": r_ms["in_auc"],
                "SM_acc": r_sm["cross_acc"], "SM_auc": r_sm["cross_auc"],
                "MS_acc": r_ms["cross_acc"], "MS_auc": r_ms["cross_auc"],
            })
        gc.collect()

    df_a = pd.DataFrame(rows_a)
    df_a.to_csv(out / "analysis_A_per_seed.csv", index=False)

    # summary + paired tests (U4 - U3)
    summ_a = []
    for direction, key in [("S->M", "SM"), ("M->S", "MS"),
                           ("in-static", "in_static"), ("in-mobile", "in_mobile")]:
        u4 = np.asarray(paired[("U4", key)])
        u3 = np.asarray(paired[("U3", key)])
        d = u4 - u3
        t, p = stats.ttest_rel(u4, u3)
        # paired Cohen's d
        cohen = d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else float("nan")
        summ_a.append({
            "direction": direction,
            "U4_mean": u4.mean(), "U4_std": u4.std(),
            "U3_mean": u3.mean(), "U3_std": u3.std(),
            "delta_U4_minus_U3": d.mean(),
            "paired_t": t, "p_value": p, "paired_cohens_d": cohen,
        })
    df_summ_a = pd.DataFrame(summ_a)
    df_summ_a.to_csv(out / "analysis_A_summary.csv", index=False)

    print("\n--- Analysis A summary (mean over seeds) ---")
    for r in summ_a:
        print(f"  {r['direction']:>10}: U4={r['U4_mean']:.4f}  U3={r['U3_mean']:.4f}  "
              f"Δ(U4-U3)={r['delta_U4_minus_U3']:+.4f}  "
              f"p={r['p_value']:.3e}  d={r['paired_cohens_d']:+.2f}")

    # =====================================================================
    # ANALYSIS B: drop-column (conditional) importance within Universal-4
    # =====================================================================
    print("\n" + "=" * 78)
    print(" ANALYSIS B: drop-column importance within Universal-4 (in-domain)")
    print(" contribution of each feature GIVEN the other three")
    print("=" * 78)

    rows_b = []
    for cfg_label, X, y, g, params in [
        ("static", Xs, ys, gs, cb_static),
        ("mobile", Xm, ym, gm, cb_mobile),
    ]:
        # full Universal-4 in-domain baseline per seed
        base_acc = {}
        for seed in seeds:
            r = fit_eval(UNIVERSAL_4, X, y, g, X, y, params, seed)  # target=source -> uses test partition for in_acc
            base_acc[seed] = r["in_acc"]

        for drop in UNIVERSAL_4:
            feats = [f for f in UNIVERSAL_4 if f != drop]
            drops = []
            for seed in seeds:
                r = fit_eval(feats, X, y, g, X, y, params, seed)
                # contribution of `drop` GIVEN the rest = base - (without drop)
                drops.append(base_acc[seed] - r["in_acc"])
            m, s = agg(drops)
            t, p = stats.ttest_1samp(drops, 0.0)
            rows_b.append({
                "config": cfg_label, "dropped_feature": drop,
                "mean_acc_drop": m, "std_acc_drop": s,
                "t_stat": t, "p_value": p,
            })
            print(f"  [{cfg_label}] drop {drop:36s} "
                  f"Δacc={m:+.4f} ± {s:.4f}  p={p:.3e}")
        gc.collect()

    df_b = pd.DataFrame(rows_b)
    df_b.to_csv(out / "analysis_B_dropcolumn.csv", index=False)

    # =====================================================================
    # Final human-readable summary
    # =====================================================================
    with open(out / "summary.txt", "w") as fh:
        fh.write("FlowThroughputStd analysis (Kiril note 1)\n")
        fh.write("=" * 60 + "\n\n")
        fh.write("ANALYSIS A: Universal-3 vs Universal-4 cross-domain\n")
        fh.write(df_summ_a.to_string(index=False) + "\n\n")
        fh.write("ANALYSIS B: drop-column importance within Universal-4\n")
        fh.write(df_b.to_string(index=False) + "\n\n")
        fh.write("Interpretation key:\n")
        fh.write("  - If U4 > U3 in cross-domain (positive delta, p<0.05), the\n")
        fh.write("    feature contributes to transfer despite univariate collapse.\n")
        fh.write("  - In Analysis B, a non-trivial drop-column contribution for\n")
        fh.write("    FlowThroughputStd in BOTH configs (especially mobile, where\n")
        fh.write("    its univariate d=0.0364) is the direct evidence that its\n")
        fh.write("    value comes from interactions, not univariate separation.\n")

    print("\n[done] outputs written to", out)
    print("  analysis_A_per_seed.csv, analysis_A_summary.csv")
    print("  analysis_B_dropcolumn.csv, summary.txt")


if __name__ == "__main__":
    main()