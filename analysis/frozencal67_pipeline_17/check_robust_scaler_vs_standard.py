#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
check_robust_scaler_vs_standard.py
============================================================================

Diagnostic: does the +0.127 improvement of Ablation_DJ_27 over Standard_33
in S->M (CatBoost) persist under RobustScaler, or is it an artifact of
StandardScaler's sensitivity to DJ feature scale shifts between domains?

Hypothesis: the 6 delay/jitter features have scale ~20x larger in mobile
than in static. StandardScaler fitted on static train then applied to
mobile produces scaled values up to |max|=508 instead of the expected
[-3, +3] range. This pushes CatBoost's learned decision thresholds into
saturated regions, hurting cross-domain accuracy. Removing the DJ features
(Ablation_DJ_27) removes the source of the scale-mismatch.

If RobustScaler (median/IQR-based) eliminates the gap → it's a scaler
artifact, and Ablation_DJ_27's improvement is methodological, not a
genuine indicator that DJ features hurt detection.

If the gap persists → there is also a signal/distribution issue beyond
scaling that warrants discussion.

Configuration:
  - Configs:    Standard_33, Ablation_DJ_27
  - Classifier: CatBoost (tuned, from hp_search_extended)
  - Direction:  S -> M only
  - Scalers:    StandardScaler (reproduce) and RobustScaler (test)
  - Seeds:      5 (42..46) — enough to confirm direction of effect

Output:
  results/robust_scaler_check/scaler_comparison.csv
  Plus a tidy printout to stdout.

Usage:
    python3 check_robust_scaler_vs_standard.py
============================================================================
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", "16")

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
import joblib

import pandas.core.series
pandas.core.series.dtype = np.dtype

from catboost import CatBoostClassifier

from feature_eng_ablation_v2 import (
    METRICS_33, DJ_FEATURES,
    load_raw_data, build_standard_33, build_ablation_dj_27,
    load_best_params, filter_catboost_params,
    split_source_60_20_20, select_threshold,
)

N_JOBS_CATBOOST = 16


def build_catboost(best_params, seed):
    p = filter_catboost_params(best_params)
    return CatBoostClassifier(
        random_seed=seed, verbose=0, allow_writing_files=False,
        thread_count=N_JOBS_CATBOOST, **p,
    )


def evaluate(F_src, y_src, g_src, F_tgt, y_tgt, best_params, scaler_cls, seed):
    tr, va, te = split_source_60_20_20(F_src, y_src, g_src, seed)

    sc = scaler_cls().fit(F_src[tr])
    X_tr = sc.transform(F_src[tr])
    X_va = sc.transform(F_src[va])
    X_te = sc.transform(F_src[te])
    X_tg = sc.transform(F_tgt)

    clf = build_catboost(best_params, seed=seed)
    clf.fit(X_tr, y_src[tr])

    p_va = clf.predict_proba(X_va)[:, 1]
    thr = select_threshold(y_src[va], p_va)

    p_te = clf.predict_proba(X_te)[:, 1]
    p_tg = clf.predict_proba(X_tg)[:, 1]

    return {
        "te_acc":  accuracy_score(y_src[te], (p_te >= thr).astype(int)),
        "te_auc":  roc_auc_score(y_src[te], p_te),
        "tgt_acc": accuracy_score(y_tgt, (p_tg >= thr).astype(int)),
        "tgt_auc": roc_auc_score(y_tgt, p_tg),
        # Diagnostic: scaled mobile DJ max value
        "mobile_DJ_max_abs_scaled": _diag_mobile_dj_max(sc, F_tgt),
    }


def _diag_mobile_dj_max(scaler, F_tgt):
    """Largest |z-score| reached by any DJ feature on the mobile target
    after applying the source-fitted scaler. Captures the scale-mismatch."""
    F_t_scaled = scaler.transform(F_tgt)
    # Need indices of DJ features in F_tgt — but F_tgt could be 33 or 27
    # columns. Caller guarantees this is only called when F_tgt has 33 cols.
    if F_t_scaled.shape[1] != 33:
        return float("nan")  # Ablation has no DJ to measure
    dj_idx = [METRICS_33.index(f) for f in DJ_FEATURES]
    return float(np.abs(F_t_scaled[:, dj_idx]).max())


def main():
    out_dir = Path("./results/robust_scaler_check")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72, flush=True)
    print("DIAGNOSTIC: StandardScaler vs RobustScaler on Ablation_DJ_27 gap",
          flush=True)
    print("=" * 72, flush=True)

    # Load data
    print("\nLoading data...", flush=True)
    X_s, y_s, g_s = load_raw_data("../simulations/features_static", "static")
    X_m, y_m, g_m = load_raw_data("../simulations/features_mobile", "mobile")
    y_s_arr = y_s.to_numpy()
    y_m_arr = y_m.to_numpy()

    # Build features (33 and 27 versions)
    F_s_33 = build_standard_33(X_s)
    F_m_33 = build_standard_33(X_m)
    F_s_27 = build_ablation_dj_27(X_s)
    F_m_27 = build_ablation_dj_27(X_m)
    print(f"  Standard_33:    F_s shape={F_s_33.shape}, F_m shape={F_m_33.shape}",
          flush=True)
    print(f"  Ablation_DJ_27: F_s shape={F_s_27.shape}, F_m shape={F_m_27.shape}",
          flush=True)

    # Load CatBoost params (static, since we're doing S->M)
    print("\nLoading CatBoost (static) hyperparameters...", flush=True)
    best_params = load_best_params("./results/hp_search_extended")
    print(f"  Static params: {filter_catboost_params(best_params['static'])}",
          flush=True)

    # Run grid: 2 configs x 2 scalers x 5 seeds
    seeds = list(range(42, 47))
    configs = [("Standard_33", F_s_33, F_m_33),
               ("Ablation_DJ_27", F_s_27, F_m_27)]
    scalers = [("StandardScaler", StandardScaler),
               ("RobustScaler", RobustScaler)]

    rows = []
    t0 = time.time()
    for cfg_name, F_s, F_m in configs:
        for scaler_name, scaler_cls in scalers:
            for seed in seeds:
                res = evaluate(F_s, y_s_arr, g_s, F_m, y_m_arr,
                               best_params=best_params["static"],
                               scaler_cls=scaler_cls, seed=seed)
                rows.append({
                    "config": cfg_name,
                    "scaler": scaler_name,
                    "seed":   seed,
                    **res,
                })
                print(f"  {cfg_name:<16} {scaler_name:<16} seed={seed}  "
                      f"tgt_acc={res['tgt_acc']:.4f}  "
                      f"tgt_auc={res['tgt_auc']:.4f}  "
                      f"DJ|z|max={res['mobile_DJ_max_abs_scaled']:.1f}",
                      flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "scaler_comparison.csv", index=False)
    print(f"\nWrote: {out_dir/'scaler_comparison.csv'}", flush=True)

    # ---- Summary ----
    print("\n" + "=" * 80, flush=True)
    print("SUMMARY: S->M cross-domain (mean over 5 seeds)", flush=True)
    print("=" * 80, flush=True)
    print(f"{'Config':<18} {'Scaler':<18} {'tgt_acc':>14} {'tgt_auc':>14} "
          f"{'DJ|z|max':>14}", flush=True)
    print("-" * 80, flush=True)
    summary = (df.groupby(["config", "scaler"])
                 .agg(tgt_acc_mean=("tgt_acc", "mean"),
                      tgt_acc_std=("tgt_acc", "std"),
                      tgt_auc_mean=("tgt_auc", "mean"),
                      tgt_auc_std=("tgt_auc", "std"),
                      dj_zmax=("mobile_DJ_max_abs_scaled", "mean")))
    for (cfg, sc), r in summary.iterrows():
        dj_z = "n/a" if np.isnan(r["dj_zmax"]) else f"{r['dj_zmax']:.1f}"
        print(f"{cfg:<18} {sc:<18} "
              f"{r['tgt_acc_mean']:.4f}+/-{r['tgt_acc_std']:.4f}  "
              f"{r['tgt_auc_mean']:.4f}+/-{r['tgt_auc_std']:.4f}  "
              f"{dj_z:>14}", flush=True)

    # ---- The key comparison ----
    print("\n" + "=" * 80, flush=True)
    print("KEY: Gap (Ablation_DJ_27 - Standard_33) under each scaler",
          flush=True)
    print("=" * 80, flush=True)
    for scaler_name, _ in scalers:
        std_acc = df[(df["config"]=="Standard_33") &
                     (df["scaler"]==scaler_name)]["tgt_acc"].mean()
        abl_acc = df[(df["config"]=="Ablation_DJ_27") &
                     (df["scaler"]==scaler_name)]["tgt_acc"].mean()
        std_auc = df[(df["config"]=="Standard_33") &
                     (df["scaler"]==scaler_name)]["tgt_auc"].mean()
        abl_auc = df[(df["config"]=="Ablation_DJ_27") &
                     (df["scaler"]==scaler_name)]["tgt_auc"].mean()
        print(f"  {scaler_name}:", flush=True)
        print(f"    acc: Standard_33={std_acc:.4f}  Ablation_DJ_27={abl_acc:.4f}  "
              f"GAP={abl_acc-std_acc:+.4f}", flush=True)
        print(f"    auc: Standard_33={std_auc:.4f}  Ablation_DJ_27={abl_auc:.4f}  "
              f"GAP={abl_auc-std_auc:+.4f}", flush=True)

    print(f"\nTotal elapsed: {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
