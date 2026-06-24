#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
dj_ablation_breakdown_MS.py
============================================================================

Mirror of dj_ablation_breakdown.py, but evaluates M -> S instead of S -> M.

This completes the asymmetry analysis: in the main feature_eng_ablation_v2
run, Ablation_DJ_27 in M->S gave Δacc=-0.0004 (negligible) while S->M
gave Δacc=+0.1365. This script tests whether the same per-group profile
appears in M->S, or whether the asymmetry holds at all granularity levels.

Setup is identical to dj_ablation_breakdown.py except:
  - Direction: M -> S
  - CatBoost params:  mobile (loaded from hp_search_extended/mobile)

Total fits: 17 * 20 = 340.  Expected runtime: ~10-15 min (CatBoost mobile
params are typically faster: depth=4 vs depth=7 for static).

Output:
  results/dj_ablation_breakdown_MS/per_seed.csv
  results/dj_ablation_breakdown_MS/summary.csv

Usage:
    python3 dj_ablation_breakdown_MS.py
============================================================================
"""

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", "16")

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy import stats

import pandas.core.series
pandas.core.series.dtype = np.dtype

from catboost import CatBoostClassifier

from feature_eng_ablation_v2 import (
    METRICS_33, DJ_FEATURES,
    load_raw_data,
    load_best_params, filter_catboost_params,
    split_source_60_20_20, select_threshold,
)

# Reuse exact same configuration definitions from the S->M breakdown
from dj_ablation_breakdown import (
    UNIVERSAL_4, PROTECTED, NON_PROTECTED_POOL,
    RAND_SEEDS, RANDOM_REMOVALS, ABLATION_SETS,
    build_features, build_catboost, evaluate, paired_test,
    N_JOBS_CATBOOST,
)


def main():
    out_dir = Path("./results/dj_ablation_breakdown_MS")
    out_dir.mkdir(parents=True, exist_ok=True)
    per_seed_path = out_dir / "per_seed.csv"

    print("=" * 78, flush=True)
    print("DJ ABLATION BREAKDOWN — M -> S direction", flush=True)
    print("=" * 78, flush=True)
    print(f"\nProtected from random removal (DJ + Universal-4): "
          f"{len(PROTECTED)} features", flush=True)
    print(f"Random-removal control sets reused from S->M run.", flush=True)
    print(f"Instability-based controls (Control 3) reused from S->M run.",
          flush=True)

    # ---- Load data ----
    print("\nLoading data...", flush=True)
    X_s, y_s, g_s = load_raw_data("../simulations/features_static", "static")
    X_m, y_m, g_m = load_raw_data("../simulations/features_mobile", "mobile")
    y_s_arr = y_s.to_numpy()
    y_m_arr = y_m.to_numpy()

    # ---- Load CatBoost (MOBILE) params ----
    print("\nLoading CatBoost (mobile) hyperparameters...", flush=True)
    best_params = load_best_params("./results/hp_search_extended")
    print(f"  Mobile params: {filter_catboost_params(best_params['mobile'])}",
          flush=True)

    # ---- Run all configs * seeds ----
    seeds = list(range(42, 62))  # 20 seeds — same as S->M for paired comparison
    rows = []
    t0 = time.time()
    total_fits = len(ABLATION_SETS) * len(seeds)
    print(f"\nRunning {len(ABLATION_SETS)} configs * {len(seeds)} seeds "
          f"= {total_fits} fits (M -> S)...\n", flush=True)

    fit_idx = 0
    for cfg_name, remove_set in ABLATION_SETS.items():
        F_s, kept = build_features(X_s, remove_set)
        F_m, _ = build_features(X_m, remove_set)
        n_kept = len(kept)
        removed_str = ", ".join(sorted(remove_set)) if remove_set else "(none)"

        for seed in seeds:
            fit_idx += 1
            t_fit = time.time()
            # M -> S : source=mobile, target=static, mobile params
            res = evaluate(F_m, y_m_arr, g_m, F_s, y_s_arr,
                           best_params=best_params["mobile"], seed=seed)
            rows.append({
                "config":    cfg_name,
                "n_kept":    n_kept,
                "removed":   removed_str,
                "seed":      seed,
                **res,
            })
            pd.DataFrame(rows).to_csv(per_seed_path, index=False)
            elapsed = time.time() - t0
            eta = elapsed / fit_idx * (total_fits - fit_idx)
            print(f"  [{fit_idx:>3}/{total_fits}] {cfg_name:<32} "
                  f"(k={n_kept})  seed={seed}  "
                  f"tgt_acc={res['tgt_acc']:.4f}  "
                  f"tgt_auc={res['tgt_auc']:.4f}  "
                  f"({time.time()-t_fit:.1f}s, eta={eta/60:.1f}min)",
                  flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(per_seed_path, index=False)
    print(f"\nWrote: {per_seed_path}", flush=True)

    # ---- Summary ----
    summary = (df.groupby("config")
                 .agg(n_kept=("n_kept", "first"),
                      removed=("removed", "first"),
                      tgt_acc_mean=("tgt_acc", "mean"),
                      tgt_acc_std=("tgt_acc", "std"),
                      tgt_auc_mean=("tgt_auc", "mean"),
                      tgt_auc_std=("tgt_auc", "std"))
                 .reset_index())

    base = df[df["config"] == "Standard_33"].sort_values("seed")
    base_acc = base.set_index("seed")["tgt_acc"]
    base_auc = base.set_index("seed")["tgt_auc"]

    deltas_rows = []
    for cfg in summary["config"]:
        if cfg == "Standard_33":
            continue
        exp = df[df["config"] == cfg].sort_values("seed")
        exp_acc = exp.set_index("seed")["tgt_acc"]
        exp_auc = exp.set_index("seed")["tgt_auc"]
        common = base_acc.index.intersection(exp_acc.index)
        d_acc_mean, d_acc_std, p_acc, cd_acc = paired_test(
            exp_acc.loc[common].values, base_acc.loc[common].values)
        d_auc_mean, d_auc_std, p_auc, cd_auc = paired_test(
            exp_auc.loc[common].values, base_auc.loc[common].values)
        deltas_rows.append({
            "config":         cfg,
            "delta_acc":      d_acc_mean,
            "delta_acc_std":  d_acc_std,
            "p_acc":          p_acc,
            "d_paired_acc":   cd_acc,
            "delta_auc":      d_auc_mean,
            "delta_auc_std":  d_auc_std,
            "p_auc":          p_auc,
            "d_paired_auc":   cd_auc,
        })

    deltas_df = pd.DataFrame(deltas_rows)
    out_summary = summary.merge(deltas_df, on="config", how="left")
    out_summary.to_csv(out_dir / "summary.csv", index=False)
    print(f"Wrote: {out_dir/'summary.csv'}", flush=True)

    # ---- Pretty stdout report ----
    print("\n" + "=" * 120, flush=True)
    print(f"RESULTS: M -> S cross-domain accuracy by ablation set "
          f"(CatBoost mobile params, {len(seeds)} seeds, "
          "mean±std, paired Δ vs Standard_33)", flush=True)
    print("=" * 120, flush=True)
    order = [
        "Standard_33",
        "LOO_AvgE2EDelay", "LOO_AvgJitter",
        "LOO_AvgFlowDelay", "LOO_AvgFlowJit",
        "LOO_FlowDelayStd", "LOO_FlowJitterStd",
        "DJ_minus2_Avg_only",
        "DJ_minus4_Flow_only",
        "DJ_minus2_Flow_extremes",
        "DJ_minus6_all",
        # instability-based controls (Control 3)
        "Top6_dShift_nonDJ",
        "Top6_dShift_nonDJ_nonU4",
    ] + list(RANDOM_REMOVALS.keys())

    def fmt(val, fmt_str):
        if isinstance(val, float) and np.isnan(val):
            return "n/a"
        return f"{val:{fmt_str}}"

    print(f"{'Config':<32} {'k':>3} {'tgt_acc':>16} {'Δacc':>10} "
          f"{'d':>7} {'p':>10}  {'tgt_auc':>16} {'Δauc':>10}", flush=True)
    print("-" * 120, flush=True)
    for cfg in order:
        if cfg not in summary["config"].values:
            continue
        srow = summary[summary["config"] == cfg].iloc[0]
        if cfg == "Standard_33":
            print(f"{cfg:<32} {srow['n_kept']:>3} "
                  f"{srow['tgt_acc_mean']:>8.4f}±{srow['tgt_acc_std']:.4f}  "
                  f"{'---':>10} {'---':>7} {'---':>10}  "
                  f"{srow['tgt_auc_mean']:>8.4f}±{srow['tgt_auc_std']:.4f}  "
                  f"{'---':>10}", flush=True)
        else:
            d = deltas_df[deltas_df["config"] == cfg].iloc[0]
            print(f"{cfg:<32} {srow['n_kept']:>3} "
                  f"{srow['tgt_acc_mean']:>8.4f}±{srow['tgt_acc_std']:.4f}  "
                  f"{fmt(d['delta_acc'], '+10.4f')} "
                  f"{fmt(d['d_paired_acc'], '+7.2f')} "
                  f"{fmt(d['p_acc'], '10.2e')}  "
                  f"{srow['tgt_auc_mean']:>8.4f}±{srow['tgt_auc_std']:.4f}  "
                  f"{fmt(d['delta_auc'], '+10.4f')}",
                  flush=True)

    # ---- Quick interpretation ----
    print("\n" + "=" * 78, flush=True)
    print("QUICK INTERPRETATION (M -> S)", flush=True)
    print("=" * 78, flush=True)

    def get_delta(cfg, col="delta_acc"):
        r = deltas_df[deltas_df["config"] == cfg]
        if len(r) == 0:
            return float("nan")
        return float(r[col].iloc[0])

    full_acc = get_delta("DJ_minus6_all", "delta_acc")
    full_auc = get_delta("DJ_minus6_all", "delta_auc")
    flow_ext_acc = get_delta("DJ_minus2_Flow_extremes", "delta_acc")
    flow_ext_auc = get_delta("DJ_minus2_Flow_extremes", "delta_auc")
    avg_only_acc = get_delta("DJ_minus2_Avg_only", "delta_acc")
    avg_only_auc = get_delta("DJ_minus2_Avg_only", "delta_auc")
    flow_only_acc = get_delta("DJ_minus4_Flow_only", "delta_acc")
    flow_only_auc = get_delta("DJ_minus4_Flow_only", "delta_auc")

    print(f"\nFull DJ_minus6 gap (M -> S):", flush=True)
    print(f"  acc: {full_acc:+.4f}    auc: {full_auc:+.4f}", flush=True)
    print(f"\n  Flow extremes (2 features):  acc={flow_ext_acc:+.4f}  "
          f"auc={flow_ext_auc:+.4f}", flush=True)
    print(f"  Avg features  (2 features):  acc={avg_only_acc:+.4f}  "
          f"auc={avg_only_auc:+.4f}", flush=True)
    print(f"  Flow-only     (4 features):  acc={flow_only_acc:+.4f}  "
          f"auc={flow_only_auc:+.4f}", flush=True)

    # Instability-based controls (Control 3)
    instab_nondj_acc = get_delta("Top6_dShift_nonDJ", "delta_acc")
    instab_nondj_auc = get_delta("Top6_dShift_nonDJ", "delta_auc")
    instab_nondj_nonu4_acc = get_delta("Top6_dShift_nonDJ_nonU4", "delta_acc")
    instab_nondj_nonu4_auc = get_delta("Top6_dShift_nonDJ_nonU4", "delta_auc")
    print(f"\nInstability-based controls (top-6 by |Δd|, M -> S):", flush=True)
    print(f"  Top6_dShift_nonDJ      (includes U4):   "
          f"acc={instab_nondj_acc:+.4f}  auc={instab_nondj_auc:+.4f}",
          flush=True)
    print(f"  Top6_dShift_nonDJ_nonU4 (primary):      "
          f"acc={instab_nondj_nonu4_acc:+.4f}  auc={instab_nondj_nonu4_auc:+.4f}",
          flush=True)
    print(f"  ^ Expected near-zero (DJ effect is also near-zero in M->S).",
          flush=True)

    rand_deltas_acc = [get_delta(f"Random_minus6_seed{rs}", "delta_acc")
                       for rs in RAND_SEEDS]
    rand_deltas_acc = [d for d in rand_deltas_acc if not np.isnan(d)]
    rand_deltas_auc = [get_delta(f"Random_minus6_seed{rs}", "delta_auc")
                       for rs in RAND_SEEDS]
    rand_deltas_auc = [d for d in rand_deltas_auc if not np.isnan(d)]
    if rand_deltas_acc:
        print(f"\nRandom controls (mean ± std, {len(rand_deltas_acc)} controls):",
              flush=True)
        print(f"  Δacc = {np.mean(rand_deltas_acc):+.4f} "
              f"± {np.std(rand_deltas_acc, ddof=1):.4f}  "
              f"(max |Δ| = {max(abs(d) for d in rand_deltas_acc):.4f})",
              flush=True)
        print(f"  Δauc = {np.mean(rand_deltas_auc):+.4f} "
              f"± {np.std(rand_deltas_auc, ddof=1):.4f}  "
              f"(max |Δ| = {max(abs(d) for d in rand_deltas_auc):.4f})",
              flush=True)

    print(f"\nTotal elapsed: {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
