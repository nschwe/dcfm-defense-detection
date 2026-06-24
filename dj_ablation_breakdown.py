#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
dj_ablation_breakdown.py  (v2 — QA-revised)
============================================================================

Decomposes the +0.1365 S->M cross-domain accuracy improvement of
Ablation_DJ_27 over Standard_33 (observed in feature_eng_ablation_v2)
into per-feature and per-group contributions, plus a random-removal
control group.

Hypothesis:  the improvement is driven by a small number of DJ features
that suffer severe distribution shift between static and mobile, not by
the act of removing features per se.

Configurations (17 total):

  Baseline (1):
    Standard_33     : 33 base metrics (the reference)

  Leave-one-out singletons (6):
    LOO_<feature>   : 33 minus one DJ feature

  Hypothesis-driven groups (4):
    DJ_minus2_Flow_extremes : 33 minus {FlowJitterStd, FlowDelayStd}
                              (the two with std ratio ~20x)
    DJ_minus2_Avg_only      : 33 minus {AverageEndToEndDelay, AverageJitter}
    DJ_minus4_Flow_only     : 33 minus the 4 flow-level DJ features
    DJ_minus6_all           : 33 minus all 6 DJ (= Ablation_DJ_27)

  Random-removal controls (4):
    Random_minus6_seed{1001,2002,3003,4004} : 33 minus 6 random features
                              from the non-DJ AND non-Universal-4 pool
                              (23 candidates). Universal-4 features are
                              protected to avoid confounding from removing
                              known-strong features.

  Instability-based controls (2) [Control 3]:
    Top6_dShift_nonDJ       : 33 minus the top-6 features by |Δd| between
                              static and mobile, from the non-DJ pool
                              (27 candidates). Includes FlowThroughputStd,
                              a Universal-4 member; serves as a sensitivity
                              check that mixes instability with removal of
                              an invariant feature.
    Top6_dShift_nonDJ_nonU4 : 33 minus the top-6 features by |Δd|, from
                              the non-DJ AND non-Universal-4 pool
                              (23 candidates). Primary instability control:
                              tests whether instability magnitude alone
                              explains the DJ-ablation effect.

Setup:
  - Classifier:  CatBoost (tuned static params)
  - Direction:   S -> M only  (the asymmetry of interest)
  - Scaler:      StandardScaler  (CatBoost is scale-invariant; same as v1)
  - Seeds:       20 per config (42..61)

Total fits: 17 * 20 = 340.  Expected runtime: ~25-45 min.

Output:
  results/dj_ablation_breakdown/per_seed.csv      (incrementally saved)
  results/dj_ablation_breakdown/summary.csv

Usage:
    python3 dj_ablation_breakdown.py
============================================================================
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", "16")

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

import pandas.core.series
pandas.core.series.dtype = np.dtype

from catboost import CatBoostClassifier
from scipy import stats

from feature_eng_ablation_v2 import (
    METRICS_33, DJ_FEATURES,
    load_raw_data,
    load_best_params, filter_catboost_params,
    split_source_60_20_20, select_threshold,
)

N_JOBS_CATBOOST = 16


# ============================================================================
# Universal-4 features — protected from random-removal controls.
# Removing these would confound the "random feature removal" comparison
# because they were independently shown in Section VI-E to be critical.
# ============================================================================

UNIVERSAL_4 = [
    "AverageAdvertisedLinksPerTCMessage",
    "AverageMprCount",
    "DataPacketRate",
    "FlowThroughputStd",
]
for f in UNIVERSAL_4:
    assert f in METRICS_33, f"Universal-4 feature {f} not in METRICS_33"


# ============================================================================
# Top-6 sets ranked by absolute change in Cohen's d between static and mobile.
# Source: strict_observable_v2/results/cohens_d/cohens_d_all_features.csv
# (computed offline by compute_cohens_d_v2.py; values hard-coded here for
# reproducibility — rerun ranking if cohens_d_all_features.csv is regenerated).
#
# Two variants:
#   - TOP6_DSHIFT_NONDJ      : top-6 over the 27 non-DJ features.
#                               Includes FlowThroughputStd (Universal-4).
#                               Sensitivity check: mixes instability with
#                               removal of an invariant Universal-4 feature.
#   - TOP6_DSHIFT_NONDJ_NONU4: top-6 over the 23 non-DJ AND non-Universal-4
#                               features. Primary instability control —
#                               tests whether cross-domain instability
#                               magnitude alone explains the DJ effect.
# ============================================================================

TOP6_DSHIFT_NONDJ = [
    "FlowThroughputStd",      # |Δd|=1.2642  (Universal-4 member)
    "AvgFlowThroughput",      # |Δd|=0.8578
    "FlowCount",              # |Δd|=0.7673
    "TcMessageRate",          # |Δd|=0.5140
    "AvgTxPacketsPerFlow",    # |Δd|=0.5115
    "NormalizedRoutingLoad",  # |Δd|=0.5098
]

TOP6_DSHIFT_NONDJ_NONU4 = [
    "AvgFlowThroughput",      # |Δd|=0.8578
    "FlowCount",              # |Δd|=0.7673
    "TcMessageRate",          # |Δd|=0.5140
    "AvgTxPacketsPerFlow",    # |Δd|=0.5115
    "NormalizedRoutingLoad",  # |Δd|=0.5098
    "AvgRxPacketsPerFlow",    # |Δd|=0.4847
]

for f in TOP6_DSHIFT_NONDJ:
    assert f in METRICS_33, f"TOP6_DSHIFT_NONDJ feature {f} not in METRICS_33"
    assert f not in DJ_FEATURES, \
        f"TOP6_DSHIFT_NONDJ should not contain DJ feature: {f}"
for f in TOP6_DSHIFT_NONDJ_NONU4:
    assert f in METRICS_33, \
        f"TOP6_DSHIFT_NONDJ_NONU4 feature {f} not in METRICS_33"
    assert f not in DJ_FEATURES, \
        f"TOP6_DSHIFT_NONDJ_NONU4 should not contain DJ feature: {f}"
    assert f not in UNIVERSAL_4, \
        f"TOP6_DSHIFT_NONDJ_NONU4 should not contain Universal-4 feature: {f}"


# ============================================================================
# Random-removal controls: protect both DJ (under study) AND Universal-4
# ============================================================================

PROTECTED = set(DJ_FEATURES) | set(UNIVERSAL_4)  # 6 + 4 = 10 features
NON_PROTECTED_POOL = [m for m in METRICS_33 if m not in PROTECTED]
assert len(NON_PROTECTED_POOL) == 23, \
    f"Expected 23 non-protected features, got {len(NON_PROTECTED_POOL)}"

RAND_SEEDS = [1001, 2002, 3003, 4004]  # 4 distinct random-removal controls
RANDOM_REMOVALS = {}
for rs in RAND_SEEDS:
    _rng = np.random.RandomState(rs)
    sample = sorted(_rng.choice(NON_PROTECTED_POOL, size=6, replace=False).tolist())
    RANDOM_REMOVALS[f"Random_minus6_seed{rs}"] = set(sample)


# ============================================================================
# Configurations to evaluate
# ============================================================================

ABLATION_SETS = {
    # Baseline
    "Standard_33":                set(),

    # Leave-one-out singletons (6 DJ features removed one at a time)
    "LOO_AvgE2EDelay":            {"AverageEndToEndDelay"},
    "LOO_AvgJitter":              {"AverageJitter"},
    "LOO_AvgFlowDelay":           {"AvgFlowDelay"},
    "LOO_AvgFlowJit":             {"AvgFlowJitter"},
    "LOO_FlowDelayStd":           {"FlowDelayStd"},
    "LOO_FlowJitterStd":          {"FlowJitterStd"},

    # Hypothesis-driven groups
    "DJ_minus2_Flow_extremes":    {"FlowDelayStd", "FlowJitterStd"},
    "DJ_minus2_Avg_only":         {"AverageEndToEndDelay", "AverageJitter"},
    "DJ_minus4_Flow_only":        {"AvgFlowDelay", "AvgFlowJitter",
                                   "FlowDelayStd", "FlowJitterStd"},
    "DJ_minus6_all":              set(DJ_FEATURES),

    # Instability-based controls (Control 3 — top-6 by |Δd| between domains)
    "Top6_dShift_nonDJ":          set(TOP6_DSHIFT_NONDJ),
    "Top6_dShift_nonDJ_nonU4":    set(TOP6_DSHIFT_NONDJ_NONU4),

    # Random-removal controls (4)
    **RANDOM_REMOVALS,
}


# ============================================================================
# Helpers
# ============================================================================

def build_features(X, remove_set):
    keep = [m for m in METRICS_33 if m not in remove_set]
    return X[keep].values.astype(np.float32), keep


def build_catboost(best_params, seed):
    p = filter_catboost_params(best_params)
    return CatBoostClassifier(
        random_seed=seed, verbose=0, allow_writing_files=False,
        thread_count=N_JOBS_CATBOOST, **p,
    )


def evaluate(F_src, y_src, g_src, F_tgt, y_tgt, best_params, seed):
    tr, va, te = split_source_60_20_20(F_src, y_src, g_src, seed)

    sc = StandardScaler().fit(F_src[tr])
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
        "thr":     thr,
    }


def paired_test(exp_vals, base_vals):
    """Returns (delta_mean, delta_std, p, cohens_d_paired). NaN-safe."""
    d = np.asarray(exp_vals) - np.asarray(base_vals)
    n = len(d)
    if n < 2:
        return float(d.mean()), 0.0, float("nan"), float("nan")
    d_std = float(d.std(ddof=1))
    d_mean = float(d.mean())
    if d_std == 0.0:
        return d_mean, 0.0, float("nan"), float("nan")
    t_stat, p_val = stats.ttest_rel(exp_vals, base_vals)
    cohens_d = d_mean / d_std
    return d_mean, d_std, float(p_val), float(cohens_d)


# ============================================================================
# main
# ============================================================================

def main():
    out_dir = Path("./results/dj_ablation_breakdown")
    out_dir.mkdir(parents=True, exist_ok=True)
    per_seed_path = out_dir / "per_seed.csv"

    print("=" * 78, flush=True)
    print("DJ ABLATION BREAKDOWN: per-feature contributions to the +0.1365 gap",
          flush=True)
    print("=" * 78, flush=True)
    print(f"\nProtected from random removal (DJ + Universal-4): "
          f"{len(PROTECTED)} features", flush=True)
    print(f"Non-protected pool: {len(NON_PROTECTED_POOL)} features", flush=True)
    print(f"\nRandom-removal controls:", flush=True)
    for name, rem in RANDOM_REMOVALS.items():
        print(f"  {name}: {sorted(rem)}", flush=True)
    print(f"\nInstability-based controls (Control 3, top-6 by |Δd|):", flush=True)
    print(f"  Top6_dShift_nonDJ:        {sorted(TOP6_DSHIFT_NONDJ)}", flush=True)
    print(f"  Top6_dShift_nonDJ_nonU4:  {sorted(TOP6_DSHIFT_NONDJ_NONU4)}", flush=True)

    # ---- Load data ----
    print("\nLoading data...", flush=True)
    X_s, y_s, g_s = load_raw_data("../simulations/features_static", "static")
    X_m, y_m, _ = load_raw_data("../simulations/features_mobile", "mobile")
    y_s_arr = y_s.to_numpy()
    y_m_arr = y_m.to_numpy()

    # ---- Load CatBoost (static) params ----
    print("\nLoading CatBoost (static) hyperparameters...", flush=True)
    best_params = load_best_params("./results/hp_search_extended")
    print(f"  Static params: {filter_catboost_params(best_params['static'])}",
          flush=True)

    # ---- Run all configs * seeds ----
    seeds = list(range(42, 62))  # 20 seeds (was 10; matches augmentation experiment)
    rows = []
    t0 = time.time()
    total_fits = len(ABLATION_SETS) * len(seeds)

    print(f"\nRunning {len(ABLATION_SETS)} configs * {len(seeds)} seeds "
          f"= {total_fits} fits...\n", flush=True)

    fit_idx = 0
    for cfg_name, remove_set in ABLATION_SETS.items():
        F_s, kept = build_features(X_s, remove_set)
        F_m, _ = build_features(X_m, remove_set)
        n_kept = len(kept)
        removed_str = ", ".join(sorted(remove_set)) if remove_set else "(none)"

        for seed in seeds:
            fit_idx += 1
            t_fit = time.time()
            res = evaluate(F_s, y_s_arr, g_s, F_m, y_m_arr,
                           best_params=best_params["static"], seed=seed)
            rows.append({
                "config":    cfg_name,
                "n_kept":    n_kept,
                "removed":   removed_str,
                "seed":      seed,
                **res,
            })
            # Incremental CSV save after each fit
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

    # Paired deltas vs Standard_33
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
    print("RESULTS: S->M cross-domain accuracy by ablation set "
          f"(CatBoost, {len(seeds)} seeds, mean±std, paired Δ vs Standard_33)",
          flush=True)
    print("=" * 120, flush=True)
    order = [
        "Standard_33",
        # singletons
        "LOO_AvgE2EDelay", "LOO_AvgJitter",
        "LOO_AvgFlowDelay", "LOO_AvgFlowJit",
        "LOO_FlowDelayStd", "LOO_FlowJitterStd",
        # groups
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
    print("QUICK INTERPRETATION", flush=True)
    print("=" * 78, flush=True)

    def get_delta(cfg):
        r = deltas_df[deltas_df["config"] == cfg]
        if len(r) == 0:
            return float("nan")
        return float(r["delta_acc"].iloc[0])

    full = get_delta("DJ_minus6_all")
    flow_ext = get_delta("DJ_minus2_Flow_extremes")
    avg_only = get_delta("DJ_minus2_Avg_only")
    flow_only = get_delta("DJ_minus4_Flow_only")

    valid = (not np.isnan(full)) and (full > 0.01)

    def pct(val, total):
        if not valid or np.isnan(val):
            return ""
        return f"  ({100*val/total:+.0f}% of full gap)"

    print(f"\nFull DJ_minus6 gap (acc):              {full:+.4f}", flush=True)
    print(f"  Flow extremes (2 features):            {flow_ext:+.4f}{pct(flow_ext, full)}",
          flush=True)
    print(f"  Avg features  (2 features):            {avg_only:+.4f}{pct(avg_only, full)}",
          flush=True)
    print(f"  Flow-only     (4 features):            {flow_only:+.4f}{pct(flow_only, full)}",
          flush=True)

    # Instability-based controls (Control 3)
    instab_nondj = get_delta("Top6_dShift_nonDJ")
    instab_nondj_nonu4 = get_delta("Top6_dShift_nonDJ_nonU4")
    print(f"\nInstability-based controls (top-6 by |Δd|):", flush=True)
    print(f"  Top6_dShift_nonDJ      (includes U4):   {instab_nondj:+.4f}{pct(instab_nondj, full)}",
          flush=True)
    print(f"  Top6_dShift_nonDJ_nonU4 (primary):      {instab_nondj_nonu4:+.4f}"
          f"{pct(instab_nondj_nonu4, full)}",
          flush=True)
    print(f"  ^ If both << DJ_minus6_all, instability magnitude alone does NOT",
          flush=True)
    print(f"    explain the DJ effect (supports family-specific interpretation).",
          flush=True)

    # Random control statistics
    rand_deltas = [get_delta(f"Random_minus6_seed{rs}") for rs in RAND_SEEDS]
    rand_deltas = [d for d in rand_deltas if not np.isnan(d)]
    if rand_deltas:
        rand_mean = np.mean(rand_deltas)
        rand_std = np.std(rand_deltas, ddof=1) if len(rand_deltas) > 1 else 0.0
        rand_max_abs = max(abs(d) for d in rand_deltas)
        print(f"\nRandom 6-feature removal controls (mean ± std across "
              f"{len(rand_deltas)} controls):",
              flush=True)
        print(f"  Δacc = {rand_mean:+.4f} ± {rand_std:.4f}  "
              f"(max |Δ| = {rand_max_abs:.4f})", flush=True)
        print(f"  Individual deltas: " +
              "  ".join(f"{d:+.4f}" for d in rand_deltas),
              flush=True)
        print(f"  ^ Noise floor of feature-removal; the DJ-specific effects "
              "should exceed this.", flush=True)

    print(f"\nTotal elapsed: {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
