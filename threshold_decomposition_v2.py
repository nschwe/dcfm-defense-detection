#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
threshold_decomposition_v2.py
============================================================================
Decomposes the cross-domain accuracy gap (UNIVERSAL-4 vs Standard-33) into a
THRESHOLD/calibration component and a RANKING/feature component, reusing the
EXACT pipeline of k_sweep_universal4_v2.py (CatBoost tuned HP, group-aware
60/20/20 source split, StandardScaler on source-train, accuracy-max threshold
scan). No new simulations.

Motivation
----------
At K=33, S->M: accuracy=0.6718 but AUC=0.8348.  At K=4: accuracy=0.8639,
AUC=0.8974.  Accuracy gap ~0.19 but AUC gap ~0.06.  AUC is threshold-free, so
most of the "collapse" is plausibly a failure to TRANSFER the decision
threshold across domains, not loss of feature separability.

Method (per direction in {S->M, M->S}, per seed 42..61)
-------------------------------------------------------
Train CatBoost on source-train, fix the source-validation threshold t_src,
and compute the target predicted probabilities ONCE per (K, seed).  Then vary
ONLY the threshold across three regimes, all scored on the SAME group-aware
target-test fold (40% of target runs), so the decomposition identity holds
exactly per seed:

  source-thr   : apply t_src to target-test                 (operational)
  oracle-calib : tune t on disjoint target-calibration pool, eval target-test
  oracle-intest: tune t on target-test, eval target-test    (upper bound only)

  Delta_total(s)     = Acc(K=4, source-thr) - Acc(K=33, source-thr)
  Delta_ranking(s)   = Acc(K=4, oracle)     - Acc(K=33, oracle)
  Delta_threshold(s) = Delta_total(s) - Delta_ranking(s)

Few-shot recalibration: sample n in {25,50,100,250,500} RUNS (groups) from the
calibration pool, tune only the threshold (isotonic arm for n>=250), eval on
the same target-test fold.  -> accuracy(n) curve.

Mechanism check: |t*_source - t*_target| and W1 on the logit of predicted
probability between source-test and target-test, per K.

Outputs (results/threshold_decomposition/)
------------------------------------------
  decomposition_per_seed.csv
  decomposition_summary.csv        (2 x 2 x 3 table)
  decomposition_components.csv     (Delta components + paired CIs + boot CIs)
  fewshot_curve.csv
  mechanism_stats.csv
  hist_data.npz                    (proba arrays for the 2x2 histogram grid)

Usage
-----
  python3 threshold_decomposition_v2.py \
      --static-root ../simulations/features_static \
      --mobile-root ../simulations/features_mobile \
      --hp-results-dir ./results/hp_search_extended \
      --out-dir ./results/threshold_decomposition
  # quick check:
  python3 threshold_decomposition_v2.py ... --n-seeds 2 --fewshot-R 5
============================================================================
"""
import argparse
import gc
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy import stats

import pandas.core.series
pandas.core.series.dtype = np.dtype  # parity with k_sweep_universal4_v2

from k_sweep_universal4_v2 import (
    UNIVERSAL_4, load_best_params, build_catboost, load_raw_data,
    split_source_60_20_20, select_threshold,
)

warnings.filterwarnings("ignore")

N_SEEDS_DEFAULT = 20
RANDOM_STATE = 42
TARGET_TEST_SIZE = 0.40
N_RUNS_LIST = [25, 50, 100, 250, 500]
FEWSHOT_R_DEFAULT = 20
ISO_MIN_N = 250
KS = [33, 4]
EPS = 1e-6  # for logit clipping
N_BOOT = 2000


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def load_headline_feature_sets(k_sweep_csv, all_features):
    """Load the EXACT feature sets the headline used, so the decomposition runs
    on identical scores.  K=4 -> UNIVERSAL-4; K=33 -> the 33-feature subset
    recorded in k_sweep_results.csv (the pivoted data has 34 columns; k_sweep's
    importance ranking drops the metadata column `_measurement_duration`, so
    K=33 != all 34 columns).  Feature order is irrelevant to CatBoost.
    """
    df = pd.read_csv(k_sweep_csv)
    row = df[(df.K == 33) & (df.direction == "S_to_M")]
    if row.empty:
        raise ValueError(f"No K=33 / S_to_M row in {k_sweep_csv}")
    feats33 = row.iloc[0]["features"].split(";")
    assert len(feats33) == 33, f"expected 33 features, got {len(feats33)}"
    missing = [f for f in feats33 if f not in all_features]
    if missing:
        raise ValueError(f"K=33 features absent from data columns: {missing}")
    missing4 = [f for f in UNIVERSAL_4 if f not in all_features]
    if missing4:
        raise ValueError(f"UNIVERSAL-4 features absent from data: {missing4}")
    return {33: feats33, 4: list(UNIVERSAL_4)}


def split_target_calib_test(g_tgt, seed, test_size=TARGET_TEST_SIZE):
    """Group-aware split of the TARGET domain into (calib_pool, test_fold).

    Mirrors split_source_60_20_20's use of GroupShuffleSplit. The 'test' part
    of GroupShuffleSplit is `test_size` of the runs -> our target-test fold.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    calib_idx, test_idx = next(gss.split(np.zeros(len(g_tgt)), groups=g_tgt))
    assert set(np.asarray(g_tgt)[calib_idx]).isdisjoint(
        set(np.asarray(g_tgt)[test_idx])), "group leakage calib/test"
    return calib_idx, test_idx


def fit_source_model_and_scores(X_src, y_src, g_src, X_tgt, features,
                                best_params, seed):
    """Refactor of evaluate_seed's first half: train on source-train, fix the
    source-validation threshold, return target probabilities (full target) and
    source-test probabilities. Identical scaler/HP/feature handling to the
    headline pipeline (raw X_tgt[features], scaler fit on source-train only).
    """
    tr, va, te = split_source_60_20_20(X_src, y_src, g_src, seed)
    Xs = X_src[features]
    Xt = X_tgt[features]

    sc = StandardScaler().fit(Xs.iloc[tr])
    clf = build_catboost(best_params, seed=seed, n_jobs=16)
    clf.fit(sc.transform(Xs.iloc[tr]), y_src.iloc[tr])

    p_va = clf.predict_proba(sc.transform(Xs.iloc[va]))[:, 1]
    t_src = select_threshold(y_src.iloc[va], p_va)
    p_te = clf.predict_proba(sc.transform(Xs.iloc[te]))[:, 1]
    p_tgt = clf.predict_proba(sc.transform(Xt))[:, 1]

    y_te = y_src.iloc[te].to_numpy()
    del clf, sc
    return {
        "t_src": float(t_src),
        "p_tgt": p_tgt,                       # full target, original order
        "p_te": p_te, "y_te": y_te,           # source-test (for mechanism)
        "acc_in": float(accuracy_score(y_te, (p_te >= t_src).astype(int))),
        "auc_in": float(roc_auc_score(y_te, p_te)),
    }


def fewshot_curve_for(calib_idx, test_idx, p_tgt, y_tgt, g_tgt,
                      n_runs_list, R, iso_min_n, rng_base):
    """accuracy(target-test) vs n labeled target RUNS. Threshold-only (primary)
    and isotonic (n>=iso_min_n) arms. Returns {n: {...}}."""
    g_tgt = np.asarray(g_tgt)
    calib_runs = np.unique(g_tgt[calib_idx])
    p_test = p_tgt[test_idx]
    y_test = y_tgt[test_idx]
    out = {}
    for n in n_runs_list:
        if n > len(calib_runs):
            continue
        accs, accs_iso = [], []
        for r in range(R):
            rng = np.random.default_rng((rng_base * 1_000_003 + n * 101 + r) % (2**32))
            sel = rng.choice(calib_runs, size=n, replace=False)
            mask = np.isin(g_tgt[calib_idx], sel)
            idx = calib_idx[mask]
            t = select_threshold(y_tgt[idx], p_tgt[idx])
            accs.append(accuracy_score(y_test, (p_test >= t).astype(int)))
            if n >= iso_min_n:
                iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0,
                                         y_max=1.0).fit(p_tgt[idx], y_tgt[idx])
                p_cal = iso.predict(p_test)
                accs_iso.append(accuracy_score(y_test, (p_cal >= 0.5).astype(int)))
        out[n] = {
            "n_windows": n * 4,  # ~4 windows/run
            "acc_thresh_mean": float(np.mean(accs)),
            "acc_thresh_std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
            "acc_iso_mean": float(np.mean(accs_iso)) if accs_iso else np.nan,
            "acc_iso_std": (float(np.std(accs_iso, ddof=1))
                            if len(accs_iso) > 1 else np.nan),
        }
    return out


def paired_ci(values, alpha=0.05):
    """Mean + two-sided t CI + (paired) Cohen's d for an array of per-seed
    paired differences (or accuracies). Matches dj_ablation_breakdown style."""
    v = np.asarray(values, dtype=np.float64)
    n = len(v)
    m = float(v.mean())
    sd = float(v.std(ddof=1)) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n > 1 else 0.0
    tcrit = stats.t.ppf(1 - alpha / 2, df=n - 1) if n > 1 else 0.0
    return {
        "mean": m, "std": sd,
        "ci_lo": m - tcrit * se, "ci_hi": m + tcrit * se,
        "cohens_d": (m / sd) if sd > 0 else np.nan,
    }


def paired_ttest_vs_zero(values):
    v = np.asarray(values, dtype=np.float64)
    if len(v) < 2 or v.std(ddof=1) == 0:
        return np.nan
    t, p = stats.ttest_1samp(v, 0.0)
    return float(p)


def averaged_cluster_boot_ci(per_seed_arrays, fn, n_boot=N_BOOT, rng_seed=12345):
    """Robustness CI via run-level cluster bootstrap, averaged across seeds.

    per_seed_arrays: list over seeds, each a dict with the arrays fn() needs,
                     plus 'run_codes' and 'n_runs' for that seed's test fold.
    fn(seed_dict, flat_idx) -> scalar component value on resampled test windows.
    For each bootstrap replicate b: for every seed independently resample its
    test runs, evaluate fn, average across seeds -> one replicate value.
    Returns (lo, hi) percentile CI. Captures both seed + test-sampling variance.
    """
    rng = np.random.default_rng(rng_seed)
    S = len(per_seed_arrays)
    # Pre-build run->window index lists per seed
    run_index_lists = []
    for d in per_seed_arrays:
        rc, nr = d["run_codes"], d["n_runs"]
        run_index_lists.append([np.where(rc == r)[0] for r in range(nr)])
    reps = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        vals = np.empty(S, dtype=np.float64)
        for si, d in enumerate(per_seed_arrays):
            nr = d["n_runs"]
            sampled = rng.integers(0, nr, size=nr)
            flat = np.concatenate([run_index_lists[si][s] for s in sampled])
            vals[si] = fn(d, flat)
        reps[b] = vals.mean()
    return float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))


# ----------------------------------------------------------------------
# Per-direction driver
# ----------------------------------------------------------------------
def run_direction(direction, X_src, y_src, g_src, X_tgt, y_tgt, g_tgt,
                  best_params_src, feats, n_seeds, fewshot_R, out_dir):
    """direction: 'S_to_M' or 'M_to_S'. src/tgt already assigned accordingly.
    feats: {33: [...], 4: [...]} headline feature sets (shared across directions).
    """
    print(f"\n{'='*70}\nDirection: {direction}\n{'='*70}")
    y_tgt_arr = np.asarray(y_tgt)
    g_tgt_arr = np.asarray(g_tgt)
    y_tgt_full = y_tgt_arr

    per_seed_rows = []
    fewshot_accum = {K: {n: {"thr": [], "iso": []} for n in N_RUNS_LIST} for K in KS}
    fewshot_refs = {K: {"src": [], "orc": []} for K in KS}
    mech_accum = {K: {"abs_t_shift": [], "w1_logit": [],
                      "t_src": [], "t_tgt": [],
                      "mean_p_src": [], "mean_p_tgt": []} for K in KS}
    hist_store = {}                       # (K) -> dict of arrays, seed 42 only
    boot_seed_data = {K: [] for K in KS}  # for averaged cluster bootstrap

    seeds = list(range(RANDOM_STATE, RANDOM_STATE + n_seeds))
    for seed in seeds:
        t0 = time.time()
        # Target split is identical across K for this seed (depends on g_tgt, seed)
        calib_idx, test_idx = split_target_calib_test(g_tgt_arr, seed)
        y_test = y_tgt_arr[test_idx]

        row = {"direction": direction, "seed": seed}
        acc_src, acc_orc, acc_orc_intest, acc_src_full = {}, {}, {}, {}
        t_src_K, t_orc_K = {}, {}

        for K in KS:
            sc = fit_source_model_and_scores(
                X_src, y_src, g_src, X_tgt, feats[K], best_params_src, seed)
            p_tgt = sc["p_tgt"]
            t_src = sc["t_src"]

            p_test = p_tgt[test_idx]
            t_orc = select_threshold(y_tgt_arr[calib_idx], p_tgt[calib_idx])
            t_intest = select_threshold(y_test, p_test)

            acc_src[K] = accuracy_score(y_test, (p_test >= t_src).astype(int))
            acc_orc[K] = accuracy_score(y_test, (p_test >= t_orc).astype(int))
            acc_orc_intest[K] = accuracy_score(y_test, (p_test >= t_intest).astype(int))
            acc_src_full[K] = accuracy_score(y_tgt_full, (p_tgt >= t_src).astype(int))
            t_src_K[K], t_orc_K[K] = t_src, t_orc

            # few-shot (per seed)
            fs = fewshot_curve_for(calib_idx, test_idx, p_tgt, y_tgt_arr,
                                   g_tgt_arr, N_RUNS_LIST, fewshot_R, ISO_MIN_N,
                                   rng_base=seed)
            for n, d in fs.items():
                fewshot_accum[K][n]["thr"].append(d["acc_thresh_mean"])
                if not np.isnan(d["acc_iso_mean"]):
                    fewshot_accum[K][n]["iso"].append(d["acc_iso_mean"])
            fewshot_refs[K]["src"].append(acc_src[K])
            fewshot_refs[K]["orc"].append(acc_orc[K])

            # mechanism (source-test vs target-test score distributions)
            p_src_test = sc["p_te"]
            mech_accum[K]["t_src"].append(t_src)
            mech_accum[K]["t_tgt"].append(t_intest)
            mech_accum[K]["abs_t_shift"].append(abs(t_src - t_intest))
            mech_accum[K]["w1_logit"].append(
                stats.wasserstein_distance(logit(p_src_test), logit(p_test)))
            mech_accum[K]["mean_p_src"].append(float(p_src_test.mean()))
            mech_accum[K]["mean_p_tgt"].append(float(p_test.mean()))

            # histogram data for representative seed
            if seed == RANDOM_STATE:
                hist_store[f"{direction}|K{K}|src_test"] = p_src_test.astype(np.float32)
                hist_store[f"{direction}|K{K}|tgt_test"] = p_test.astype(np.float32)
                hist_store[f"{direction}|K{K}|t_src"] = np.float32(t_src)
                hist_store[f"{direction}|K{K}|t_tgt"] = np.float32(t_intest)

            # bootstrap bookkeeping: store test-fold preds under src & oracle thr
            uniq, run_codes = np.unique(g_tgt_arr[test_idx], return_inverse=True)
            boot_seed_data[K].append({
                "run_codes": run_codes, "n_runs": len(uniq),
                "y_test": y_test,
                "pred_src": (p_test >= t_src).astype(int),
                "pred_orc": (p_test >= t_orc).astype(int),
            })

            del sc, p_tgt
            gc.collect()

        d_total = acc_src[4] - acc_src[33]
        d_ranking = acc_orc[4] - acc_orc[33]
        d_threshold = d_total - d_ranking
        assert abs(d_total - (d_ranking + d_threshold)) < 1e-12, "identity broke"

        row.update({
            "acc_src_K33": acc_src[33], "acc_src_K4": acc_src[4],
            "acc_orc_K33": acc_orc[33], "acc_orc_K4": acc_orc[4],
            "acc_orc_intest_K33": acc_orc_intest[33],
            "acc_orc_intest_K4": acc_orc_intest[4],
            "delta_total": d_total, "delta_ranking": d_ranking,
            "delta_threshold": d_threshold,
            "t_src_K33": t_src_K[33], "t_src_K4": t_src_K[4],
            "t_orc_K33": t_orc_K[33], "t_orc_K4": t_orc_K[4],
            "acc_src_fulltarget_K33": acc_src_full[33],
            "acc_src_fulltarget_K4": acc_src_full[4],
        })
        per_seed_rows.append(row)

        # ordering sanity (informational): in-test >= oracle-calib >= source
        viol = []
        for K in KS:
            if acc_orc_intest[K] + 5e-3 < acc_orc[K]:
                viol.append(f"K{K}:intest<calib")
            if acc_orc[K] + 5e-3 < acc_src[K]:
                viol.append(f"K{K}:calib<src")
        flag = f"  [order-flag: {','.join(viol)}]" if viol else ""
        print(f"  seed {seed}: d_total={d_total:+.4f} d_rank={d_ranking:+.4f} "
              f"d_thr={d_threshold:+.4f} | acc_src(K33/K4)={acc_src[33]:.4f}/"
              f"{acc_src[4]:.4f} ({time.time()-t0:.0f}s){flag}")

        # incremental save
        pd.DataFrame(per_seed_rows).to_csv(
            Path(out_dir) / "decomposition_per_seed.csv", index=False)

    return {
        "per_seed_rows": per_seed_rows,
        "fewshot_accum": fewshot_accum,
        "fewshot_refs": fewshot_refs,
        "mech_accum": mech_accum,
        "hist_store": hist_store,
        "boot_seed_data": boot_seed_data,
    }


# ----------------------------------------------------------------------
# Aggregation / output
# ----------------------------------------------------------------------
def build_summary_and_components(direction, res, out_rows_summary,
                                 out_rows_components):
    df = pd.DataFrame(res["per_seed_rows"])
    # 2x2x3 summary table
    regime_cols = {
        "source_thr": ("acc_src_K33", "acc_src_K4"),
        "oracle_calib": ("acc_orc_K33", "acc_orc_K4"),
        "oracle_intest": ("acc_orc_intest_K33", "acc_orc_intest_K4"),
    }
    for regime, (c33, c4) in regime_cols.items():
        for K, col in ((33, c33), (4, c4)):
            ci = paired_ci(df[col].to_numpy())
            out_rows_summary.append({
                "direction": direction, "K": K, "regime": regime,
                "acc_mean": ci["mean"], "acc_std": ci["std"],
                "ci_lo": ci["ci_lo"], "ci_hi": ci["ci_hi"],
                "n_seeds": len(df),
            })

    # Delta components with paired t CI + paired cluster-bootstrap robustness CI
    comp_cols = {"delta_total": "delta_total",
                 "delta_ranking": "delta_ranking",
                 "delta_threshold": "delta_threshold"}
    bsd = res["boot_seed_data"]

    def boot_fn_total(seed_d_pair, flat):
        d33, d4 = seed_d_pair
        a4 = (d4["pred_src"][flat] == d4["y_test"][flat]).mean()
        a33 = (d33["pred_src"][flat] == d33["y_test"][flat]).mean()
        return a4 - a33

    def boot_fn_ranking(seed_d_pair, flat):
        d33, d4 = seed_d_pair
        a4 = (d4["pred_orc"][flat] == d4["y_test"][flat]).mean()
        a33 = (d33["pred_orc"][flat] == d33["y_test"][flat]).mean()
        return a4 - a33

    # Build per-seed paired structures sharing one run resample (same test fold
    # across K, so run_codes/n_runs taken from K=33 entry).
    paired_seed_data = []
    for si in range(len(bsd[33])):
        d33, d4 = bsd[33][si], bsd[4][si]
        paired_seed_data.append({
            "run_codes": d33["run_codes"], "n_runs": d33["n_runs"],
            "_pair": (d33, d4),
        })

    def mk_fn(component):
        if component == "delta_total":
            inner = boot_fn_total
        elif component == "delta_ranking":
            inner = boot_fn_ranking
        else:
            inner = None
        def f(d, flat):
            if inner is not None:
                return inner(d["_pair"], flat)
            return boot_fn_total(d["_pair"], flat) - boot_fn_ranking(d["_pair"], flat)
        return f

    for comp, col in comp_cols.items():
        vals = df[col].to_numpy()
        ci = paired_ci(vals)
        p = paired_ttest_vs_zero(vals)
        blo, bhi = averaged_cluster_boot_ci(paired_seed_data, mk_fn(comp))
        out_rows_components.append({
            "direction": direction, "component": comp,
            "mean": ci["mean"], "ci_lo": ci["ci_lo"], "ci_hi": ci["ci_hi"],
            "p_value": p, "cohens_d": ci["cohens_d"],
            "boot_ci_lo": blo, "boot_ci_hi": bhi,
        })


def build_fewshot_rows(direction, res, out_rows):
    fa, fr = res["fewshot_accum"], res["fewshot_refs"]
    for K in KS:
        src_ref = float(np.mean(fr[K]["src"]))
        orc_ref = float(np.mean(fr[K]["orc"]))
        for n in N_RUNS_LIST:
            thr = fa[K][n]["thr"]
            iso = fa[K][n]["iso"]
            if not thr:
                continue
            out_rows.append({
                "direction": direction, "K": K, "n_runs": n,
                "n_windows": n * 4,
                "acc_thresh_mean": float(np.mean(thr)),
                "acc_thresh_std": float(np.std(thr, ddof=1)) if len(thr) > 1 else 0.0,
                "acc_iso_mean": float(np.mean(iso)) if iso else np.nan,
                "acc_iso_std": float(np.std(iso, ddof=1)) if len(iso) > 1 else np.nan,
                "acc_source_thr_ref": src_ref,
                "acc_oracle_ref": orc_ref,
            })


def build_mechanism_rows(direction, res, out_rows):
    ma = res["mech_accum"]
    for K in KS:
        out_rows.append({
            "direction": direction, "K": K,
            "t_star_source": float(np.mean(ma[K]["t_src"])),
            "t_star_target": float(np.mean(ma[K]["t_tgt"])),
            "abs_t_shift": float(np.mean(ma[K]["abs_t_shift"])),
            "abs_t_shift_std": float(np.std(ma[K]["abs_t_shift"], ddof=1)),
            "w1_logit": float(np.mean(ma[K]["w1_logit"])),
            "w1_logit_std": float(np.std(ma[K]["w1_logit"], ddof=1)),
            "mean_p_source_test": float(np.mean(ma[K]["mean_p_src"])),
            "mean_p_target_test": float(np.mean(ma[K]["mean_p_tgt"])),
        })


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--static-root", required=True)
    ap.add_argument("--mobile-root", required=True)
    ap.add_argument("--hp-results-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-seeds", type=int, default=N_SEEDS_DEFAULT)
    ap.add_argument("--fewshot-R", type=int, default=FEWSHOT_R_DEFAULT)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    t_start = time.time()

    print("=" * 70)
    print("Threshold-vs-Ranking decomposition of the cross-domain gap")
    print("=" * 70)
    print(f"  seeds:     {args.n_seeds} (42..{42 + args.n_seeds - 1})")
    print(f"  few-shot:  R={args.fewshot_R}, n_runs={N_RUNS_LIST}, iso>= {ISO_MIN_N}")
    print(f"  target-test fold: {TARGET_TEST_SIZE:.0%} (group-aware)")
    print(f"  out:       {args.out_dir}")

    best_params = load_best_params(args.hp_results_dir)
    X_static, y_static, g_static = load_raw_data(args.static_root, "static")
    X_mobile, y_mobile, g_mobile = load_raw_data(args.mobile_root, "mobile")
    if list(X_mobile.columns) != list(X_static.columns):
        raise ValueError("Static and mobile feature columns differ!")
    all_features = list(X_static.columns)
    k_sweep_csv = "./results/k_sweep_universal4/k_sweep_results.csv"
    feats = load_headline_feature_sets(k_sweep_csv, all_features)
    print(f"  total columns: {len(all_features)} | "
          f"K=33 set: {len(feats[33])} | K=4 set: {len(feats[4])}")

    directions = [
        ("S_to_M", X_static, y_static, g_static, X_mobile, y_mobile, g_mobile,
         best_params["static"]),
        ("M_to_S", X_mobile, y_mobile, g_mobile, X_static, y_static, g_static,
         best_params["mobile"]),
    ]

    summary_rows, component_rows, fewshot_rows, mech_rows = [], [], [], []
    hist_all = {}

    for (dname, Xs, ys, gs, Xt, yt, gt, bp) in directions:
        res = run_direction(dname, Xs, ys, gs, Xt, yt, gt, bp, feats,
                            args.n_seeds, args.fewshot_R, args.out_dir)
        build_summary_and_components(dname, res, summary_rows, component_rows)
        build_fewshot_rows(dname, res, fewshot_rows)
        build_mechanism_rows(dname, res, mech_rows)
        hist_all.update(res["hist_store"])

        # incremental writes after each direction
        pd.DataFrame(summary_rows).to_csv(
            Path(args.out_dir) / "decomposition_summary.csv", index=False)
        pd.DataFrame(component_rows).to_csv(
            Path(args.out_dir) / "decomposition_components.csv", index=False)
        pd.DataFrame(fewshot_rows).to_csv(
            Path(args.out_dir) / "fewshot_curve.csv", index=False)
        pd.DataFrame(mech_rows).to_csv(
            Path(args.out_dir) / "mechanism_stats.csv", index=False)
        np.savez_compressed(Path(args.out_dir) / "hist_data.npz", **hist_all)

    # ---- console summary + route verdict ----
    comp_df = pd.DataFrame(component_rows)
    print(f"\n{'='*70}\nDECOMPOSITION SUMMARY\n{'='*70}")
    for dname in ("S_to_M", "M_to_S"):
        sub = comp_df[comp_df.direction == dname].set_index("component")
        dt = sub.loc["delta_total"]
        dr = sub.loc["delta_ranking"]
        dh = sub.loc["delta_threshold"]
        print(f"\n  {dname}:")
        print(f"    Delta_total     = {dt['mean']:+.4f}  "
              f"[{dt['ci_lo']:+.4f}, {dt['ci_hi']:+.4f}]")
        print(f"    Delta_ranking   = {dr['mean']:+.4f}  "
              f"[{dr['ci_lo']:+.4f}, {dr['ci_hi']:+.4f}]  (feature/AUC part)")
        print(f"    Delta_threshold = {dh['mean']:+.4f}  "
              f"[{dh['ci_lo']:+.4f}, {dh['ci_hi']:+.4f}]  (calibration artifact)")
        if abs(dt["mean"]) > 1e-9:
            frac_thr = dh["mean"] / dt["mean"]
            verdict = ("ROUTE A (gap is mostly threshold)" if frac_thr >= 0.5
                       else "ROUTE B (feature story survives)")
            print(f"    threshold share of gap = {frac_thr:.0%}  ->  {verdict}")

    mech_df = pd.DataFrame(mech_rows)
    print(f"\n  Threshold-shift / score-drift (mechanism):")
    for _, r in mech_df.iterrows():
        print(f"    {r['direction']} K={int(r['K']):2d}: "
              f"|t*_src - t*_tgt|={r['abs_t_shift']:.3f}  "
              f"W1(logit)={r['w1_logit']:.3f}")

    print(f"\n  Total elapsed: {(time.time()-t_start)/60:.1f} min")
    print(f"  Outputs written to: {args.out_dir}")


if __name__ == "__main__":
    main()
