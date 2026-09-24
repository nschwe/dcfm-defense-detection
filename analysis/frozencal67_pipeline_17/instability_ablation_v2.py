#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
instability_ablation_v2.py
============================================================================

The single-vantage equivalent of the paper's delay/jitter ablation
(Sec. 6.7 / 7.2, Table 9). Plan and rationale: STATE 25.77.10.

WHY THIS SCRIPT EXISTS
----------------------
The submitted paper asked: is the cross-domain gain from dropping features
driven by removing metrics that suffer static<->mobile DISTRIBUTION SHIFT,
rather than by removing features per se? It answered that with the six
delay/jitter metrics (delta = +0.1307, three controls).

Those six are not observable from a single passive vantage. On the listener
arm they arrive zero-filled, so `Ablation_DJ_27` in feature_eng_ablation_v2
now removes six constant-zero columns and measures nothing (STATE 25.77.3).
The QUESTION is still valid, so it is asked again on the 17-observable space.

*** THE EQUIVALENCE CONDITION -- do not break it. ***
In the paper the ablated group was defined A PRIORI BY SEMANTIC CATEGORY
(delay/jitter), and Delta-d was the EXPLANATORY hypothesis tested against it.
The two were independent; that is what made the controls meaningful.

Defining the ablation group BY Delta-d and then testing whether Delta-d
explains the result is CIRCULAR and is NOT the equivalent analysis. So the
groups below are semantic families, fixed before looking at Delta-d, and
Delta-d enters only as the explanatory variable joined in at reporting time.

WHAT THE ANSWER IS EXPECTED TO LOOK LIKE
----------------------------------------
In the global space the unstable metrics were also the uninformative ones, so
removing them was a free gain. Here the most-shifted feature by a factor of
2.5 is AverageMprCount (Delta-d = -0.5505, separability 1.1043 -> 0.5538),
which is ALSO the feature that buys the +8.5-point jump at K=3. The trade-off
that did not exist in the global space is unavoidable here. That is the
finding this script is built to measure.

*** THE PROTECTION COLLISION IS THE RESULT. ***
The original protected the Universal-4 so the random controls would not
remove known-strong features. Here AverageMprCount is simultaneously the
most-shifted feature and a member of the protected K=3 set. Protecting it
makes the headline question unaskable; not protecting it breaks parity with
the original. Both variants are therefore run for the control-plane family,
and the difference between them is the section's central number.

Configurations (18)
-------------------
  baseline (1)          Full_17
  semantic families (5) minus_{Overhead,FlowDuration,PacketSize,Volume,ControlPlane}
  protection variant(1) minus_ControlPlane_protected  (K=3 members retained)
  leave-one-out (6)     each member of the two most-shifted non-selected
                        families, plus LOO_AverageMprCount (the collision)
  random controls (4)   Random3_seed{1001,2002,3003,4004}, size-matched to
                        minus_Overhead, drawn from the LIVE non-family,
                        non-protected pool
  dead-column control(1) minus_Dead -- MUST come out at exactly 0.0000

*** minus_Dead is a positive control on the design itself. ***
MidMessageRate and HnaMessageRate are constant zero in both configurations
(verified on the bundle, STATE 25.77.10). Removing them must change nothing.
If that config returns anything but 0.0000 for CatBoost, the harness is
wrong and no other row can be trusted. This control exists precisely because
Ablation_DJ_27 silently became a no-op and nobody noticed for months.

Pipeline parity
---------------
Every evaluation primitive is IMPORTED from feature_eng_ablation_v2 rather
than reimplemented -- same 60/20/20 group-aware split, same StandardScaler,
same tuned CatBoost, same source-validation threshold selection, same target
evaluation. Any change there applies here automatically.

Output (its own directory; no existing schema is touched)
  instability_per_seed.csv
  instability_summary.csv
  instability_paired.csv
  instability_vs_shift.csv   <- the hypothesis test: gain vs |Delta-d| removed

Usage
  python3 instability_ablation_v2.py \
      --static-root ./features_static --mobile-root ./features_mobile \
      --hp-results-dir <...>/hp_search_extended \
      --out-dir <arm>/results/instability_ablation_v2 \
      --cohens-d-csv <arm>/results/compute_cohens_d_v2/cohens_d_all_features.csv
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Pipeline parity: reuse, never reimplement (see header).
from feature_eng_ablation_v2 import (        # noqa: E402
    CLASSIFIERS,
    DIRECTIONS,
    RANDOM_STATE_OFFSET,
    build_catboost,          # noqa: F401  (used via evaluate_seed_config_clf)
    evaluate_seed_config_clf,
    load_best_params,
    load_raw_data,
    paired_analysis,
)

warnings.filterwarnings("ignore")


# ============================================================================
# The feature space
# ============================================================================

# The 12 metrics the listener has no analogue for; zero-filled by
# preprocess_data_enhanced (STATE 25.52 B1 / 25.77.3). Excluded here entirely.
PHANTOM_12 = [
    "AverageEndToEndDelay", "AverageJitter", "AvgFlowDelay", "AvgFlowJitter",
    "AvgFlowLossRate", "FlowCount", "FlowDelayStd", "FlowJitterStd",
    "FlowLossRateStd", "PacketDeliveryRatio", "PacketLossRatio",
    "RxTxPacketRatio",
]

# Semantic families. *** FIXED A PRIORI, BEFORE CONSULTING Delta-d. ***
# Changing these after seeing the results breaks the equivalence condition.
FAMILIES = {
    "ControlPlane": [
        "TcMessageRate", "MidMessageRate", "HnaMessageRate",
        "AverageAdvertisedLinksPerTCMessage", "AverageMprCount",
        "AverageHopCount",
    ],
    "Overhead": [
        "NormalizedRoutingLoad", "RoutingOverheadBytesRatio",
    ],
    "FlowDuration": ["AvgFlowDuration", "FlowDurationStd"],
    "PacketSize": ["AvgTxPacketSize"],
    "Volume": [
        "Throughput", "AvgFlowThroughput", "FlowThroughputStd",
        "AvgTxBytesPerFlow", "AvgTxPacketsPerFlow", "DataPacketRate",
    ],
}

# 17 distinct observables. The four canonical duplicates
# (AvgRxPacketSize, AvgRxBytesPerFlow, AvgRxPacketsPerFlow,
# RoutingOverheadBytesRatio) share an obs: expression with a sibling that stays
# and are excluded, so the space matches the one the paper reports.
METRICS_21 = [m for fam in FAMILIES.values() for m in fam]
assert len(METRICS_21) == 17, f"expected 17 distinct observables, got {len(METRICS_21)}"
assert len(set(METRICS_21)) == 17, "duplicate metric across families"

# Constant zero in BOTH configurations -- verified on the bundle 21/8/26.
# Re-verified at runtime; see assert_dead_columns().
DEAD = ["MidMessageRate", "HnaMessageRate"]

# The selected set (STATE 25.77.2, decision closed 21/8). "Protected" in the
# sense the original protected Universal-4.
SELECTED_K3 = [
    "AverageAdvertisedLinksPerTCMessage", "AverageMprCount", "TcMessageRate",
]

# The two most-shifted families that contain no selected feature. Named here
# rather than derived, so the LOO block is also a priori.
LOO_FAMILIES = ["Overhead", "FlowDuration"]

RANDOM_SEEDS = [1001, 2002, 3003, 4004]
RANDOM_K = len(FAMILIES["Overhead"])   # size-matched to the headline family

BASELINE = "Full_17"


def build_configs() -> "dict[str, list]":
    """config name -> list of features REMOVED from the 17. Order is stable."""
    cfg: "dict[str, list]" = {BASELINE: []}

    for fam, members in FAMILIES.items():
        cfg[f"minus_{fam}"] = list(members)

    # The protection variant exists only where a family meets the selected
    # set. Only ControlPlane does; for the others protected == unprotected,
    # so emitting them would be duplicate work presented as a comparison.
    for fam, members in FAMILIES.items():
        kept = [m for m in members if m not in SELECTED_K3]
        if len(kept) != len(members):
            cfg[f"minus_{fam}_protected"] = kept

    for fam in LOO_FAMILIES:
        for m in FAMILIES[fam]:
            cfg[f"LOO_{m}"] = [m]
    # the collision case, stated explicitly rather than folded into a family
    cfg["LOO_AverageMprCount"] = ["AverageMprCount"]

    # Random controls: drawn from LIVE features only, excluding the protected
    # set and every family used as a headline group, so the control is not
    # accidentally re-running one of the treatments.
    excluded = set(SELECTED_K3) | set(DEAD)
    for fam in LOO_FAMILIES:
        excluded |= set(FAMILIES[fam])
    pool = [m for m in METRICS_21 if m not in excluded]
    if len(pool) < RANDOM_K:
        raise SystemExit(f"random pool too small: {len(pool)} < {RANDOM_K}")
    for rs in RANDOM_SEEDS:
        rng = np.random.RandomState(rs)
        pick = sorted(rng.choice(pool, size=RANDOM_K, replace=False).tolist())
        cfg[f"Random{RANDOM_K}_seed{rs}"] = pick

    cfg["minus_Dead"] = list(DEAD)
    return cfg


def assert_dead_columns(X_s: pd.DataFrame, X_m: pd.DataFrame) -> None:
    """Re-verify the dead columns instead of trusting the note in the header.

    If a future bundle does carry MID/HNA traffic this must fail loudly: the
    random pool and the minus_Dead control both depend on it.
    """
    bad = []
    for col in DEAD:
        for name, X in (("static", X_s), ("mobile", X_m)):
            n = X[col].nunique()
            if n != 1:
                bad.append(f"{col}/{name}: nunique={n} (expected 1)")
    if bad:
        raise SystemExit(
            "DEAD column assumption violated -- this bundle carries MID/HNA "
            "traffic:\n  " + "\n  ".join(bad) +
            "\nFix DEAD, the random pool and minus_Dead before trusting any row."
        )
    print(f"  dead-column check: {', '.join(DEAD)} constant in both configs -- OK",
          flush=True)


# ============================================================================
# Evaluation
# ============================================================================

def features_for(cfg_removed: list, X: pd.DataFrame) -> np.ndarray:
    keep = [m for m in METRICS_21 if m not in set(cfg_removed)]
    if not keep:
        raise ValueError("configuration removes every feature")
    return X[keep].values.astype(np.float32)


def run_one_seed(X_s, y_s, g_s, X_m, y_m, g_m, configs, best_params, seed):
    """All configs x classifiers x directions for one seed."""
    row = {"seed": seed}
    feats_s = {c: features_for(r, X_s) for c, r in configs.items()}
    feats_m = {c: features_for(r, X_m) for c, r in configs.items()}
    y_s_a, y_m_a = y_s.values, y_m.values

    for cfg in configs:
        for clf in CLASSIFIERS:
            for src, _tgt, dlabel in DIRECTIONS:
                if src == "S":
                    F_src, y_src, g_src, F_tgt, y_tgt = (
                        feats_s[cfg], y_s_a, g_s, feats_m[cfg], y_m_a)
                    bp = best_params["static"]
                else:
                    F_src, y_src, g_src, F_tgt, y_tgt = (
                        feats_m[cfg], y_m_a, g_m, feats_s[cfg], y_s_a)
                    bp = best_params["mobile"]

                res = evaluate_seed_config_clf(
                    F_src, y_src, g_src, F_tgt, y_tgt, bp, clf, seed)
                for k, v in res.items():
                    row[f"{cfg}_{clf}_{dlabel}_{k}"] = v
    return row


# ============================================================================
# Reporting
# ============================================================================

def shift_table(configs, cohens_d_csv: str) -> pd.DataFrame:
    """Join |Delta-d| of the REMOVED set onto each config.

    This is where Delta-d enters -- at reporting time, as the explanatory
    variable. It is never used to choose what to remove (see header).
    """
    d = pd.read_csv(cohens_d_csv).set_index("feature")["delta"]
    rows = []
    for cfg, removed in configs.items():
        vals = [abs(float(d[m])) for m in removed if m in d.index]
        rows.append({
            "config": cfg,
            "n_removed": len(removed),
            "removed": ";".join(removed),
            "mean_abs_delta_d": float(np.mean(vals)) if vals else 0.0,
            "max_abs_delta_d": float(np.max(vals)) if vals else 0.0,
            "removes_selected": any(m in SELECTED_K3 for m in removed),
        })
    return pd.DataFrame(rows)


def build_summary(df: pd.DataFrame, configs) -> pd.DataFrame:
    rows = []
    for cfg in configs:
        for clf in CLASSIFIERS:
            for _s, _t, dlabel in DIRECTIONS:
                col = f"{cfg}_{clf}_{dlabel}_tgt_acc"
                base = f"{BASELINE}_{clf}_{dlabel}_tgt_acc"
                if col not in df.columns:
                    continue
                st = paired_analysis(df[base].values, df[col].values)
                rows.append({
                    "config": cfg, "classifier": clf, "direction": dlabel,
                    "tgt_acc_mean": df[col].mean(),
                    "tgt_acc_std": df[col].std(ddof=1),
                    # paired on per-seed deltas, not a difference of means
                    "delta_vs_baseline": st["delta_mean"],
                    "delta_ci_lo": st["delta_ci_lo"],
                    "delta_ci_hi": st["delta_ci_hi"],
                    "p_value": st["paired_p_twosided"],
                    "cohens_d": st["cohens_d_paired"],
                    "n_seeds": st["n_seeds"],
                })
    return pd.DataFrame(rows)


def print_report(summary: pd.DataFrame, shift: pd.DataFrame, df: pd.DataFrame):
    print("\n" + "=" * 78, flush=True)
    print("DESIGN CONTROL — minus_Dead must be exactly 0.0000 for CatBoost",
          flush=True)
    print("=" * 78, flush=True)
    ok = True
    for _s, _t, dlabel in DIRECTIONS:
        a = df.get(f"{BASELINE}_CatBoost_{dlabel}_tgt_acc")
        b = df.get(f"minus_Dead_CatBoost_{dlabel}_tgt_acc")
        if a is None or b is None:
            continue
        same = bool((a == b).all())
        ok &= same
        print(f"  CatBoost/{dlabel}: identical to {BASELINE} in all "
              f"{len(df)} seeds: {same}", flush=True)
    if not ok:
        print("\n  *** CONTROL FAILED. Removing two constant-zero columns changed",
              flush=True)
        print("  the result, so the harness is wrong and NO row below can be",
              flush=True)
        print("  trusted. Do not report anything from this run.", flush=True)
    print("=" * 78, flush=True)

    m = summary.merge(shift, on="config", how="left")
    for clf in CLASSIFIERS:
        for _s, _t, dlabel in DIRECTIONS:
            sub = (m[(m.classifier == clf) & (m.direction == dlabel)]
                   .sort_values("mean_abs_delta_d", ascending=False))
            print(f"\n  Classifier: {clf}    Direction: {dlabel}", flush=True)
            print(f"  {'configuration':<30}{'tgt acc':>10}{'delta':>10}"
                  f"{'p':>11}{'mean|dd|':>10}  sel", flush=True)
            print("  " + "-" * 74, flush=True)
            for _, r in sub.iterrows():
                p = "" if pd.isna(r["p_value"]) else f"{r['p_value']:.2e}"
                sel = "*" if r.get("removes_selected") else ""
                print(f"  {r['config']:<30}{r['tgt_acc_mean']:>10.4f}"
                      f"{r['delta_vs_baseline']:>+10.4f}{p:>11}"
                      f"{r['mean_abs_delta_d']:>10.4f}  {sel}", flush=True)
    print("\n  * = the configuration removes a member of the selected K=3 set.",
          flush=True)
    print("  The hypothesis is tested by whether delta tracks mean|dd| ACROSS",
          flush=True)
    print("  rows, not by any single row.", flush=True)


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--static-root", required=True)
    ap.add_argument("--mobile-root", required=True)
    ap.add_argument("--hp-results-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cohens-d-csv", required=True,
                    help="compute_cohens_d_v2/cohens_d_all_features.csv")
    ap.add_argument("--n-seeds", type=int, default=20)
    ap.add_argument("--base-seed", type=int, default=RANDOM_STATE_OFFSET)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    configs = build_configs()
    print("=" * 78, flush=True)
    print("INSTABILITY ABLATION v2 — single-vantage equivalent of Sec. 6.7/7.2",
          flush=True)
    print("=" * 78, flush=True)
    print(f"  seeds       : {args.n_seeds} "
          f"({args.base_seed}..{args.base_seed + args.n_seeds - 1})", flush=True)
    print(f"  configs     : {len(configs)}", flush=True)
    print(f"  total fits  : {args.n_seeds * len(configs) * len(CLASSIFIERS) * len(DIRECTIONS)}",
          flush=True)
    for c, r in configs.items():
        print(f"    {c:<32} removes {len(r):>2}: {';'.join(r) if r else '-'}",
              flush=True)

    print("\n[1/4] Loading hyperparameters and data...", flush=True)
    best_params = load_best_params(args.hp_results_dir)
    X_s, y_s, g_s = load_raw_data(args.static_root, "static")
    X_m, y_m, g_m = load_raw_data(args.mobile_root, "mobile")

    missing = [m for m in METRICS_21 if m not in X_s.columns or m not in X_m.columns]
    if missing:
        raise SystemExit(f"metrics missing from the bundle: {missing}")
    assert_dead_columns(X_s, X_m)

    print(f"\n[2/4] Running {args.n_seeds} seeds...", flush=True)
    t0 = time.time()
    rows = []
    for i, seed in enumerate(range(args.base_seed, args.base_seed + args.n_seeds)):
        ts = time.time()
        rows.append(run_one_seed(X_s, y_s, g_s, X_m, y_m, g_m,
                                 configs, best_params, seed))
        el = (time.time() - t0) / 60
        print(f"  seed {seed} ({i+1}/{args.n_seeds}) done in "
              f"{time.time()-ts:.1f}s  elapsed={el:.1f}min  "
              f"eta={el/(i+1)*(args.n_seeds-i-1):.1f}min", flush=True)
    df = pd.DataFrame(rows)

    print("\n[3/4] Writing CSVs...", flush=True)
    shift = shift_table(configs, args.cohens_d_csv)
    summary = build_summary(df, configs)
    df.to_csv(out / "instability_per_seed.csv", index=False)
    summary.to_csv(out / "instability_summary.csv", index=False)
    shift.to_csv(out / "instability_vs_shift.csv", index=False)
    summary.merge(shift, on="config", how="left").to_csv(
        out / "instability_paired.csv", index=False)
    for f in ("instability_per_seed.csv", "instability_summary.csv",
              "instability_vs_shift.csv", "instability_paired.csv"):
        print(f"  Wrote: {out / f}", flush=True)

    print("\n[4/4] Report...", flush=True)
    print_report(summary, shift, df)
    print(f"\nTotal elapsed: {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
