#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""seed_bootstrap_ci.py -- R#3.14: replace every remaining parametric test with a
bootstrap interval computed over THE SAME UNITS the test used.

WHY THIS AND NOT THE CLUSTER BOOTSTRAP ALREADY ON DISK
    threshold_decomposition_v2 stores two intervals per component: a t-interval
    over the 20 seeds (`ci_lo/ci_hi`) and a run-level cluster-bootstrap interval
    averaged across seeds (`boot_ci_lo/boot_ci_hi`). They are NOT interchangeable
    -- the second averages the seeds away and therefore ignores seed-to-seed
    variability, which makes it markedly narrower. On this campaign it flips two
    S->M components from "interval contains zero" to "excludes zero".

    Swapping a t-interval for that narrower interval would strengthen claims on a
    smaller basis, in the very answer where we promise more conservative
    inference. So the replacement here resamples THE SEEDS: same estimand, same
    units, same width scale as the t-interval, but no normality assumption --
    which is exactly what R#3.14 asks for. The cluster bootstrap stays on record
    as a separate sensitivity analysis answering a different question.

WHAT IT DOES
    1. threshold_decomposition_v2/decomposition_per_seed.csv -> percentile CI per
       (direction, component) by resampling seeds.
    2. instability_ablation_v2/instability_per_seed.csv -> same per
       (config, classifier, direction).
    3. compute_separability_v2/separability_bootstrap.csv -> percentile CI of the
       mobile-minus-static difference, replacing the Welch t-test that is applied
       to bootstrap distributions (its p depends on n_boot, not on the data).

    Nothing is retrained and no campaign is re-run; every input is already on
    disk. Outputs land beside the inputs as *_seedboot.csv.

Usage:
    python seed_bootstrap_ci.py [--arm arms_r34_17feat_frozencal] [--n-boot 10000]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

AN = os.path.dirname(os.path.abspath(__file__))


def boot_ci(values, n_boot, seed, lo=2.5, hi=97.5):
    """Percentile CI of the mean, resampling the given units with replacement."""
    v = np.asarray(values, dtype=np.float64)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return float("nan"), float("nan"), float(v.mean()) if len(v) else float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    return float(np.percentile(means, lo)), float(np.percentile(means, hi)), float(v.mean())


def pick(df, *names):
    for n in names:
        if n in df.columns:
            return n
    return None


def do_threshold(res, n_boot, seed):
    """decomposition_per_seed.csv is WIDE: one row per (direction, seed) with the
    components as COLUMNS. Bootstrap each component over the seeds."""
    f = os.path.join(res, "threshold_decomposition_v2", "decomposition_per_seed.csv")
    if not os.path.isfile(f):
        print("  [skip] %s" % f); return None
    d = pd.read_csv(f)
    comps = [c for c in ("delta_total", "delta_ranking", "delta_threshold") if c in d.columns]
    if not comps or "direction" not in d.columns:
        print("  [skip] threshold: columns are %s" % list(d.columns)[:8]); return None
    rows = []
    for direction, g in d.groupby("direction"):
        for c in comps:
            lo, hi, m = boot_ci(g[c], n_boot, seed)
            rows.append({"direction": direction, "component": c,
                         "n_seeds": len(g), "mean": m,
                         "seedboot_ci_lo": lo, "seedboot_ci_hi": hi,
                         "excludes_zero": bool(lo > 0 or hi < 0)})
    out = os.path.join(res, "threshold_decomposition_v2", "decomposition_seedboot.csv")
    df = pd.DataFrame(rows); df.to_csv(out, index=False)
    print("  wrote %s" % out)
    return df


def do_instability(res, n_boot, seed):
    """instability_per_seed.csv is WIDE: columns are
    {config}_{classifier}_{SM|MS}_{metric}. The quantity of interest is the
    per-seed delta of tgt_acc against the Full_17 baseline, same classifier and
    direction, then bootstrapped over seeds."""
    f = os.path.join(res, "instability_ablation_v2", "instability_per_seed.csv")
    if not os.path.isfile(f):
        print("  [skip] %s" % f); return None
    d = pd.read_csv(f)
    tgt = [c for c in d.columns if c.endswith("_tgt_acc")]
    if not tgt:
        print("  [skip] instability: no *_tgt_acc columns"); return None
    parsed = {}
    for c in tgt:
        stem = c[:-len("_tgt_acc")]
        for direction in ("SM", "MS"):
            suf = "_" + direction
            if stem.endswith(suf):
                rest = stem[: -len(suf)]
                for clf in ("CatBoost", "LogReg"):
                    if rest.endswith("_" + clf):
                        parsed[c] = (rest[: -(len(clf) + 1)], clf, direction)
                break
    rows = []
    for col, (cfg, clf, direction) in parsed.items():
        base = "Full_17_%s_%s_tgt_acc" % (clf, direction)
        if base not in d.columns or cfg == "Full_17":
            continue
        delta = d[col].to_numpy() - d[base].to_numpy()
        lo, hi, m = boot_ci(delta, n_boot, seed)
        rows.append({"config": cfg, "classifier": clf, "direction": direction,
                     "n_seeds": int(len(delta)), "mean_delta": m,
                     "seedboot_ci_lo": lo, "seedboot_ci_hi": hi,
                     "excludes_zero": bool(lo > 0 or hi < 0)})
    if not rows:
        print("  [skip] instability: nothing parsed"); return None
    out = os.path.join(res, "instability_ablation_v2", "instability_seedboot.csv")
    df = pd.DataFrame(rows).sort_values(["config", "classifier", "direction"])
    df.to_csv(out, index=False)
    print("  wrote %s" % out)
    return df


def do_separability(res):
    """Replace the Welch test on bootstrap distributions with the percentile
    interval of their difference. The distributions are already saved."""
    f = os.path.join(res, "compute_separability_v2", "separability_bootstrap.csv")
    if not os.path.isfile(f):
        print("  [skip] %s" % f); return None
    d = pd.read_csv(f)
    rows = []
    for measure, s_col, m_col in [
            ("Mahalanobis distance", "static_mahalanobis", "mobile_mahalanobis"),
            ("LDA separation ratio", "static_lda_ratio", "mobile_lda_ratio")]:
        if s_col not in d.columns or m_col not in d.columns:
            continue
        diff = d[m_col].to_numpy() - d[s_col].to_numpy()   # mobile minus static
        rows.append({
            "measure": measure, "n_bootstrap": len(diff),
            "mean_diff_mobile_minus_static": float(diff.mean()),
            "diff_ci_lo": float(np.percentile(diff, 2.5)),
            "diff_ci_hi": float(np.percentile(diff, 97.5)),
            "excludes_zero": bool(np.percentile(diff, 2.5) > 0
                                  or np.percentile(diff, 97.5) < 0),
        })
    if not rows:
        print("  [skip] separability: columns are %s" % list(d.columns)); return None
    out = os.path.join(res, "compute_separability_v2", "separability_diff_ci.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print("  wrote %s" % out)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="arms_r34_17feat_frozencal")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    res = os.path.join(AN, args.arm, "listener", "results")
    if not os.path.isdir(res):
        sys.exit("no results dir: %s" % res)
    print("arm     : %s" % args.arm)
    print("n_boot  : %d   (resampling SEEDS, not runs)" % args.n_boot)

    print("\n[1] threshold decomposition")
    t = do_threshold(res, args.n_boot, args.seed)
    print("\n[2] instability ablation")
    i = do_instability(res, args.n_boot, args.seed)
    print("\n[3] separability (replaces the Welch test on bootstrap distributions)")
    s = do_separability(res)

    for name, df in [("threshold", t), ("instability", i), ("separability", s)]:
        if df is not None and len(df):
            print("\n===== %s =====" % name)
            print(df.to_string(index=False))


if __name__ == "__main__":
    main()
