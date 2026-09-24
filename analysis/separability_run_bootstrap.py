#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""separability_run_bootstrap.py -- R#3.14: a RUN-LEVEL interval for the
static-vs-mobile separability difference (Table 5).

WHY
    compute_separability_v2 resamples ROWS. Each row is one (scenario,
    file_source) pair -- windows are already averaged away by to_wide -- so a
    row is a run x scenario, and every `file_source` contributes FOUR correlated
    rows (two with defense_active=0, two with =1). Resampling rows independently
    breaks that clustering and yields an interval narrower than a run-level one.

    That analysis also reports two statistics computed FROM the bootstrap
    distributions: a Welch t-test and a Cohen's d whose denominator is the
    pooled SD of those distributions. Both treat the number of resamples as
    though it were the experimental sample size, so both can be driven to any
    value by raising n_bootstrap. Neither is reproduced here.

WHAT THIS DOES
    Resamples `file_source` WITH REPLACEMENT, carrying all of a run's rows, and
    recomputes the separability measures inside each replicate exactly as the
    original does (StandardScaler fitted on the replicate only). Static and
    mobile are independent campaigns, so each is resampled independently with
    its own generator; the difference of the two independent replicate sequences
    is a draw from the difference distribution.

    Reports, per measure: point estimates, the difference, and the percentile
    interval of the difference.

USAGE
    python separability_run_bootstrap.py --arm arms_r34_17feat [--n-boot 2000]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

AN = os.path.dirname(os.path.abspath(__file__))
PIPELINE = os.environ.get("SEP_PIPELINE", os.path.join(AN, "frozencal", "pipeline_17"))
sys.path.insert(0, PIPELINE)

from sklearn.preprocessing import StandardScaler          # noqa: E402
import compute_separability_v2 as sep                     # noqa: E402


def load_with_groups(data_root):
    """Same load + wide pivot as the original, but KEEPING file_source."""
    tall = sep.load_dataset(data_root)
    tall = tall[tall["Metric"].isin(sep.METRICS)].copy()
    agg = (tall.groupby(["scenario", "file_source", "Metric"], sort=False)["Value"]
               .mean().reset_index())
    wide = agg.pivot_table(index=["scenario", "file_source"],
                           columns="Metric", values="Value", aggfunc="mean")
    wide.columns.name = None
    label_map = (tall[["scenario", "file_source", "defense_active"]]
                 .drop_duplicates(["scenario", "file_source"])
                 .set_index(["scenario", "file_source"])["defense_active"])
    y = label_map.reindex(wide.index).astype(int)
    idx = wide.index.to_frame(index=False)
    for m in sep.METRICS:
        if m not in wide.columns:
            wide[m] = 0.0
    X = wide[sep.METRICS].fillna(0.0).reset_index(drop=True)
    return X.values, y.to_numpy(), idx["file_source"].to_numpy()


def replicate(X, y, groups, uniq, rng):
    """One run-level replicate: resample file_source, carry all its rows."""
    picked = rng.choice(uniq, size=len(uniq), replace=True)
    # index rows per run once, then concatenate
    rows = np.concatenate([groups_index[g] for g in picked])
    Xb, yb = X[rows], y[rows]
    if len(np.unique(yb)) < 2:
        return None
    Xs = StandardScaler().fit_transform(Xb)
    return sep._compute_both_measures(Xs, yb)


def run_domain(X, y, groups, n_boot, seed, label):
    global groups_index
    uniq = np.unique(groups)
    groups_index = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    point = sep._compute_both_measures(StandardScaler().fit_transform(X), y)
    out = []
    for _ in range(n_boot):
        r = replicate(X, y, groups, uniq, rng)
        if r is not None:
            out.append(r)
    arr = np.asarray(out)
    print("  %-7s rows=%d runs=%d  point: maha %.4f  lda %.4f  (%d replicates)"
          % (label, len(y), len(uniq), point[0], point[1], len(arr)))
    return point, arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="arms_r34_17feat")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    arm = os.path.join(AN, args.arm)
    os.environ["DCFM_DATA_BUNDLE"] = os.path.join(arm, "listener", "colab_data")
    out_dir = os.path.join(arm, "listener", "results", "compute_separability_v2")
    os.makedirs(out_dir, exist_ok=True)

    print("run-level separability bootstrap  (n_boot=%d)" % args.n_boot)
    print("  resampling unit: file_source (a simulation run, all its rows)")

    Xs, ys, gs = load_with_groups("./features_static")
    Xm, ym, gm = load_with_groups("./features_mobile")
    p_s, b_s = run_domain(Xs, ys, gs, args.n_boot, args.seed, "static")
    p_m, b_m = run_domain(Xm, ym, gm, args.n_boot, args.seed + 100_000, "mobile")

    n = min(len(b_s), len(b_m))
    rows = []
    for j, name in enumerate(["Mahalanobis distance", "LDA separation ratio"]):
        diff = b_m[:n, j] - b_s[:n, j]
        lo, hi = np.percentile(diff, 2.5), np.percentile(diff, 97.5)
        rows.append({"measure": name,
                     "static_point": float(p_s[j]), "mobile_point": float(p_m[j]),
                     "observed_diff": float(p_m[j] - p_s[j]),
                     "boot_mean_diff": float(diff.mean()),
                     "diff_ci_lo": float(lo), "diff_ci_hi": float(hi),
                     "excludes_zero": bool(lo > 0 or hi < 0),
                     "n_boot": int(n), "unit": "file_source"})
        print("  %-24s diff %+.4f  95%% CI [%+.4f, %+.4f]  excludes0=%s"
              % (name, p_m[j] - p_s[j], lo, hi, rows[-1]["excludes_zero"]))

    df = pd.DataFrame(rows)
    out = os.path.join(out_dir, "separability_run_level_ci.csv")
    df.to_csv(out, index=False)
    print("\n  wrote %s" % out)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
