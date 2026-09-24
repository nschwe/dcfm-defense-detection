#!/usr/bin/env python3
"""
R#5.9 --- enumerate the engineered feature space, exactly as the pipeline builds
it, and emit the supplementary table the reviewer asks for.

    "The full feature list should appear in a supplementary table, or the
     engineering code should be made available along with the processed data."

The expansion is NOT a black box: engineer_advanced_features is deterministic
arithmetic over named base metrics. This script imports the very pipeline module
the reported campaign runs, applies it to the reported bundle, and reports every
resulting column together with what happens to it downstream.

⛔ TWO THINGS THE REVIEWER'S NUMBERS PREDATE
  1. "141-dimensional space from 33 base metrics" is the OLD feature space. The
     reported campaign is the 17-observable arm: 77 engineered from 18 columns.
  2. engineer_advanced_features resolves a missing base metric to ZEROS, via
         def col(name): return X[name] if name in X.columns else zeros
     Several engineered features reference metrics that do not exist in the
     17-observable space (PacketDeliveryRatio, PacketLossRatio,
     AverageEndToEndDelay, AverageJitter, FlowCount, ...), so they are built from
     zeros. VarianceThreshold(0.01) drops them later, but they are counted in the
     "77" the log reports. The table below marks them.

USAGE
    enumerate_engineered.py <wide_bundle.csv.gz> <mode> [out.csv]
"""
import os
import sys

import numpy as np
import pandas as pd

# Resolved against this file, so a clone of the repository runs as it stands;
# the same pattern rename_67.py uses beside it. It was an absolute home
# path until 16/9, which no clone could follow.
_HERE = os.path.dirname(os.path.abspath(__file__))
PIPE = os.environ.get("ENG_PIPELINE",
                      os.path.join(os.path.dirname(_HERE),
                                   "frozencal67_pipeline_17"))
sys.path.insert(0, PIPE)

from defense_detection_v2 import DefenseDetector, Config as DetectionConfig  # noqa: E402


def main(bundle, mode, out=None):
    cfg = DetectionConfig()
    det = DefenseDetector(cfg)
    base = list(det.METRICS)

    print(f"pipeline   : {PIPE}")
    print(f"base metrics ({len(base)}): {', '.join(base)}")
    print()

    df = pd.read_csv(bundle)
    print(f"bundle     : {bundle}")
    print(f"             {len(df)} rows, {len(df.columns)} columns")

    # which base metrics the engineering references but the arm does not carry
    present = [m for m in base if m in df.columns]
    missing = [m for m in base if m not in df.columns]
    print(f"base metrics present in this bundle: {len(present)}")
    if missing:
        print(f"  ⚠️ absent, will resolve to zeros: {missing}")
    print()

    # Build the base frame the way load_data does: one row per window, the base
    # metric columns only.
    X = df[[c for c in base if c in df.columns]].copy()
    for m in base:
        if m not in X.columns:
            X[m] = 0.0
    X = X[base]
    X["_measurement_duration"] = df["_Duration"] if "_Duration" in df.columns else 40.0

    X_eng = det.engineer_advanced_features(X)

    engineered = [c for c in X_eng.columns if c not in base]
    print(f"engineered : {X_eng.shape[1]} columns total "
          f"({len(base)} base + {len(engineered)} derived)")
    print()

    # classify each column
    rows = []
    for c in X_eng.columns:
        v = X_eng[c].astype(float)
        var = float(v.var())
        rows.append(dict(
            feature=c,
            kind=("base" if c in base else classify(c)),
            variance=var,
            constant=bool(var <= 1e-12),
            drops_at_variance_filter=bool(var < 0.01),
            min=float(v.min()), max=float(v.max()), mean=float(v.mean()),
        ))
    tab = pd.DataFrame(rows)

    # the correlation filter the pipeline applies next, reproduced
    kept = tab.loc[~tab.drops_at_variance_filter, "feature"].tolist()
    corr = X_eng[kept].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    dropped_corr = [c for c in upper.columns if (upper[c] > 0.95).any()]
    tab["drops_at_correlation_filter"] = tab.feature.isin(dropped_corr)
    tab["survives_both_filters"] = (~tab.drops_at_variance_filter
                                   & ~tab.drops_at_correlation_filter)

    print(tab.kind.value_counts().to_string())
    print()
    print(f"constant (all one value)          : {int(tab.constant.sum())}")
    print(f"dropped by VarianceThreshold(0.01): {int(tab.drops_at_variance_filter.sum())}")
    print(f"dropped by |r|>0.95               : {int(tab.drops_at_correlation_filter.sum())}")
    print(f"⭐ surviving both filters          : {int(tab.survives_both_filters.sum())}")
    print()
    dead = tab.loc[tab.constant, "feature"].tolist()
    if dead:
        print(f"⚠️ constant features, i.e. built from absent base metrics ({len(dead)}):")
        for d in dead:
            print(f"     {d}")

    if out:
        tab.to_csv(out, index=False)
        print(f"\nwrote {out}")


def classify(name):
    if name.startswith("log1p_"):
        return "statistical: log1p"
    if name.startswith("sqrt_"):
        return "statistical: sqrt"
    if name.endswith("_squared") or name.endswith("_cubed"):
        return "statistical: power"
    if name.startswith("row_"):
        return "statistical: row-wise"
    if name.startswith("interact_"):
        return "interaction: product"
    if name.startswith("ratio_"):
        return "interaction: ratio"
    if name.startswith("Duration_") or name == "Time_Efficiency":
        return "temporal"
    return "network-performance indicator"


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
