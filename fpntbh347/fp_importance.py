#!/usr/bin/env python3
"""fp_importance.py -- which listener features actually carry the defence signal?

The run gave listener 0.9481 static / 0.9469 mobile, the highest of any arm this
system has produced, and recall jumped from GCOP's 0.80-0.89 to 0.94. The
suspicion is that the Section 5.2 piggyback -- 4 bytes per advertised neighbour
on every TC, attack or no attack -- is being read as the defence.

Per-feature separation of defense_active, straight off the bundle, no model:
  * AUC of the single feature
  * the standardised mean difference between defended and undefended windows

The three byte-volume expressions are flagged, since they are the ablation
targets:
    ObsControlBytesRatio        -> RoutingOverheadRatio, RoutingOverheadBytesRatio
    SniffedBytes/SniffedFrames  -> AvgTxPacketSize, AvgRxPacketSize
    ObsAvgBytesPerSource        -> AvgTxBytesPerFlow, AvgRxBytesPerFlow,
                                   AvgFlowThroughput
"""
import gzip
import os
import sys

import numpy as np
import pandas as pd

# The bundle directory to read: it holds wide_static.csv.gz and
# wide_mobile.csv.gz. Give it as the first argument; the default is the
# reported campaign, resolved against this file so a clone runs as it stands.
_HERE = os.path.dirname(os.path.abspath(__file__))
BASE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    _HERE, "analysis", "arms_fpnt_17_frozencal", "listener", "colab_data")

BYTE_VOLUME = {
    "RoutingOverheadRatio": "ObsControlBytesRatio",
    "RoutingOverheadBytesRatio": "ObsControlBytesRatio",
    "AvgTxPacketSize": "SniffedBytes/SniffedFrames",
    "AvgRxPacketSize": "SniffedBytes/SniffedFrames",
    "AvgTxBytesPerFlow": "ObsAvgBytesPerSource",
    "AvgRxBytesPerFlow": "ObsAvgBytesPerSource",
    "AvgFlowThroughput": "ObsAvgBytesPerSource",
}

META = {"scenario", "file_source", "defense_active",
        "_Duration", "_StartTime", "_EndTime"}


def auc(x, y):
    """Rank-based AUC of a single feature against a binary label."""
    ok = np.isfinite(x)
    x, y = x[ok], y[ok]
    if len(np.unique(y)) < 2:
        return float("nan")
    r = pd.Series(x).rank().values
    n1 = (y == 1).sum()
    n0 = (y == 0).sum()
    a = (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return max(a, 1 - a)          # direction-free separation


for mode in ("static", "mobile"):
    df = pd.read_csv(f"{BASE}/wide_{mode}.csv.gz")
    y = df["defense_active"].values.astype(int)
    feats = [c for c in df.columns if c not in META]

    rows = []
    for c in feats:
        x = pd.to_numeric(df[c], errors="coerce").values.astype(float)
        a = auc(x, y)
        d0, d1 = x[y == 0], x[y == 1]
        d0, d1 = d0[np.isfinite(d0)], d1[np.isfinite(d1)]
        sd = np.sqrt((d0.var(ddof=1) + d1.var(ddof=1)) / 2) if len(d0) > 1 and len(d1) > 1 else np.nan
        smd = (d1.mean() - d0.mean()) / sd if sd and np.isfinite(sd) and sd > 0 else np.nan
        rows.append((c, a, smd))

    rows.sort(key=lambda r: (-r[1] if np.isfinite(r[1]) else 0))

    print(f"\n===== {mode}   n={len(df)} windows, "
          f"{int((y==1).sum())} defended / {int((y==0).sum())} not =====")
    print(f"  {'feature':<36} {'AUC':>7} {'std.mean.diff':>14}   byte-volume source")
    print("  " + "-" * 84)
    for c, a, smd in rows:
        tag = BYTE_VOLUME.get(c, "")
        mark = "  <== " + tag if tag else ""
        print(f"  {c:<36} {a:>7.4f} {smd:>14.3f}{mark}")

    bv = [r for r in rows if r[0] in BYTE_VOLUME]
    other = [r for r in rows if r[0] not in BYTE_VOLUME]
    print(f"\n  best byte-volume feature : {bv[0][0]} AUC={bv[0][1]:.4f}"
          if bv else "\n  no byte-volume features found")
    print(f"  best other feature       : {other[0][0]} AUC={other[0][1]:.4f}")
    top5 = [r[0] for r in rows[:5]]
    nbv = sum(1 for c in top5 if c in BYTE_VOLUME)
    print(f"  byte-volume features in the top 5: {nbv} of 5  ({', '.join(top5)})")
