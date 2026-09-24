#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_arm_bundles.py -- build the single-vantage listener bundle from a simulation tree.

For each configuration this writes
    <DCFM_ARMS_ROOT>/listener/colab_data/wide_{static,mobile}.csv.gz
in the schema the pipeline's loader expects (defense_detection_v2.py, bundle
path): one row per measurement window, one column per observable under the
canonical name arm_spec.py assigns it, plus the six meta columns
    scenario, file_source, defense_active, _Duration, _StartTime, _EndTime.

Every row is computed from ONE observer_metrics-<id>.csv row: the vantage point
chosen for that run, deterministically from the run id (see _observer_row_for).
The simulator also writes a network-wide summary row into the same file
(ObserverIdx == -1); it is not part of any single vantage point's observation
and is dropped before the pick, so it can never be selected and never changes
the pick.

Run:
    DCFM_SIM_ROOT=<tree with features_{static,mobile}/> \
    DCFM_ARMS_ROOT=<arms root> \
    python make_arm_bundles.py --arm listener              # both configurations
    python make_arm_bundles.py --arm listener --mode static
    python make_arm_bundles.py --arm listener --limit 200  # quick rehearsal

The reported campaign's bundle was built with --limit 10000 from the two C_all
stagings (static and mobile); see the README.
"""
import argparse
import glob
import gzip
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arm_spec import ARMS, LISTENER, LISTENER_ORDER  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))          # analysis/
# Input simulation root. Defaults to the ns-3.47 campaign output; override with
# DCFM_SIM_ROOT to build bundles from a different campaign. Must contain
# features_{static,mobile}/<scenario>/*.csv.
V4 = os.environ.get(
    "DCFM_SIM_ROOT",
    os.path.abspath(os.path.join(_HERE, "..", "simulations_v347")))
# Where the bundle is written: <ARMS_ROOT>/<arm>/colab_data/.
OUT_ROOT = os.environ.get("DCFM_ARMS_ROOT", os.path.join(_HERE, "arms"))
SCENARIOS = {"baseline": 0, "attack_only": 0, "defense_only": 1, "defense_vs_attack": 1}
META = ["scenario", "file_source", "defense_active", "_Duration", "_StartTime", "_EndTime"]
OBS_PICK = 2654435761   # Knuth multiplicative hash constant: the observer pick


def log(m):
    print(m, flush=True)


def _eval_obs(expr, row):
    """Evaluate a 'a/b', 'a*8/b', or bare-column observer expression on one row."""
    expr = expr[4:]  # strip 'obs:'
    # tokens are column names, numeric literals, and the operators * /
    import re
    def repl(m):
        tok = m.group(0)
        if re.fullmatch(r"[0-9.]+", tok):
            return tok
        return f"row['{tok}']"
    py = re.sub(r"[A-Za-z_][A-Za-z0-9_]*", repl, expr)
    try:
        return float(eval(py, {"__builtins__": {}}, {"row": row}))
    except (ZeroDivisionError, KeyError, ValueError, TypeError):
        return 0.0


def _vantage_rows(op):
    """The vantage-point rows of one observer_metrics file. The simulator's
    network-wide summary row (ObserverIdx == -1) is excluded first, so the
    deterministic pick and its row count are those of the vantage points only."""
    df = pd.read_csv(op)
    if "ObserverIdx" in df.columns:
        df = df[df["ObserverIdx"] != -1].reset_index(drop=True)
    return df


def _observer_row_for(op, rid):
    """The single chosen vantage-point row (dict) for run rid, deterministic in
    the run id: index (rid * OBS_PICK) mod <number of vantage points>."""
    df = _vantage_rows(op)
    if df.empty:
        return None
    idx = (rid * OBS_PICK) % len(df)
    return df.iloc[idx].to_dict()


def build_observer(mode, limit, space, order, row_pick, prefix=""):
    """Build the bundle for one configuration: row_pick(op, rid) selects the
    observer row of each run, and every canonical name in `order` is evaluated
    from that row with the expression `space` gives it."""
    base = os.path.join(V4, f"features_{mode}")
    rows = []
    for scen, label in SCENARIOS.items():
        files = sorted(glob.glob(os.path.join(base, scen, "observer_metrics-*.csv")),
                       key=lambda p: int(os.path.basename(p).split("-")[1].split(".")[0]))
        if limit:
            files = files[:limit]
        for op in files:
            rid = int(os.path.basename(op).split("-")[1].split(".")[0])
            orow = row_pick(op, rid)
            if orow is None:
                continue
            rec = {}
            for canon in order:
                rec[prefix + canon] = _eval_obs(space[canon], orow)
            rec["_Duration"] = float(orow.get("Duration", 40.0))
            rec["_StartTime"] = float(orow.get("StartTime", 0.0))
            rec["_EndTime"] = float(orow.get("EndTime", 0.0))
            rec["scenario"] = scen
            # file_source names the run; the pipeline groups its splits by it.
            rec["file_source"] = f"metrics_output-{rid}.csv"
            rec["defense_active"] = label
            rows.append(rec)
    return pd.DataFrame(rows)


def order_columns(df):
    metric_cols = sorted(c for c in df.columns if c not in META)
    return df[metric_cols + [c for c in META if c in df.columns]]


def write_bundle(df, arm, mode):
    outdir = os.path.join(OUT_ROOT, arm, "colab_data")
    os.makedirs(outdir, exist_ok=True)
    df = order_columns(df)
    path = os.path.join(outdir, f"wide_{mode}.csv.gz")
    # %.17g is the shortest format that round-trips every float64 exactly. %.10g
    # silently truncated values above ~1e4 (QA caught AvgTx/RxBytesPerFlow at
    # 11032.444444 -> 11032.444440), and this project makes bit-exactness claims,
    # so no precision is traded for file size here.
    with gzip.open(path, "wt", newline="") as f:
        df.to_csv(f, index=False, float_format="%.17g")
    log(f"  [{arm}/{mode}] {df.shape[0]} rows x {df.shape[1]} cols -> {path}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(ARMS), default="listener")
    ap.add_argument("--mode", choices=["static", "mobile"], default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    modes = [args.mode] if args.mode else ["static", "mobile"]
    for mode in modes:
        log(f"building {args.arm}/{mode} ...")
        df = build_observer(mode, args.limit, LISTENER, LISTENER_ORDER, _observer_row_for)
        write_bundle(df, args.arm, mode)


if __name__ == "__main__":
    main()
