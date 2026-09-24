#!/usr/bin/env python3
"""
fraction_frozen_score.py -- R#3.11 in the 17-observable space, frozen detector.

WHY THIS EXISTS. The fraction sweep (STATE 25.104, 26/8) is DONE: 12,000 runs,
six cells f in {0.25,0.50,0.75} x {static,mobile}, exactly paired on 2,000 seeds
(25.109.1). But every cell was LEARNED in the 33-metric space -- f0.5_static is
(8000, 34) -- so none of its accuracies may be quoted under the revision's
17-observable constraint. Nothing has to be simulated again: the pipeline, not
the bundle, decides the feature space (25.162.1), and run_prop17.sh re-derives
arms by copying the same bundle and pointing DCFM_PIPELINE at pipeline_17.

WHAT IT DOES. Scores the existing fraction bundles with the REPORTED campaign's
detector, frozen:

    detector trained under ordinary DCFM (f=1)  ->  performance at f < 1

No fit, no feature selection, no calibration, no threshold tuning. The scaler,
the selected column list, the fitted estimator and its tuned threshold all come
from arms_r34_17feat's stored pickle.

OUT-OF-SAMPLE RESTRICTION, AND WHY IT IS NOT OPTIONAL. The swept seeds are a
subset of the 10,000-run campaign the detector was TRAINED on, so scoring every
swept row would put training topologies into the evaluation. Every cell is
therefore restricted to the run ids in the reported model's own TEST split,
reconstructed with the pipeline's own splitter. The f=1.00 comparator is the
reported campaign restricted to the SAME run ids, which is the exactly-matched
endpoint 25.109.1 left open.

THREE GATES, THE SCRIPT ABORTS ON ANY.
  1. the pipeline exposes 17 metrics;
  2. the reported arm's stored accuracy is reproduced to 1e-9 from the
     reconstruction (the r23_paired_bootstrap.py fidelity check);
  3. the swept run ids intersected with the test split are IDENTICAL across all
     four f values -- otherwise the series is not paired.

Reads:  arms_r34_17feat/listener/{colab_data,results}/...
        arms_fractionsweep/f{F}_{mode}/listener/colab_data/wide_{mode}.csv.gz
Writes: only --out-dir (default arms_fractionsweep/frozen17/).

  python3 fraction_frozen_score.py --mode static
  python3 fraction_frozen_score.py --mode both
"""
import argparse
import glob
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

HERE = Path(__file__).resolve().parent          # .../analysis/arms_fractionsweep
ANALYSIS = HERE.parent                          # .../analysis
# 2/9 (STATE 25.188.9): the campaign arm is selectable via FRAC_REPORTED so the
# sweep can be scored against the CORRECTED (frozencal) models. The default is
# unchanged -- the originally reported arm. ⚠️ Both arms are 77-column, which is
# what the frozen scaler requires (see the note below).
REPORTED = Path(os.environ.get("FRAC_REPORTED", str(ANALYSIS / "arms_r34_17feat")))
# ⛔ The generator version is not free to choose here. A frozen scaler was fitted
# on a fixed column count, so the pipeline that built those columns is the only
# one that can feed it. The reported arm's own log reads "Created 77 features";
# analysis/cleanfeat/pipeline_17 is the later 67-column generator, which skips 10
# formulas whose inputs are absent. Both are 17-observable and 25.179.4 measured
# the difference between them as exactly zero -- but 67 != 77 to a fitted scaler,
# and the first run of this script aborted on precisely that.
PIPELINE = os.environ.get("FRAC_PIPELINE", str(REPORTED / "pipeline_17"))
FRACS = [0.25, 0.5, 0.75]
RUN_RE = re.compile(r"metrics_output-(\d+)\.csv")


def load_pipeline():
    path = os.path.join(PIPELINE, "defense_detection_v2.py")
    if not os.path.isfile(path):
        sys.exit("ABORT: no defense_detection_v2.py under %s" % PIPELINE)
    spec = importlib.util.spec_from_file_location("dd", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["dd"] = mod
    spec.loader.exec_module(mod)
    n = len(getattr(mod.DefenseDetector, "METRICS", []))
    if n != 17:
        sys.exit("ABORT: %s exposes %d metrics, expected 17" % (PIPELINE, n))
    print("pipeline: %s (METRICS=17 ok)" % PIPELINE)
    return mod


def build_frame(dd, arm_listener, mode):
    """X_eng, y, groups for one arm, using the pipeline's own loader."""
    bundle = arm_listener / "colab_data"
    if not bundle.is_dir():
        sys.exit("ABORT: no bundle dir %s" % bundle)
    os.environ["DCFM_DATA_BUNDLE"] = str(bundle)
    cfg = dd.Config()
    cfg.data_root = str(bundle / mode)          # mode-bearing, only parsed
    cfg.results_dir = str(arm_listener / "results" / mode)
    det = dd.DefenseDetector(cfg)
    det.verbosity = 0
    tall = det.load_simulation_data_enhanced()
    X, y, groups = det.preprocess_data_enhanced(tall)
    return det.engineer_advanced_features(X), y, np.array(groups), cfg


def run_ids(groups):
    out = []
    for g in groups:
        m = RUN_RE.search(str(g))
        out.append(int(m.group(1)) if m else -1)
    arr = np.array(out)
    if (arr < 0).any():
        sys.exit("ABORT: %d group ids do not match the run-id pattern"
                 % int((arr < 0).sum()))
    return arr


def load_reported_model(mode):
    import joblib
    hits = glob.glob(str(REPORTED / "listener" / "results" / mode / "best_model_*.pkl"))
    if len(hits) != 1:
        sys.exit("ABORT: %d model pickles for the reported arm/%s, expected 1"
                 % (len(hits), mode))
    md = joblib.load(hits[0])
    for key in ("model", "scaler", "feature_names", "threshold"):
        if key not in md:
            sys.exit("ABORT: %s has no '%s'" % (hits[0], key))
    name = os.path.basename(hits[0])[len("best_model_"):-len(".pkl")]
    return md, name


def score(md, X, y):
    scaler, cols, model = md["scaler"], list(md["feature_names"]), md["model"]
    n_seen = getattr(scaler, "n_features_in_", None)
    if n_seen is not None and n_seen != X.shape[1]:
        sys.exit("ABORT: scaler was fitted on %s columns, this frame has %d.\n"
                 "       This is a generator-version mismatch, not a data problem:\n"
                 "       77 = the arm's own pipeline_17, 67 = cleanfeat/pipeline_17.\n"
                 "       Point FRAC_PIPELINE at the one the arm was learned with."
                 % (n_seen, X.shape[1]))
    Xs = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
    missing = [c for c in cols if c not in Xs.columns]
    if missing:
        sys.exit("ABORT: %d selected columns missing, e.g. %s"
                 % (len(missing), missing[:5]))
    proba = model.predict_proba(Xs[cols])[:, 1]
    pred = (proba >= float(md["threshold"])).astype(int)
    yv = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
    return proba, pred, yv


def metrics(proba, pred, yv):
    pos = yv == 1
    return {
        "n_rows": int(len(yv)),
        "accuracy": float((pred == yv).mean()),
        "auc": float(roc_auc_score(yv, proba)) if len(np.unique(yv)) > 1 else float("nan"),
        "defense_recall": float(pred[pos].mean()) if pos.any() else float("nan"),
        "false_positive_rate": float(pred[~pos].mean()) if (~pos).any() else float("nan"),
    }


def paired_bootstrap(correct_by_f, ids_by_f, ids, B=2000, seed=42):
    """Paired over run ids: one resampled id vector applied to every f."""
    rng = np.random.default_rng(seed)
    idx = {f: {i: np.where(ids_by_f[f] == i)[0] for i in ids} for f in correct_by_f}
    draws = {f: [] for f in correct_by_f}
    for _ in range(B):
        take = rng.choice(ids, size=len(ids), replace=True)
        for f in correct_by_f:
            rows = np.concatenate([idx[f][i] for i in take])
            draws[f].append(float(correct_by_f[f][rows].mean()))
    for f in correct_by_f:
        draws[f] = np.array(draws[f])
    out = {}
    for f in FRACS:
        d = draws[f] - draws[1.0]
        out["f%g_minus_f1" % f] = {
            "point": float(correct_by_f[f].mean() - correct_by_f[1.0].mean()),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_diff_ge_0": float((d >= 0).mean()),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="static", choices=["static", "mobile", "both"])
    ap.add_argument("--out-dir", default=str(HERE / "frozen17"))
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    dd = load_pipeline()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    modes = ["static", "mobile"] if args.mode == "both" else [args.mode]

    for mode in modes:
        print("\n================ %s ================" % mode)
        md, model_name = load_reported_model(mode)
        print("frozen detector: %s, threshold %.4f, %d selected columns"
              % (model_name, float(md["threshold"]), len(md["feature_names"])))

        # --- the reported arm, and GATE 2: reproduce its stored accuracy ------
        Xr, yr, gr, cfg = build_frame(dd, REPORTED / "listener", mode)
        splitter = GroupShuffleSplit(n_splits=1, test_size=cfg.test_size,
                                     random_state=cfg.random_state)
        _, test_idx = next(splitter.split(Xr, yr, gr))
        Xr_t, yr_t, gr_t = Xr.iloc[test_idx], yr.iloc[test_idx], gr[test_idx]
        proba, pred, yv = score(md, Xr_t, yr_t)
        acc = float((pred == yv).mean())
        stored = pd.read_csv(REPORTED / "listener" / "results" / mode / "results.csv")
        row = stored[stored["Model"] == model_name]
        if row.empty:
            sys.exit("ABORT: results.csv has no row for %s" % model_name)
        stored_acc = float(row.iloc[0]["Accuracy"])
        if abs(acc - stored_acc) > 1e-9:
            sys.exit("ABORT: reconstruction gives %.10f, results.csv says %.10f"
                     % (acc, stored_acc))
        print("gate 2 ok: reconstruction reproduces %.4f to 1e-9" % stored_acc)
        test_ids = set(run_ids(gr_t).tolist())

        # --- every cell, restricted to the test run ids ------------------------
        frames = {1.0: (Xr_t, yr_t, run_ids(gr_t))}
        for f in FRACS:
            arm = HERE / ("f%g_%s" % (f, mode)) / "listener"
            if not arm.is_dir():
                sys.exit("ABORT: no cell %s" % arm)
            Xf, yf, gf, _ = build_frame(dd, arm, mode)
            frames[f] = (Xf, yf, run_ids(gf))

        common = None
        for f, (_, _, rid) in frames.items():
            s = set(rid.tolist()) & test_ids
            common = s if common is None else (common & s)
        common = np.array(sorted(common))
        print("paired run ids after the test-split restriction: %d" % len(common))
        if len(common) == 0:
            sys.exit("ABORT: the swept ids and the test split do not intersect")

        cells, correct_by_f, ids_by_f = {}, {}, {}
        for f, (Xf, yf, rid) in sorted(frames.items()):
            keep = np.isin(rid, common)
            # GATE 3: identical id set in every cell
            if set(rid[keep].tolist()) != set(common.tolist()):
                sys.exit("ABORT: f=%g does not cover the paired id set" % f)
            proba, pred, yv = score(md, Xf.iloc[keep], yf.iloc[keep])
            m = metrics(proba, pred, yv)
            m["n_run_ids"] = int(len(common))
            cells["f%g" % f] = m
            correct_by_f[f] = (pred == yv).astype(float)
            ids_by_f[f] = rid[keep]
            print("  f=%.2f  n=%5d  acc=%.4f  auc=%.4f  defense recall=%.4f  fpr=%.4f"
                  % (f, m["n_rows"], m["accuracy"], m["auc"],
                     m["defense_recall"], m["false_positive_rate"]))

        boot = paired_bootstrap(correct_by_f, ids_by_f, common, B=args.boot)
        for k, v in boot.items():
            print("  %-16s %+.4f  95%% CI [%+.4f, %+.4f]  P(diff>=0)=%.4f"
                  % (k, v["point"], v["ci95"][0], v["ci95"][1], v["p_diff_ge_0"]))

        payload = {"mode": mode, "pipeline": PIPELINE, "frozen_model": model_name,
                   "threshold": float(md["threshold"]),
                   "reported_stored_accuracy": stored_acc,
                   "n_paired_run_ids": int(len(common)),
                   "cells": cells, "paired_bootstrap": boot,
                   "bootstrap_B": args.boot}
        dest = out_dir / ("frozen17_%s.json" % mode)
        dest.write_text(json.dumps(payload, indent=2))
        print("wrote %s" % dest)


if __name__ == "__main__":
    main()
