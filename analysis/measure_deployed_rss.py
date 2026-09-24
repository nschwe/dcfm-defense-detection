#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""measure_deployed_rss.py -- R#3.19: runtime memory of the DEPLOYED detector.

WHY THIS EXISTS
    inference_cost_report.py reports two memory numbers and neither is what the
    reviewer asked for:

      712.90 MB   the serialised ensemble -- a STORAGE footprint, not RAM
      1761 MB     peak RSS of a process that deserialised the whole results
                  bundle, i.e. all seventeen models

    A deployment loads one model. This script measures that: it extracts the
    deployed artefact (the stacking ensemble, the scaler, the selected feature
    names and the threshold), then measures resident memory in a FRESH process
    that loads nothing else -- before the load, after the load, and while
    scoring windows one at a time.

    ⛔ Nothing is fitted. The estimator, scaler, feature list and threshold all
    come from the arm's stored pickle.

GATES (hard aborts, all printed)
    SPACE       the pipeline at R23_PIPELINE has METRICS == 17
    CALIBRATION that pipeline contains FrozenEstimator
    VANTAGE     --arm-root is a "listener" directory and it is the only arm
    BUNDLE      wide_<mode>.csv.gz has 27 columns
    MODEL       the model has a row in the arm's results.csv
    LOG         every stage-1 log claiming this arm and mode is read, and they
                must agree about the selected feature count
    SPACE+      the deployed artefact carries exactly that count
    IDENTITY    the threshold stored in the pickle equals the threshold
                results.csv reports for this model

    ⭐ IDENTITY is sharper than it looks. The stacking threshold defaults to
    0.5 (pipeline_17/defense_detection_v2.py:1261, still there); the reported
    campaign overrode it with --calibrate-stacking, picking 0.64 static and
    0.48 mobile on 4,000 held-out rows. A pickle carrying 0.5 is therefore some
    other run, and this gate refuses to measure it.

USAGE
    cd ~/ns3/nv347/ns-3.47/analysis
    R23_PIPELINE=$PWD/frozencal/pipeline_17 python measure_deployed_rss.py \
        --arm-root $PWD/arms_r34_17feat_frozencal/listener --mode static
"""
from __future__ import annotations

import argparse
import glob
import gzip
import io
import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PAGE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096


def rss_mb() -> float:
    """Current resident set size, from /proc, in MB."""
    with io.open("/proc/self/statm") as fh:
        pages = int(fh.read().split()[1])
    return pages * PAGE / (1024.0 * 1024.0)


def peak_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


# --------------------------------------------------------------------- child
def child(path: str, n_calls: int) -> None:
    """A fresh process that loads ONE model and scores with it."""
    before = rss_mb()
    t0 = time.perf_counter()
    art = joblib.load(path)
    load_s = time.perf_counter() - t0
    after = rss_mb()

    model = art["model"]
    feats = list(art["feature_names"])
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.standard_normal((1, len(feats))), columns=feats)

    model.predict_proba(X)              # warm up
    during_peak = rss_mb()
    lat = []
    for _ in range(n_calls):
        t = time.perf_counter()
        model.predict_proba(X)
        lat.append(time.perf_counter() - t)
        during_peak = max(during_peak, rss_mb())

    print(json.dumps({
        "rss_before_load_mb": before,
        "rss_after_load_mb": after,
        "rss_delta_mb": after - before,
        "rss_during_inference_mb": during_peak,
        # ⛔ Inherited from the parent across fork+exec, NOT this process.
        "ru_maxrss_inherited_mb": peak_mb(),
        "load_s": load_s,
        "median_ms_per_window": float(np.median(lat)) * 1e3,
        "n_features": len(feats),
        "n_calls": n_calls,
    }))


# --------------------------------------------------------------------- gates
def gate(arm_root: Path, mode: str, model: str) -> int:
    pipeline = Path(os.environ.get("R23_PIPELINE", HERE / "frozencal" / "pipeline_17"))
    dd_path = pipeline / "defense_detection_v2.py"
    if not dd_path.is_file():
        sys.exit(f"ABORT: no defense_detection_v2.py under {pipeline}")
    text = dd_path.read_text(encoding="utf-8", errors="replace")
    start = text.find(chr(10) + "    METRICS = [")
    end = text.find(chr(10) + "    ]", start)
    n_metrics = text.count('"', start, end) // 2 if start >= 0 else -1
    n_frozen = text.count("FrozenEstimator")
    print(f"  gate SPACE   : METRICS = {n_metrics}   ({pipeline})")
    print(f"  gate CALIB   : FrozenEstimator = {n_frozen}")
    if n_metrics != 17:
        sys.exit(f"ABORT: {n_metrics} metrics, expected 17")
    if n_frozen < 1:
        sys.exit("ABORT: pre-fix pipeline")

    parent = arm_root.parent
    arms = sorted(d.name for d in parent.iterdir()
                  if d.is_dir() and (d / "colab_data").is_dir())
    print(f"  gate VANTAGE : {arms}")
    if arms != ["listener"] or arm_root.name != "listener":
        sys.exit(f"ABORT: expected one arm 'listener' under {parent}, found {arms}")

    with gzip.open(arm_root / "colab_data" / f"wide_{mode}.csv.gz", "rt") as fh:
        ncol = len(fh.readline().rstrip(chr(10)).split(","))
    print(f"  gate BUNDLE  : wide_{mode} has {ncol} columns")
    if ncol != 27:
        sys.exit(f"ABORT: bundle has {ncol} columns, expected 27")

    res = pd.read_csv(arm_root / "results" / mode / "results.csv")
    hit = res[res["Model"] == model]
    if hit.empty:
        sys.exit(f"ABORT: {model!r} has no row in results.csv")
    row = hit.iloc[0]
    print(f"  gate MODEL   : {model} acc {float(row['Accuracy']):.5f} "
          f"auc {float(row['AUC']):.5f} thr {float(row['Threshold']):.4f}")

    # What the arm's own logs say was selected -- the SPACE+ reference. Every
    # log that claims this arm and mode is read, and they must agree: two do
    # claim each mode here (frozencal and frozencal2), and taking whichever
    # came last would hide a disagreement.
    want = str((arm_root / "results" / mode).resolve())
    claims = {}
    for root, _d, files in os.walk(HERE):
        for fn in files:
            if not fn.endswith(".log"):
                continue
            try:
                t = io.open(os.path.join(root, fn), encoding="utf-8",
                            errors="replace").read()
            except OSError:
                continue
            if "Results saved to: " + want not in t:
                continue
            for line in t.splitlines():
                line = line.strip()
                if line.startswith("Selected ") and "features" in line:
                    try:
                        claims[fn] = int(line.split()[1])
                    except ValueError:
                        pass
    for fn, k in sorted(claims.items()):
        print(f"  gate LOG     : {fn} -- selected {k}")
    values = set(claims.values())
    if len(values) > 1:
        sys.exit(f"⛔ ABORT [space]: the logs claiming this arm/mode disagree "
                 f"about the selected count: {claims}")
    return (values.pop() if values else None), float(row["Threshold"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--arm-root")
    ap.add_argument("--mode", choices=["static", "mobile"])
    ap.add_argument("--model", default="Stacking_Ensemble")
    ap.add_argument("--calls", type=int, default=100)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    if args.child:
        child(args.child, args.calls)
        return

    arm_root = Path(args.arm_root).resolve()
    print("=" * 92)
    print(f"DEPLOYED-PROCESS MEMORY -- {args.model}, {args.mode}")
    print("=" * 92)
    selected, stored_thr = gate(arm_root, args.mode, args.model)

    pkls = sorted(glob.glob(str(arm_root / "results" / args.mode /
                                "best_model_*.pkl")))
    if not pkls:
        sys.exit("ABORT: no best_model_*.pkl")
    md = joblib.load(pkls[0])
    est = (md.get("all_models") or {}).get(args.model)
    if est is None:
        sys.exit(f"ABORT: {args.model!r} not in the pickle")
    feats = list(md["feature_names"])
    if selected is not None and selected != len(feats):
        sys.exit(f"⛔ ABORT [space]: the arm's log recorded {selected} selected "
                 f"features, the artefact carries {len(feats)}")
    print(f"  gate SPACE+  : artefact carries {len(feats)} features, matching "
          f"the log")

    # ⛔ Tie the PICKLE to results.csv: the threshold the campaign reported must
    # be the threshold this model carries, or the artefact whose memory we are
    # about to measure is not the one behind the reported numbers.
    thr = float((md.get("all_thresholds") or {}).get(
        args.model, md.get("threshold", float("nan"))))
    if not (abs(thr - stored_thr) < 1e-9):
        sys.exit(f"⛔ ABORT [identity]: the pickle's threshold for "
                 f"{args.model} is {thr!r}, results.csv reports {stored_thr!r}")
    print(f"  gate IDENTITY: pickle threshold {thr:.4f} == results.csv")

    tmp = tempfile.mkdtemp(prefix="deployed_")
    free_gb = shutil.disk_usage(tmp).free / (1024.0 ** 3)
    if free_gb < 3:
        shutil.rmtree(tmp, ignore_errors=True)
        sys.exit(f"ABORT: only {free_gb:.1f} GB free in {tmp}")
    art_path = os.path.join(tmp, "deployed.pkl")
    try:
        joblib.dump({"model": est, "scaler": md["scaler"],
                     "feature_names": feats,
                     "threshold": thr}, art_path)
        art_mb = os.path.getsize(art_path) / (1024.0 * 1024.0)
        print(f"  deployed artefact on disk : {art_mb:9.2f} MB "
              f"(model + scaler + feature list + threshold)")
        print(f"  this process, having loaded all {len(md.get('all_models') or {})} "
              f"models  : peak {peak_mb():9.2f} MB   <- what NOT to quote")
        print()
        print("  measuring in a fresh process that loads only the artefact...")
        out = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--child", art_path,
             "--calls", str(args.calls)],
            capture_output=True, text=True, check=True)
        rec = json.loads(out.stdout.strip().splitlines()[-1])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    print(f"  interpreter before load   : {rec['rss_before_load_mb']:9.2f} MB")
    print(f"  after loading the model   : {rec['rss_after_load_mb']:9.2f} MB "
          f"(delta {rec['rss_delta_mb']:+.2f} MB)")
    print(f"  while scoring windows     : {rec['rss_during_inference_mb']:9.2f} MB")
    print(f"  (ru_maxrss, inherited from the parent, not a measurement of "
          f"the child: {rec['ru_maxrss_inherited_mb']:.2f} MB)")
    print(f"  deserialisation           : {rec['load_s']:9.3f} s")
    print(f"  one window, median of {rec['n_calls']}: "
          f"{rec['median_ms_per_window']:9.3f} ms")

    rec.update({"mode": args.mode, "model": args.model,
                "artefact_mb": art_mb})
    out_dir = Path(args.out_dir or (HERE / "inference_cost_out" /
                                    arm_root.parent.name / "deployed_rss"))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"deployed_rss_{args.mode}_{args.model}.csv"
    pd.DataFrame([rec]).to_csv(path, index=False)
    print(f"  written: {path}")


if __name__ == "__main__":
    main()
