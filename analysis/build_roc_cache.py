#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_roc_cache.py -- R#3.18: the prediction cache Figure 1 needs, built from
the CORRECTED campaign by reconstruction, never by refitting.

WHY THIS EXISTS
    generate_roc_figure.py loads
    <results_root>/hp_search_final/{static,mobile}/test_predictions_cache.pkl.
    The only such caches on disk are dated 13 May and belong to the old
    33-metric campaign. This script writes the same structure for the
    frozen-calibration 17-observable campaign:

        {"y_test": ndarray,
         "per_model": {name: {"scores_test": ndarray, "point_auc": float}}}

    ALL models are cached (16 + the stacking ensemble). Which of them a figure
    draws is a plotting decision, not a caching one -- see plot_roc_17.py.

HOW
    The test split and the scores come from r23_paired_bootstrap.py's own
    functions, so this cannot drift from the campaign by reimplementation:
    the pipeline's load / preprocess / engineer, its GroupShuffleSplit settings,
    the arm's stored scaler, its stored feature selection, its stored
    per-model estimators and thresholds. ⛔ Nothing is fitted here.

GATES -- all four are hard aborts, and every one of them is printed
    1. SPACE      the pipeline exposes exactly 17 METRICS
    2. CALIBRATION the pipeline contains FrozenEstimator (the corrected copy)
    3. VANTAGE    the arm holds exactly one arm directory, "listener"
    4. BUNDLE     the wide bundle has 27 columns (21 canonical + 6 meta)
    plus, per model, a reproduction gate: the accuracy recomputed at the stored
    threshold must equal results.csv to 1e-9, and the AUC likewise.

USAGE
    cd ~/ns3/nv347/ns-3.47/analysis
    R23_PIPELINE=$PWD/frozencal67_pipeline_17 \
      ~/miniconda3/envs/manet/bin/python -u build_roc_cache.py \
      --arm arms_r34_17feat_frozencal67
"""
from __future__ import annotations

import argparse
import gzip
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# The pipeline must be chosen before r23_paired_bootstrap is imported: it reads
# R23_PIPELINE at module level.
os.environ.setdefault("R23_PIPELINE", str(HERE / "frozencal67_pipeline_17"))

import joblib                     # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

import r23_paired_bootstrap as rp  # noqa: E402


def gate_pipeline(pipeline: Path) -> None:
    dd_path = pipeline / "defense_detection_v2.py"
    if not dd_path.is_file():
        sys.exit(f"ABORT: no defense_detection_v2.py under {pipeline}")
    text = dd_path.read_text(encoding="utf-8", errors="replace")

    start = text.find("\n    METRICS = [")
    if start < 0:
        sys.exit("ABORT: no METRICS block in the pipeline copy")
    end = text.find("\n    ]", start)
    n_metrics = text.count('"', start, end) // 2
    n_frozen = text.count("FrozenEstimator")

    print(f"  pipeline    : {pipeline}")
    print(f"  gate SPACE  : METRICS = {n_metrics}")
    print(f"  gate CALIB  : FrozenEstimator = {n_frozen}")
    if n_metrics != 17:
        sys.exit(f"ABORT: pipeline exposes {n_metrics} metrics, expected 17 "
                 f"(21 or 33 means the wrong copy)")
    if n_frozen < 1:
        sys.exit("ABORT: no FrozenEstimator -- this is the pre-fix pipeline")


def gate_arm(arm_dir: Path, mode: str) -> Path:
    arms = sorted(p.name for p in arm_dir.iterdir()
                  if p.is_dir() and (p / "colab_data").is_dir())
    print(f"  gate VANTAGE: arm directories = {arms}")
    if arms != ["listener"]:
        sys.exit(f"ABORT: expected exactly one arm, 'listener'; found {arms}")

    bundle = arm_dir / "listener" / "colab_data" / f"wide_{mode}.csv.gz"
    if not bundle.is_file():
        sys.exit(f"ABORT: no bundle {bundle}")
    with gzip.open(bundle, "rt") as fh:
        header = fh.readline().rstrip("\n").split(",")
    print(f"  gate BUNDLE : {bundle.name} has {len(header)} columns")
    if len(header) != 27:
        sys.exit(f"ABORT: bundle has {len(header)} columns, expected 27 "
                 f"(21 canonical + 6 meta). A 33-metric bundle is the wrong data")
    return arm_dir / "listener"


def build_mode(dd, arm_dir: Path, mode: str) -> dict:
    print(f"\n=== {mode} ===")
    listener = gate_arm(arm_dir, mode)

    results = pd.read_csv(listener / "results" / mode / "results.csv")
    hits = sorted((listener / "results" / mode).glob("best_model_*.pkl"))
    if len(hits) != 1:
        sys.exit(f"ABORT: {len(hits)} model pickles under {listener}/results/{mode}")
    md = joblib.load(hits[0])
    all_models = md.get("all_models") or {}
    all_thr = md.get("all_thresholds") or {}
    if not all_models:
        sys.exit(f"ABORT: {hits[0].name} carries no 'all_models'")

    X_test, y_test, _ = rp.rebuild_test_split(dd, listener, mode)
    scaler, cols = md["scaler"], list(md["feature_names"])
    n_seen = getattr(scaler, "n_features_in_", None)
    print(f"  rebuilt test: {X_test.shape[0]} rows x {X_test.shape[1]} engineered "
          f"columns; scaler was fitted on {n_seen}; {len(cols)} selected")
    if n_seen is not None and n_seen != X_test.shape[1]:
        sys.exit(f"ABORT: scaler expects {n_seen} columns, rebuilt frame has "
                 f"{X_test.shape[1]}")

    X_scaled = pd.DataFrame(scaler.transform(X_test),
                            columns=X_test.columns, index=X_test.index)
    X_sel = X_scaled[cols]
    y_arr = y_test.to_numpy()

    per_model = {}
    for _, row in results.iterrows():
        name = str(row["Model"])
        est = all_models.get(name)
        if est is None:
            print(f"  ⚠️  {name}: not in all_models, skipped")
            continue
        thr = float(all_thr.get(name, row["Threshold"]))
        proba = est.predict_proba(X_sel)[:, 1]
        acc = float(((proba >= thr).astype(int) == y_arr).mean())
        auc = float(roc_auc_score(y_arr, proba))

        d_acc = acc - float(row["Accuracy"])
        d_auc = auc - float(row["AUC"])
        flag = "ok "
        if abs(d_acc) > 1e-9 or abs(d_auc) > 1e-9:
            flag = "⛔ "
        print(f"  {flag}{name:<22} acc {acc:.6f} (Δ{d_acc:+.1e})  "
              f"auc {auc:.6f} (Δ{d_auc:+.1e})")
        if flag != "ok ":
            sys.exit(f"ABORT [reproduction gate] {name}: the reconstruction does "
                     f"not match results.csv. Do not write a cache from it.")
        per_model[name] = {"scores_test": proba, "point_auc": auc,
                           "threshold": thr, "accuracy": acc}

    print(f"  cached {len(per_model)} models")
    return {"y_test": y_arr, "per_model": per_model,
            "arm": arm_dir.name, "mode": mode,
            "pipeline": os.environ["R23_PIPELINE"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="arms_r34_17feat_frozencal67")
    ap.add_argument("--out", default=None,
                    help="results_root for generate_roc_figure.py; default "
                         "<arm>/listener/results/roc_cache")
    args = ap.parse_args()

    arm_dir = HERE / args.arm
    if not arm_dir.is_dir():
        sys.exit(f"ABORT: no such arm {arm_dir}")

    pipeline = Path(os.environ["R23_PIPELINE"])
    print("=== gates ===")
    gate_pipeline(pipeline)

    dd = rp.load_pipeline_module()

    out_root = Path(args.out) if args.out else (
        arm_dir / "listener" / "results" / "roc_cache")
    for mode in ("static", "mobile"):
        cache = build_mode(dd, arm_dir, mode)
        dest = out_root / "hp_search_final" / mode
        dest.mkdir(parents=True, exist_ok=True)
        joblib.dump(cache, dest / "test_predictions_cache.pkl")
        print(f"  wrote {dest / 'test_predictions_cache.pkl'}")

    print(f"\n=== done. results_root for the figure: {out_root} ===")


if __name__ == "__main__":
    main()
