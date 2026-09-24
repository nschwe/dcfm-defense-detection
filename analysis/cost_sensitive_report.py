#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cost_sensitive_report.py — answers R#2's request for threshold-free and
cost-sensitive reporting (STATE §25.10, line 3396):

    "...a brief, more explicit explanation of why accuracy is the primary
     criterion for threshold tuning and what the results would look like with
     threshold-free decision reporting or cost-sensitive settings."

Nothing is retrained. Part A needs only the confusion counts already present
in every results.csv row. Part B re-scores stored probabilities from the saved
model pickle to trace cost against threshold.

Cost model: a false positive (flagging an undefended network as defended)
costs 1; a false negative (missing an active defence) costs C. Total expected
cost per window is (FP + C*FN) / N. C = 1 is symmetric, i.e. plain error rate.

Usage
-----
  python3 cost_sensitive_report.py --results <arm>/results/<mode>/results.csv
  python3 cost_sensitive_report.py --results ... --sweep      # adds part B
  # options: --costs 0.25,0.5,1,2,4,10   --model Stacking_Ensemble
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

DEFAULT_COSTS = [0.25, 0.5, 1.0, 2.0, 4.0, 10.0]


def read_results(path: str) -> list[dict]:
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        sys.exit(f"ERROR: no rows in {path}")
    need = {"Model", "TN", "FP", "FN", "TP"}
    missing = need - set(rows[0])
    if missing:
        sys.exit(f"ERROR: {path} lacks {sorted(missing)}")
    return rows


def part_a(rows: list[dict], costs: list[float]) -> None:
    """Expected cost at each model's own operating point — confusion counts only."""
    print("=" * 100)
    print("PART A — expected cost per window at the operating point already reported")
    print("         (no retraining, no rescoring: derived from TN/FP/FN/TP)")
    print("=" * 100)
    head = f"{'model':<22}{'thr':>6}{'acc':>8}" + "".join(f"{'C=' + str(c):>9}" for c in costs)
    print(head)
    print("-" * len(head))
    best_per_cost = {c: (None, float("inf")) for c in costs}
    for r in rows:
        tn, fp, fn, tp = (int(r[k]) for k in ("TN", "FP", "FN", "TP"))
        n = tn + fp + fn + tp
        acc = (tn + tp) / n if n else float("nan")
        thr = float(r.get("Threshold", 0.5))
        line = f"{r['Model']:<22}{thr:>6.2f}{acc:>8.4f}"
        for c in costs:
            cost = (fp + c * fn) / n if n else float("nan")
            line += f"{cost:>9.4f}"
            if cost < best_per_cost[c][1]:
                best_per_cost[c] = (r["Model"], cost)
        print(line)
    print("-" * len(head))
    print("lowest-cost model per C:")
    for c in costs:
        m, v = best_per_cost[c]
        print(f"    C={c:<5} {m:<24} cost={v:.4f}")
    print()
    print("  C is the cost of a missed defence relative to a false alarm.")
    print("  C=1 reduces to the error rate, so that column mirrors accuracy.")


def part_b(results_path: str, model_name: str, costs: list[float]) -> None:
    """Cost against threshold, from stored probabilities. Requires the pkl."""
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import GroupShuffleSplit

    res_dir = os.path.dirname(os.path.abspath(results_path))
    mode = os.path.basename(res_dir)
    arm_root = os.path.dirname(os.path.dirname(res_dir))
    here = os.path.dirname(os.path.abspath(__file__))
    # COST_PIPELINE selects the pipeline copy whose feature space part B must
    # reproduce. ⛔ This is NOT cosmetic: DefenseDetector.METRICS is a CLASS
    # constant, and it differs between copies -- analysis/pipeline carries the
    # stale 33-name network-wide list, frozencal/pipeline_17 the listener's 17.
    # self.metrics drives which log1p/sqrt columns are generated and which
    # columns feed the row-wise statistics, so the wrong copy engineers a
    # different feature set than the arm's frozen scaler expects.
    # Default unchanged, so existing invocations reproduce.
    pipe_dir = os.environ.get("COST_PIPELINE") or os.path.join(here, "pipeline")
    if not os.path.isfile(os.path.join(pipe_dir, "defense_detection_v2.py")):
        sys.exit(f"ERROR: no defense_detection_v2.py under {pipe_dir}")
    sys.path.insert(0, pipe_dir)
    print(f"  [part B] pipeline: {pipe_dir}")

    pkls = [f for f in os.listdir(res_dir)
            if f.startswith("best_model_") and f.endswith(".pkl")]
    if not pkls:
        print(f"\n[part B skipped] no best_model_*.pkl in {res_dir}")
        return
    md = joblib.load(os.path.join(res_dir, pkls[0]))
    models = md.get("all_models") or {}
    if model_name not in models:
        print(f"\n[part B skipped] {model_name!r} not in {pkls[0]}")
        return

    import defense_detection_v2 as dd
    os.environ["DCFM_DATA_BUNDLE"] = os.path.join(arm_root, "colab_data")
    cfg = dd.Config(data_root=f"./features_{mode}/",
                    results_dir="/tmp/_cost", verbose=0)
    cfg.group_split_by_file_source = True
    pipe = dd.DefenseDetector(cfg)
    X, y, groups = pipe.preprocess_data_enhanced(pipe.load_simulation_data_enhanced())
    Xe = pipe.engineer_advanced_features(X)
    g = np.array(groups)
    _, test_idx = next(GroupShuffleSplit(
        n_splits=1, test_size=cfg.test_size,
        random_state=cfg.random_state).split(Xe, y, g))

    feats = list(md["feature_names"])
    sc = md["scaler"]
    sf = list(sc.feature_names_in_) if hasattr(sc, "feature_names_in_") else feats
    # Gate: every column the arm's frozen scaler was fitted on must exist in the
    # space just engineered. If any is absent the pipeline copy is the wrong one
    # and the scores would come from a different feature space -- stop, do not
    # score. (Without this the failure is a bare pandas KeyError, or worse, a
    # silent mismatch when the missing names happen to be engineered anyway.)
    absent = [c for c in sf if c not in Xe.columns]
    print(f"  [part B] base metrics {len(pipe.metrics)} -> engineered {Xe.shape[1]}; "
          f"scaler expects {len(sf)}")
    if absent:
        sys.exit(
            f"ERROR: {len(absent)} of {len(sf)} scaler features are absent from the "
            f"engineered space, e.g. {absent[:5]}.\n"
            f"       The pipeline copy does not match the arm. Set COST_PIPELINE to "
            f"the copy the arm was learned with\n"
            f"       (the corrected campaign: analysis/frozencal/pipeline_17).")
    Xs = pd.DataFrame(sc.transform(Xe[sf]), columns=sf, index=Xe.index)
    Xt = Xs.iloc[test_idx][feats]
    yt = np.asarray(y)[test_idx]
    prob = models[model_name].predict_proba(Xt)[:, 1]

    print()
    print("=" * 100)
    print(f"PART B — cost-minimising threshold per cost ratio ({model_name}, {mode})")
    print("         probabilities rescored from the saved model; nothing retrained")
    print("=" * 100)
    grid = np.arange(0.05, 0.96, 0.01)
    print(f"{'C':>7}{'best thr':>11}{'cost':>10}{'acc there':>12}"
          f"{'cost@0.5':>11}{'saving':>10}")
    print("-" * 61)
    for c in costs:
        best_t, best_cost = 0.5, float("inf")
        for t in grid:
            pred = (prob >= t).astype(int)
            fp = int(((pred == 1) & (yt == 0)).sum())
            fn = int(((pred == 0) & (yt == 1)).sum())
            cost = (fp + c * fn) / len(yt)
            if cost < best_cost:
                best_cost, best_t = cost, float(t)
        pred = (prob >= best_t).astype(int)
        acc = float((pred == yt).mean())
        p05 = (prob >= 0.5).astype(int)
        fp5 = int(((p05 == 1) & (yt == 0)).sum())
        fn5 = int(((p05 == 0) & (yt == 1)).sum())
        c05 = (fp5 + c * fn5) / len(yt)
        print(f"{c:>7}{best_t:>11.2f}{best_cost:>10.4f}{acc:>12.4f}"
              f"{c05:>11.4f}{c05 - best_cost:>10.4f}")
    print("-" * 61)
    print("  'saving' is what tuning the threshold to the cost ratio would buy")
    print("  over the fixed 0.5 the paper reports. Small savings support the")
    print("  choice of accuracy as the tuning criterion.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="path to a results.csv")
    ap.add_argument("--costs", default=",".join(str(c) for c in DEFAULT_COSTS))
    ap.add_argument("--model", default="Stacking_Ensemble")
    ap.add_argument("--sweep", action="store_true",
                    help="also trace cost against threshold (needs the pkl)")
    args = ap.parse_args()

    costs = [float(c) for c in args.costs.split(",") if c.strip()]
    rows = read_results(args.results)
    print(f"source: {args.results}\n")
    part_a(rows, costs)
    if args.sweep:
        part_b(args.results, args.model, costs)


if __name__ == "__main__":
    main()
