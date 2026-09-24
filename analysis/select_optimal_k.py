#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
select_optimal_k.py — pick the optimal K from a completed k_sweep run.

The paper's rule, as decided in README_project_summary part 21 (12/5/2026,
"החלטה 3: K=4 הוא ה-optimal האמיתי (לא K=5)"):

    the SMALLEST K whose performance is not distinguishable from the best.

There, K=5 scored marginally higher than K=4 on S->M (0.8658 vs 0.8639), but
the gap sat inside one standard deviation, so K=4 won on parsimony — and it
also had the lower cross-domain asymmetry (0.005 vs 0.014).

This script applies that rule mechanically instead of by eye:

  1. score(K) = mean of the two cross-domain accuracies (S->M, M->S)
     — cross-domain is what the paper's claim rests on; in-domain rows are
     reported alongside but do not drive the choice.
  2. best = max score. tolerance = the std of the best K (one-SD rule), or an
     explicit --tolerance.
  3. candidates = every K within tolerance of best.
  4. chosen K = the SMALLEST candidate. Ties on K are broken by lower
     asymmetry, exactly as the paper did.

Nothing about K is hard-coded: whatever the sweep produced is what gets chosen.

Usage:
  python3 select_optimal_k.py --k-sweep-csv <...>/k_sweep_universal4/k_sweep_results.csv
  # options: --tolerance 0.01   --metric acc|auc
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

CROSS = {"S_to_M", "M_to_S", "SM", "MS", "S->M", "M->S"}


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """k_sweep writes long-form rows: K, direction, acc_mean, acc_std, ..."""
    need = {"K", "direction"}
    if not need.issubset(df.columns):
        sys.exit(f"ERROR: expected columns {need}, got {list(df.columns)}")
    df = df.copy()
    df["direction"] = df["direction"].astype(str)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-sweep-csv", required=True)
    ap.add_argument("--metric", default="acc", choices=["acc", "auc"])
    ap.add_argument("--tolerance", type=float, default=None,
                    help="absolute tolerance; default = std of the best K (one-SD rule)")
    ap.add_argument("--out", default=None, help="where to write optimal_k.json")
    args = ap.parse_args()

    csv_path = Path(args.k_sweep_csv).resolve()
    if not csv_path.is_file():
        sys.exit(f"ERROR: not found: {csv_path}")
    df = normalise(pd.read_csv(csv_path))

    mean_col, std_col = f"{args.metric}_mean", f"{args.metric}_std"
    for c in (mean_col, std_col):
        if c not in df.columns:
            sys.exit(f"ERROR: column {c} missing; have {list(df.columns)}")

    cross = df[df["direction"].isin(CROSS)]
    if cross.empty:
        dirs = sorted(df["direction"].unique())
        sys.exit(f"ERROR: no cross-domain rows. directions present: {dirs}")

    # score per K = mean over the two cross-domain directions
    per_k = (cross.groupby("K")
                  .agg(score=(mean_col, "mean"),
                       spread=(std_col, "mean"),
                       n_rows=(mean_col, "size"))
                  .reset_index()
                  .sort_values("K"))

    # asymmetry = |S->M - M->S|, the paper's tie-break
    asym = {}
    for K, sub in cross.groupby("K"):
        vals = sub.groupby("direction")[mean_col].mean()
        asym[K] = float(abs(vals.iloc[0] - vals.iloc[1])) if len(vals) >= 2 else float("nan")
    per_k["asymmetry"] = per_k["K"].map(asym)

    best_row = per_k.loc[per_k["score"].idxmax()]
    tol = args.tolerance if args.tolerance is not None else float(best_row["spread"])
    cand = per_k[per_k["score"] >= best_row["score"] - tol].sort_values(
        ["K", "asymmetry"])
    chosen = cand.iloc[0]

    print("=" * 78)
    print(f"K selection — metric={args.metric}, cross-domain mean over "
          f"{sorted(cross['direction'].unique())}")
    print("=" * 78)
    print(f"{'K':>5}  {'score':>8}  {'sd':>7}  {'asym':>7}   within tol?")
    print("-" * 78)
    for _, r in per_k.iterrows():
        mark = "  <-- CHOSEN" if int(r["K"]) == int(chosen["K"]) else ""
        inside = "yes" if r["score"] >= best_row["score"] - tol else "no"
        peak = " *peak*" if int(r["K"]) == int(best_row["K"]) else ""
        print(f"{int(r['K']):>5}  {r['score']:>8.4f}  {r['spread']:>7.4f}  "
              f"{r['asymmetry']:>7.4f}   {inside}{peak}{mark}")
    print("-" * 78)
    print(f"  peak K       = {int(best_row['K'])}  (score {best_row['score']:.4f})")
    print(f"  tolerance    = {tol:.4f} "
          f"({'explicit' if args.tolerance is not None else 'one-SD of peak'})")
    print(f"  OPTIMAL K    = {int(chosen['K'])}  "
          f"(score {chosen['score']:.4f}, asymmetry {chosen['asymmetry']:.4f})")
    print(f"  rule: smallest K within tolerance of the peak; ties -> lower asymmetry")

    out = Path(args.out) if args.out else csv_path.parent / "optimal_k.json"
    out.write_text(json.dumps({
        "optimal_K": int(chosen["K"]),
        "peak_K": int(best_row["K"]),
        "metric": args.metric,
        "score_at_optimal": float(chosen["score"]),
        "score_at_peak": float(best_row["score"]),
        "tolerance": tol,
        "tolerance_source": "explicit" if args.tolerance is not None else "one_sd_of_peak",
        "asymmetry_at_optimal": float(chosen["asymmetry"]),
        "per_k": per_k.to_dict(orient="records"),
        "source_csv": str(csv_path),
    }, indent=2))
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
