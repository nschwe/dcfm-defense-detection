#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
make_colab_dataset.py
============================================================================
Build a compact, git-shippable dataset bundle for the Colab notebook.

The raw simulation output is ~80,080 tiny "tall" CSV files (one per
measurement window) totalling ~314 MB across two configurations. Uploading
that to git / Colab is impractical. This script collapses each configuration
into a single WIDE parquet table (one row per measurement window, all 46
base metrics pivoted to columns), which is ~5 MB per config and a drop-in
replacement for DefenseDetector.load_simulation_data_enhanced().

It also copies the small support files and the cheap caches needed for the
Tier-1 (no-retrain) reproduction path in the notebook.

The huge trained-model pickles (best_models.pkl, ~18.7 GB) are intentionally
NOT copied: the notebook either reproduces Table 4 from the shipped result
CSVs / prediction caches (default) or retrains from scratch (RETRAIN=True).

Output layout (./colab_data/):
    wide_static.parquet
    wide_mobile.parquet
    support/topology_probes_static.csv      (Table 2, accepted-vs-rejected)
    support/topology_probes_mobile.csv
    support/run_status_static.csv
    support/run_status_mobile.csv
    cache/feature_importance/...            (Universal-4 selection, §6.5)
    cache/hp_search_final/<cfg>/results...  (Table 4 numbers, ROC, CI)
    MANIFEST.txt

Run from inside strict_observable_v2/.  For speed run it INSIDE WSL (the raw
files are local there); over the Windows 9p mount it is much slower.
============================================================================
"""
import os
import sys
import glob
import shutil
import argparse

import pandas as pd

# Mirrors DefenseDetector.load_simulation_data_enhanced scenario->label map.
SCENARIOS = {
    "baseline": 0,
    "attack_only": 0,
    "defense_only": 1,
    "defense_vs_attack": 1,
}

# Columns of the wide table that are NOT pivoted metric values.
META_COLS = ["scenario", "file_source", "defense_active",
             "_Duration", "_StartTime", "_EndTime"]


def build_wide(data_root: str, verbose: bool = True) -> pd.DataFrame:
    """Pivot every per-window tall CSV under data_root into one wide row.

    Each raw file is a single measurement window: the rows are (Metric, Value)
    pairs with constant StartTime/EndTime/Duration columns. We pivot
    Metric -> column so that one file becomes one row, preserving scenario,
    file_source (basename) and the defense_active label.
    """
    rows = []
    n_read, n_bad = 0, 0
    for scenario, label in SCENARIOS.items():
        sdir = os.path.join(data_root, scenario)
        if not os.path.isdir(sdir):
            if verbose:
                print(f"  [skip] missing scenario dir: {sdir}")
            continue
        files = sorted(glob.glob(os.path.join(sdir, "*.csv")))
        if verbose:
            print(f"  {scenario}: {len(files)} files")
        for fp in files:
            try:
                df = pd.read_csv(fp)
                # Average duplicate metric rows defensively (normally 1 each),
                # matching preprocess_data_enhanced's group-mean semantics.
                wide = df.groupby("Metric")["Value"].mean().to_dict()
            except Exception:
                n_bad += 1
                continue
            # Window timing is constant within a file; carry the mean.
            if "Duration" in df.columns:
                wide["_Duration"] = float(df["Duration"].mean())
            if "StartTime" in df.columns:
                wide["_StartTime"] = float(df["StartTime"].mean())
            if "EndTime" in df.columns:
                wide["_EndTime"] = float(df["EndTime"].mean())
            wide["scenario"] = scenario
            wide["file_source"] = os.path.basename(fp)
            wide["defense_active"] = label
            rows.append(wide)
            n_read += 1
    if verbose:
        print(f"  -> {n_read} windows pivoted ({n_bad} unreadable)")
    wide_df = pd.DataFrame(rows)
    # Stable column order: metrics (sorted) then meta.
    metric_cols = sorted(c for c in wide_df.columns if c not in META_COLS)
    return wide_df[metric_cols + [c for c in META_COLS if c in wide_df.columns]]


def save_wide(wide: pd.DataFrame, out: str, cfg: str) -> str:
    """Write the wide table as parquet if an engine is available, else csv.gz.

    csv.gz needs no extra dependency and is read transparently by the notebook
    loader, which auto-detects the extension.
    """
    pq = os.path.join(out, f"wide_{cfg}.parquet")
    try:
        wide.to_parquet(pq, index=False)
        return pq
    except Exception as e:
        print(f"    [parquet unavailable: {e}; falling back to csv.gz]")
        gz = os.path.join(out, f"wide_{cfg}.csv.gz")
        wide.to_csv(gz, index=False, compression="gzip")
        return gz


def copy_if_exists(src: str, dst: str, verbose: bool = True) -> bool:
    if os.path.exists(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
        if verbose:
            print(f"  [copy] {src} -> {dst}")
        return True
    if verbose:
        print(f"  [miss] {src}")
    return False


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def main():
    ap = argparse.ArgumentParser(description="Build git-shippable Colab dataset bundle.")
    ap.add_argument("--sim-root", default="../simulations",
                    help="Path to simulations/ (default: ../simulations).")
    ap.add_argument("--out", default="./colab_data",
                    help="Output bundle directory (default: ./colab_data).")
    ap.add_argument("--results-root", default="./results",
                    help="Path to results/ for cheap caches (default: ./results).")
    args = ap.parse_args()

    sim_root = args.sim_root
    out = args.out
    os.makedirs(out, exist_ok=True)

    manifest = []

    # --- 1. Wide parquet per configuration ---------------------------------
    for cfg, sub in (("static", "features_static"), ("mobile", "features_mobile")):
        data_root = os.path.join(sim_root, sub)
        print(f"\n[1] Aggregating {cfg} from {data_root} ...")
        wide = build_wide(data_root)
        pq = save_wide(wide, out, cfg)
        sz = os.path.getsize(pq)
        print(f"    wrote {pq}  shape={wide.shape}  size={human(sz)}")
        manifest.append(f"{os.path.basename(pq)}  shape={wide.shape}  {human(sz)}")

    # --- 2. Small support files (Table 2 / Sec 4.2) ------------------------
    print("\n[2] Copying support CSVs ...")
    for name in ("topology_probes_static.csv", "topology_probes_mobile.csv",
                 "run_status_static.csv", "run_status_mobile.csv"):
        if copy_if_exists(os.path.join(sim_root, name),
                          os.path.join(out, "support", name)):
            manifest.append(f"support/{name}")

    # --- 3. Cheap caches for Tier-1 (no-retrain) reproduction --------------
    print("\n[3] Copying cheap caches ...")
    rr = args.results_root
    # Universal-4 importance cache (Sec 6.5)
    copy_if_exists(os.path.join(rr, "feature_importance", "importance_cache"),
                   os.path.join(out, "cache", "feature_importance", "importance_cache"))
    # Per-config small artifacts: scaler, prediction cache, result CSVs/JSON.
    for cfg in ("static", "mobile"):
        base = os.path.join(rr, "hp_search_final", cfg)
        dst = os.path.join(out, "cache", "hp_search_final", cfg)
        for fn in ("scaler_v2.pkl", "test_predictions_cache.pkl",
                   "combined_results.csv", f"combined_final_{cfg}.csv",
                   "stacking_summary.txt"):
            copy_if_exists(os.path.join(base, fn), os.path.join(dst, fn))

    # --- 4. Manifest -------------------------------------------------------
    with open(os.path.join(out, "MANIFEST.txt"), "w", encoding="utf-8") as fh:
        fh.write("Colab dataset bundle for DCFM defense-detection notebook\n")
        fh.write("=" * 60 + "\n\n")
        fh.write("\n".join(manifest) + "\n")
    print(f"\n[done] bundle at {out}")


if __name__ == "__main__":
    main()
