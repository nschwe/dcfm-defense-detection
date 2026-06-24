#!/usr/bin/env python3
"""
Targeted hyperparameter grid extension for models with edge warnings.

This script runs RandomizedSearchCV with extended grids for the specific
parameters that hit grid boundaries in the main extended run.
It does NOT modify or overwrite any files in:
  - results/hp_search_extended/
  - results/hp_search_focused/
  - results/hp_search_final/
  - unified_hp_search_v2.py

Output: results/hp_search_edge_extension/{static,mobile}/

USAGE:

  # 1. Dry-run first (no tuning, validates everything):
  cd ~/ns3/Final_Project_NS3-master/strict_observable_v2
  MAX_JOBS=12 python3 -u extend_hp_grids.py --dry-run

  # 2. After dry-run passes, run for real:
  cd ~/ns3/Final_Project_NS3-master/strict_observable_v2
  OMP_NUM_THREADS=12 MAX_JOBS=12 \
      nohup python3 -u extend_hp_grids.py > extend_hp_grids.log 2>&1 &
  disown
"""

import argparse
import os
import sys
import json
import time
from pathlib import Path

# CRITICAL: MAX_JOBS env var must be set BEFORE importing the modules below,
# because N_JOBS_OUTER is captured at import time.
if "MAX_JOBS" not in os.environ:
    print("[fatal] MAX_JOBS env var must be set before running this script.")
    print("        Try: MAX_JOBS=12 python3 extend_hp_grids.py")
    sys.exit(1)

print(f"[setup] MAX_JOBS env: {os.environ['MAX_JOBS']}")
print(f"[setup] cwd: {os.getcwd()}")

# Make sure we can import from the strict_observable_v2 directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Imports
import numpy as np
import pandas as pd
import joblib

# Import the unified_hp_search_v2 module so we can override N_ITER_RANDOM globally
import unified_hp_search_v2 as u
# Override n_iter to match the original paper run (n_iter=80, not the default 30)
u.N_ITER_RANDOM = 80
print(f"[setup] u.N_ITER_RANDOM overridden to {u.N_ITER_RANDOM}")
print(f"[setup] u.N_JOBS_OUTER = {u.N_JOBS_OUTER}")
print(f"[setup] u.RANDOM_STATE = {u.RANDOM_STATE}")

# Config comes from defense_detection_v2 (NOT unified_hp_search_v2)
from defense_detection_v2 import Config


# ----------------------------------------------------------------------
# Custom grids — only the targeted edge parameters are extended.
# Everything else matches the existing 'extended' grid in unified_hp_search_v2.
# ----------------------------------------------------------------------
CUSTOM_GRIDS = {
    "static": {
        "randomforest": {
            "n_estimators":      [300, 500, 800, 1200, 1600],
            "max_depth":         [10, 15, 20, 30, None],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf":  [1, 2, 5, 10, 15, 20, 30],   # +15,20,30 vs extended
            "max_features":      ["sqrt", "log2", 0.3, 0.5, 0.7],
        },
        "extratrees": {
            "n_estimators":      [100, 150, 200, 300, 500, 800, 1200, 1600],  # +100,150,200
            "max_depth":         [10, 15, 20, 30, None],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf":  [1, 2, 5, 10],
            "max_features":      ["sqrt", "log2", 0.3, 0.5, 0.7],
        },
        "bagging_et": {
            "n_estimators":              [10, 15, 20, 30, 50, 100, 200],   # +10,15
            "max_samples":               [0.3, 0.5, 0.7, 0.8, 1.0],
            "estimator__n_estimators":   [10, 20, 30, 50, 75, 100],        # +75,100
            "estimator__max_depth":      [10, 15, 20, None],
        },
    },
    "mobile": {
        "randomforest": {
            "n_estimators":      [300, 500, 800, 1200, 1600],
            "max_depth":         [10, 15, 20, 30, None],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf":  [1, 2, 5, 10],
            "max_features":      ["sqrt", "log2", 0.3, 0.5, 0.7, 0.85, 1.0],  # +0.85,1.0
        },
        "extratrees": {
            "n_estimators":      [300, 500, 800, 1200, 1600],
            "max_depth":         [10, 15, 20, 30, None],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf":  [1, 2, 5, 10],
            "max_features":      ["sqrt", "log2", 0.3, 0.5, 0.7, 0.85, 1.0],  # +0.85,1.0
        },
    },
}


def make_config(data_root: str, results_dir: str) -> Config:
    """Build a Config matching the original paper run."""
    cfg = Config()
    cfg.data_root = data_root
    cfg.results_dir = results_dir
    # Defaults already set in defense_detection_v2.py:
    #   random_state = 42, test_size = 0.20, group_split_by_file_source = True,
    #   n_features_target = 150, etc.
    return cfg


def backup_existing_outputs():
    """If hp_search_edge_extension already exists, back it up."""
    out_dir = Path("results/hp_search_edge_extension")
    if out_dir.exists():
        ts = time.strftime("%Y%m%d_%H%M%S")
        backup = out_dir.parent / f"hp_search_edge_extension.backup_{ts}"
        print(f"[backup] {out_dir} -> {backup}")
        out_dir.rename(backup)
        return backup
    return None


def run_for_config(label: str, data_root: str, out_dir: Path, dry_run: bool = False):
    print(f"\n{'='*70}\n  {label.upper()}\n{'='*70}")

    cfg = make_config(data_root, str(out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[prepare_data] Loading and preprocessing for {label}...")
    t_prep_start = time.time()
    data = u.prepare_data(cfg)
    t_prep = time.time() - t_prep_start
    print(f"  prepare_data took {t_prep:.1f}s")

    # Sanity-check the keys we expect
    required_keys = ["X_train_smote", "y_train_smote", "X_test", "y_test", "feature_names"]
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise RuntimeError(f"prepare_data did not return expected keys: missing {missing}")

    X_tr, y_tr = data["X_train_smote"], data["y_train_smote"]
    X_test, y_test = data["X_test"], data["y_test"]
    feature_names = data["feature_names"]
    print(f"  X_train_smote: {X_tr.shape}")
    print(f"  X_test:        {X_test.shape}")
    print(f"  features:      {len(feature_names)}")

    # Cross-check feature_names against the previously trained models
    prev_pkl_path = Path(f"results/hp_search_final/{label}/best_models.pkl")
    if prev_pkl_path.exists():
        try:
            prev_models = joblib.load(prev_pkl_path)
            # Pick any base model (not Stacking_Ensemble) to inspect
            for n, m in prev_models.items():
                if n == "Stacking_Ensemble":
                    continue
                if hasattr(m, "feature_names_in_"):
                    prev_feats = list(m.feature_names_in_)
                    if set(prev_feats) == set(feature_names):
                        print(f"  feature_names match prev ({n}): YES")
                    else:
                        print(f"  feature_names match prev ({n}): NO")
                        print(f"    prev only: {sorted(set(prev_feats) - set(feature_names))[:5]}")
                        print(f"    new only:  {sorted(set(feature_names) - set(prev_feats))[:5]}")
                    break
        except Exception as e:
            print(f"  [warn] cross-check failed: {e}")
    else:
        print(f"  [warn] {prev_pkl_path} not found — skipping cross-check")

    grids = CUSTOM_GRIDS[label]
    print(f"\nModels to extend: {list(grids.keys())}")

    if dry_run:
        print("\n[dry-run] Would run the following searches:")
        for name, space in grids.items():
            n_combos = int(np.prod([len(v) for v in space.values()]))
            n_iter = min(u.N_ITER_RANDOM, n_combos)
            print(f"  {name}: n_iter={n_iter} of {n_combos} combos")
            for k, vals in space.items():
                print(f"    {k} ({len(vals)} values): {vals}")
        return None

    results = {}
    for name, space in grids.items():
        print(f"\n{'-'*60}\n  Running {name}\n{'-'*60}")
        n_combos = int(np.prod([len(v) for v in space.values()]))
        n_iter = min(u.N_ITER_RANDOM, n_combos)
        print(f"  Grid: {n_combos} combos, sampling n_iter={n_iter}")

        try:
            search, elapsed = u.run_search_for_model(name, space, X_tr, y_tr)
        except Exception as e:
            print(f"  [FAILED] {name}: {e}")
            results[name] = {"error": str(e)}
            with open(out_dir / "extension_results.json", "w") as f:
                json.dump(results, f, indent=2, default=str)
            continue

        best = search.best_estimator_
        test_score = float(best.score(X_test, y_test))

        results[name] = {
            "best_params":   {k: (str(v) if v is None else v) for k, v in search.best_params_.items()},
            "best_cv_score": float(search.best_score_),
            "test_accuracy": test_score,
            "elapsed_sec":   float(elapsed),
        }

        print(f"  best_cv_score: {search.best_score_:.4f}")
        print(f"  test_accuracy: {test_score:.4f}")
        print(f"  elapsed:       {elapsed:.1f}s")
        print(f"  best_params:   {search.best_params_}")

        # Compare to previous run's test_accuracy from CSV (NOT by re-scoring,
        # to avoid feature-mismatch issues)
        prev_acc = _read_prev_test_accuracy(label, name)
        if prev_acc is not None:
            delta = test_score - prev_acc
            sign = "+" if delta >= 0 else ""
            note = ""
            if abs(delta) < 0.005:
                note = " (within noise)"
            elif delta >= 0.005:
                note = " ** improvement >= 0.005 **"
            else:
                note = " ** regression <= -0.005 **"
            print(f"  vs previous:   prev={prev_acc:.4f}  new={test_score:.4f}  "
                  f"delta={sign}{delta:.4f}{note}")
            results[name]["prev_test_accuracy"] = prev_acc
            results[name]["delta_vs_prev"] = float(delta)

        # Save best model for this single classifier
        joblib.dump(best, out_dir / f"best_model_{name}.pkl")

        # Incremental save in case the next model crashes
        with open(out_dir / "extension_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

    print(f"\n[saved] {out_dir / 'extension_results.json'}")
    return results


def _read_prev_test_accuracy(label: str, model_name: str):
    """Read previous test_accuracy from combined_results.csv files."""
    import csv
    candidates = [
        f"results/hp_search_final/{label}/combined_results.csv",
        f"results/hp_search_extended/{label}/combined_results.csv",
    ]
    for path in candidates:
        if not Path(path).exists():
            continue
        try:
            with open(path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("model") == model_name:
                        val = row.get("test_accuracy")
                        if val and val.strip():
                            try:
                                return float(val)
                            except ValueError:
                                pass
        except Exception:
            continue
    return None


def print_summary(all_results):
    """Print a cross-config summary."""
    print("\n\n")
    print("="*78)
    print("  SUMMARY: edge-extension search vs previous best (test accuracy)")
    print("="*78)
    print(f"{'config':<10s}{'model':<20s}{'previous':>12s}{'new':>12s}{'delta':>10s}{'note':>14s}")
    print("-"*78)

    for label, results in all_results.items():
        if results is None:
            continue
        for name, info in results.items():
            if "error" in info:
                print(f"{label:<10s}{name:<20s}{'?':>12s}{'ERROR':>12s}{'?':>10s}{'failed':>14s}")
                continue
            new_acc = info["test_accuracy"]
            prev_acc = info.get("prev_test_accuracy")
            delta = info.get("delta_vs_prev")
            if prev_acc is None:
                print(f"{label:<10s}{name:<20s}{'?':>12s}{new_acc:>12.4f}{'?':>10s}")
            else:
                sign = "+" if delta >= 0 else ""
                if abs(delta) < 0.005:
                    note = "within noise"
                elif delta >= 0.005:
                    note = "improvement"
                else:
                    note = "regression"
                print(f"{label:<10s}{name:<20s}{prev_acc:>12.4f}{new_acc:>12.4f}"
                      f"{sign+f'{delta:.4f}':>10s}{note:>14s}")

    print("\nFull JSON: results/hp_search_edge_extension/{static,mobile}/extension_results.json")
    print("Best models saved as: results/hp_search_edge_extension/{static,mobile}/best_model_<name>.pkl")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Validate setup without running tuning")
    ap.add_argument("--config", choices=["static", "mobile", "both"], default="both",
                    help="Which config to run")
    args = ap.parse_args()

    base_out = Path("results/hp_search_edge_extension")

    # Backup any prior run output
    if not args.dry_run:
        backup_existing_outputs()

    base_out.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print(f"  TARGETED EDGE-EXTENSION GRID SEARCH ({'DRY RUN' if args.dry_run else 'LIVE'})")
    print("="*70)

    all_results = {}
    if args.config in ("static", "both"):
        all_results["static"] = run_for_config(
            "static", "../simulations/features_static",
            base_out / "static", dry_run=args.dry_run
        )
    if args.config in ("mobile", "both"):
        all_results["mobile"] = run_for_config(
            "mobile", "../simulations/features_mobile",
            base_out / "mobile", dry_run=args.dry_run
        )

    if not args.dry_run:
        print_summary(all_results)
    else:
        print("\n[dry-run complete] No tuning was performed. Re-run without --dry-run to execute.")