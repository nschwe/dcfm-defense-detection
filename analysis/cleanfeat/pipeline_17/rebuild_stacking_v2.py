#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
rebuild_stacking_v2.py
============================================================================

Rebuild the Stacking_Ensemble using:
    - 11 base learners from hp_search_extended/best_models.pkl
    - SVM from hp_search_focused/best_models.pkl (replaces extended SVM)
    - LogisticRegression is excluded (no predict_proba in our stacking design,
      and was excluded in the original run too)

Both static and mobile configs are processed.

Output files (per config) in --out-dir/<config>/:
    - best_models.pkl       (12 base learners + Stacking_Ensemble)
    - combined_results.csv  (one row per model, with Stacking added/updated)
    - stacking_summary.txt  (human-readable summary of base learners used)

============================================================================
"""

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# Match environment setup of unified_hp_search_v2 / defense_detection_v2
import os
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score)

# Import v2 detector (uses 33 features)
from defense_detection_v2 import DefenseDetector, Config

RANDOM_STATE = 42


# ============================================================
# DATA PREP — duplicated from unified_hp_search_v2.prepare_data
# ============================================================
def prepare_data(data_root: str) -> Dict[str, Any]:
    """Replicates the exact 60/20/20 split + scaling + feature selection
    used by unified_hp_search_v2 and defense_detection_v2."""
    print(f"\nPreparing data from: {data_root}")

    config = Config()
    config.data_root = data_root
    config.random_state = RANDOM_STATE
    config.test_size = 0.2

    detector = DefenseDetector(config)
    tall_df = detector.load_simulation_data_enhanced()
    X, y, groups = detector.preprocess_data_enhanced(tall_df)
    X_eng = detector.engineer_advanced_features(X)

    groups_arr = np.array(groups)
    outer = GroupShuffleSplit(n_splits=1, test_size=config.test_size,
                              random_state=config.random_state)
    train_val_idx, test_idx = next(outer.split(X_eng, y, groups_arr))

    X_tv, y_tv = X_eng.iloc[train_val_idx], y.iloc[train_val_idx]
    X_test, y_test = X_eng.iloc[test_idx], y.iloc[test_idx]
    groups_tv = groups_arr[train_val_idx]

    inner = GroupShuffleSplit(n_splits=1, test_size=0.25,
                              random_state=config.random_state)
    train_idx, val_idx = next(inner.split(X_tv, y_tv, groups_tv))

    X_train = X_tv.iloc[train_idx]
    y_train = y_tv.iloc[train_idx]

    print(f"  Split sizes: train={len(X_train)}, test={len(X_test)}")
    print(f"  Train class balance: {y_train.value_counts().to_dict()}")

    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train),
                                   columns=X_train.columns,
                                   index=X_train.index)
    X_test_scaled  = pd.DataFrame(scaler.transform(X_test),
                                   columns=X_test.columns,
                                   index=X_test.index)

    X_train_sel, _ = detector.select_features_intelligent(
        X_train_scaled, X_test_scaled, y_train
    )
    X_test_sel = X_test_scaled[detector.feature_names]
    print(f"  Features selected: {len(detector.feature_names)}")

    X_train_smote, y_train_smote = detector.augment_data_aggressive(
        X_train_sel, y_train
    )
    print(f"  SMOTE: {len(X_train_sel)} -> {len(X_train_smote)} samples")

    return {
        "X_train": X_train_smote,
        "y_train": y_train_smote,
        "X_test":  X_test_sel,
        "y_test":  y_test,
        "feature_names": detector.feature_names,
    }


# ============================================================
# STACKING BUILDER — duplicated from unified_hp_search_v2
# ============================================================
def build_stacking_with_tuned_bases(tuned_models: Dict[str, Any], n_jobs: int = 16) -> Any:
    """Build Stacking_Ensemble from tuned base learners that have predict_proba.

    Note: original unified_hp_search_v2 used n_jobs=1 to avoid OOM with full HP search.
    Here we override to n_jobs=16 because we are only building Stacking once
    (no parallel HP search running), so we can use all cores.
    """
    base_pairs: List[Tuple[str, Any]] = []
    for n, m in tuned_models.items():
        if n == "Stacking_Ensemble":
            continue
        if hasattr(m, "predict_proba"):
            base_pairs.append((n, m))

    if not base_pairs:
        raise RuntimeError("No tuned base learners available for Stacking.")

    meta = LogisticRegression(C=1.0, penalty="l2",
                               max_iter=5000, n_jobs=1,
                               random_state=RANDOM_STATE)
    return StackingClassifier(
        estimators=base_pairs,
        final_estimator=meta,
        cv=5,
        stack_method="predict_proba",
        n_jobs=n_jobs,
        passthrough=False,
    )


# ============================================================
# EVALUATION — duplicated from unified_hp_search_v2
# ============================================================
def evaluate_on_test(model, X_test, y_test) -> Dict[str, float]:
    """Compute test-set metrics."""
    y_pred = model.predict(X_test)
    try:
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            y_score = model.decision_function(X_test)
    except Exception:
        y_score = None

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, zero_division=0)
    n = len(y_test)
    se = np.sqrt(acc * (1 - acc) / n)
    ci_low  = max(0.0, acc - 1.96 * se)
    ci_high = min(1.0, acc + 1.96 * se)

    auc = None
    if y_score is not None:
        try:
            auc = float(roc_auc_score(y_test, y_score))
        except Exception:
            auc = None

    return {
        "test_accuracy": float(acc),
        "test_ci_low":   float(ci_low),
        "test_ci_high":  float(ci_high),
        "test_auc":      auc,
        "test_f1":       float(f1),
    }


# ============================================================
# MAIN PROCESSING PER CONFIG
# ============================================================
def process_config(config_name: str,
                   data_root: str,
                   extended_dir: Path,
                   focused_dir: Path,
                   out_dir: Path,
                   n_jobs: int = 16) -> None:
    """Process one configuration (static or mobile)."""
    print("\n" + "=" * 70)
    print(f"REBUILDING STACKING — {config_name}")
    print("=" * 70)

    out_config_dir = out_dir / config_name
    out_config_dir.mkdir(parents=True, exist_ok=True)

    # Load extended best_models (12 base learners)
    extended_pkl = extended_dir / config_name / "best_models.pkl"
    if not extended_pkl.exists():
        raise FileNotFoundError(f"Missing: {extended_pkl}")
    print(f"\nLoading extended models from: {extended_pkl}")
    with extended_pkl.open("rb") as f:
        extended_models = pickle.load(f)
    print(f"  Loaded {len(extended_models)} models from extended:")
    for name in extended_models:
        print(f"    - {name}")

    # Load focused best_models (SVM only)
    focused_pkl = focused_dir / config_name / "best_models.pkl"
    if not focused_pkl.exists():
        raise FileNotFoundError(f"Missing: {focused_pkl}")
    print(f"\nLoading focused models from: {focused_pkl}")
    with focused_pkl.open("rb") as f:
        focused_models = pickle.load(f)
    print(f"  Loaded {len(focused_models)} models from focused:")
    for name in focused_models:
        print(f"    - {name}")

    # Build the merged set: take all from extended, then OVERWRITE SVM with focused
    merged_models = dict(extended_models)
    if "svm" in focused_models:
        old_svm = merged_models.get("svm")
        merged_models["svm"] = focused_models["svm"]
        print(f"\n  Replaced SVM with focused version.")
        if old_svm is not None:
            try:
                print(f"    Old SVM: C={old_svm.C}, gamma={old_svm.gamma}")
            except Exception:
                pass
            try:
                print(f"    New SVM: C={focused_models['svm'].C}, "
                       f"gamma={focused_models['svm'].gamma}")
            except Exception:
                pass
    else:
        print(f"\n  WARNING: focused/best_models.pkl has no 'svm' key — "
               f"keeping extended SVM as-is.")

    # Remove any pre-existing Stacking_Ensemble (we will rebuild it)
    if "Stacking_Ensemble" in merged_models:
        del merged_models["Stacking_Ensemble"]
        print(f"  Removed pre-existing Stacking_Ensemble (will rebuild).")

    # Prepare data
    data = prepare_data(data_root)
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]

    # Build & fit Stacking_Ensemble
    print("\n" + "-" * 70)
    print("Building Stacking_Ensemble with merged base learners")
    print("-" * 70)

    stack = build_stacking_with_tuned_bases(merged_models, n_jobs=n_jobs)
    n_bases = len(stack.estimators)
    print(f"  Number of base learners (with predict_proba): {n_bases}")
    print(f"  StackingClassifier n_jobs: {n_jobs}")

    t0 = time.time()
    stack.fit(X_train, y_train)
    elapsed = time.time() - t0

    ev = evaluate_on_test(stack, X_test, y_test)
    print(f"\n  Stacking test accuracy: {ev['test_accuracy']:.4f}")
    print(f"  Stacking test AUC:      {ev['test_auc']:.4f}")
    print(f"  Stacking test F1:       {ev['test_f1']:.4f}")
    print(f"  Build time:             {elapsed:.1f}s")

    # Add Stacking_Ensemble to merged models
    merged_models["Stacking_Ensemble"] = stack

    # Save merged models
    out_pkl = out_config_dir / "best_models.pkl"
    with out_pkl.open("wb") as f:
        pickle.dump(merged_models, f)
    print(f"\n  Saved merged models to: {out_pkl}")

    # Save combined_results.csv (only the new Stacking entry — extended/focused
    # already have their own combined_results.csv)
    rows = [{
        "model": "Stacking_Ensemble",
        "cv_accuracy": None,
        "test_accuracy": ev["test_accuracy"],
        "test_ci_low":  ev["test_ci_low"],
        "test_ci_high": ev["test_ci_high"],
        "test_auc":     ev["test_auc"],
        "test_f1":      ev["test_f1"],
        "elapsed_s":    elapsed,
        "best_params":  json.dumps({"meta": "L2-LR C=1.0", "n_bases": n_bases}),
        "edge_warnings": "",
        "grid_mode":    "post-tuning-rebuild-with-focused-svm",
    }]
    df = pd.DataFrame(rows)
    csv_path = out_config_dir / "combined_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved combined_results.csv to: {csv_path}")

    # Human-readable summary
    summary_lines = [
        "=" * 70,
        f"Stacking_Ensemble REBUILD SUMMARY — {config_name}",
        "=" * 70,
        "",
        "Source pkls:",
        f"  Extended: {extended_pkl}",
        f"  Focused:  {focused_pkl}",
        "",
        f"Number of base learners (with predict_proba): {n_bases}",
        "",
        "Base learners used:",
    ]
    for name, _ in stack.estimators:
        summary_lines.append(f"  - {name}")
    summary_lines += [
        "",
        f"Test accuracy: {ev['test_accuracy']:.4f}",
        f"Test AUC:      {ev['test_auc']:.4f}",
        f"Test F1:       {ev['test_f1']:.4f}",
        f"95% CI:        [{ev['test_ci_low']:.4f}, {ev['test_ci_high']:.4f}]",
        f"Build time:    {elapsed:.1f}s",
    ]
    txt_path = out_config_dir / "stacking_summary.txt"
    txt_path.write_text("\n".join(summary_lines) + "\n")
    print(f"  Saved summary to: {txt_path}")


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Rebuild Stacking_Ensemble combining extended base "
                    "learners with focused SVM."
    )
    parser.add_argument("--static-root",
                         default="../simulations/features_static/",
                         help="Path to static features directory.")
    parser.add_argument("--mobile-root",
                         default="../simulations/features_mobile/",
                         help="Path to mobile features directory.")
    parser.add_argument("--extended-dir",
                         default="./results/hp_search_extended",
                         help="Directory with extended HP search results.")
    parser.add_argument("--focused-dir",
                         default="./results/hp_search_focused",
                         help="Directory with focused (SVM-only) HP search results.")
    parser.add_argument("--out-dir",
                         default="./results/hp_search_final",
                         help="Output directory for rebuilt Stacking results.")
    parser.add_argument("--config",
                         default="both",
                         choices=["static", "mobile", "both"],
                         help="Which config to process (default: both).")
    parser.add_argument("--n-jobs", type=int, default=16,
                         help="n_jobs for StackingClassifier (default: 16).")
    args = parser.parse_args()

    extended_dir = Path(args.extended_dir)
    focused_dir = Path(args.focused_dir)
    out_dir = Path(args.out_dir)

    print("=" * 70)
    print("REBUILD STACKING — v2 (strict observability)")
    print("=" * 70)
    print(f"  Extended source: {extended_dir}")
    print(f"  Focused source:  {focused_dir}")
    print(f"  Output:          {out_dir}")
    print(f"  Config:          {args.config}")

    if args.config in ("static", "both"):
        process_config("static", args.static_root,
                       extended_dir, focused_dir, out_dir,
                       n_jobs=args.n_jobs)

    if args.config in ("mobile", "both"):
        process_config("mobile", args.mobile_root,
                       extended_dir, focused_dir, out_dir,
                       n_jobs=args.n_jobs)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()