#!/usr/bin/env python3
"""
generate_scaler_v2.py

Recovers the RobustScaler and the 60/20/20 group-aware split used by
unified_hp_search_v2.py. Saves scaler + indices + feature names so that
downstream scripts can reproduce inference without retraining.

Reproduces the pipeline of unified_hp_search_v2.py exactly:
  - random_state=42
  - GroupShuffleSplit at run level (file_source)
  - 80/20 outer split (train_val vs test)
  - 75/25 inner split (train vs val)  -> 60/20/20 overall
  - Selected feature names are READ from full_log.json (not recomputed),
    to guarantee a 1:1 match with the features the saved models were trained on.
  - RobustScaler.fit on the selected training features.

Outputs (per config), written to results/hp_search_final/{config}/:
  scaler_v2.pkl
  split_indices_v2.npz       (train_idx, val_idx, test_idx)
  feature_names_v2.json
  run_ids_test_v2.npy        (run_id per test window)

Usage:
  python3 generate_scaler_v2.py --config static
  python3 generate_scaler_v2.py --config mobile
  python3 generate_scaler_v2.py --config both
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import RobustScaler

import defense_detection_v2 as dd


def restore_split(n_rows, y_arr, groups_arr, random_state=42):
    """
    Reproduces the 60/20/20 group-aware split of unified_hp_search_v2.
    Returns integer index arrays.
    """
    outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=random_state)
    train_val_idx, test_idx = next(
        outer.split(np.arange(n_rows), y_arr, groups_arr)
    )

    inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=random_state)
    sub_train, sub_val = next(
        inner.split(
            np.arange(len(train_val_idx)),
            y_arr[train_val_idx],
            groups_arr[train_val_idx],
        )
    )
    train_idx = train_val_idx[sub_train]
    val_idx = train_val_idx[sub_val]
    return train_idx, val_idx, test_idx


def process_config(config_name, results_root):
    print(f"\n{'='*70}\nProcessing config: {config_name}\n{'='*70}")

    # 1) Load selected feature names from full_log.json
    log_path = Path(results_root) / "hp_search_extended" / config_name / "full_log.json"
    if not log_path.exists():
        sys.exit(f"ERROR: {log_path} not found. Cannot determine selected features.")
    with open(log_path) as f:
        log = json.load(f)
    selected_features = log["feature_names"]
    expected_train = log.get("n_train_pre_smote")
    expected_val = log.get("n_val")
    expected_test = log.get("n_test")
    print(f"[1/7] Loaded {len(selected_features)} selected feature names "
          f"from full_log.json")

    # 2) Build Config and detector — matches unified_hp_search_v2.py lines 1028-1032
    cfg = dd.Config()
    cfg.random_state = 42
    cfg.group_split_by_file_source = True
    cfg.use_validation = True
    cfg.test_size = 0.20
    cfg.train_size = 0.75
    if config_name == "static":
        cfg.data_root = "../simulations/features_static/"
    else:
        cfg.data_root = "../simulations/features_mobile/"
    detector = dd.DefenseDetector(cfg)

    # 3) Load raw simulation data (tall format) and pivot to wide
    print(f"[2/7] Loading simulation data from {cfg.data_root} ...")
    tall_df = detector.load_simulation_data_enhanced()
    if not isinstance(tall_df, pd.DataFrame):
        sys.exit(f"ERROR: expected tall_df to be a DataFrame, got {type(tall_df).__name__}")
    print(f"      tall_df shape: {tall_df.shape}")

    print("      Pivoting to wide (preprocess_data_enhanced) ...")
    X, y, groups = detector.preprocess_data_enhanced(tall_df)
    if not isinstance(X, pd.DataFrame):
        sys.exit(f"ERROR: expected X to be a DataFrame, got {type(X).__name__}")
    y_arr = np.asarray(y)
    groups_arr = np.asarray(groups)
    print(f"      X shape: {X.shape}, y shape: {y_arr.shape}, "
          f"unique runs: {len(set(groups_arr))}")

    # 4) Engineer features (DataFrame in, DataFrame out)
    print("[3/7] Engineering advanced features ...")
    X_eng = detector.engineer_advanced_features(X)
    if not isinstance(X_eng, pd.DataFrame):
        sys.exit(f"ERROR: expected X_eng to be a DataFrame, got {type(X_eng).__name__}")
    print(f"      X_eng shape: {X_eng.shape}")

    # 5) Verify all selected features exist
    missing = [f for f in selected_features if f not in X_eng.columns]
    if missing:
        sys.exit(f"ERROR: {len(missing)} selected features missing from X_eng. "
                 f"First few: {missing[:5]}")
    print(f"      All {len(selected_features)} selected features present in X_eng")

    # 6) Reproduce 60/20/20 split
    print("[4/7] Reproducing 60/20/20 group-aware split (random_state=42) ...")
    train_idx, val_idx, test_idx = restore_split(
        len(X_eng), y_arr, groups_arr, random_state=cfg.random_state
    )
    print(f"      train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}")

    if expected_train is not None:
        if (len(train_idx) == expected_train and
            len(val_idx) == expected_val and
            len(test_idx) == expected_test):
            print("      Split sizes match full_log.json")
        else:
            print(f"      WARNING: split sizes differ! "
                  f"got ({len(train_idx)},{len(val_idx)},{len(test_idx)}) "
                  f"vs expected ({expected_train},{expected_val},{expected_test})")

    # 7) Subset to selected features on training rows, fit RobustScaler
    print("[5/7] Subsetting to selected features and fitting RobustScaler ...")
    X_train_selected = X_eng.iloc[train_idx][selected_features]
    print(f"      X_train_selected shape: {X_train_selected.shape}")

    scaler = RobustScaler()
    scaler.fit(X_train_selected.values)
    print(f"      Scaler fitted: center shape {scaler.center_.shape}, "
          f"scale shape {scaler.scale_.shape}")

    # 8) Build run_ids for test windows
    run_ids_test = groups_arr[test_idx]
    n_unique_test_runs = len(set(run_ids_test))
    windows_per_run = len(test_idx) / n_unique_test_runs
    print(f"[6/7] Test set: {len(test_idx)} windows, {n_unique_test_runs} unique runs "
          f"(avg {windows_per_run:.2f} windows/run)")

    # 9) Save outputs
    out_dir = Path(results_root) / "hp_search_final" / config_name
    out_dir.mkdir(parents=True, exist_ok=True)

    scaler_path = out_dir / "scaler_v2.pkl"
    joblib.dump(scaler, scaler_path)

    indices_path = out_dir / "split_indices_v2.npz"
    np.savez(indices_path, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)

    fnames_path = out_dir / "feature_names_v2.json"
    with open(fnames_path, "w") as f:
        json.dump(
            {"selected": selected_features, "n_selected": len(selected_features)},
            f,
            indent=2,
        )

    runids_path = out_dir / "run_ids_test_v2.npy"
    np.save(runids_path, run_ids_test, allow_pickle=True)

    print(f"[7/7] Saved:")
    print(f"        {scaler_path}")
    print(f"        {indices_path}")
    print(f"        {fnames_path}")
    print(f"        {runids_path}")
    print(f"\nDone for {config_name}.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        choices=["static", "mobile", "both"],
        default="both",
    )
    parser.add_argument(
        "--results-root",
        default="./results",
    )
    args = parser.parse_args()

    configs = ["static", "mobile"] if args.config == "both" else [args.config]
    for c in configs:
        process_config(c, args.results_root)


if __name__ == "__main__":
    main()
