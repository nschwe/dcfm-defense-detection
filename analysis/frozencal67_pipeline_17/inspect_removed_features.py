#!/usr/bin/env python3
"""
Inspect which features are removed at each filtering stage.
Replicates exact pipeline of defense_detection_v2.py: split 60/20/20 -> RobustScaler -> VarianceThreshold -> correlation pruning.

Outputs:
  - List of features removed by VarianceThreshold (static / mobile)
  - List of features removed by correlation pruning (static / mobile)
  - Which features differ between the two configs
"""

import sys, os
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from defense_detection_v2 import DefenseDetector, Config


def split_60_20_20(X_eng, y, groups, cfg):
    groups_arr = np.array(groups)
    splitter = GroupShuffleSplit(n_splits=1, test_size=cfg.test_size, random_state=cfg.random_state)
    train_val_idx, _ = next(splitter.split(X_eng, y, groups_arr))
    X_temp = X_eng.iloc[train_val_idx]
    y_temp = y.iloc[train_val_idx]
    groups_temp = groups_arr[train_val_idx]
    splitter_val = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=cfg.random_state)
    train_idx, _ = next(splitter_val.split(X_temp, y_temp, groups_temp))
    return X_temp.iloc[train_idx]


def analyze_config(data_root, label):
    print(f"\n{'='*70}\n  {label.upper()}\n{'='*70}")
    cfg = Config()
    cfg.data_root = data_root
    cfg.results_dir = f"./_inspect_tmp_{label}"
    os.makedirs(cfg.results_dir, exist_ok=True)

    detector = DefenseDetector(cfg)
    tall_df = detector.load_simulation_data_enhanced()
    out = detector.preprocess_data_enhanced(tall_df)
    X, y, groups = out if len(out) == 3 else (*out, None)
    X_eng = detector.engineer_advanced_features(X)
    X_train = split_60_20_20(X_eng, y, groups, cfg)

    # RobustScaler
    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns, index=X_train.index
    )

    # Variance threshold
    var_selector = VarianceThreshold(threshold=0.01)
    var_selector.fit(X_train_scaled)
    var_kept = X_train_scaled.columns[var_selector.get_support()].tolist()
    var_removed = [f for f in X_train_scaled.columns if f not in var_kept]

    # Correlation pruning on the survivors
    X_train_var = X_train_scaled[var_kept]
    corr_matrix = X_train_var.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    corr_removed = [col for col in upper.columns if any(upper[col] > 0.95)]
    final_kept = [f for f in var_kept if f not in corr_removed]

    # Output
    print(f"\nFEATURES REMOVED BY VARIANCE THRESHOLD ({len(var_removed)} features):")
    for i, f in enumerate(sorted(var_removed), 1):
        print(f"  {i:2d}. {f}")

    print(f"\nFEATURES REMOVED BY CORRELATION PRUNING ({len(corr_removed)} features):")
    for i, f in enumerate(sorted(corr_removed), 1):
        print(f"  {i:2d}. {f}")

    print(f"\nFINAL KEPT FEATURES ({len(final_kept)} features):")
    for i, f in enumerate(sorted(final_kept), 1):
        print(f"  {i:2d}. {f}")

    return {
        "label": label,
        "var_removed": set(var_removed),
        "corr_removed": set(corr_removed),
        "final_kept": set(final_kept),
        "all_engineered": set(X_train_scaled.columns.tolist()),
    }


if __name__ == "__main__":
    static = analyze_config("../simulations/features_static", "static")
    mobile = analyze_config("../simulations/features_mobile", "mobile")

    print(f"\n{'='*78}\n  COMPARISON: VARIANCE-FILTER REMOVALS\n{'='*78}")

    only_static = sorted(static["var_removed"] - mobile["var_removed"])
    only_mobile = sorted(mobile["var_removed"] - static["var_removed"])
    both = sorted(static["var_removed"] & mobile["var_removed"])

    print(f"\nRemoved in BOTH configs ({len(both)} features):")
    for f in both: print(f"  - {f}")
    print(f"\nRemoved ONLY in STATIC ({len(only_static)} features) -- 'static-specific dead features':")
    for f in only_static: print(f"  - {f}")
    print(f"\nRemoved ONLY in MOBILE ({len(only_mobile)} features):")
    for f in only_mobile: print(f"  - {f}")

    # Categorize the static-specific dead features
    print(f"\n{'='*78}\n  CATEGORIZATION OF STATIC-SPECIFIC DEAD FEATURES\n{'='*78}")
    print(f"\n{len(only_static)} features dead under static but alive under mobile:")
    print("These are the features that drive the 21 vs 9 difference in variance pruning.\n")

    categories = {
        "delay/jitter related": [],
        "loss-rate related": [],
        "flow-count/duration related": [],
        "energy/MAC related": [],
        "row_*/aggregate stats": [],
        "other": [],
    }
    for f in only_static:
        fl = f.lower()
        if any(k in fl for k in ["delay", "jitter"]):
            categories["delay/jitter related"].append(f)
        elif any(k in fl for k in ["loss", "lossrate"]):
            categories["loss-rate related"].append(f)
        elif any(k in fl for k in ["flowcount", "duration", "flowdur"]):
            categories["flow-count/duration related"].append(f)
        elif any(k in fl for k in ["energy", "mac", "drop"]):
            categories["energy/MAC related"].append(f)
        elif fl.startswith("row_"):
            categories["row_*/aggregate stats"].append(f)
        else:
            categories["other"].append(f)

    for cat, feats in categories.items():
        if feats:
            print(f"\n  [{cat}] ({len(feats)}):")
            for f in feats:
                print(f"    - {f}")