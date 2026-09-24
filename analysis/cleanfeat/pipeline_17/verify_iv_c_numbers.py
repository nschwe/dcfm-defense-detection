#!/usr/bin/env python3
"""
Verification script for Section IV-C numerical claims.

Replicates the EXACT pipeline order of defense_detection_v2.py
(see lines 1078-1097 of run_full_pipeline):
  1. 60/20/20 GroupShuffleSplit (two sequential splits)
  2. RobustScaler.fit_transform(X_train)         <-- CRITICAL
  3. VarianceThreshold(threshold=0.01) on X_train_scaled
  4. Pearson correlation pruning |r|>0.95 on X_train_var

The previous version of this script omitted step 2 (scaling), which
caused the variance threshold to operate on raw-scale features. Many
features that pass threshold=0.01 on scaled data fail it on raw-scale
(e.g., features in [0,1] range have raw variance < 0.01). Correcting
the order should reproduce the 25 / 37 final feature counts.

Run from strict_observable_v2/ directory:
    python3 -u verify_iv_c_numbers.py 2>&1 | tee verify_iv_c.log
"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GroupShuffleSplit, train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from defense_detection_v2 import DefenseDetector, Config


def split_60_20_20(X_eng, y, groups, cfg):
    """Replicates the 60/20/20 split logic (defense_detection_v2.py:1026-1075)."""
    groups_arr = np.array(groups)

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=cfg.test_size, random_state=cfg.random_state,
    )
    train_val_idx, test_idx = next(splitter.split(X_eng, y, groups_arr))

    X_temp = X_eng.iloc[train_val_idx]
    y_temp = y.iloc[train_val_idx]
    groups_temp = groups_arr[train_val_idx]

    splitter_val = GroupShuffleSplit(
        n_splits=1, test_size=0.25, random_state=cfg.random_state,
    )
    train_idx, val_idx = next(splitter_val.split(X_temp, y_temp, groups_temp))

    X_train = X_temp.iloc[train_idx]
    X_val = X_temp.iloc[val_idx]
    X_test = X_eng.iloc[test_idx]
    y_train = y_temp.iloc[train_idx]
    y_val = y_temp.iloc[val_idx]
    y_test = y.iloc[test_idx]

    return X_train, X_val, X_test, y_train, y_val, y_test


def run_for_config(data_root: str, label: str):
    print(f"\n{'='*70}")
    print(f"  {label.upper()}")
    print(f"{'='*70}")

    cfg = Config()
    cfg.data_root = data_root
    cfg.results_dir = f"./_verify_tmp_{label}"
    os.makedirs(cfg.results_dir, exist_ok=True)

    detector = DefenseDetector(cfg)

    print("\n[stage 1] Loading data...")
    tall_df = detector.load_simulation_data_enhanced()

    print("\n[stage 2] Preprocessing...")
    out = detector.preprocess_data_enhanced(tall_df)
    if len(out) == 3:
        X, y, groups = out
    else:
        X, y = out
        groups = None
    print(f"  X.shape={X.shape}, groups={'yes' if groups is not None else 'no'}")
    n_base = X.shape[1]

    print("\n[stage 3] Engineering features...")
    X_eng = detector.engineer_advanced_features(X)
    n_eng = X_eng.shape[1]
    print(f"  After engineering: {n_eng} features (from {n_base})")

    print("\n[stage 3.5] 60/20/20 group-aware split...")
    if cfg.group_split_by_file_source and groups is not None:
        X_train, X_val, X_test, y_train, y_val, y_test = split_60_20_20(X_eng, y, groups, cfg)
    else:
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_eng, y, test_size=cfg.test_size,
            random_state=cfg.random_state, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25,
            random_state=cfg.random_state, stratify=y_temp
        )
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val:   {X_val.shape}")
    print(f"  X_test:  {X_test.shape}")

    # Stage 4: RobustScaler — fit on X_train, transform train (mirrors line 1078-1083)
    print("\n[stage 4] RobustScaler.fit_transform(X_train)...")
    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns, index=X_train.index
    )
    print(f"  Scaled. X_train_scaled.shape={X_train_scaled.shape}")

    # Stage 5a: variance threshold on SCALED data
    print("\n[stage 5a] Variance threshold (threshold=0.01) — on SCALED X_train...")
    var_selector = VarianceThreshold(threshold=0.01)
    var_selector.fit(X_train_scaled)
    n_after_var = int(var_selector.get_support().sum())
    n_removed_var = n_eng - n_after_var
    print(f"  Removed by variance:  {n_removed_var:3d}/{n_eng}  "
          f"({100*n_removed_var/n_eng:.1f}%)")
    print(f"  Remaining after var:  {n_after_var}")

    remaining_cols = X_train_scaled.columns[var_selector.get_support()]
    X_train_var = pd.DataFrame(
        var_selector.transform(X_train_scaled),
        columns=remaining_cols,
        index=X_train_scaled.index,
    )

    # Stage 5b: correlation analysis
    print("\n[stage 5b] Correlation analysis (|r|>0.95)...")
    corr_matrix = X_train_var.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    n_feats_var = X_train_var.shape[1]
    total_pairs = n_feats_var * (n_feats_var - 1) // 2
    pairs_above = int((upper > 0.95).sum().sum())
    pct_pairs_above = 100.0 * pairs_above / total_pairs if total_pairs > 0 else 0
    print(f"  Total feature pairs:                       {total_pairs}")
    print(f"  Pairs with |r|>0.95:                       {pairs_above}")
    print(f"  Fraction of pairs above |r|>0.95:          {pct_pairs_above:.2f}%")

    # Stage 5c: correlation pruning
    to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
    n_after_corr = n_feats_var - len(to_drop)
    print(f"\n[stage 5c] Correlation pruning...")
    print(f"  Features dropped:       {len(to_drop)}")
    print(f"  Remaining after corr:   {n_after_corr}")

    # Cumulative summary
    print(f"\n[summary for {label}]")
    print(f"  Engineered:                  {n_eng}")
    print(f"  After variance threshold:    {n_after_var:>3d}  "
          f"(removed {n_removed_var}, {100*n_removed_var/n_eng:.1f}% of engineered)")
    print(f"  After correlation pruning:   {n_after_corr:>3d}  "
          f"(removed {len(to_drop)} of {n_feats_var}, {100*len(to_drop)/n_feats_var:.1f}% of survivors)")
    print(f"  % of pairs |r|>0.95:         {pct_pairs_above:.2f}%")
    print(f"  Final:                       {n_after_corr}")

    # Sanity check vs trained pkl
    pkl_path = f"results/{label}/best_model_Stacking_Ensemble.pkl"
    pkl_match = None
    if os.path.exists(pkl_path):
        try:
            import joblib
            d = joblib.load(pkl_path)
            n_pkl = len(d['feature_names'])
            pkl_match = (n_pkl == n_after_corr)
            tag = "MATCH" if pkl_match else f"MISMATCH (pkl={n_pkl}, here={n_after_corr})"
            print(f"  Cross-check vs trained model pkl: {tag}")
        except Exception as e:
            print(f"  [warning] could not cross-check pkl: {e}")

    return {
        "label": label,
        "engineered": n_eng,
        "after_variance": n_after_var,
        "removed_variance": n_removed_var,
        "pct_removed_variance_of_engineered": 100*n_removed_var/n_eng,
        "after_correlation": n_after_corr,
        "removed_correlation": len(to_drop),
        "pct_removed_correlation_of_survivors": 100*len(to_drop)/n_feats_var,
        "total_pairs": total_pairs,
        "pairs_above_threshold": pairs_above,
        "pct_pairs_above_threshold": pct_pairs_above,
        "pkl_match": pkl_match,
    }


if __name__ == "__main__":
    results = []
    results.append(run_for_config("../simulations/features_static", "static"))
    results.append(run_for_config("../simulations/features_mobile", "mobile"))

    print("\n\n")
    print("="*78)
    print("  COMPARATIVE SUMMARY (for IV-C in paper)")
    print("="*78)
    print(f"{'Quantity':<60s}{'Static':>9s}{'Mobile':>9s}")
    print("-"*78)
    rows = [
        ("engineered",                              "engineered features"),
        ("after_variance",                          "after variance threshold"),
        ("removed_variance",                        "  removed by variance"),
        ("pct_removed_variance_of_engineered",      "  % removed by variance (of engineered)"),
        ("after_correlation",                       "after correlation pruning (final)"),
        ("removed_correlation",                     "  removed by correlation"),
        ("pct_removed_correlation_of_survivors",    "  % removed by correlation (of survivors)"),
        ("total_pairs",                             "total feature pairs (post-variance)"),
        ("pairs_above_threshold",                   "feature pairs |r|>0.95"),
        ("pct_pairs_above_threshold",               "  % pairs |r|>0.95"),
    ]
    for key, label in rows:
        s = results[0][key]
        m = results[1][key]
        if isinstance(s, float):
            print(f"{label:<60s}{s:>9.2f}{m:>9.2f}")
        else:
            print(f"{label:<60s}{s:>9d}{m:>9d}")

    s_match = results[0]["pkl_match"]
    m_match = results[1]["pkl_match"]
    if s_match is True and m_match is True:
        print("\n  >>> Both pkl cross-checks PASSED. Numbers above are authoritative for IV-C. <<<")
    else:
        print(f"\n  >>> Static pkl match: {s_match}, Mobile pkl match: {m_match} <<<")
    print()