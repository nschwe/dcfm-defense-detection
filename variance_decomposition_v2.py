"""
Variance decomposition R = sigma^2_within / sigma^2_between
for the final retained features in static and mobile configurations.

Replicates the analysis described in Section IV-A of the paper.

This version reuses the trained pipeline's load_simulation_data_enhanced
and preprocess_data_enhanced so that engineering output matches exactly
the features the model was trained on.

Usage (from strict_observable_v2/):
    python3 -u variance_decomposition_v2.py 2>&1 | tee variance_decomposition.log
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from defense_detection_v2 import DefenseDetector, Config


def compute_R(X, run_ids):
    """
    For each column:
      sigma^2_within  = mean over runs of (variance across the 4 windows of a run)
      sigma^2_between = variance of per-run means
      R = within / between
    """
    df = pd.DataFrame(X.copy())
    df["__run__"] = run_ids
    g = df.groupby("__run__")
    within = g.var(ddof=1).mean(axis=0).values   # numpy array, ordered by columns
    between = g.mean().var(axis=0, ddof=1).values
    eps = 1e-12
    R = within / (between + eps)
    return R, within, between


def analyze_config(name, data_root, model_path):
    print(f"\n{'=' * 60}\n  {name.upper()}\n{'=' * 60}")

    md = joblib.load(model_path)
    feat_names = list(md["feature_names"])
    print(f"Final features in model: {len(feat_names)}")

    # 1) load + preprocess via the original pipeline
    cfg = Config(data_root=data_root, results_dir="/tmp/_va", verbose=1)
    pipe = DefenseDetector(cfg)

    tall_df = pipe.load_simulation_data_enhanced()
    X_wide, y, groups = pipe.preprocess_data_enhanced(tall_df)
    # groups[i] is the file_source (e.g. "metrics_output-1.csv") of row i

    # 2) feature engineering — same function used in training
    X_eng = pipe.engineer_advanced_features(X_wide)
    print(f"After engineering: {X_eng.shape[1]} features")

    # 3) apply the trained scaler — it expects ALL engineered features (141),
    #    not just the final 25/37 selected ones
    scaler = md["scaler"]
    scaler_features = list(scaler.feature_names_in_) if hasattr(scaler, "feature_names_in_") else None

    if scaler_features is not None:
        missing_for_scaler = [f for f in scaler_features if f not in X_eng.columns]
        if missing_for_scaler:
            print(f"WARNING: {len(missing_for_scaler)} features missing for scaler:")
            for f in missing_for_scaler[:15]:
                print(f"  - {f}")
            return
        X_for_scaler = X_eng[scaler_features].copy()
    else:
        X_for_scaler = X_eng.copy()

    X_scaled_full = pd.DataFrame(scaler.transform(X_for_scaler),
                                 columns=X_for_scaler.columns,
                                 index=X_for_scaler.index)

    # 4) keep only retained features for the variance analysis
    missing_final = [f for f in feat_names if f not in X_scaled_full.columns]
    if missing_final:
        print(f"WARNING: {len(missing_final)} final features missing:")
        for f in missing_final[:15]:
            print(f"  - {f}")
        return

    X_scaled = X_scaled_full[feat_names].copy()

    # 5) variance decomposition
    R_arr, w_arr, b_arr = compute_R(X_scaled.values, np.array(groups))
    R = pd.Series(R_arr, index=feat_names)

    n_pass = int((R >= 0.5).sum())
    n_fail = int((R < 0.5).sum())
    print(f"\nR >= 0.5 : {n_pass} / {len(feat_names)} features")
    print(f"R <  0.5 : {n_fail} feature(s)")

    if n_fail > 0:
        print("\nFailing features (sorted by R, ascending):")
        for f, r in R[R < 0.5].sort_values().items():
            print(f"  {f:<50s} R = {r:.4f}")

    # 6) Random Forest importance
    print("\nFitting Random Forest (300 trees) for importance...")
    rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42)
    rf.fit(X_scaled.values, y.values)
    imp = pd.Series(rf.feature_importances_, index=feat_names)
    imp_total = float(imp.sum())

    if n_fail > 0:
        failing_idx = R[R < 0.5].index
        fail_share = float(imp[failing_idx].sum() / imp_total)
        print(f"\nFailing features contribute "
              f"{100 * fail_share:.2f}% of total RF importance")
        print("Per-feature breakdown:")
        for f in failing_idx:
            print(f"  {f:<50s} importance = {100*imp[f]/imp_total:.2f}%")

    # 7) save full table
    out = pd.DataFrame({
        "feature": feat_names,
        "R": R_arr,
        "var_within": w_arr,
        "var_between": b_arr,
        "rf_importance_pct": (100 * imp / imp_total).values,
    }).sort_values("R")
    out_path = f"results/{name}/variance_decomposition.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    analyze_config("static",
                   "../simulations/features_static",
                   "results/static/best_model_Stacking_Ensemble.pkl")
    analyze_config("mobile",
                   "../simulations/features_mobile",
                   "results/mobile/best_model_Stacking_Ensemble.pkl")