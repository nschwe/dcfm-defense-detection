#!/usr/bin/env python3
"""
cluster_bootstrap_ci.py

Computes cluster-aware bootstrap confidence intervals for Table IV by
resampling at the run level (clusters of 4 windows) rather than at the
window level.

Pipeline:
  1. Load scaler, indices, run_ids from generate_scaler_v2.py outputs.
  2. Load best_models.pkl (12 base learners + Stacking).
  3. Engineer features on the FULL dataset (DataFrame), subset to the
     saved selected features, scale with the saved RobustScaler.
  4. Inference on validation set -> tune threshold per model
     (scan [0.1, 0.9] step 0.01 for predict_proba models; for Ridge,
     tune on decision_function scores directly). Stacking fixed at 0.5.
  5. Inference on test set -> cache scores to disk.
  6. Cluster bootstrap: 2000 resamples of N_runs test runs (with replacement),
     accuracy and AUC per model under fixed (validation-tuned) threshold.
  7. Output: bootstrap_results.csv + table_iv_cluster_ci.tex.

Threshold strategy: validation-tuned, held FIXED across resamples.
This is consistent with the Wilson CIs in the paper, which describe sampling
variability of test accuracy given a single trained model + threshold.

Usage:
  python3 cluster_bootstrap_ci.py --config both
  python3 cluster_bootstrap_ci.py --config static --n-resamples 2000
"""

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import defense_detection_v2 as dd


TABLE_IV_MODELS = [
    "Stacking_Ensemble",
    "xgboost",
    "catboost",
    "randomforest",
    "adaboost",
    "logisticregression",
    "ridge",
    "svm",
]

DISPLAY_NAMES = {
    "Stacking_Ensemble": "Stacking Ensemble",
    "xgboost": "\\textsc{XGBoost}",
    "catboost": "\\textsc{CatBoost}",
    "randomforest": "Random Forest",
    "adaboost": "AdaBoost",
    "logisticregression": "Logistic Regression",
    "ridge": "Ridge",
    "svm": "SVM (RBF)",
}


def get_scores(model, X):
    """
    Returns positive-class scores.

    - predict_proba models  -> probas of positive class in [0,1]
    - decision_function only (Ridge) -> raw decision_function values
                                         (NOT normalized; we threshold on
                                         the raw scale, consistent with how
                                         a Ridge classifier is typically used.)
    - fallback (hard predict) -> 0/1 float
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1], "proba"
    elif hasattr(model, "decision_function"):
        return model.decision_function(X), "decision"
    else:
        return model.predict(X).astype(float), "hard"


def tune_threshold(y_val, scores_val, score_type):
    """
    Tune accuracy-maximizing threshold on validation.

    For 'proba': scan [0.10, 0.90] step 0.01 (paper's setting).
    For 'decision': scan over percentiles of validation scores so that we
                    cover the actual score range without assuming [0,1].
    For 'hard': threshold doesn't matter, return 0.5.
    """
    if score_type == "hard":
        return 0.5, float((scores_val == y_val).mean())

    if score_type == "proba":
        thresholds = np.arange(0.10, 0.90 + 1e-9, 0.01)
    else:  # decision
        # 81 candidate cuts over the empirical distribution
        thresholds = np.quantile(scores_val, np.linspace(0.05, 0.95, 81))

    best_t = float(thresholds[len(thresholds) // 2])
    best_acc = -1.0
    for t in thresholds:
        preds = (scores_val >= t).astype(int)
        acc = float((preds == y_val).mean())
        if acc > best_acc:
            best_acc = acc
            best_t = float(t)
    return best_t, best_acc


def cluster_bootstrap(
    preds, scores, y_test, run_codes, n_runs, n_resamples, rng_seed
):
    """
    Cluster bootstrap over runs. Returns (accs, aucs) arrays of length n_resamples.

    preds:      (n_test,) 0/1
    scores:     (n_test,) float — for AUC computation
    y_test:     (n_test,) 0/1
    run_codes:  (n_test,) int code of run_id for each window
    n_runs:     total number of unique runs
    """
    rng = np.random.default_rng(rng_seed)

    # Build run_id -> indices, then convert to a list of arrays for fast lookup
    run_indices = [np.where(run_codes == r)[0] for r in range(n_runs)]

    correct = (preds == y_test).astype(np.int32)

    accs = np.empty(n_resamples, dtype=np.float64)
    aucs = np.empty(n_resamples, dtype=np.float64)

    for b in range(n_resamples):
        sampled = rng.integers(0, n_runs, size=n_runs)
        # Vectorise the gather
        flat = np.concatenate([run_indices[s] for s in sampled])
        accs[b] = correct[flat].mean()

        # AUC requires both classes present; if degenerate, mark NaN
        y_r = y_test[flat]
        s_r = scores[flat]
        if y_r.min() == y_r.max():
            aucs[b] = np.nan
        else:
            aucs[b] = roc_auc_score(y_r, s_r)

    return accs, aucs


def process_config(config_name, results_root, n_resamples, random_state):
    print(f"\n{'='*70}\nConfig: {config_name}\n{'='*70}")

    out_dir = Path(results_root) / "hp_search_final" / config_name

    # 1) Required artifact files
    scaler_path = out_dir / "scaler_v2.pkl"
    indices_path = out_dir / "split_indices_v2.npz"
    fnames_path = out_dir / "feature_names_v2.json"
    runids_path = out_dir / "run_ids_test_v2.npy"
    models_path = out_dir / "best_models.pkl"

    for p in [scaler_path, indices_path, fnames_path, runids_path, models_path]:
        if not p.exists():
            sys.exit(f"ERROR: missing {p}\nRun generate_scaler_v2.py first.")

    scaler = joblib.load(scaler_path)
    idx_data = np.load(indices_path)
    val_idx = idx_data["val_idx"]
    test_idx = idx_data["test_idx"]
    with open(fnames_path) as f:
        fnames = json.load(f)
    selected_features = fnames["selected"]
    run_ids_test = np.load(runids_path, allow_pickle=True)
    models_dict = joblib.load(models_path)

    print(f"  Loaded artifacts. val: {len(val_idx)}, test: {len(test_idx)} windows, "
          f"{len(np.unique(run_ids_test))} unique test runs")

    # 2) Reload data and engineer features — Config matches unified_hp_search_v2.py
    cfg = dd.Config()
    cfg.random_state = 42
    cfg.group_split_by_file_source = True
    cfg.use_validation = True
    cfg.data_root = (
        "../simulations/features_static/" if config_name == "static"
        else "../simulations/features_mobile/"
    )
    detector = dd.DefenseDetector(cfg)

    print("  Loading simulation data ...")
    tall_df = detector.load_simulation_data_enhanced()
    print("  Pivoting to wide (preprocess_data_enhanced) ...")
    X, y, _ = detector.preprocess_data_enhanced(tall_df)
    y_arr = np.asarray(y)

    print("  Engineering features ...")
    X_eng = detector.engineer_advanced_features(X)

    # Subset and scale
    X_eng_sel = X_eng[selected_features]
    X_eng_scaled_np = scaler.transform(X_eng_sel.values)

    # Build val/test DataFrames so models that expect DataFrame input still work
    X_val_df = pd.DataFrame(
        X_eng_scaled_np[val_idx], columns=selected_features
    )
    X_test_df = pd.DataFrame(
        X_eng_scaled_np[test_idx], columns=selected_features
    )
    y_val = y_arr[val_idx]
    y_test = y_arr[test_idx]

    # 3) Inference per model: tune threshold on val, predict on test
    print("\n  Tuning thresholds on validation + inference on test ...")
    cache = {}
    for name in TABLE_IV_MODELS:
        if name not in models_dict:
            print(f"    [SKIP] {name}: not in best_models.pkl")
            continue

        model = models_dict[name]
        t0 = time.time()

        if name == "Stacking_Ensemble":
            scores_val, score_type_val = get_scores(model, X_val_df)
            scores_test, score_type_test = get_scores(model, X_test_df)
            threshold = 0.5
        else:
            scores_val, score_type_val = get_scores(model, X_val_df)
            scores_test, score_type_test = get_scores(model, X_test_df)
            threshold, _ = tune_threshold(y_val, scores_val, score_type_val)

        preds_test = (scores_test >= threshold).astype(int)
        point_acc = float((preds_test == y_test).mean())
        point_auc = (
            float(roc_auc_score(y_test, scores_test))
            if y_test.min() != y_test.max() else float("nan")
        )

        cache[name] = {
            "threshold": threshold,
            "score_type": score_type_test,
            "scores_test": scores_test,
            "preds_test": preds_test,
            "point_acc": point_acc,
            "point_auc": point_auc,
        }

        print(f"    {name:22s}  t={threshold:8.4f}  ({score_type_test:8s})  "
              f"acc={point_acc:.4f}  auc={point_auc:.4f}  ({time.time()-t0:.1f}s)")

    # 4) Cache predictions to disk for reproducibility / re-bootstrap
    cache_path = out_dir / "test_predictions_cache.pkl"
    joblib.dump(
        {
            "config": config_name,
            "y_test": y_test,
            "run_ids_test": run_ids_test,
            "per_model": {
                name: {
                    "threshold": d["threshold"],
                    "score_type": d["score_type"],
                    "scores_test": d["scores_test"],
                    "point_acc": d["point_acc"],
                    "point_auc": d["point_auc"],
                }
                for name, d in cache.items()
            },
        },
        cache_path,
    )
    print(f"\n  Cached predictions -> {cache_path}")

    # 5) Encode run_ids -> integer codes for fast lookup
    unique_runs, run_codes = np.unique(run_ids_test, return_inverse=True)
    n_runs = len(unique_runs)
    n_windows = len(test_idx)
    print(f"\n  Cluster bootstrap (n_resamples={n_resamples}, "
          f"resampling {n_runs} runs) ...")

    rows = []
    for name, entry in cache.items():
        t0 = time.time()
        accs, aucs = cluster_bootstrap(
            entry["preds_test"],
            entry["scores_test"],
            y_test,
            run_codes,
            n_runs=n_runs,
            n_resamples=n_resamples,
            rng_seed=random_state,
        )
        acc_lo = float(np.percentile(accs, 2.5))
        acc_hi = float(np.percentile(accs, 97.5))
        # Use nanpercentile for AUC because of possible degenerate resamples
        auc_lo = float(np.nanpercentile(aucs, 2.5))
        auc_hi = float(np.nanpercentile(aucs, 97.5))

        rows.append({
            "config": config_name,
            "model": name,
            "display_name": DISPLAY_NAMES.get(name, name),
            "threshold": entry["threshold"],
            "score_type": entry["score_type"],
            "point_accuracy": entry["point_acc"],
            "acc_ci_low": acc_lo,
            "acc_ci_high": acc_hi,
            "acc_ci_width": acc_hi - acc_lo,
            "point_auc": entry["point_auc"],
            "auc_ci_low": auc_lo,
            "auc_ci_high": auc_hi,
            "n_resamples": n_resamples,
            "n_test_runs": n_runs,
            "n_test_windows": n_windows,
            "elapsed_s": time.time() - t0,
        })
        print(f"    {name:22s}  "
              f"acc={entry['point_acc']:.4f} [{acc_lo:.4f}, {acc_hi:.4f}]  "
              f"auc={entry['point_auc']:.4f} [{auc_lo:.4f}, {auc_hi:.4f}]  "
              f"({time.time()-t0:.1f}s)")

    return rows


def write_outputs(all_rows, out_path_csv, out_path_tex):
    df = pd.DataFrame(all_rows)
    df.to_csv(out_path_csv, index=False)
    print(f"\nWrote results CSV -> {out_path_csv}")

    # LaTeX in Table IV format with cluster CIs on accuracy and AUC point values
    lines = [
        "% Cluster-aware bootstrap CIs (2000 resamples at run level).",
        "% Accuracy intervals are 95% percentile CIs over the bootstrap distribution.",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Model & Static Acc. & Mobile Acc. & Static AUC & Mobile AUC \\\\",
        "\\midrule",
    ]
    by_model = {}
    for r in all_rows:
        by_model.setdefault(r["model"], {})[r["config"]] = r

    for name in TABLE_IV_MODELS:
        if name not in by_model:
            continue
        entry = by_model[name]
        s = entry.get("static")
        m = entry.get("mobile")
        disp = DISPLAY_NAMES.get(name, name)

        def fmt_acc(r):
            return (f"{r['point_accuracy']:.4f} "
                    f"[{r['acc_ci_low']:.3f}, {r['acc_ci_high']:.3f}]")

        def fmt_auc(r):
            return f"{r['point_auc']:.4f}"

        s_acc = fmt_acc(s) if s else "--"
        m_acc = fmt_acc(m) if m else "--"
        s_auc = fmt_auc(s) if s else "--"
        m_auc = fmt_auc(m) if m else "--"
        lines.append(f"{disp} & {s_acc} & {m_acc} & {s_auc} & {m_auc} \\\\")

    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(out_path_tex, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote LaTeX snippet -> {out_path_tex}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=["static", "mobile", "both"], default="both")
    parser.add_argument("--results-root", default="./results")
    parser.add_argument("--n-resamples", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--out-dir", default="./results/cluster_bootstrap")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = ["static", "mobile"] if args.config == "both" else [args.config]
    all_rows = []
    for c in configs:
        rows = process_config(c, args.results_root, args.n_resamples, args.random_state)
        all_rows.extend(rows)

    write_outputs(
        all_rows,
        out_dir / "bootstrap_results.csv",
        out_dir / "table_iv_cluster_ci.tex",
    )
    print("\nAll done.")


if __name__ == "__main__":
    main()
