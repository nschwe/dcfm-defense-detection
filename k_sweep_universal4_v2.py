#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
k_sweep_universal4_v2.py
============================================================================
K-sweep cross-domain evaluation: Universal-4 as the core, with progressively
more features added (top-K by catboost_pvc strict importance).

This script extends `confusion_matrices_universal4_v2.py` to a sweep over
K. It uses the IDENTICAL pipeline: CatBoost (tuned), 60/20/20 group-aware
split, StandardScaler on source train, threshold tuning on source
validation, evaluation on full source + full target. The only difference
is that K varies and features are chosen accordingly:

  - K=4:  Universal-4 (anchored, identical to confusion_matrices_*)
  - K>4:  Universal-4 + top-(K-4) features by catboost_pvc strict ranking
          (mean importance across 20 seeds, intersected static/mobile top-K
          using the same ranking that produced Universal-4)

Output:
  - results/k_sweep_universal4/k_sweep_results.csv
      Columns: K, classifier, direction, acc_mean, acc_std, auc_mean,
               auc_std, n_seeds, features
  - results/k_sweep_universal4/k_sweep_summary.txt

Usage:
    python3 k_sweep_universal4_v2.py \
        --static-root ../simulations/features_static \
        --mobile-root ../simulations/features_mobile \
        --hp-results-dir ./results/hp_search_extended \
        --out-dir ./results/k_sweep_universal4 \
        --k-list 4,5,9,13,15,17,24,33

The K-list defaults to the values relevant to the paper's Table VIII.
============================================================================
"""
import argparse
import gc
import os
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib

import pandas.core.series
pandas.core.series.dtype = np.dtype

from catboost import CatBoostClassifier

from defense_detection_v2 import DefenseDetector, Config

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# Universal-4 (intersected from catboost_pvc Top-9 of static and mobile,
# strict observability variant). MUST match confusion_matrices_*_v2.
# ----------------------------------------------------------------------
UNIVERSAL_4 = [
    "AverageAdvertisedLinksPerTCMessage",
    "AverageMprCount",
    "DataPacketRate",
    "FlowThroughputStd",
]

N_SEEDS = 20
RANDOM_STATE = 42


# ----------------------------------------------------------------------
# Hyperparameter loading (copied verbatim from confusion_matrices_*_v2)
# ----------------------------------------------------------------------
def load_best_params(hp_results_dir: str) -> dict:
    out = {}
    for cfg in ("static", "mobile"):
        pkl_path = Path(hp_results_dir) / cfg / "best_models.pkl"
        if not pkl_path.exists():
            raise FileNotFoundError(f"Missing: {pkl_path}")
        d = joblib.load(pkl_path)
        if "catboost" not in d:
            raise KeyError(f"No 'catboost' key in {pkl_path}")
        entry = d["catboost"]
        if hasattr(entry, "get_params"):
            params = entry.get_params()
        elif isinstance(entry, dict):
            if "best_params" in entry:
                params = entry["best_params"]
            elif "params" in entry:
                params = entry["params"]
            else:
                params = entry
        else:
            params = {}
        out[cfg] = params
    return out


def filter_catboost_params(p: dict) -> dict:
    keep = {"iterations", "learning_rate", "depth", "l2_leaf_reg",
            "border_count", "bagging_temperature", "random_strength",
            "subsample"}
    return {k: v for k, v in p.items() if k in keep}


def build_catboost(best_params: dict, seed: int, n_jobs: int = 16):
    p = filter_catboost_params(best_params)
    return CatBoostClassifier(
        random_seed=seed, verbose=0, allow_writing_files=False,
        thread_count=n_jobs, **p,
    )


# ----------------------------------------------------------------------
# Data loading (no feature filter — we filter to top-K later)
# ----------------------------------------------------------------------
def load_raw_data(data_root: str, config_name: str):
    print(f"\n  Loading {config_name} from: {data_root}")
    config = Config()
    config.data_root = data_root
    config.random_state = RANDOM_STATE
    detector = DefenseDetector(config)
    tall_df = detector.load_simulation_data_enhanced()
    X, y, groups = detector.preprocess_data_enhanced(tall_df)
    missing = [f for f in UNIVERSAL_4 if f not in X.columns]
    if missing:
        raise ValueError(f"Missing Universal-4 features in {config_name}: {missing}")
    print(f"    Loaded: {X.shape}, classes={dict(y.value_counts())}, groups={len(set(groups))}")
    return X, y, np.array(groups)


# ----------------------------------------------------------------------
# Feature ranking: computed in-script via CatBoost PredictionValuesChange
# (the same importance method that produced Universal-4 originally).
# ----------------------------------------------------------------------
def compute_importance_ranking(X: pd.DataFrame, y: pd.Series, groups: np.ndarray,
                               best_params: dict, n_seeds: int = 5,
                               base_seed: int = 42) -> list:
    """Compute CatBoost PredictionValuesChange importance, averaged across
    seeds, return features sorted by mean importance descending.

    Uses a 60/20/20 split per seed and trains on the 60% train partition,
    consistent with the rest of the pipeline. The importance for K-selection
    only needs a stable ranking — 5 seeds suffice (vs. 20 for the cross-domain
    evaluation that follows).
    """
    from sklearn.model_selection import GroupShuffleSplit
    imps = []
    for seed in range(base_seed, base_seed + n_seeds):
        outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_val_idx, _ = next(outer.split(X, y, groups))
        inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
        train_idx, _ = next(inner.split(
            X.iloc[train_val_idx], y.iloc[train_val_idx], groups[train_val_idx]
        ))
        tr = train_val_idx[train_idx]

        sc = StandardScaler().fit(X.iloc[tr])
        X_tr_s = sc.transform(X.iloc[tr])

        clf = build_catboost(best_params, seed=seed, n_jobs=16)
        clf.fit(X_tr_s, y.iloc[tr])
        imp = clf.get_feature_importance(type="PredictionValuesChange")
        imps.append(imp)

    imps_mean = np.mean(imps, axis=0)
    s = pd.Series(imps_mean, index=X.columns).sort_values(ascending=False)
    return s.index.tolist()


def select_features_for_K(K: int, static_ranking: list, mobile_ranking: list,
                          all_features: list) -> list:
    """Select K features:
       - K == 4: returns Universal-4 exactly.
       - K  > 4: Universal-4 + (K-4) extras by avg rank across static/mobile.
       - K  < 4: top-K features from Universal-4 ranked by avg static+mobile
                 importance position (worst-ranked Universal-4 members
                 dropped first).
    """
    if K < 1:
        raise ValueError(f"K must be >= 1. Got K={K}")
    if K > len(all_features):
        raise ValueError(f"K={K} exceeds total features {len(all_features)}")

    n = len(static_ranking)
    s_rank = {f: i for i, f in enumerate(static_ranking)}
    m_rank = {f: i for i, f in enumerate(mobile_ranking)}

    if K == 4:
        return list(UNIVERSAL_4)

    if K < 4:
        # Rank Universal-4 features by avg static+mobile rank, keep top-K.
        u4_scored = []
        for f in UNIVERSAL_4:
            sr = s_rank.get(f, n)
            mr = m_rank.get(f, n)
            u4_scored.append((f, (sr + mr) / 2.0))
        u4_scored.sort(key=lambda x: x[1])
        return [f for f, _ in u4_scored[:K]]

    # K > 4: Universal-4 + extras
    selected = list(UNIVERSAL_4)
    n_extra_needed = K - 4
    candidates = [f for f in all_features if f not in UNIVERSAL_4]
    scores = []
    for f in candidates:
        sr = s_rank.get(f, n)
        mr = m_rank.get(f, n)
        avg = (sr + mr) / 2.0
        scores.append((f, avg))
    scores.sort(key=lambda x: x[1])
    for f, _ in scores[:n_extra_needed]:
        selected.append(f)

    if len(selected) != K:
        raise RuntimeError(
            f"Internal: expected {K} features, got {len(selected)}: {selected}"
        )
    return selected


# ----------------------------------------------------------------------
# Train/eval (copied from confusion_matrices_*_v2 — identical pipeline)
# ----------------------------------------------------------------------
def split_source_60_20_20(X, y, groups, seed):
    from sklearn.model_selection import GroupShuffleSplit
    outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_val_idx, test_idx = next(outer.split(X, y, groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    train_idx, val_idx = next(inner.split(
        X.iloc[train_val_idx], y.iloc[train_val_idx],
        groups[train_val_idx]
    ))
    train_idx_full = train_val_idx[train_idx]
    val_idx_full = train_val_idx[val_idx]
    return train_idx_full, val_idx_full, test_idx


def select_threshold(y_val, p_val):
    thresholds = np.arange(0.1, 0.91, 0.01)
    best_thr, best_acc = 0.5, -1.0
    for t in thresholds:
        y_pred = (p_val >= t).astype(int)
        acc = accuracy_score(y_val, y_pred)
        if acc > best_acc:
            best_acc, best_thr = acc, t
    return best_thr


def evaluate_seed(X_src, y_src, g_src, X_tgt, y_tgt,
                  features, best_params, seed):
    """Run one seed for one K-feature configuration.

    Returns dict with: acc_in, auc_in (source test partition),
                       acc_cross, auc_cross (full target).
    """
    tr, va, te = split_source_60_20_20(X_src, y_src, g_src, seed)

    Xs = X_src[features]
    Xt = X_tgt[features]

    sc = StandardScaler().fit(Xs.iloc[tr])
    X_tr_s = sc.transform(Xs.iloc[tr])
    X_va_s = sc.transform(Xs.iloc[va])
    X_te_s = sc.transform(Xs.iloc[te])
    X_tgt_s = sc.transform(Xt)

    clf = build_catboost(best_params, seed=seed, n_jobs=16)
    clf.fit(X_tr_s, y_src.iloc[tr])

    p_va = clf.predict_proba(X_va_s)[:, 1]
    thr = select_threshold(y_src.iloc[va], p_va)

    p_te = clf.predict_proba(X_te_s)[:, 1]
    p_tgt = clf.predict_proba(X_tgt_s)[:, 1]

    y_pred_te = (p_te >= thr).astype(int)
    y_pred_tgt = (p_tgt >= thr).astype(int)

    acc_in = accuracy_score(y_src.iloc[te], y_pred_te)
    auc_in = roc_auc_score(y_src.iloc[te], p_te)
    acc_cross = accuracy_score(y_tgt, y_pred_tgt)
    auc_cross = roc_auc_score(y_tgt, p_tgt)

    return {
        "acc_in": acc_in, "auc_in": auc_in,
        "acc_cross": acc_cross, "auc_cross": auc_cross,
        "thr": thr,
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--static-root", required=True,
                    help="Directory with static features")
    ap.add_argument("--mobile-root", required=True,
                    help="Directory with mobile features")
    ap.add_argument("--hp-results-dir", required=True,
                    help="Dir with best_models.pkl for static/mobile (e.g. hp_search_extended)")
    ap.add_argument("--out-dir", required=True,
                    help="Output directory for k_sweep_results.csv")
    ap.add_argument("--k-list", default="4,5,9,13,15,17,24,33",
                    help="Comma-separated K values to sweep (default: 4,5,9,13,15,17,24,33)")
    ap.add_argument("--n-seeds", type=int, default=N_SEEDS,
                    help=f"Bootstrap seeds for cross-domain eval (default {N_SEEDS})")
    ap.add_argument("--n-importance-seeds", type=int, default=5,
                    help="Seeds for importance-ranking computation (default 5)")
    args = ap.parse_args()

    k_list = [int(k) for k in args.k_list.split(",")]
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 70)
    print("K-sweep Universal-4 anchored, CatBoost (tuned)")
    print("=" * 70)
    print(f"  K values:        {k_list}")
    print(f"  N seeds:         {args.n_seeds}")
    print(f"  HP results:      {args.hp_results_dir}")
    print(f"  Output:          {args.out_dir}")
    print()

    # Load hyperparameters
    best_params = load_best_params(args.hp_results_dir)
    print(f"  Loaded HP for static and mobile.")

    # Load data
    X_static, y_static, g_static = load_raw_data(args.static_root, "static")
    X_mobile, y_mobile, g_mobile = load_raw_data(args.mobile_root, "mobile")
    all_features = list(X_static.columns)
    if list(X_mobile.columns) != all_features:
        raise ValueError("Static and mobile feature columns differ!")
    print(f"  Total features: {len(all_features)}")

    # Compute importance ranking (CatBoost PVC, in-script)
    print(f"\n  Computing importance ranking (CatBoost PVC, {args.n_importance_seeds} seeds)...")
    t0 = time.time()
    static_ranking = compute_importance_ranking(
        X_static, y_static, g_static, best_params["static"],
        n_seeds=args.n_importance_seeds, base_seed=RANDOM_STATE
    )
    print(f"    Static top-5: {static_ranking[:5]}")
    mobile_ranking = compute_importance_ranking(
        X_mobile, y_mobile, g_mobile, best_params["mobile"],
        n_seeds=args.n_importance_seeds, base_seed=RANDOM_STATE
    )
    print(f"    Mobile top-5: {mobile_ranking[:5]}")
    print(f"    Importance ranking elapsed: {(time.time()-t0)/60:.1f} min")

    # Build K -> features map
    print("\n  Feature sets per K:")
    feature_sets = {}
    for K in k_list:
        feats = select_features_for_K(K, static_ranking, mobile_ranking, all_features)
        feature_sets[K] = feats
        print(f"    K={K:2d}: {feats}")

    # Sweep
    results = []
    t_total_start = time.time()
    for K in k_list:
        feats = feature_sets[K]
        print(f"\n  Running K={K} ({len(feats)} features), {args.n_seeds} seeds...")
        t_k_start = time.time()

        sm_results = []  # static -> mobile
        ms_results = []  # mobile -> static
        s_in_results = []  # static in-domain (test set)
        m_in_results = []  # mobile in-domain (test set)

        for seed in range(RANDOM_STATE, RANDOM_STATE + args.n_seeds):
            # Static -> Mobile
            r_sm = evaluate_seed(X_static, y_static, g_static,
                                 X_mobile, y_mobile,
                                 feats, best_params["static"], seed)
            sm_results.append(r_sm)
            s_in_results.append(r_sm)  # in-domain comes from same training

            # Mobile -> Static
            r_ms = evaluate_seed(X_mobile, y_mobile, g_mobile,
                                 X_static, y_static,
                                 feats, best_params["mobile"], seed)
            ms_results.append(r_ms)
            m_in_results.append(r_ms)

        # Aggregate
        for direction, rs in [("S_in_domain", s_in_results),
                              ("M_in_domain", m_in_results),
                              ("S_to_M",      sm_results),
                              ("M_to_S",      ms_results)]:
            if direction in ("S_in_domain", "M_in_domain"):
                acc_key, auc_key = "acc_in", "auc_in"
            else:
                acc_key, auc_key = "acc_cross", "auc_cross"
            accs = [r[acc_key] for r in rs]
            aucs = [r[auc_key] for r in rs]
            results.append({
                "K": K,
                "n_features": len(feats),
                "classifier": "catboost_tuned",
                "direction": direction,
                "acc_mean": np.mean(accs),
                "acc_std": np.std(accs, ddof=1),
                "auc_mean": np.mean(aucs),
                "auc_std": np.std(aucs, ddof=1),
                "n_seeds": args.n_seeds,
                "features": ";".join(feats),
            })

        elapsed = time.time() - t_k_start
        sm_acc = np.mean([r["acc_cross"] for r in sm_results])
        ms_acc = np.mean([r["acc_cross"] for r in ms_results])
        asym = abs(sm_acc - ms_acc)
        print(f"    K={K}: S->M={sm_acc:.4f}, M->S={ms_acc:.4f}, "
              f"asym={asym:.4f}, elapsed={elapsed:.0f}s")

        # Free CatBoost memory before next K iteration
        del sm_results, ms_results, s_in_results, m_in_results
        gc.collect()

        # Save incremental progress: write CSV after each K so we have
        # partial results if the run is interrupted.
        df_partial = pd.DataFrame(results)
        df_partial.to_csv(Path(args.out_dir) / "k_sweep_results.csv", index=False)

    # Save
    df = pd.DataFrame(results)
    out_csv = Path(args.out_dir) / "k_sweep_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n  Saved: {out_csv}")

    # Summary
    summary_path = Path(args.out_dir) / "k_sweep_summary.txt"
    with open(summary_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("K-sweep Universal-4 anchored — summary\n")
        f.write("=" * 70 + "\n")
        f.write(f"  Classifier:    CatBoost (tuned)\n")
        f.write(f"  Bootstrap seeds: {args.n_seeds}\n")
        f.write(f"  K values:      {k_list}\n\n")
        f.write(f"  K  | S->S    | M->M    | S->M    | M->S    | asym\n")
        f.write(f"  ---|---------|---------|---------|---------|------\n")
        for K in k_list:
            s_in = df[(df.K==K) & (df.direction=="S_in_domain")]["acc_mean"].iloc[0]
            m_in = df[(df.K==K) & (df.direction=="M_in_domain")]["acc_mean"].iloc[0]
            sm = df[(df.K==K) & (df.direction=="S_to_M")]["acc_mean"].iloc[0]
            ms = df[(df.K==K) & (df.direction=="M_to_S")]["acc_mean"].iloc[0]
            f.write(f"  {K:2d} | {s_in:.4f}  | {m_in:.4f}  | "
                    f"{sm:.4f}  | {ms:.4f}  | {abs(sm-ms):.4f}\n")
    print(f"  Saved: {summary_path}")

    total_elapsed = time.time() - t_total_start
    print(f"\n  Total elapsed: {total_elapsed/60:.1f} min")


if __name__ == "__main__":
    main()