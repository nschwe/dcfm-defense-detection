#!/usr/bin/env python3
"""
feature_importance_sensitivity_v2.py
====================================

Universal Feature Set K selection with sensitivity across importance methods,
classifiers, and stability thresholds.

VERSION 2 NOTES:
This is a strict-observability variant. Three features removed because the
underlying signal does not travel across the network:
    HelloMessageRate, ControlPacketRate, AverageRoutingTableSize.

Feature space: 33 raw observable metrics (down from 36 in v1).
The previous Universal Set (Universal-5) included AverageRoutingTableSize and
is therefore invalid; this script recomputes the Universal Set from scratch.

Default args adjusted for execution from inside strict_observable_v2/:
  --static-root  ../simulations/features_static
  --mobile-root  ../simulations/features_mobile
  --hp-results-dir ./results/hp_search_extended  (was: ./unified_hp_search_results_FINAL)
  --out-dir      ./results/feature_importance
  --k-max        33  (was 36)

Methodology
-----------
Feature space: 33 raw observable metrics from defense_detection_v2.py METRICS.

Importance methods (6, all bootstrap-stable across 20 seeds):
  1. RF native (feature_importances_)
  2. XGBoost gain
  3. CatBoost PredictionValuesChange
  4. Permutation importance on RandomForest
  5. Permutation importance on XGBoost
  6. Permutation importance on CatBoost

Universal Set construction (two variants reported in parallel):
  - Soft  : top-(K_static) intersect top-(K_mobile) per seed; supplement to
            exactly K via avg importance if fewer than K features pass the
            stability threshold
  - Strict: top-(K_static) intersect top-(K_mobile) per seed; final set is
            only those features that pass the stability threshold (size <= K)

Stability thresholds explored: {0.6, 0.7, 0.8, 0.9}.

Cross-domain evaluation:
  - GroupShuffleSplit by file_source on the source domain (60/20/20).
  - Three classifiers, each loaded with best_params from
    unified_hp_search_results_FINAL/{static,mobile}/best_models.pkl:
        XGBoost, CatBoost, Stacking_Ensemble.
  - 20 random seeds per (K, method, variant, classifier, direction).
  - Two transfer directions: S->M and M->S.

Outputs
-------
  results_main.csv          : K x method x variant x classifier x direction
  stability_per_threshold.csv : feature stability counts at each threshold
  importance_cache_*.pkl    : cached importance scores (one per method)
  progress_state.pkl        : checkpoint for resume
  summary.txt               : recommended K per (method, variant)
  optimal_k_curves.png      : 6-panel plot (one per importance method)

Usage
-----
  python3 feature_importance_sensitivity.py \\
      --static-root ./simulations/features_static \\
      --mobile-root ./simulations/features_mobile \\
      --hp-results-dir ./unified_hp_search_results_FINAL \\
      --out-dir ./feature_sensitivity_output

  # Resume after crash:
  python3 feature_importance_sensitivity.py ... --resume
"""

from __future__ import annotations

import os
import sys
import glob
import time
import signal
import argparse
import warnings
import pickle
import traceback
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import joblib as jl
from joblib import Parallel, delayed

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.inspection import permutation_importance

import xgboost as xgb
from catboost import CatBoostClassifier

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ============================================================
# CONSTANTS
# ============================================================

# 33 raw metrics — must match defense_detection_v2.py:METRICS exactly
METRICS = [
    # Control-plane metrics observable from network-traveling traffic
    # (TC, MID, HNA flooded; HELLO/ControlPacketRate removed in v2)
    "TcMessageRate", "MidMessageRate", "HnaMessageRate",
    "AverageAdvertisedLinksPerTCMessage",
    "NormalizedRoutingLoad", "RoutingOverheadRatio", "RoutingOverheadBytesRatio",
    # Data-plane metrics (observable from traffic)
    "PacketDeliveryRatio", "PacketLossRatio", "AverageEndToEndDelay", "AverageJitter",
    "Throughput", "AverageHopCount", "DataPacketRate", "RxTxPacketRatio",
    "FlowCount", "AvgFlowDuration", "FlowDurationStd", "AvgFlowThroughput",
    "AvgFlowDelay", "AvgFlowJitter", "AvgFlowLossRate", "FlowThroughputStd",
    "FlowDelayStd", "FlowJitterStd", "FlowLossRateStd",
    "AvgTxBytesPerFlow", "AvgRxBytesPerFlow", "AvgTxPacketsPerFlow", "AvgRxPacketsPerFlow",
    "AvgTxPacketSize", "AvgRxPacketSize",
    # Inferable from TC messages (AverageRoutingTableSize removed in v2)
    "AverageMprCount",
]

SCENARIOS = {
    "baseline": 0,
    "attack_only": 0,
    "defense_only": 1,
    "defense_vs_attack": 1,
}

# Methods: 3 native + 3 permutation
NATIVE_METHODS = ["rf_native", "xgb_gain", "catboost_pvc"]
PERMUTATION_METHODS = ["perm_rf", "perm_xgb", "perm_catboost"]
ALL_METHODS = NATIVE_METHODS + PERMUTATION_METHODS

# Variants of universal set construction
VARIANTS = ["soft", "strict"]

# Stability thresholds for sensitivity analysis
THRESHOLDS = [0.6, 0.7, 0.8, 0.9]

# Evaluation classifiers
EVAL_CLASSIFIERS = ["xgboost", "catboost", "stacking"]

# Default bootstrap parameters
DEFAULT_N_SEEDS = 20
DEFAULT_K_RANGE = list(range(1, 34))  # K = 1..33
DEFAULT_PERM_REPEATS = 5

# Reduced parallelism for memory-heavy classifiers
REDUCED_PARALLELISM_CLASSIFIERS = {"stacking"}
DEFAULT_MAX_JOBS = 16
REDUCED_MAX_JOBS = 12

# Train/test split for source domain
TRAIN_VAL_TEST = (0.6, 0.2, 0.2)


# ============================================================
# DATA LOADING
# ============================================================

def _load_one_csv(csv_path: str, label: int, scenario: str):
    try:
        df = pd.read_csv(csv_path)
        df["defense_active"] = label
        df["scenario"] = scenario
        df["file_source"] = os.path.basename(csv_path)
        return df
    except Exception:
        return None


def load_dataset(data_root: str, n_jobs: int = -1) -> pd.DataFrame:
    tasks = []
    for scenario, label in SCENARIOS.items():
        scenario_dir = os.path.join(data_root, scenario)
        if not os.path.exists(scenario_dir):
            continue
        for csv_path in glob.glob(os.path.join(scenario_dir, "*.csv")):
            tasks.append((csv_path, label, scenario))
    if not tasks:
        raise ValueError(f"No CSV files found in: {data_root}")
    print(f"    Loading {len(tasks)} files from {data_root} ...")
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_load_one_csv)(p, l, s) for p, l, s in tasks
    )
    parts = [r for r in results if r is not None]
    return pd.concat(parts, ignore_index=True)


def to_wide(tall_df: pd.DataFrame):
    """Convert tall metric format to wide table of features.
    Returns X (DataFrame), y (Series), groups (Series of file_source)."""
    tall_df = tall_df[tall_df["Metric"].isin(METRICS)].copy()
    agg = (tall_df.groupby(["scenario", "file_source", "Metric"], sort=False)["Value"]
                  .mean().reset_index())
    wide = agg.pivot_table(
        index=["scenario", "file_source"],
        columns="Metric",
        values="Value",
        aggfunc="mean",
    )
    wide.columns.name = None
    label_map = (tall_df[["scenario", "file_source", "defense_active"]]
                 .drop_duplicates(["scenario", "file_source"])
                 .set_index(["scenario", "file_source"])["defense_active"])
    y = label_map.reindex(wide.index).astype(int)
    # Ensure every metric is a column (fill missing with 0.0)
    for m in METRICS:
        if m not in wide.columns:
            wide[m] = 0.0
    X = wide[METRICS].fillna(0.0).reset_index()
    groups = X["file_source"].copy()
    X = X[METRICS]
    y = y.reset_index(drop=True)
    return X, y, groups


# ============================================================
# BEST PARAMS LOADING
# ============================================================

def load_best_params(hp_results_dir: str) -> dict:
    """Load best hyperparameters from unified_hp_search_results_FINAL.

    Returns: {config: {model_name: dict_of_params}}.
    Falls back to library defaults if a model is missing.
    """
    out = {}
    for cfg in ["static", "mobile"]:
        pkl_path = Path(hp_results_dir) / cfg / "best_models.pkl"
        if not pkl_path.exists():
            raise FileNotFoundError(
                f"Missing tuned models: {pkl_path}\n"
                f"Run unified_hp_search.py first, or pass --hp-results-dir <path>"
            )
        with open(pkl_path, "rb") as f:
            best_models = pickle.load(f)
        # Each entry is typically {model_name: trained_estimator}
        # We extract get_params() from each estimator.
        cfg_params = {}
        for name, est in best_models.items():
            try:
                cfg_params[name.lower()] = est.get_params()
            except Exception:
                cfg_params[name.lower()] = {}
        out[cfg] = cfg_params
    return out


def _filter_xgb_params(p: dict) -> dict:
    """Keep only parameters that XGBClassifier accepts on the user's stack."""
    keep = {"n_estimators", "learning_rate", "max_depth", "subsample",
            "colsample_bytree", "min_child_weight", "gamma", "reg_alpha",
            "reg_lambda", "scale_pos_weight"}
    return {k: v for k, v in p.items() if k in keep}


def _filter_catboost_params(p: dict) -> dict:
    keep = {"iterations", "learning_rate", "depth", "l2_leaf_reg",
            "border_count", "bagging_temperature", "random_strength"}
    return {k: v for k, v in p.items() if k in keep}


def build_xgboost(best_params: dict, seed: int, n_jobs: int = 1):
    p = _filter_xgb_params(best_params.get("xgboost", {}))
    return xgb.XGBClassifier(
        random_state=seed, n_jobs=n_jobs, verbosity=0,
        eval_metric="logloss",
        tree_method="hist", **p
    )


def build_catboost(best_params: dict, seed: int, n_jobs: int = 1):
    p = _filter_catboost_params(best_params.get("catboost", {}))
    return CatBoostClassifier(
        random_seed=seed, verbose=0, allow_writing_files=False,
        thread_count=n_jobs, **p
    )


def build_rf(seed: int, n_jobs: int = 1):
    """RF for permutation baseline. Uses a fixed reasonable config —
    not tuned, since it's only an importance probe (consistent with
    the original optimal_k_bootstrap.py design)."""
    return RandomForestClassifier(
        n_estimators=300, random_state=seed, n_jobs=n_jobs
    )


def build_stacking(best_params: dict, seed: int, n_jobs: int = 1,
                    cv_splits=None):
    """Build a Stacking ensemble for K-search evaluation.

    Uses 3 base learners spanning the dominant tree-based paradigms
    (gradient boosting tree, gradient boosting ordered, bagging tree),
    with L2 Logistic Regression as meta-learner. The full 11-base
    ensemble used for in-domain classification is reported separately;
    a reduced 3-base setup is used here to keep K-search tractable.
    Empirical gap to the full ensemble in prior in-domain analyses
    is small (Delta acc < 0.005).

    Parallelism strategy:
        StackingClassifier outer n_jobs=3 (3 base learners run in
        parallel — one per worker). Each base learner internally
        uses n_jobs=4 threads. Total: 3 outer * 4 inner = 12 cores
        active concurrently. This matches the typical core budget
        and avoids both nested oversubscription (when outer >> 3)
        and serial execution (when outer = 1).

    cv_splits: optional pre-computed list of (train, test) index pairs
    for group-aware cross-validation (prevents leakage across windows
    of the same simulation run). If None, falls back to default cv=3.
    """
    bases = [
        ("xgb", build_xgboost(best_params, seed, n_jobs=4)),
        ("catboost", build_catboost(best_params, seed, n_jobs=4)),
        ("rf", build_rf(seed, n_jobs=4)),
    ]
    meta = LogisticRegression(C=1.0, penalty="l2", max_iter=5000,
                              random_state=seed, n_jobs=1)
    cv_param = cv_splits if cv_splits is not None else 3
    return StackingClassifier(
        estimators=bases, final_estimator=meta,
        stack_method="predict_proba", cv=cv_param,
        n_jobs=3,  # parallelize across 3 base learners
    )


def build_classifier(name: str, best_params: dict, seed: int,
                     n_jobs: int = 1, cv_splits=None):
    name = name.lower()
    if name == "xgboost":
        return build_xgboost(best_params, seed, n_jobs=n_jobs)
    if name == "catboost":
        return build_catboost(best_params, seed, n_jobs=n_jobs)
    if name == "stacking":
        return build_stacking(best_params, seed, n_jobs=n_jobs,
                              cv_splits=cv_splits)
    raise ValueError(f"Unknown classifier: {name}")


# ============================================================
# IMPORTANCE COMPUTATION (one method, one seed)
# ============================================================

def _importance_one_seed_native(method: str, X: pd.DataFrame, y: pd.Series,
                                 best_params: dict, seed: int) -> pd.Series:
    """Native importance: RF, XGBoost gain, or CatBoost PVC."""
    Xs = StandardScaler().fit_transform(X)

    if method == "rf_native":
        m = build_rf(seed, n_jobs=1)
        m.fit(Xs, y)
        return pd.Series(m.feature_importances_, index=X.columns)

    if method == "xgb_gain":
        m = build_xgboost(best_params, seed, n_jobs=1)
        m.fit(Xs, y)
        booster = m.get_booster()
        gain = booster.get_score(importance_type="gain")
        scores = np.zeros(len(X.columns))
        for k, v in gain.items():
            # XGBoost returns 'f0', 'f1', ... — index by position
            idx = int(k[1:])
            scores[idx] = v
        if scores.sum() > 0:
            scores = scores / scores.sum()
        return pd.Series(scores, index=X.columns)

    if method == "catboost_pvc":
        m = build_catboost(best_params, seed)
        m.fit(Xs, y)
        # PredictionValuesChange is the default importance type
        imp = m.get_feature_importance(type="PredictionValuesChange")
        if imp.sum() > 0:
            imp = imp / imp.sum()
        return pd.Series(imp, index=X.columns)

    raise ValueError(f"Unknown native method: {method}")


def _importance_one_seed_permutation(method: str, X: pd.DataFrame, y: pd.Series,
                                      best_params: dict, seed: int,
                                      n_repeats: int = DEFAULT_PERM_REPEATS
                                      ) -> pd.Series:
    """Permutation importance on RF / XGBoost / CatBoost.

    Uses an internal 80/20 stratified split: model is fit on 80% and
    permutation importance is measured on the held-out 20%. This avoids
    the inflation that occurs when permutation importance is measured
    on the same data the model was trained on.
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(sss.split(X, y))

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X.iloc[train_idx])
    X_test = scaler.transform(X.iloc[test_idx])
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    if method == "perm_rf":
        m = build_rf(seed, n_jobs=1)
    elif method == "perm_xgb":
        m = build_xgboost(best_params, seed, n_jobs=1)
    elif method == "perm_catboost":
        m = build_catboost(best_params, seed, n_jobs=1)
    else:
        raise ValueError(f"Unknown permutation method: {method}")

    m.fit(X_train, y_train)
    r = permutation_importance(
        m, X_test, y_test,
        n_repeats=n_repeats, random_state=seed, n_jobs=1,
        scoring="accuracy",
    )
    imp = r.importances_mean
    # Clip negatives to 0 (negative permutation imp = noise)
    imp = np.clip(imp, 0, None)
    if imp.sum() > 0:
        imp = imp / imp.sum()
    return pd.Series(imp, index=X.columns)


def importance_one_seed(method: str, X: pd.DataFrame, y: pd.Series,
                         best_params: dict, seed: int) -> pd.Series:
    if method in NATIVE_METHODS:
        return _importance_one_seed_native(method, X, y, best_params, seed)
    if method in PERMUTATION_METHODS:
        return _importance_one_seed_permutation(method, X, y, best_params, seed)
    raise ValueError(f"Unknown method: {method}")


# ============================================================
# UNIVERSAL SET CONSTRUCTION
# ============================================================

def build_feature_set(k: int,
                       all_imp_static: list[pd.Series],
                       all_imp_mobile: list[pd.Series],
                       threshold: float,
                       variant: str) -> tuple[list[str], int]:
    """Construct the universal feature set for given K, threshold, variant.

    Soft variant: always returns exactly K features (supplements via avg imp).
    Strict variant: returns only stable features (size <= K).

    The returned feature list is always sorted alphabetically. This is
    important for two reasons: (a) deterministic ordering makes the
    pipeline reproducible, and (b) tree-based classifiers (RF, XGBoost)
    are sensitive to column order (tie-breaking in splits depends on
    feature evaluation order), so feeding them columns in the same
    canonical order as the cache key ensures cached results match what
    a re-run would produce.

    Returns: (feature_list_sorted, n_passing_threshold)
    """
    n = len(all_imp_static)
    feature_counts: dict[str, int] = {}
    for imp_s, imp_m in zip(all_imp_static, all_imp_mobile):
        top_s = set(imp_s.nlargest(k).index)
        top_m = set(imp_m.nlargest(k).index)
        for f in top_s & top_m:
            feature_counts[f] = feature_counts.get(f, 0) + 1

    stable = [f for f, cnt in feature_counts.items()
              if cnt / n >= threshold]
    n_passing = len(stable)

    if variant == "strict":
        # Use only the features that passed the threshold; size <= K
        return sorted(stable), n_passing

    # Soft variant: always return exactly K features (supplement if needed)
    avg_imp = sum([(s + m) / 2 for s, m in
                    zip(all_imp_static, all_imp_mobile)]) / n
    if n_passing >= k:
        # Truncate to K (keep highest-avg-importance among stable)
        ranked = avg_imp.loc[stable].sort_values(ascending=False)
        feat_set = list(ranked.head(k).index)
    else:
        # Stable + supplement to exactly K via avg importance
        avg_ranked = avg_imp.sort_values(ascending=False)
        feat_set = list(stable)
        for f in avg_ranked.index:
            if f not in feat_set:
                feat_set.append(f)
            if len(feat_set) >= k:
                break
    return sorted(feat_set), n_passing


# ============================================================
# CROSS-DOMAIN EVALUATION
# ============================================================

def split_source(X: pd.DataFrame, y: pd.Series, groups: pd.Series,
                 seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """GroupShuffleSplit by file_source: 60/20/20."""
    train_frac, val_frac, test_frac = TRAIN_VAL_TEST
    # First split: (train+val) vs test
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_frac,
                              random_state=seed)
    trainval_idx, test_idx = next(gss1.split(X, y, groups))
    # Second split: train vs val from the train+val portion
    rel_val = val_frac / (train_frac + val_frac)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=rel_val,
                              random_state=seed)
    train_idx_rel, val_idx_rel = next(
        gss2.split(X.iloc[trainval_idx], y.iloc[trainval_idx],
                    groups.iloc[trainval_idx]))
    train_idx = trainval_idx[train_idx_rel]
    val_idx = trainval_idx[val_idx_rel]
    return train_idx, val_idx, test_idx


def _select_threshold(y_true, p_proba):
    """Select decision threshold maximizing accuracy on validation.

    Sweeps thresholds in [0.1, 0.9] with step 0.01 (matches Section V-C
    of the paper). Returns the threshold value.
    """
    candidates = np.arange(0.1, 0.901, 0.01)
    best_thr, best_acc = 0.5, -1.0
    for thr in candidates:
        acc = accuracy_score(y_true, p_proba >= thr)
        if acc > best_acc:
            best_acc, best_thr = acc, float(thr)
    return best_thr


def _make_group_cv_splits(groups: np.ndarray, n_splits: int = 3) -> list:
    """Build group-aware CV splits for use as `cv` in StackingClassifier.

    Returns a list of (train_idx, test_idx) pairs that respect the
    file_source grouping (no run is split across train/test of the
    inner CV used to build meta-features).

    Note: GroupKFold splits are deterministic by group ordering; no seed
    parameter is supported by sklearn's GroupKFold.
    """
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=n_splits)
    return list(gkf.split(np.zeros(len(groups)), np.zeros(len(groups)),
                           groups=groups))


def evaluate_one_cell(K: int, method: str, variant: str, classifier_name: str,
                       seed: int,
                       X_s: pd.DataFrame, y_s: pd.Series, g_s: pd.Series,
                       X_m: pd.DataFrame, y_m: pd.Series, g_m: pd.Series,
                       feat_set: list[str],
                       best_params_static: dict,
                       best_params_mobile: dict,
                       n_jobs_classifier: int) -> dict:
    """Evaluate one (K, method, variant, classifier, seed) cell.

    Pipeline (matches Section V of the paper):
      1. Group-aware split source domain into train (60%) / val (20%) /
         test (20%).
      2. Fit classifier on train. For Stacking, the inner cv is also
         group-aware (prevents within-run leakage in meta-feature
         construction).
      3. Tune decision threshold on val to maximize accuracy.
      4. Report acc on source test (in-domain) and on full target
         (cross-domain).

    Returns dict with acc/auc for both directions and source-test scores.
    """
    if not feat_set:
        return {
            "K": K, "method": method, "variant": variant,
            "classifier": classifier_name, "seed": seed,
            "n_features": 0,
            "acc_S_test": np.nan, "auc_S_test": np.nan,
            "acc_M_test": np.nan, "auc_M_test": np.nan,
            "acc_SM": np.nan, "auc_SM": np.nan,
            "acc_MS": np.nan, "auc_MS": np.nan,
            "asymmetry": np.nan,
            "thr_static": np.nan, "thr_mobile": np.nan,
        }

    # Subset features
    X_s_f = X_s[feat_set]
    X_m_f = X_m[feat_set]

    # ── Source: static -> in-domain test + cross-domain transfer to mobile
    tr_s, va_s, te_s = split_source(X_s_f, y_s, g_s, seed=seed)
    sc_s = StandardScaler().fit(X_s_f.iloc[tr_s])
    Xs_tr = sc_s.transform(X_s_f.iloc[tr_s])
    Xs_va = sc_s.transform(X_s_f.iloc[va_s])
    Xs_te = sc_s.transform(X_s_f.iloc[te_s])
    Xs_mall = sc_s.transform(X_m_f)

    # Group-aware CV splits for Stacking inner CV (prevents within-run leakage)
    cv_splits_s = None
    if classifier_name == "stacking":
        cv_splits_s = _make_group_cv_splits(
            g_s.iloc[tr_s].to_numpy(), n_splits=3)

    clf_s = build_classifier(classifier_name, best_params_static, seed,
                              n_jobs=n_jobs_classifier, cv_splits=cv_splits_s)
    clf_s.fit(Xs_tr, y_s.iloc[tr_s])

    # Threshold selection on validation set
    p_s_va = clf_s.predict_proba(Xs_va)[:, 1]
    thr_s = _select_threshold(y_s.iloc[va_s], p_s_va)

    p_s_te = clf_s.predict_proba(Xs_te)[:, 1]
    p_s_to_m = clf_s.predict_proba(Xs_mall)[:, 1]
    acc_S_te = accuracy_score(y_s.iloc[te_s], p_s_te >= thr_s)
    auc_S_te = roc_auc_score(y_s.iloc[te_s], p_s_te)
    acc_SM = accuracy_score(y_m, p_s_to_m >= thr_s)
    auc_SM = roc_auc_score(y_m, p_s_to_m)

    # ── Source: mobile -> in-domain test + cross-domain transfer to static
    tr_m, va_m, te_m = split_source(X_m_f, y_m, g_m, seed=seed)
    sc_m = StandardScaler().fit(X_m_f.iloc[tr_m])
    Xm_tr = sc_m.transform(X_m_f.iloc[tr_m])
    Xm_va = sc_m.transform(X_m_f.iloc[va_m])
    Xm_te = sc_m.transform(X_m_f.iloc[te_m])
    Xm_sall = sc_m.transform(X_s_f)

    cv_splits_m = None
    if classifier_name == "stacking":
        cv_splits_m = _make_group_cv_splits(
            g_m.iloc[tr_m].to_numpy(), n_splits=3)

    clf_m = build_classifier(classifier_name, best_params_mobile, seed,
                              n_jobs=n_jobs_classifier, cv_splits=cv_splits_m)
    clf_m.fit(Xm_tr, y_m.iloc[tr_m])

    p_m_va = clf_m.predict_proba(Xm_va)[:, 1]
    thr_m = _select_threshold(y_m.iloc[va_m], p_m_va)

    p_m_te = clf_m.predict_proba(Xm_te)[:, 1]
    p_m_to_s = clf_m.predict_proba(Xm_sall)[:, 1]
    acc_M_te = accuracy_score(y_m.iloc[te_m], p_m_te >= thr_m)
    auc_M_te = roc_auc_score(y_m.iloc[te_m], p_m_te)
    acc_MS = accuracy_score(y_s, p_m_to_s >= thr_m)
    auc_MS = roc_auc_score(y_s, p_m_to_s)

    return {
        "K": K, "method": method, "variant": variant,
        "classifier": classifier_name, "seed": seed,
        "n_features": len(feat_set),
        "acc_S_test": float(acc_S_te), "auc_S_test": float(auc_S_te),
        "acc_M_test": float(acc_M_te), "auc_M_test": float(auc_M_te),
        "acc_SM": float(acc_SM), "auc_SM": float(auc_SM),
        "acc_MS": float(acc_MS), "auc_MS": float(auc_MS),
        "asymmetry": float(abs(acc_SM - acc_MS)),
        "thr_static": float(thr_s), "thr_mobile": float(thr_m),
    }


# ============================================================
# CHECKPOINT / RESUME
# ============================================================

@dataclass
class ProgressState:
    completed_cells: set = field(default_factory=set)  # set of tuples
    importance_done: dict = field(default_factory=dict)  # method -> True

    def mark_cell_done(self, K, method, variant, classifier, seed):
        self.completed_cells.add((K, method, variant, classifier, seed))

    def is_cell_done(self, K, method, variant, classifier, seed) -> bool:
        return (K, method, variant, classifier, seed) in self.completed_cells

    def mark_importance_done(self, method: str):
        self.importance_done[method] = True


def save_progress(state: ProgressState, path: Path):
    tmp_path = path.with_suffix(".pkl.tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(state, f)
    os.replace(tmp_path, path)  # atomic on POSIX


def load_progress(path: Path) -> ProgressState:
    if not path.exists():
        return ProgressState()
    with open(path, "rb") as f:
        return pickle.load(f)


# ============================================================
# IMPORTANCE COMPUTATION (all seeds, with cache)
# ============================================================

def compute_importance_for_method(method: str,
                                   X_s: pd.DataFrame, y_s: pd.Series,
                                   X_m: pd.DataFrame, y_m: pd.Series,
                                   best_params_static: dict,
                                   best_params_mobile: dict,
                                   n_seeds: int,
                                   max_jobs: int,
                                   cache_path: Path) -> tuple[list[pd.Series], list[pd.Series]]:
    """Compute importance scores for all seeds; cache to disk.

    Permutation methods are run sequentially per seed (each seed already
    parallelizes internally over n_repeats x n_features). Native methods
    are parallelized over seeds.
    """
    if cache_path.exists():
        print(f"    [{method}] cache hit: {cache_path.name}")
        cache = jl.load(cache_path)
        return cache["static"], cache["mobile"]

    print(f"    [{method}] computing importance over {n_seeds} seeds ...")
    t0 = time.time()

    if method in NATIVE_METHODS:
        # Parallel over seeds; each seed runs lightweight native importance
        results_s = Parallel(n_jobs=max_jobs, backend="loky")(
            delayed(importance_one_seed)(method, X_s, y_s,
                                         best_params_static, seed)
            for seed in range(n_seeds)
        )
        results_m = Parallel(n_jobs=max_jobs, backend="loky")(
            delayed(importance_one_seed)(method, X_m, y_m,
                                         best_params_mobile, seed)
            for seed in range(n_seeds)
        )
    else:
        # Permutation: parallelize over seeds with reduced jobs to keep memory in check
        perm_jobs = min(max_jobs, 8)
        results_s = Parallel(n_jobs=perm_jobs, backend="loky")(
            delayed(importance_one_seed)(method, X_s, y_s,
                                         best_params_static, seed)
            for seed in range(n_seeds)
        )
        results_m = Parallel(n_jobs=perm_jobs, backend="loky")(
            delayed(importance_one_seed)(method, X_m, y_m,
                                         best_params_mobile, seed)
            for seed in range(n_seeds)
        )

    elapsed = time.time() - t0
    print(f"    [{method}] done in {elapsed:.1f}s")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    jl.dump({"static": results_s, "mobile": results_m}, cache_path)
    return results_s, results_m


# ============================================================
# MAIN EVALUATION LOOP
# ============================================================

def run_main_evaluation(X_s, y_s, g_s, X_m, y_m, g_m,
                         imp_by_method: dict,
                         best_params_static: dict,
                         best_params_mobile: dict,
                         out_dir: Path,
                         n_seeds: int,
                         k_values: list[int],
                         threshold: float,
                         max_jobs: int,
                         state: ProgressState,
                         resume: bool):
    """Iterate over all (K, method, variant, classifier, seed) cells and
    evaluate. Append results to CSV incrementally."""
    csv_path = out_dir / "results_main.csv"
    progress_path = out_dir / "progress_state.pkl"

    # Build feat_sets up front (cheap)
    feat_sets: dict = {}  # (method, variant, K) -> (feat_list, n_pass)
    for method in ALL_METHODS:
        all_imp_s = imp_by_method[method]["static"]
        all_imp_m = imp_by_method[method]["mobile"]
        for variant in VARIANTS:
            for K in k_values:
                feat_list, n_pass = build_feature_set(
                    K, all_imp_s, all_imp_m, threshold, variant)
                feat_sets[(method, variant, K)] = (feat_list, n_pass)

    # Compute total cells
    total_cells = (len(k_values) * len(ALL_METHODS) * len(VARIANTS) *
                   len(EVAL_CLASSIFIERS) * n_seeds)
    done_at_start = sum(
        1 for K in k_values for method in ALL_METHODS for variant in VARIANTS
        for clf in EVAL_CLASSIFIERS for seed in range(n_seeds)
        if state.is_cell_done(K, method, variant, clf, seed)
    )
    print(f"\n  Total cells: {total_cells}  (already done: {done_at_start})")

    # Prepare CSV: write header if not resuming
    CSV_COLS = ["K", "method", "variant", "classifier", "seed",
                 "n_features", "n_features_passing_threshold",
                 "acc_S_test", "auc_S_test", "acc_M_test", "auc_M_test",
                 "acc_SM", "auc_SM", "acc_MS", "auc_MS", "asymmetry",
                 "thr_static", "thr_mobile"]
    if not csv_path.exists() or not resume:
        with open(csv_path, "w") as f:
            f.write(",".join(CSV_COLS) + "\n")

    t_start = time.time()
    cells_done_this_run = 0
    last_save_t = time.time()
    SAVE_INTERVAL = 60.0  # seconds

    # Optimization: cache evaluation results keyed by (frozenset(feat_set),
    # classifier, seed). Many (method, variant, K) cells produce the same
    # feat_set — we evaluate it once and reuse the result.
    eval_cache: dict = {}

    # Order: outer loop over expensive things first so that early failures
    # surface quickly. Stacking is heaviest, so put it last.
    classifier_order = ["xgboost", "catboost", "stacking"]

    for classifier_name in classifier_order:
        n_jobs_clf = (REDUCED_MAX_JOBS
                       if classifier_name in REDUCED_PARALLELISM_CLASSIFIERS
                       else max_jobs)
        # We do not parallelize over the cell loop (stable resume + simpler
        # memory profile). Parallelism is inside the classifier where it can.
        for method in ALL_METHODS:
            for variant in VARIANTS:
                for K in k_values:
                    feat_list, n_pass = feat_sets[(method, variant, K)]
                    for seed in range(n_seeds):
                        if state.is_cell_done(K, method, variant,
                                              classifier_name, seed):
                            continue
                        try:
                            cache_key = (frozenset(feat_list),
                                          classifier_name, seed)
                            if cache_key in eval_cache:
                                cached = eval_cache[cache_key]
                                # Override the labelling (K, method, variant)
                                # but keep the metric values
                                res = dict(cached)
                                res["K"] = K
                                res["method"] = method
                                res["variant"] = variant
                                res["classifier"] = classifier_name
                                res["seed"] = seed
                            else:
                                res = evaluate_one_cell(
                                    K, method, variant, classifier_name,
                                    seed,
                                    X_s, y_s, g_s, X_m, y_m, g_m,
                                    feat_list,
                                    best_params_static, best_params_mobile,
                                    n_jobs_classifier=n_jobs_clf,
                                )
                                eval_cache[cache_key] = dict(res)
                            res["n_features_passing_threshold"] = n_pass
                            with open(csv_path, "a") as f:
                                f.write(",".join(
                                    str(res.get(c, "")) for c in CSV_COLS
                                ) + "\n")
                            state.mark_cell_done(K, method, variant,
                                                  classifier_name, seed)
                            cells_done_this_run += 1
                        except Exception as e:
                            err_msg = (f"FAILED cell K={K} method={method} "
                                        f"variant={variant} clf={classifier_name} "
                                        f"seed={seed}: {type(e).__name__}: {e}")
                            print(f"    [WARN] {err_msg}")
                            with open(out_dir / "errors.log", "a") as ef:
                                ef.write(err_msg + "\n")
                                ef.write(traceback.format_exc() + "\n")

                        # Periodic checkpoint save
                        now = time.time()
                        if now - last_save_t > SAVE_INTERVAL:
                            save_progress(state, progress_path)
                            last_save_t = now
                            elapsed = now - t_start
                            done_total = done_at_start + cells_done_this_run
                            rate = cells_done_this_run / max(elapsed, 1e-9)
                            remaining = total_cells - done_total
                            eta_s = remaining / max(rate, 1e-9) if rate > 0 else float("inf")
                            print(f"    progress: {done_total}/{total_cells}  "
                                  f"({100*done_total/total_cells:.1f}%)  "
                                  f"rate={rate:.2f}/s  eta={eta_s/3600:.1f}h")

    # Final save
    save_progress(state, progress_path)


# ============================================================
# AGGREGATION & K-SELECTION
# ============================================================

def aggregate_results(out_dir: Path, k_values: list[int]):
    """Compute mean/std over seeds per (K, method, variant, classifier);
    additionally average over classifiers; recommend K per (method, variant)."""
    csv_path = out_dir / "results_main.csv"
    if not csv_path.exists():
        print("  [aggregate] no results_main.csv — skipping")
        return None

    df = pd.read_csv(csv_path)
    if df.empty:
        print("  [aggregate] results_main.csv is empty")
        return None

    # Mean/std over seeds per (K, method, variant, classifier)
    grp_cols = ["K", "method", "variant", "classifier"]
    metric_cols = ["acc_S_test", "auc_S_test", "acc_M_test", "auc_M_test",
                   "acc_SM", "auc_SM", "acc_MS", "auc_MS", "asymmetry"]
    agg = df.groupby(grp_cols, sort=False)[metric_cols].agg(["mean", "std"])
    agg.columns = [f"{m}_{stat}" for m, stat in agg.columns]
    agg = agg.reset_index()
    n_feat_avg = df.groupby(grp_cols, sort=False)[
        ["n_features", "n_features_passing_threshold"]].mean().reset_index()
    agg = agg.merge(n_feat_avg, on=grp_cols)
    agg.to_csv(out_dir / "results_aggregated_per_classifier.csv", index=False)

    # Average over classifiers per (K, method, variant)
    grp2_cols = ["K", "method", "variant"]
    agg2 = (agg.groupby(grp2_cols, sort=False)
              [[f"{m}_mean" for m in metric_cols]]
              .mean()
              .reset_index())
    agg2.to_csv(out_dir / "results_aggregated_avg_classifiers.csv",
                index=False)

    # Recommend K per (method, variant): smallest K reaching plateau
    # in (acc_SM_mean + acc_MS_mean) / 2 across all 3 classifiers
    recommendations = []
    for (method, variant), sub in agg2.groupby(["method", "variant"]):
        sub = sub.sort_values("K")
        # average accuracy across both directions
        sub = sub.assign(
            acc_avg=(sub["acc_SM_mean"] + sub["acc_MS_mean"]) / 2.0
        )
        max_acc = sub["acc_avg"].max()
        # K reaching at least 99% of max
        plateau = sub[sub["acc_avg"] >= 0.99 * max_acc]
        if len(plateau) == 0:
            continue
        rec_k = int(plateau["K"].min())
        rec_row = sub[sub["K"] == rec_k].iloc[0]
        recommendations.append({
            "method": method,
            "variant": variant,
            "recommended_K": rec_k,
            "max_acc_avg": float(max_acc),
            "rec_acc_SM": float(rec_row["acc_SM_mean"]),
            "rec_acc_MS": float(rec_row["acc_MS_mean"]),
            "rec_asymmetry": float(rec_row["asymmetry_mean"]),
        })
    rec_df = pd.DataFrame(recommendations)
    rec_df.to_csv(out_dir / "recommended_K_per_method.csv", index=False)
    return agg, agg2, rec_df


def write_summary(out_dir: Path, threshold: float, n_seeds: int,
                   k_values: list[int]):
    """Human-readable summary."""
    summary_path = out_dir / "summary.txt"
    rec_path = out_dir / "recommended_K_per_method.csv"
    if not rec_path.exists():
        return
    rec = pd.read_csv(rec_path)
    lines = []
    lines.append("=" * 70)
    lines.append("Universal Feature Set K — sensitivity summary")
    lines.append("=" * 70)
    lines.append(f"  bootstrap seeds: {n_seeds}")
    lines.append(f"  stability threshold (primary): {threshold}")
    lines.append(f"  K range: {min(k_values)}..{max(k_values)}")
    lines.append("")
    lines.append("Recommended K per (importance method, variant):")
    lines.append("")
    lines.append(f"  {'method':<16}{'variant':<8}{'K':>4}  "
                 f"{'acc_avg':>8}  {'acc_SM':>8}  {'acc_MS':>8}  {'asym':>6}")
    lines.append("  " + "-" * 64)
    for _, r in rec.iterrows():
        lines.append(f"  {r['method']:<16}{r['variant']:<8}"
                     f"{int(r['recommended_K']):>4}  "
                     f"{r['max_acc_avg']:>8.4f}  "
                     f"{r['rec_acc_SM']:>8.4f}  "
                     f"{r['rec_acc_MS']:>8.4f}  "
                     f"{r['rec_asymmetry']:>6.4f}")
    lines.append("")
    text = "\n".join(lines)
    with open(summary_path, "w") as f:
        f.write(text)
    print("\n" + text)


# ============================================================
# THRESHOLD SENSITIVITY
# ============================================================

def threshold_sensitivity(imp_by_method: dict, k_values: list[int],
                            out_dir: Path):
    """For each (method, K, threshold), report n_features_passing_threshold."""
    rows = []
    for method in ALL_METHODS:
        all_imp_s = imp_by_method[method]["static"]
        all_imp_m = imp_by_method[method]["mobile"]
        for K in k_values:
            for thr in THRESHOLDS:
                _, n_pass = build_feature_set(
                    K, all_imp_s, all_imp_m, thr, variant="strict")
                rows.append({
                    "method": method, "K": K,
                    "threshold": thr,
                    "n_features_passing": n_pass,
                })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "stability_per_threshold.csv", index=False)


# ============================================================
# PLOTTING
# ============================================================

def plot_optimal_k(out_dir: Path, threshold: float):
    csv_path = out_dir / "results_aggregated_avg_classifiers.csv"
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    if df.empty:
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    for idx, method in enumerate(ALL_METHODS):
        ax = axes[idx]
        for variant, color in [("soft", "#4C72B0"), ("strict", "#DD8452")]:
            sub = df[(df["method"] == method) & (df["variant"] == variant)]
            sub = sub.sort_values("K")
            if sub.empty:
                continue
            ax.plot(sub["K"], sub["acc_SM_mean"], "o-", color=color,
                     label=f"S->M ({variant})", alpha=0.85)
            ax.plot(sub["K"], sub["acc_MS_mean"], "s--", color=color,
                     label=f"M->S ({variant})", alpha=0.85)
        ax.set_title(method)
        ax.set_xlabel("K")
        ax.set_ylabel("cross-domain accuracy")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle(f"Cross-domain accuracy vs K  "
                  f"(threshold={threshold}, avg over 3 classifiers)",
                  fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_dir / "optimal_k_curves.png", dpi=140)
    plt.close(fig)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Universal Feature Set K with sensitivity across methods "
                    "and classifiers.",
    )
    parser.add_argument("--static-root", required=True,
                         help="Directory with static dataset CSVs.")
    parser.add_argument("--mobile-root", required=True,
                         help="Directory with mobile dataset CSVs.")
    parser.add_argument("--hp-results-dir",
                         default="./results/hp_search_extended",
                         help="Directory with static/best_models.pkl and "
                              "mobile/best_models.pkl from unified_hp_search_v2.")
    parser.add_argument("--out-dir", default="./results/feature_importance")
    parser.add_argument("--n-seeds", type=int, default=DEFAULT_N_SEEDS)
    parser.add_argument("--threshold", type=float, default=0.8,
                         help="Primary stability threshold for main evaluation.")
    parser.add_argument("--k-min", type=int, default=1)
    parser.add_argument("--k-max", type=int, default=33)
    parser.add_argument("--max-jobs", type=int, default=DEFAULT_MAX_JOBS)
    parser.add_argument("--load-jobs", type=int, default=-1,
                         help="n_jobs for CSV loading (-1 = all cores).")
    parser.add_argument("--resume", action="store_true",
                         help="Resume from existing progress_state.pkl.")
    parser.add_argument("--skip-importance", action="store_true",
                         help="Skip importance computation (require caches "
                              "to exist).")
    parser.add_argument("--skip-evaluation", action="store_true",
                         help="Compute importance and threshold sensitivity, "
                              "but do not run cross-domain evaluation.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "importance_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    k_values = list(range(args.k_min, args.k_max + 1))
    n_seeds = args.n_seeds

    print("=" * 70)
    print("feature_importance_sensitivity.py")
    print("=" * 70)
    print(f"  static-root     : {args.static_root}")
    print(f"  mobile-root     : {args.mobile_root}")
    print(f"  hp-results-dir  : {args.hp_results_dir}")
    print(f"  out-dir         : {args.out_dir}")
    print(f"  n_seeds         : {n_seeds}")
    print(f"  threshold       : {args.threshold}")
    print(f"  K range         : {args.k_min}..{args.k_max}")
    print(f"  max_jobs        : {args.max_jobs}")
    print(f"  resume          : {args.resume}")
    print()

    # ── Step 1: Load datasets ─────────────────────────────────
    t0 = time.time()
    print("Step 1/5: Loading datasets ...")
    print("  Static:")
    X_s, y_s, g_s = to_wide(load_dataset(args.static_root,
                                          n_jobs=args.load_jobs))
    print(f"    static: X={X_s.shape}  y_pos={int(y_s.sum())}/{len(y_s)}  "
          f"groups={g_s.nunique()}")
    print("  Mobile:")
    X_m, y_m, g_m = to_wide(load_dataset(args.mobile_root,
                                          n_jobs=args.load_jobs))
    print(f"    mobile: X={X_m.shape}  y_pos={int(y_m.sum())}/{len(y_m)}  "
          f"groups={g_m.nunique()}")
    print(f"  loading elapsed: {time.time()-t0:.1f}s")

    # ── Step 2: Load best params ───────────────────────────────
    print("\nStep 2/5: Loading tuned hyperparameters ...")
    bp = load_best_params(args.hp_results_dir)
    bp_static = bp["static"]
    bp_mobile = bp["mobile"]
    print(f"  loaded best_params for {len(bp_static)} static models, "
          f"{len(bp_mobile)} mobile models.")
    for k in ("xgboost", "catboost"):
        s_keys = list(_filter_xgb_params(bp_static.get(k, {})).keys()) \
                  if k == "xgboost" else \
                  list(_filter_catboost_params(bp_static.get(k, {})).keys())
        print(f"    static.{k}: {len(s_keys)} relevant params")

    # ── Step 3: Importance computation ─────────────────────────
    print("\nStep 3/5: Bootstrap importance over 6 methods x "
          f"{n_seeds} seeds ...")
    imp_by_method: dict = {}
    for method in ALL_METHODS:
        cache_path = cache_dir / f"importance_{method}.pkl"
        if args.skip_importance and not cache_path.exists():
            raise FileNotFoundError(
                f"--skip-importance set but cache missing: {cache_path}")
        results_s, results_m = compute_importance_for_method(
            method, X_s, y_s, X_m, y_m,
            bp_static, bp_mobile,
            n_seeds=n_seeds, max_jobs=args.max_jobs,
            cache_path=cache_path,
        )
        imp_by_method[method] = {"static": results_s, "mobile": results_m}

    # ── Step 4: Threshold sensitivity (cheap; always run) ───────
    print("\nStep 4/5: Threshold sensitivity (n_features_passing) ...")
    threshold_sensitivity(imp_by_method, k_values, out_dir)
    print(f"  saved: {out_dir/'stability_per_threshold.csv'}")

    if args.skip_evaluation:
        print("\n--skip-evaluation set; stopping here.")
        return

    # ── Step 5: Main cross-domain evaluation ─────────────────────
    print("\nStep 5/5: Cross-domain evaluation ...")
    progress_path = out_dir / "progress_state.pkl"
    state = load_progress(progress_path) if args.resume else ProgressState()

    # SIGINT handler: save and exit
    def handle_sigint(signum, frame):
        print("\n[SIGINT] saving progress_state and exiting ...")
        save_progress(state, progress_path)
        sys.exit(130)
    signal.signal(signal.SIGINT, handle_sigint)

    run_main_evaluation(
        X_s, y_s, g_s, X_m, y_m, g_m,
        imp_by_method, bp_static, bp_mobile,
        out_dir=out_dir,
        n_seeds=n_seeds, k_values=k_values, threshold=args.threshold,
        max_jobs=args.max_jobs, state=state, resume=args.resume,
    )

    # ── Aggregation, summary, plots ─────────────────────────────
    print("\nAggregating results ...")
    aggregate_results(out_dir, k_values)
    write_summary(out_dir, args.threshold, n_seeds, k_values)
    plot_optimal_k(out_dir, args.threshold)
    print("\nDone.")
    print(f"  outputs: {out_dir}")


if __name__ == "__main__":
    main()