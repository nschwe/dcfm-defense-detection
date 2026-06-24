#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
Unified Hyperparameter Search for Defense Detection Models (v2 - strict obs.)
===============================================================================

VERSION 2 NOTES:
This is a strict-observability variant of unified_hp_search.py.
It imports from defense_detection_v2.py (instead of defense_detection.py),
which uses 33 observable features (instead of 36) - removed are:
    HelloMessageRate, ControlPacketRate, AverageRoutingTableSize.

All HP search logic, search spaces, and pipeline behavior remain identical.
Only the data pipeline differs (via the v2 DefenseDetector).

Default paths assume execution from inside strict_observable_v2/:
  --static-root  ../simulations/features_static/
  --mobile-root  ../simulations/features_mobile/
  --out-dir      ./results/hp_search

===============================================================================

Purpose
-------
Single, unified pipeline that replaces three separate scripts:
    hp_search.py + svm_extended.py + svm_focused.py

Tunes every model family that appears in defense_detection.py
(15 base learners + Stacking_Ensemble), so that the final selection of
representative models for Table IV can be made on the basis of empirical
post-tuning performance rather than on default values.

Design decisions (aligned with accepted ML practice)
-----------------------------------------------------
* random_state = 42 everywhere (matches defense_detection.py and Tables III/IV)
* Same data pipeline as defense_detection.py via DefenseDetector:
  load -> engineer -> GroupShuffleSplit 60/20/20 -> RobustScaler ->
  feature selection -> SMOTE
* cv = 3 for HP search (Bergstra & Bengio 2012; standard for moderate data)
* n_iter = 30 for RandomizedSearchCV (extension of the original n_iter = 20)
* scoring = accuracy (matches Table IV primary metric)
* Best model refit on training; evaluated once on held-out test set
* Incremental saves after every model survive long runs
* Stacking_Ensemble is built last, using the tuned base learners
* Edge-of-grid detection: numeric hyperparameters are checked for boundary
  values; warnings are emitted to edge_warnings.txt for follow-up runs

Search-space variants (--grid-mode)
------------------------------------
narrow    : initial grids matching hp_search.py (the default)
extended  : wider grids for RF/XGB/CatBoost/LGBM/SVM
            (intended as a follow-up if narrow-mode best params hit edges)
focused   : SVM-only narrow grid for high-C convergence verification
            (matches svm_focused.py)

Usage
-----
    # First-pass: every model, narrow grids
    python3 unified_hp_search.py --config both --grid-mode narrow

    # Follow-up if any best params hit grid edges (read edge_warnings.txt)
    python3 unified_hp_search.py --config both --grid-mode extended \\
        --models svm,randomforest,xgboost

    # SVM-only convergence check (replaces svm_focused.py)
    python3 unified_hp_search.py --config static --grid-mode focused \\
        --models svm

Outputs (per configuration, under <out_dir>/<config>/)
------------------------------------------------------
* combined_results.csv    : best params, CV acc, test acc, CIs, AUC per model
* full_log.json           : full search history (every candidate evaluated)
* best_models.pkl         : tuned models, ready to load
* edge_warnings.txt       : list of models whose best params hit grid edges
* methodology.txt         : full methodology/justification text
* environment.txt         : Python / library / hardware fingerprint
* runtime.log             : human-readable progress log
"""

import os
# Match environment of defense_detection.py before any heavy imports
os.environ.setdefault("KMP_DISABLE_SHARED_MEM", "1")

import sys
import json
import pickle
import argparse
import datetime
import platform
import warnings
import time
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.model_selection import (
    GroupShuffleSplit, StratifiedKFold,
    RandomizedSearchCV, GridSearchCV,
)
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, BaggingClassifier,
    AdaBoostClassifier, StackingClassifier,
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Optional boosting libraries
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[WARNING] XGBoost not available; will be skipped")

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("[WARNING] CatBoost not available; will be skipped")

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("[WARNING] LightGBM not available; will be skipped")

# Reuse the data pipeline from the main detector
from defense_detection_v2 import DefenseDetector, Config


# ============================================================
# CONSTANTS
# ============================================================
RANDOM_STATE   = 42
N_ITER_RANDOM  = 30          # was 20 in hp_search.py; widened here.
                             # Can be overridden from CLI via --n-iter.
CV_FOLDS       = 3
SCORING        = "accuracy"
N_JOBS_OUTER   = int(os.environ.get("MAX_JOBS",
                                     max(1, (os.cpu_count() or 2) // 2)))
N_JOBS_INNER   = 1   # estimator runs single-threaded inside parallel search


# ============================================================
# SEARCH SPACES
# ============================================================
# Each model has up to two grids: 'narrow' and 'extended'.
# 'narrow' matches hp_search.py / standard practice; 'extended' is a wider
# grid used when 'narrow' best params hit a numerical boundary.
# SVM has an additional 'focused' grid (matches svm_focused.py).

SEARCH_SPACES: Dict[str, Dict[str, Dict[str, list]]] = {

    # -------------------------- Tree-based --------------------------
    "randomforest": {
        "narrow": {
            "n_estimators":       [300, 500, 800],
            "max_depth":          [10, 15, 20, None],
            "min_samples_split":  [2, 5, 10],
            "min_samples_leaf":   [1, 2, 5],
            "max_features":       ["sqrt", "log2", 0.5],
        },
        "extended": {
            "n_estimators":       [300, 500, 800, 1200, 1600],
            "max_depth":          [10, 15, 20, 30, None],
            "min_samples_split":  [2, 5, 10, 20],
            "min_samples_leaf":   [1, 2, 5, 10],
            "max_features":       ["sqrt", "log2", 0.3, 0.5, 0.7],
        },
    },
    "extratrees": {
        "narrow": {
            "n_estimators":       [300, 500, 800],
            "max_depth":          [10, 15, 20, None],
            "min_samples_split":  [2, 5, 10],
            "min_samples_leaf":   [1, 2, 5],
            "max_features":       ["sqrt", "log2", 0.5],
        },
        "extended": {
            "n_estimators":       [300, 500, 800, 1200, 1600],
            "max_depth":          [10, 15, 20, 30, None],
            "min_samples_split":  [2, 5, 10, 20],
            "min_samples_leaf":   [1, 2, 5, 10],
            "max_features":       ["sqrt", "log2", 0.3, 0.5, 0.7],
        },
    },
    "gradientboosting": {
        "narrow": {
            "n_estimators":  [300, 500, 800],
            "learning_rate": [0.03, 0.05, 0.1],
            "max_depth":     [4, 6, 7, 9],
            "subsample":     [0.7, 0.8, 1.0],
            "max_features":  ["sqrt", "log2", 0.5],
        },
        "extended": {
            "n_estimators":  [300, 500, 800, 1200],
            "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
            "max_depth":     [3, 4, 6, 7, 9, 11],
            "subsample":     [0.5, 0.7, 0.8, 1.0],
            "max_features":  ["sqrt", "log2", 0.3, 0.5, 0.7],
        },
    },

    # -------------------------- Boosting libraries --------------------------
    "xgboost": {
        "narrow": {
            "n_estimators":     [500, 800, 1000],
            "learning_rate":    [0.03, 0.05, 0.1],
            "max_depth":        [4, 6, 7, 9],
            "subsample":        [0.7, 0.8, 1.0],
            "colsample_bytree": [0.7, 0.8, 1.0],
            "min_child_weight": [1, 3, 5],
            "reg_lambda":       [1, 3, 5],
        },
        "extended": {
            "n_estimators":     [500, 800, 1000, 1500, 2000],
            "learning_rate":    [0.01, 0.03, 0.05, 0.1, 0.2],
            "max_depth":        [3, 4, 6, 7, 9, 11],
            "subsample":        [0.5, 0.7, 0.8, 1.0],
            "colsample_bytree": [0.5, 0.7, 0.8, 1.0],
            "min_child_weight": [1, 3, 5, 10],
            "reg_lambda":       [0.1, 1, 3, 5, 10],
        },
    },
    "catboost": {
        "narrow": {
            "iterations":    [500, 800, 1000],
            "learning_rate": [0.03, 0.05, 0.1],
            "depth":         [4, 6, 7, 8],
            "l2_leaf_reg":   [1, 3, 5, 10],
            "subsample":     [0.7, 0.8, 1.0],
        },
        "extended": {
            "iterations":    [500, 800, 1000, 1500],
            "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
            "depth":         [4, 6, 7, 8, 9, 10],
            "l2_leaf_reg":   [0.5, 1, 3, 5, 10, 20],
            "subsample":     [0.5, 0.7, 0.8, 1.0],
        },
    },
    "lightgbm": {
        "narrow": {
            "n_estimators":      [500, 800, 1000],
            "learning_rate":     [0.03, 0.05, 0.1],
            "max_depth":         [6, 8, 10, 12],
            "num_leaves":        [31, 63, 127],
            "subsample":         [0.7, 0.8, 1.0],
            "colsample_bytree":  [0.7, 0.8, 1.0],
            "min_child_samples": [10, 20, 50],
            "reg_lambda":        [0.0, 0.1, 0.5],
        },
        "extended": {
            "n_estimators":      [500, 800, 1000, 1500, 2000],
            "learning_rate":     [0.01, 0.03, 0.05, 0.1, 0.2],
            "max_depth":         [4, 6, 8, 10, 12, -1],
            "num_leaves":        [15, 31, 63, 127, 255],
            "subsample":         [0.5, 0.7, 0.8, 1.0],
            "colsample_bytree":  [0.5, 0.7, 0.8, 1.0],
            "min_child_samples": [5, 10, 20, 50, 100],
            "reg_lambda":        [0.0, 0.1, 0.5, 1.0, 5.0],
        },
    },

    # -------------------------- Adaptive / Bagging --------------------------
    "adaboost": {
        "narrow": {
            "n_estimators":  [100, 200, 300],
            "learning_rate": [0.05, 0.1, 0.5, 1.0],
            "estimator__max_depth": [3, 5, 7],
        },
        "extended": {
            "n_estimators":  [50, 100, 200, 300, 500],
            "learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0, 1.5],
            "estimator__max_depth": [1, 3, 5, 7, 10],
        },
    },
    "bagging_rf": {
        "narrow": {
            "n_estimators":            [50, 100, 200],
            "max_samples":             [0.5, 0.7, 0.8, 1.0],
            "max_features":            [0.5, 0.7, 0.8, 1.0],
            "estimator__max_depth":    [10, 15, 20],
        },
        "extended": {
            "n_estimators":            [50, 100, 200, 300, 500],
            "max_samples":             [0.3, 0.5, 0.7, 0.8, 1.0],
            "max_features":            [0.3, 0.5, 0.7, 0.8, 1.0],
            "estimator__max_depth":    [10, 15, 20, 30, None],
        },
    },
    "bagging_et": {
        "narrow": {
            "n_estimators":                [30, 50, 100],
            "max_samples":                 [0.5, 0.7, 0.8, 1.0],
            "estimator__n_estimators":     [10, 20],
            "estimator__max_depth":        [10, 15, None],
        },
        "extended": {
            "n_estimators":                [20, 30, 50, 100, 200],
            "max_samples":                 [0.3, 0.5, 0.7, 0.8, 1.0],
            "estimator__n_estimators":     [10, 20, 30, 50],
            "estimator__max_depth":        [10, 15, 20, None],
        },
    },

    # -------------------------- Linear --------------------------
    "logisticregression": {
        "narrow": {
            "C":       [0.01, 0.1, 1, 10, 100],
            "penalty": ["l2"],
            "solver":  ["lbfgs", "liblinear"],
        },
        "extended": {
            "C":       [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
            "penalty": ["l2"],
            "solver":  ["lbfgs", "liblinear"],
        },
    },
    "ridge": {
        "narrow": {
            "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
        },
        "extended": {
            "alpha": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0],
        },
    },

    # -------------------------- Kernel --------------------------
    "svm": {
        "narrow": {
            "C":     [0.1, 1, 10, 100],
            "gamma": ["scale", 0.001, 0.01, 0.1, 1],
        },
        "extended": {
            "C":     [1, 10, 100, 1000, 10000],
            "gamma": [0.0001, 0.001, 0.01, 0.1, 1, "scale"],
        },
        "focused_static_legacy": {
            # Historical grid used during March-April 2026 round on the OLD
            # data. Designed to verify static SVM convergence at high C with
            # gamma fixed at 0.01. Best result on old data: C=10000,
            # gamma=0.01, test=0.9021.
            # Kept here so the original svm_focused.py round can be
            # reproduced bit-for-bit by passing --grid-mode focused_static_legacy.
            "C":     [10000, 100000, 1000000],
            "gamma": [0.01],
        },
        "focused": {
            # Wide search introduced 27/4/2026 (NEW data round).
            # Covers both directions discovered by HP search:
            # - High-C direction: matches the static "focused_static_legacy"
            #   grid (C up to 100000).
            # - Low-gamma direction: matches the new mobile data behavior
            #   where extended SVM mobile chose gamma=0.0001 at the edge.
            # 7 C values x 9 gamma values = 63 combinations.
            "C":     [0.1, 1, 10, 100, 1000, 10000, 100000],
            "gamma": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, "scale"],
        },
        "focused_high_C": {
            # Targeted exploration of the C asymptote, run 27/4/2026 round 2.
            # Previous focused round found best at C=100000 (edge of grid).
            # This grid pushes C up to 10^7 while restricting gamma to the
            # values that performed best in focused (scale, 0.001, 0.0001).
            # 5 C values x 3 gamma values = 15 combinations.
            "C":     [100000, 300000, 1000000, 3000000, 10000000],
            "gamma": ["scale", 0.001, 0.0001],
        },
    },
}


# ============================================================
# MODEL BUILDERS
# ============================================================
# Each builder returns a fresh estimator instance. Keys must match
# SEARCH_SPACES keys (lower-case canonical names).

def _make_dt(max_depth: int) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(max_depth=max_depth, random_state=RANDOM_STATE)


def build_estimator(name: str) -> Any:
    """Build a fresh, untrained estimator for the given canonical model name."""
    name = name.lower()
    if name == "randomforest":
        return RandomForestClassifier(random_state=RANDOM_STATE,
                                       n_jobs=N_JOBS_INNER)
    if name == "extratrees":
        return ExtraTreesClassifier(random_state=RANDOM_STATE,
                                     n_jobs=N_JOBS_INNER)
    if name == "gradientboosting":
        return GradientBoostingClassifier(random_state=RANDOM_STATE)
    if name == "xgboost":
        if not HAS_XGB:
            raise ImportError("XGBoost is not installed")
        return XGBClassifier(random_state=RANDOM_STATE,
                             n_jobs=N_JOBS_INNER, verbosity=0,
                             tree_method="hist", eval_metric="logloss")
    if name == "catboost":
        if not HAS_CATBOOST:
            raise ImportError("CatBoost is not installed")
        return CatBoostClassifier(random_state=RANDOM_STATE,
                                   task_type="CPU", verbose=0)
    if name == "lightgbm":
        if not HAS_LGBM:
            raise ImportError("LightGBM is not installed")
        return LGBMClassifier(random_state=RANDOM_STATE,
                              n_jobs=N_JOBS_INNER, verbose=-1)
    if name == "adaboost":
        # AdaBoost wraps a DecisionTreeClassifier base estimator; the
        # base depth is part of the search space ("estimator__max_depth").
        try:
            return AdaBoostClassifier(estimator=_make_dt(5),
                                       random_state=RANDOM_STATE)
        except TypeError:
            # Older sklearn API
            return AdaBoostClassifier(base_estimator=_make_dt(5),
                                       random_state=RANDOM_STATE)
    if name == "bagging_rf":
        try:
            return BaggingClassifier(estimator=_make_dt(15),
                                      random_state=RANDOM_STATE,
                                      n_jobs=N_JOBS_INNER)
        except TypeError:
            return BaggingClassifier(base_estimator=_make_dt(15),
                                      random_state=RANDOM_STATE,
                                      n_jobs=N_JOBS_INNER)
    if name == "bagging_et":
        base = ExtraTreesClassifier(n_estimators=10, max_depth=10,
                                     random_state=RANDOM_STATE,
                                     n_jobs=N_JOBS_INNER)
        try:
            return BaggingClassifier(estimator=base,
                                      random_state=RANDOM_STATE,
                                      n_jobs=N_JOBS_INNER)
        except TypeError:
            return BaggingClassifier(base_estimator=base,
                                      random_state=RANDOM_STATE,
                                      n_jobs=N_JOBS_INNER)
    if name == "logisticregression":
        # max_iter=5000 is kept only for headroom; it is NOT what fixes the mobile
        # behavior. Verified empirically (9/6/2026, mobile, 37 selected features):
        #   - The mobile collapse to ~0.5473 is INDEPENDENT of C: all 9 C values in
        #     {1e-4..1e4} x {lbfgs, liblinear} (18 configs) return 0.5473 with an
        #     identical confusion matrix. C=1e-4 is "selected" only as the first
        #     tie when all candidates tie. So it is NOT a regularization effect.
        #   - It is NOT a convergence failure: the solver finishes in ~4 iterations
        #     with no ConvergenceWarning even at max_iter=5000.
        #   - LR and LinearSVC (hinge) both collapse and fit near-degenerate models
        #     (||w|| ~ 1e-10, i.e. effectively a constant predictor), whereas Ridge
        #     and LDA recover non-trivial weight vectors (||w|| ~ units) and reach
        #     ~0.89. A 1-D projection on the LDA axis reaches ~0.8935, so the signal
        #     IS present in a linear direction.
        # Conclusion: the collapse is an empirical dissociation between linear
        # learning criteria (margin/likelihood vs least-squares/discriminant), NOT
        # a convergence, regularization, or linear-capacity problem. See paper
        # Section VII-A. (Earlier comment calling this a convergence issue was wrong.)
        return LogisticRegression(random_state=RANDOM_STATE,
                                   max_iter=5000, n_jobs=N_JOBS_INNER)
    if name == "ridge":
        return RidgeClassifier(random_state=RANDOM_STATE)
    if name == "svm":
        return SVC(kernel="rbf", probability=True,
                    random_state=RANDOM_STATE)
    raise ValueError(f"Unknown model name: {name}")


# Bagging variants only have a 'narrow' grid; AdaBoost, LR, Ridge similarly.
# Ridge is a bit thin but its search space is intrinsically small.
MODELS_WITH_EXTENDED = {
    "randomforest", "extratrees", "gradientboosting",
    "xgboost", "catboost", "lightgbm",
    "adaboost", "svm",
}


# ============================================================
# HELPERS
# ============================================================
class Tee:
    """Write to stdout and a log file simultaneously."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, msg):
        for s in self.streams:
            s.write(msg)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()


def wilson_ci(p: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95% Wilson confidence interval for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def evaluate_on_test(model, X_test, y_test) -> Dict[str, Any]:
    """Evaluate a trained model on the test set. AUC uses decision_function
    when predict_proba is not available (e.g. RidgeClassifier)."""
    y_pred = model.predict(X_test)
    y_score = None
    if hasattr(model, "predict_proba"):
        try:
            y_score = model.predict_proba(X_test)[:, 1]
        except Exception:
            y_score = None
    if y_score is None and hasattr(model, "decision_function"):
        try:
            y_score = model.decision_function(X_test)
        except Exception:
            y_score = None

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, zero_division=0)
    n = len(y_test)
    ci_low, ci_high = wilson_ci(acc, n)

    auc = None
    if y_score is not None:
        try:
            auc = float(roc_auc_score(y_test, y_score))
        except Exception:
            auc = None

    return {
        "test_accuracy":          float(acc),
        "test_f1":                float(f1),
        "test_auc":               auc,
        "test_accuracy_ci95_low":  float(ci_low),
        "test_accuracy_ci95_high": float(ci_high),
        "n_test":                 int(n),
    }


def is_edge_value(value, candidate_list) -> bool:
    """True if `value` is at the min or max NUMERIC element of
    `candidate_list`, AND no 'open-ended' marker (None) is present.
    Categorical-only lists never trigger an edge warning. If None or -1
    are present (e.g. max_depth=None or LightGBM max_depth=-1 mean
    'no limit'), the maximum numeric value is not considered an edge:
    the search has effectively unbounded headroom via the open marker.
    The open marker itself is excluded from min/max computation."""
    numerics_all = [v for v in candidate_list if isinstance(v, (int, float))
                    and not isinstance(v, bool)]
    has_open_marker = (None in candidate_list) or (-1 in numerics_all)
    # Exclude the open marker from min/max computation
    numerics = [v for v in numerics_all if v != -1]
    if len(numerics) < 2:
        return False
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    if value == -1:
        return False  # the open marker itself is never an edge
    if has_open_marker:
        # Only flag as edge if at the LOWER numeric bound.
        return value == min(numerics)
    return value == min(numerics) or value == max(numerics)


def detect_edge_params(best_params: Dict[str, Any],
                        space: Dict[str, list]) -> List[str]:
    """Return list of hyperparameter names whose best value sits at the
    numeric boundary of their search range."""
    edges = []
    for k, v in best_params.items():
        if k in space and is_edge_value(v, space[k]):
            edges.append(f"{k}={v} (range {space[k]})")
    return edges


def cv_splitter() -> StratifiedKFold:
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                            random_state=RANDOM_STATE)


# ============================================================
# DATA PIPELINE
# ============================================================
def prepare_data(config: Config) -> Dict[str, Any]:
    """
    Run the pipeline from defense_detection.py up to SMOTE, yielding the same
    splits, scaling, and feature selection as Tables III/IV.
    """
    print(f"\nPreparing data from: {config.data_root}")

    detector = DefenseDetector(config)

    tall_df = detector.load_simulation_data_enhanced()
    X, y, groups = detector.preprocess_data_enhanced(tall_df)
    X_eng = detector.engineer_advanced_features(X)

    # 60/20/20 group split matching run_full_pipeline
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
    X_val   = X_tv.iloc[val_idx]
    y_val   = y_tv.iloc[val_idx]

    print(f"  Split sizes: train={len(X_train)}, val={len(X_val)}, "
          f"test={len(X_test)}")
    print(f"  Train class balance: {y_train.value_counts().to_dict()}")

    # Robust scaling, fitted on train only
    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train),
                                   columns=X_train.columns,
                                   index=X_train.index)
    X_val_scaled   = pd.DataFrame(scaler.transform(X_val),
                                   columns=X_val.columns,
                                   index=X_val.index)
    X_test_scaled  = pd.DataFrame(scaler.transform(X_test),
                                   columns=X_test.columns,
                                   index=X_test.index)

    # Feature selection (matches detector pipeline)
    X_train_sel, X_val_sel = detector.select_features_intelligent(
        X_train_scaled, X_val_scaled, y_train
    )
    X_test_sel = X_test_scaled[detector.feature_names]
    print(f"  Features selected: {len(detector.feature_names)}")

    # SMOTE (matches detector pipeline)
    X_train_smote, y_train_smote = detector.augment_data_aggressive(
        X_train_sel, y_train
    )
    print(f"  SMOTE: {len(X_train_sel)} -> {len(X_train_smote)} samples")

    return {
        "X_train_pre_smote": X_train_sel,
        "y_train_pre_smote": y_train,
        "X_train_smote":     X_train_smote,
        "y_train_smote":     y_train_smote,
        "X_val":   X_val_sel,
        "y_val":   y_val,
        "X_test":  X_test_sel,
        "y_test":  y_test,
        "feature_names": detector.feature_names,
    }


# ============================================================
# SEARCH RUNNERS
# ============================================================
def run_random_search(estimator, param_space, X, y, name: str,
                       n_jobs_override: int = None):
    """RandomizedSearchCV with N_ITER_RANDOM (or fewer if grid is small)."""
    n_combos = int(np.prod([len(v) for v in param_space.values()]))
    n_iter   = min(N_ITER_RANDOM, n_combos)
    n_jobs   = n_jobs_override if n_jobs_override is not None else N_JOBS_OUTER
    print(f"  [Random search] {name}: "
          f"n_iter={n_iter} of {n_combos} combos, cv={CV_FOLDS}, n_jobs={n_jobs}")

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_space,
        n_iter=n_iter,
        scoring=SCORING,
        cv=cv_splitter(),
        n_jobs=n_jobs,
        random_state=RANDOM_STATE,
        return_train_score=False,
        refit=True,
        verbose=0,
        error_score=np.nan,
    )

    t0 = time.time()
    search.fit(X, y)
    elapsed = time.time() - t0

    print(f"    Best CV accuracy: {search.best_score_:.4f}")
    print(f"    Best params:      {search.best_params_}")
    print(f"    Elapsed:          {elapsed:.1f}s")
    return search, elapsed


def run_grid_search(estimator, param_grid, X, y, name: str,
                     n_jobs_override: int = None):
    """GridSearchCV (used when grid is small or coverage is desired)."""
    n_combos = int(np.prod([len(v) for v in param_grid.values()]))
    n_jobs   = n_jobs_override if n_jobs_override is not None else N_JOBS_OUTER
    print(f"  [Grid search]   {name}: "
          f"{n_combos} combos x {CV_FOLDS}-fold, n_jobs={n_jobs}")

    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=SCORING,
        cv=cv_splitter(),
        n_jobs=n_jobs,
        return_train_score=False,
        refit=True,
        verbose=0,
        error_score=np.nan,
    )

    t0 = time.time()
    search.fit(X, y)
    elapsed = time.time() - t0

    print(f"    Best CV accuracy: {search.best_score_:.4f}")
    print(f"    Best params:      {search.best_params_}")
    print(f"    Elapsed:          {elapsed:.1f}s")
    return search, elapsed


# Models that always use grid search (small grids or full coverage desired):
GRID_SEARCH_MODELS = {"logisticregression", "ridge", "svm"}


def run_search_for_model(name: str, space: Dict[str, list],
                          X, y) -> Tuple[Any, float]:
    """Dispatch to grid or random search based on model name. Models in
    REDUCED_PARALLELISM_MODELS get n_jobs=REDUCED_MAX_JOBS automatically
    to avoid OOM-driven worker kills."""
    estimator = build_estimator(name)
    n_jobs_override = None
    if name in REDUCED_PARALLELISM_MODELS:
        # Cap parallelism for memory-hungry models. Only reduce, never increase.
        n_jobs_override = min(REDUCED_MAX_JOBS, N_JOBS_OUTER)
        if n_jobs_override < N_JOBS_OUTER:
            print(f"  [info] {name} uses reduced parallelism: "
                   f"n_jobs={n_jobs_override} (default={N_JOBS_OUTER}) "
                   f"to prevent OOM")
    if name in GRID_SEARCH_MODELS:
        return run_grid_search(estimator, space, X, y, name, n_jobs_override)
    return run_random_search(estimator, space, X, y, name, n_jobs_override)


# ============================================================
# STACKING
# ============================================================
def build_stacking_with_tuned_bases(tuned_models: Dict[str, Any]) -> Any:
    """
    Build a Stacking_Ensemble whose base learners are the tuned models that
    have a probability interface. The meta-learner is the same regularized
    Logistic Regression as in defense_detection.py (matches Section V-D).
    Out-of-fold predictions from a 5-fold CV avoid information leakage.
    """
    base_pairs: List[Tuple[str, Any]] = []
    for n, m in tuned_models.items():
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
        n_jobs=1,
        passthrough=False,
    )


# ============================================================
# OUTPUT FILES
# ============================================================
def capture_environment() -> str:
    """Snapshot of Python, libraries, and hardware for reproducibility."""
    import sklearn
    lines = [
        f"Timestamp:    {datetime.datetime.now().isoformat()}",
        f"Python:       {platform.python_version()}",
        f"Platform:     {platform.platform()}",
        f"Processor:    {platform.processor() or 'unknown'}",
        f"CPU count:    {os.cpu_count()}",
        f"MAX_JOBS:     {N_JOBS_OUTER}",
        f"random_state: {RANDOM_STATE}",
        f"numpy:        {np.__version__}",
        f"pandas:       {pd.__version__}",
        f"sklearn:      {sklearn.__version__}",
    ]
    if HAS_XGB:
        import xgboost
        lines.append(f"xgboost:      {xgboost.__version__}")
    if HAS_CATBOOST:
        import catboost
        lines.append(f"catboost:     {catboost.__version__}")
    if HAS_LGBM:
        import lightgbm
        lines.append(f"lightgbm:     {lightgbm.__version__}")
    return "\n".join(lines) + "\n"


METHODOLOGY_TEXT = """\
Unified Hyperparameter Search Methodology
==========================================

This document records the methodology used by unified_hp_search.py.

1. Pipeline alignment
   The data pipeline is reused verbatim from defense_detection.py
   (DefenseDetector class): load tall CSV -> preprocess -> engineer features
   -> 60/20/20 GroupShuffleSplit at the run level (random_state=42) ->
   RobustScaler fitted on train only -> intelligent feature selection ->
   SMOTE/Borderline-SMOTE/SVM-SMOTE augmentation. Tuning therefore evaluates
   exactly the same training distribution as Tables III/IV.

2. Search procedure
   - RandomizedSearchCV with n_iter = 30 is used for tree-based and boosting
     models. n_iter is set to min(30, total combinations); we therefore never
     waste iterations on a smaller grid.
   - GridSearchCV is used for Logistic Regression, Ridge, and SVM, whose
     grids are small enough to enumerate exhaustively.
   - All searches use 3-fold StratifiedKFold (Bergstra & Bengio 2012;
     standard for moderate sample sizes), accuracy as the scoring criterion
     (matching Table IV's primary metric), and refit on the full training
     partition.

3. Search-space variants
   - 'narrow' grids match the original hp_search.py and standard practice.
   - 'extended' grids widen ranges where boundary best params have been
     observed (matching svm_extended.py).
   - 'focused' is an SVM-only high-C grid (matching svm_focused.py).
   The script automatically detects when best params land at the numeric
   boundary of a 'narrow' grid and emits warnings to edge_warnings.txt,
   prompting an 'extended'-mode follow-up if needed.

4. Stacking
   Stacking_Ensemble is constructed AFTER all base learners are tuned. Base
   learners are the tuned models that expose predict_proba. The meta-learner
   is L2-regularized Logistic Regression (C=1.0). Five-fold cross-validation
   produces out-of-fold predictions to avoid information leakage, matching
   Section V-D of the paper.

5. Reporting
   For each model: best params, mean CV accuracy, and held-out test accuracy
   with 95% Wilson confidence intervals (n_test approx 8,008). Test set is
   never touched during search.

References
----------
- Bergstra & Bengio (2012) Random Search for Hyper-Parameter Optimization.
- Probst, Boulesteix & Bischl (2019) Tunability: Importance of Hyperparameters
  of Machine Learning Algorithms.
- Wolpert (1992) Stacked generalization.
"""


# ============================================================
# INCREMENTAL SAVE
# ============================================================
def save_incremental(out_dir: Path, payload: Dict[str, Any]) -> None:
    """Persist after every model so a crash never loses prior progress."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # full_log.json (everything searched so far)
    with (out_dir / "full_log.json").open("w") as f:
        json.dump(payload, f, indent=2, default=_json_default)

    # combined_results.csv (per-model summary)
    rows = []
    for name, info in payload.get("search_results", {}).items():
        if "error" in info:
            rows.append({"model": name, "error": info["error"]})
            continue
        row = {
            "model":           name,
            "cv_accuracy":     info.get("cv_accuracy"),
            "test_accuracy":   info.get("test_accuracy"),
            "test_ci_low":     info.get("test_accuracy_ci95_low"),
            "test_ci_high":    info.get("test_accuracy_ci95_high"),
            "test_auc":        info.get("test_auc"),
            "test_f1":         info.get("test_f1"),
            "elapsed_s":       info.get("elapsed_s"),
            "best_params":     json.dumps(info.get("best_params", {}),
                                            default=_json_default),
            "edge_warnings":   "; ".join(info.get("edge_params", [])),
            "grid_mode":       info.get("grid_mode", ""),
        }
        rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "combined_results.csv",
                                    index=False)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# ============================================================
# MAIN PIPELINE PER CONFIG
# ============================================================
def is_model_already_done(out_dir: Path, model_name: str,
                            tuned_models_pkl: Dict[str, Any]) -> bool:
    """Return True if the model has a valid completed result in the output
    directory, indicating it can be skipped on resume.

    A model is 'done' iff:
      1. It appears in combined_results.csv with a non-null test_accuracy
         and no error column populated.
      2. It exists in best_models.pkl (already loaded via tuned_models_pkl).

    Both conditions must hold. If either is missing, the model needs to be
    re-tuned (we cannot construct Stacking from a CSV alone, we need the
    pkl too).
    """
    csv_path = out_dir / "combined_results.csv"
    if not csv_path.exists():
        return False

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return False

    rows = df[df["model"] == model_name]
    if rows.empty:
        return False

    row = rows.iloc[0]

    # Test accuracy must be a valid number (not NaN, not error string).
    test_acc = row.get("test_accuracy")
    if test_acc is None or (isinstance(test_acc, float) and np.isnan(test_acc)):
        return False

    # If the row has a populated 'error' column, it's a failed run.
    if "error" in df.columns:
        err_val = row.get("error")
        if isinstance(err_val, str) and err_val.strip() != "":
            return False

    # Must also exist in pkl. Stacking_Ensemble doesn't need to be in the
    # base learner pkl; it gets rebuilt separately.
    if model_name == "Stacking_Ensemble":
        return True
    if model_name not in tuned_models_pkl:
        return False

    return True


def load_existing_tuned_models(out_dir: Path) -> Dict[str, Any]:
    """Load best_models.pkl if it exists, return empty dict otherwise."""
    pkl_path = out_dir / "best_models.pkl"
    if not pkl_path.exists():
        return {}
    try:
        with pkl_path.open("rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"  [warn] Could not load existing pkl: {e}")
        return {}


def load_existing_payload(out_dir: Path, config_name: str,
                           grid_mode: str) -> Dict[str, Any]:
    """Load full_log.json if it exists, return fresh payload otherwise.

    On resume, we want to PRESERVE prior search_results entries so that
    save_incremental writes a complete CSV. New models get appended to
    the same payload. Only entries with errors or NaN test_accuracy will
    be re-tuned and overwritten.
    """
    log_path = out_dir / "full_log.json"
    if log_path.exists():
        try:
            with log_path.open("r") as f:
                payload = json.load(f)
            print(f"  [resume] Loaded existing payload with "
                   f"{len(payload.get('search_results', {}))} model entries")
            return payload
        except Exception as e:
            print(f"  [warn] Could not load existing payload: {e}; "
                   f"starting fresh")
    return {}


def run_for_config(config_name: str, data_root: str,
                    out_dir: Path,
                    requested_models: List[str],
                    grid_mode: str,
                    resume: bool = False,
                    skip_on_failure: bool = True) -> None:
    """Run unified HP search for a single configuration (static or mobile)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "runtime.log"
    log_file = open(log_path, "w")
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, log_file)

    try:
        print("\n" + "#" * 70)
        print(f"# Configuration: {config_name}")
        print(f"# Data root:     {data_root}")
        print(f"# Output:        {out_dir}")
        print(f"# Grid mode:     {grid_mode}")
        print(f"# Models:        {requested_models}")
        print(f"# Resume:        {resume}")
        print(f"# Started:       {datetime.datetime.now().isoformat()}")
        print("#" * 70)

        (out_dir / "environment.txt").write_text(capture_environment())
        (out_dir / "methodology.txt").write_text(METHODOLOGY_TEXT)

        # ------- Prepare data -------
        cfg = Config()
        cfg.data_root = data_root
        cfg.group_split_by_file_source = True
        cfg.use_validation = True
        cfg.random_state = RANDOM_STATE
        data = prepare_data(cfg)

        X_tr, y_tr = data["X_train_smote"], data["y_train_smote"]
        X_test, y_test = data["X_test"], data["y_test"]

        # ------- Resume support: load existing payload + tuned_models if any -------
        if resume:
            existing_payload = load_existing_payload(out_dir, config_name,
                                                       grid_mode)
            existing_tuned = load_existing_tuned_models(out_dir)
        else:
            existing_payload = {}
            existing_tuned = {}

        payload: Dict[str, Any] = existing_payload if existing_payload else {
            "config":            config_name,
            "grid_mode":         grid_mode,
            "timestamp":         datetime.datetime.now().isoformat(),
            "n_train_pre_smote":  int(len(data["X_train_pre_smote"])),
            "n_train_post_smote": int(len(X_tr)),
            "n_val":  int(len(data["X_val"])),
            "n_test": int(len(X_test)),
            "feature_names":   data["feature_names"],
            "search_results":  {},
        }
        # Always update timestamp on (re-)start so we know the latest run time
        payload["timestamp"] = datetime.datetime.now().isoformat()
        # Ensure search_results dict exists (defensive, in case payload was
        # loaded from an older format)
        payload.setdefault("search_results", {})

        # ------- Run search per model -------
        tuned_models: Dict[str, Any] = dict(existing_tuned)  # start with any preserved models
        edge_lines: List[str] = []

        # Track whether any base learner was newly tuned (to decide if Stacking
        # needs rebuild, even if a Stacking entry already exists)
        any_base_tuned_this_run = False

        for name in requested_models:
            print("\n" + "=" * 70)
            print(f"Tuning model: {name}  (grid_mode={grid_mode})")
            print("=" * 70)

            # ----- Resume check: skip if already done -----
            if resume and is_model_already_done(out_dir, name, existing_tuned):
                print(f"  [resume] {name} already completed; loading from pkl, "
                      f"skipping tuning.")
                # tuned_models already contains the pre-loaded estimator
                # via existing_tuned. Just confirm and continue.
                if name not in tuned_models:
                    # Should not happen given is_model_already_done's checks,
                    # but be safe.
                    print(f"  [warn] {name} marked done but missing from pkl; "
                           f"will re-tune.")
                else:
                    continue
            elif resume and name in existing_tuned:
                # Pkl had an old entry but result was incomplete (NaN/error).
                # Drop it before retuning so we don't carry stale state.
                print(f"  [resume] {name} present in pkl but result is "
                       f"incomplete; will re-tune and overwrite.")
                tuned_models.pop(name, None)

            # Pick the right grid for this model+mode
            spaces_for_model = SEARCH_SPACES.get(name, {})
            if grid_mode in spaces_for_model:
                space = spaces_for_model[grid_mode]
            elif grid_mode != "narrow" and "narrow" in spaces_for_model:
                # Fall back to narrow if extended/focused is unavailable
                # (e.g. ridge has no extended grid)
                print(f"  [info] '{grid_mode}' grid not defined for {name}; "
                      f"falling back to 'narrow'.")
                space = spaces_for_model["narrow"]
            else:
                msg = f"No search space defined for model '{name}'"
                print(f"  [SKIP] {msg}")
                payload["search_results"][name] = {"error": msg}
                save_incremental(out_dir, payload)
                continue

            try:
                search, elapsed = run_search_for_model(name, space,
                                                        X_tr, y_tr)
            except Exception as e:
                print(f"  [FAILED] {name}: {e}")
                payload["search_results"][name] = {"error": str(e)}
                save_incremental(out_dir, payload)
                if not skip_on_failure:
                    raise
                # With skip_on_failure=True (default), proceed to next model
                continue

            ev = evaluate_on_test(search.best_estimator_, X_test, y_test)
            edges = detect_edge_params(search.best_params_, space)
            if edges:
                edge_msg = (f"[{config_name}] {name} (grid={grid_mode}) "
                             f"best params at edge: " + ", ".join(edges))
                print(f"  [EDGE] {edge_msg}")
                edge_lines.append(edge_msg)

            payload["search_results"][name] = {
                "cv_accuracy":      float(search.best_score_),
                "best_params":      dict(search.best_params_),
                "elapsed_s":        float(elapsed),
                "n_candidates":     int(len(search.cv_results_["params"])),
                "grid_mode":        grid_mode,
                "edge_params":      edges,
                "all_candidates":   [
                    {"params": p, "mean_cv": float(s)}
                    for p, s in zip(search.cv_results_["params"],
                                      search.cv_results_["mean_test_score"])
                ],
                **ev,
            }
            tuned_models[name] = search.best_estimator_
            any_base_tuned_this_run = True
            save_incremental(out_dir, payload)

        # ------- Persist tuned base learners -------
        with (out_dir / "best_models.pkl").open("wb") as f:
            pickle.dump(tuned_models, f)

        # ------- Stacking with tuned bases -------
        # Build a fresh Stacking_Ensemble whenever we have >=3 base learners
        # AND either:
        #   (a) Stacking was never built, OR
        #   (b) we tuned a new base learner this run (so the existing
        #       Stacking is stale and needs to be rebuilt over the updated
        #       set of bases).
        # On --resume, when no base learners were re-tuned and a valid
        # Stacking entry already exists, we skip the rebuild.

        # Filter to only base learners (exclude any pre-existing Stacking
        # in tuned_models; we always reconstruct it from the bases).
        base_learners = {n: m for n, m in tuned_models.items()
                          if n != "Stacking_Ensemble"}

        existing_stacking_done = is_model_already_done(
            out_dir, "Stacking_Ensemble", tuned_models
        )

        if len(base_learners) < 3:
            print(f"\n[info] Skipping Stacking "
                   f"(only {len(base_learners)} tuned base learners; "
                   f"need >= 3).")
        elif resume and existing_stacking_done and not any_base_tuned_this_run:
            print(f"\n[resume] Stacking_Ensemble already complete and no base "
                   f"learners were re-tuned this run; skipping rebuild.")
            # Keep the existing entry in payload (it's already there from
            # existing_payload load) and the existing pkl entry.
        else:
            if existing_stacking_done and any_base_tuned_this_run:
                print(f"\n[resume] Stacking_Ensemble exists but base learner(s) "
                       f"were re-tuned; rebuilding.")
            print("\n" + "=" * 70)
            print("Building Stacking_Ensemble with tuned base learners")
            print("=" * 70)
            try:
                stack = build_stacking_with_tuned_bases(base_learners)
                t0 = time.time()
                stack.fit(X_tr, y_tr)
                elapsed = time.time() - t0
                ev = evaluate_on_test(stack, X_test, y_test)
                payload["search_results"]["Stacking_Ensemble"] = {
                    "cv_accuracy":   None,
                    "best_params":   {"meta": "L2-LR C=1.0",
                                       "n_bases": len(stack.estimators_)},
                    "elapsed_s":     float(elapsed),
                    "grid_mode":     "post-tuning",
                    "edge_params":   [],
                    "all_candidates": [],
                    **ev,
                }
                tuned_models["Stacking_Ensemble"] = stack
                with (out_dir / "best_models.pkl").open("wb") as f:
                    pickle.dump(tuned_models, f)
                save_incremental(out_dir, payload)
                print(f"  Stacking test accuracy: {ev['test_accuracy']:.4f}")
            except Exception as e:
                print(f"  [FAILED] Stacking: {e}")
                payload["search_results"]["Stacking_Ensemble"] = {
                    "error": str(e)
                }
                save_incremental(out_dir, payload)

        # ------- Edge-warning file -------
        if edge_lines:
            (out_dir / "edge_warnings.txt").write_text("\n".join(edge_lines)
                                                         + "\n")
            print(f"\n[info] Edge warnings written to "
                   f"{out_dir / 'edge_warnings.txt'}")
        else:
            (out_dir / "edge_warnings.txt").write_text(
                "No edge-of-grid warnings.\n"
            )

        print("\n" + "#" * 70)
        print(f"# Configuration {config_name} complete: "
               f"{datetime.datetime.now().isoformat()}")
        print("#" * 70)

    finally:
        sys.stdout = original_stdout
        log_file.close()


# ============================================================
# CLI
# ============================================================
# Canonical names of all 12 base learners. Used for CLI parsing and
# default 'all' expansion. Ordering matters: the list defines the
# default execution order. bagging_et is intentionally LAST because it
# has caused worker SIGKILL (OOM) in past runs; placing it last means
# all the other models will be saved before any failure.
# After-bagging_et order is then: build Stacking_Ensemble.
ALL_MODELS = [
    # Fast / cheap models first (sub-minute fits)
    "logisticregression",
    "ridge",
    "extratrees",
    "lightgbm",
    "xgboost",
    # Medium-cost models
    "bagging_rf",
    "adaboost",
    "gradientboosting",
    "catboost",
    "randomforest",
    # SVM can hang on high C; place it after the cheap/medium tier
    "svm",
    # bagging_et can OOM under high MAX_JOBS; place it LAST so all other
    # models are persisted before any potential SIGKILL.
    "bagging_et",
]

# Models that automatically use a reduced MAX_JOBS to avoid OOM.
# bagging_et with extended grids can spawn 10K+ trees in memory per CV
# fold; with MAX_JOBS=16 this overwhelms 31 GB systems.
REDUCED_PARALLELISM_MODELS = {"bagging_et"}
REDUCED_MAX_JOBS = 8


def parse_models(arg: str) -> List[str]:
    """Comma-separated list -> list of canonical model names. 'all' expands."""
    arg = arg.strip().lower()
    if arg == "all":
        return list(ALL_MODELS)
    requested = [m.strip() for m in arg.split(",") if m.strip()]
    invalid = [m for m in requested if m not in ALL_MODELS]
    if invalid:
        raise ValueError(f"Unknown model(s): {invalid}. "
                         f"Valid: {ALL_MODELS}")
    return requested


def main():
    parser = argparse.ArgumentParser(
        description=("Unified hyperparameter search for defense detection "
                      "models (replaces hp_search + svm_extended + "
                      "svm_focused)."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", choices=["static", "mobile", "both"],
                         default="both",
                         help="Which configuration(s) to run.")
    parser.add_argument("--static-root",
                         default="../simulations/features_static/",
                         help="Path to static features directory.")
    parser.add_argument("--mobile-root",
                         default="../simulations/features_mobile/",
                         help="Path to mobile features directory.")
    parser.add_argument("--out-dir", "--results-dir",
                         dest="out_dir",
                         default="./results/hp_search",
                         help="Root output directory.")
    parser.add_argument("--models", default="all",
                         help="Comma-separated model names "
                               "(or 'all'). See --list-models.")
    parser.add_argument("--grid-mode",
                         choices=["narrow", "extended", "focused",
                                   "focused_static_legacy", "focused_high_C"],
                         default="narrow",
                         help="Search-space variant.")
    parser.add_argument("--n-iter", type=int, default=None,
                         help=("Number of RandomizedSearchCV iterations "
                                "for non-grid models. Overrides the default "
                                "of 30. Recommended: 50-80 for extended grids "
                                "with very large search spaces (e.g. lightgbm)."))
    parser.add_argument("--resume", action="store_true",
                         help=("Resume an interrupted run: skip any model "
                                "that already has a complete result in "
                                "<out-dir>/<config>/combined_results.csv "
                                "AND best_models.pkl. Models with errors or "
                                "missing entries are re-tuned. Stacking is "
                                "rebuilt only if a base learner was re-tuned "
                                "during this resume run."))
    parser.add_argument("--no-skip-on-failure", action="store_true",
                         help=("By default, if a single model fails to tune, "
                                "the rest of the run continues and the failure "
                                "is recorded. Pass this flag to abort the run "
                                "on first failure."))
    parser.add_argument("--list-models", action="store_true",
                         help="List all canonical model names and exit.")
    args = parser.parse_args()

    # Honor --n-iter override (must be set before any model is searched)
    if args.n_iter is not None:
        if args.n_iter < 1:
            parser.error("--n-iter must be a positive integer")
        global N_ITER_RANDOM
        N_ITER_RANDOM = args.n_iter
        print(f"  N_ITER_RANDOM overridden to {N_ITER_RANDOM} via CLI")

    if args.list_models:
        print("Available models:")
        for m in ALL_MODELS:
            modes = list(SEARCH_SPACES.get(m, {}).keys())
            print(f"  {m:<22}  grids: {modes}")
        return

    requested_models = parse_models(args.models)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Unified HP search starting at "
           f"{datetime.datetime.now().isoformat()}")
    print(f"  Models:    {requested_models}")
    print(f"  Grid mode: {args.grid_mode}")
    print(f"  Configs:   {args.config}")
    print(f"  Output:    {out_root.resolve()}")
    print(f"  MAX_JOBS:  {N_JOBS_OUTER}")
    print(f"  Resume:    {args.resume}")
    print(f"  Skip on failure: {not args.no_skip_on_failure}")

    skip_on_failure = not args.no_skip_on_failure

    if args.config in ("static", "both"):
        run_for_config(
            "static", args.static_root,
            out_root / "static",
            requested_models, args.grid_mode,
            resume=args.resume,
            skip_on_failure=skip_on_failure,
        )
    if args.config in ("mobile", "both"):
        run_for_config(
            "mobile", args.mobile_root,
            out_root / "mobile",
            requested_models, args.grid_mode,
            resume=args.resume,
            skip_on_failure=skip_on_failure,
        )

    print(f"\nUnified HP search finished at "
           f"{datetime.datetime.now().isoformat()}")


if __name__ == "__main__":
    main()