#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hp_search_spaces.py -- the hyperparameter grids of the unified search, as data.

This is the SEARCH_SPACES table of the search script that selected the
configurations in the appendix (unified_hp_search_v2.py, lines 163-378),
copied verbatim: per model, the 'narrow' and 'extended' grids, and for the
SVM the additional focused grids. The search itself is not part of this
package; catboost_retune.py imports the table from here to re-tune CatBoost
and XGBoost on the reported campaign (the control in Section 5.2), so that it
searches the same grid the appendix reports.
"""

SEARCH_SPACES = {

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
