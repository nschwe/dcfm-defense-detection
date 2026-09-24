#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
feature_eng_ablation_v2.py
============================================================================

Feature engineering ablation and augmentation for Section VI-G of the paper
(strict observability variant).

Verifies that for cross-domain detection of Fictive Mitigation under strict
observability:
  (a) the 33 base observable metrics are sufficient — additional engineered
      features (interactions, log/sqrt transforms, polynomial expansion) do
      not yield meaningful improvement and may hurt linear models;
  (b) [VOID on this arm, 21/8/26] removing the six delay/jitter features
      (Ablation_DJ_27) has a markedly larger effect than augmenting with
      engineered features.
      *** That was measured on the OLD GLOBAL feature space (delta = +0.1307,
      Table 9). On the listener arm all six DJ_FEATURES are among the 12
      columns the bundle has no analogue for, so preprocess_data_enhanced
      zero-fills them into the canonical 33 slots and this arm removes six
      constant-zero columns. Measured: CatBoost bit-identical in both
      directions, LogReg differing only by solver conditioning on 33 vs 27
      dimensions (p = 0.98, d = 0.01).
      The arm is KEPT deliberately -- dropping it would change the output
      schema -- and it is still evidence that the listener carries no
      delay/jitter information. It cannot support a comparative claim.
      See STATE 25.77.3; manuscript side: r2_1_answer.tex:63 (Sec. 6.7/7.2,
      "structural, decision required") and r2_10_answer.tex:27. ***

Pipeline (matches Section VI-F / confusion_matrices_universal4_v2.py):
  - Source: 60/20/20 group-aware split (GroupShuffleSplit twice)
  - StandardScaler fit on source train (applied to BOTH classifiers, for
    parity with VI-F where CatBoost also receives scaled inputs)
  - CatBoost (tuned hyperparameters per source config) OR LogReg
  - Threshold selection on source validation by accuracy in [0.10, 0.90]
  - Evaluate on source test (in-domain reference) AND full target
    (cross-domain — the primary metric reported)
  - 20 seeds, paired statistics on per-seed deltas

Configurations:
  1. Standard_33     : 33 strict-observable base metrics             (BASELINE)
  2. Plus_Engineered : 33 + CDR + TDR                                (35)
  3. Plus_Log        : 33 + log1p of skewed non-negative features    (33 + n_skew)
  4. Plus_Poly2      : PolynomialFeatures(degree=2, include_bias=False) (594)
  5. Plus_All        : Standard + Engineered + Log + Poly2-extras
  6. Ablation_DJ_27  : 33 minus 6 delay/jitter features              (27)

CDR/TDR (strict observability, matches defense_detection_v2.py):
  data_packet_rate = (AvgTxPacketsPerFlow * FlowCount) / measurement_duration
  CDR = TcMessageRate / (data_packet_rate + eps)
  TDR = TcMessageRate * AverageAdvertisedLinksPerTCMessage

Classifiers:
  - CatBoost : hyperparameters loaded per source config from
                 hp_search_extended/{static,mobile}/best_models.pkl
  - LogReg   : LogisticRegression(max_iter=1000) on top of the scaled inputs
                 (same scaler as CatBoost — see pipeline note above)

Total fits: 20 seeds * 6 configs * 2 classifiers * 2 directions = 480

Note about reported numbers
---------------------------
Standard_33 here is an INTERNAL baseline for VI-G. It uses the same
evaluation pipeline as the K sweep and should reproduce the sweep's largest K
within sampling noise. It is NOT meant to replace VI-D's numbers. The script
prints a sanity check comparing Standard_33 to that reference; |delta| > 0.01
prints a warning.

*** The name overstates the width. On the listener arm Standard_33 is 21 real
columns plus 12 zero-filled slots, which is why it must be compared against the
sweep at K=21 and not against a 33-column number. ***

*** The reference was stale until 21/8/26: it held (0.6718, 0.8425) from the
PRE-phantom-fix sweep, which made every run print "pipeline may diverge" on a
0.2039 delta that did not exist. Whenever the K sweep is re-run, SWEEP_K21_REF
below must be regenerated from its k_sweep_results.csv or this recurs. ***

Output:
  results/feature_eng_ablation_v2/feature_eng_per_seed.csv
  results/feature_eng_ablation_v2/feature_eng_summary.csv
  results/feature_eng_ablation_v2/feature_eng_paired.csv
  (sanity-check report printed to stdout and written to summary header)

Usage:
    python3 feature_eng_ablation_v2.py \\
        --static-root ../simulations/features_static \\
        --mobile-root ../simulations/features_mobile \\
        --hp-results-dir ./results/hp_search_extended \\
        --out-dir ./results/feature_eng_ablation_v2 \\
        --n-seeds 20

============================================================================
"""

import json
import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Set thread limits BEFORE importing numerical libraries.
os.environ.setdefault("OMP_NUM_THREADS", "16")
os.environ.setdefault("MKL_NUM_THREADS", "16")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy import stats
import joblib

# Patch pandas for old pickle compatibility (matches confusion_matrices_universal4_v2.py).
import pandas.core.series
pandas.core.series.dtype = np.dtype

from catboost import CatBoostClassifier

# Import data loading from v2 detector.
from defense_detection_v2 import DefenseDetector, Config

warnings.filterwarnings("ignore")


# ============================================================================
# Constants
# ============================================================================

# 33 strict-observable base metrics (matches DefenseDetector.METRICS in
# defense_detection_v2.py). Order preserved for reproducibility.
# PATCHED for the isolated 17-observable tree. The name is kept so the
# imports of other modules do not break; the CONTENT is the 17 distinct
# observables, derived from defense_detection_v2.METRICS in this same
# directory. The stale 33-name list is in the .bak_33names backup.
METRICS_33 = [
    "TcMessageRate",
    "AverageAdvertisedLinksPerTCMessage",
    "MidMessageRate",
    "HnaMessageRate",
    "RoutingOverheadBytesRatio",
    "AverageMprCount",
    "AverageHopCount",
    "DataPacketRate",
    "Throughput",
    "NormalizedRoutingLoad",
    "AvgTxPacketSize",
    "AvgFlowThroughput",
    "FlowThroughputStd",
    "AvgFlowDuration",
    "FlowDurationStd",
    "AvgTxBytesPerFlow",
    "AvgTxPacketsPerFlow",
]
assert len(METRICS_33) == 17, f"17-observable tree: expected 17, got {len(METRICS_33)}"

# Six delay/jitter features (subset of METRICS_33), removed in Ablation_DJ_27.
# The six delay/jitter metrics are NOT observable from a single vantage and
# are absent from this space. The DJ ablation therefore has no subject here;
# its single-vantage successor is instability_ablation_v2.py (STATE 25.77.10).
DJ_FEATURES = []
# (DJ_FEATURES is empty in this tree; the membership assert is vacuous.)
assert len(DJ_FEATURES) == 0, '17-observable tree: the DJ family does not exist here'

# Configs evaluated. Standard_33 must be FIRST (it's the paired baseline).
CONFIGS = [
    "Standard_33",
    "Plus_Engineered",
    "Plus_Log",
    "Plus_Poly2",
    "Plus_All",
    "Ablation_DJ_27",
]

CLASSIFIERS = ["CatBoost", "LogReg"]
DIRECTIONS = [("S", "M", "SM"), ("M", "S", "MS")]

# Reference: the K sweep's largest-K cross-domain accuracy, which is what this
# script's Standard_33 baseline should reproduce (21 real columns + 12 zeros).
# Source run: k_sweep_universal4_v2/k_sweep_results.csv, K=21, 20 seeds, 21/8/26.
# Sanity check only; not authoritative. REGENERATE when the sweep is re-run.
SWEEP_K21_REF = {
    "SM": 0.8764,
    "MS": 0.8542,
}
SANITY_TOL = 0.01

EPS = 1e-10
N_JOBS_CATBOOST = 16
RANDOM_STATE_OFFSET = 42  # seeds run RANDOM_STATE_OFFSET .. RANDOM_STATE_OFFSET+n-1


# ============================================================================
# CatBoost hyperparameter loading (mirrors confusion_matrices_universal4_v2.py)
# ============================================================================

def load_best_params(hp_results_dir: str) -> dict:
    """Load best CatBoost hyperparameters from best_models.pkl, per config.

    Returns {"static": params_dict, "mobile": params_dict}.
    """
    out = {}
    for cfg in ("static", "mobile"):
        js_path = Path(hp_results_dir) / cfg / "best_params.json"
        pkl_path = Path(hp_results_dir) / cfg / "best_models.pkl"
        if js_path.exists():            # the shipped parameter file (dump_hp_params.py)
            with open(js_path) as fh:
                d = json.load(fh)
        elif pkl_path.exists():         # the search's pickle of fitted estimators
            d = joblib.load(pkl_path)
        else:
            raise FileNotFoundError(f"Missing: {js_path} (or {pkl_path})")
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
            print(f"  [warn] unexpected entry type {type(entry).__name__}; "
                  "using CatBoost defaults", flush=True)
            params = {}

        out[cfg] = params
    return out


def filter_catboost_params(p: dict) -> dict:
    """Keep only relevant CatBoost hyperparameters (matches VI-F's filter)."""
    keep = {"iterations", "learning_rate", "depth", "l2_leaf_reg",
            "border_count", "bagging_temperature", "random_strength",
            "subsample"}
    return {k: v for k, v in p.items() if k in keep}


def build_catboost(best_params: dict, seed: int) -> CatBoostClassifier:
    """Build CatBoost with tuned hyperparameters (matches VI-F)."""
    p = filter_catboost_params(best_params)
    return CatBoostClassifier(
        random_seed=seed,
        verbose=0,
        allow_writing_files=False,
        thread_count=N_JOBS_CATBOOST,
        **p,
    )


def build_logreg(seed: int) -> LogisticRegression:
    """Plain LogReg; standardization applied separately for parity with VI-F."""
    return LogisticRegression(
        max_iter=1000,
        random_state=seed,
        # default L2 with C=1.0; this is a baseline, not tuned.
    )


# ============================================================================
# Data loading
# ============================================================================

def load_raw_data(data_root: str, config_name: str):
    """Load 33-feature dataset + _measurement_duration column.

    Returns:
        X      : DataFrame with the 33 metrics + '_measurement_duration'
        y      : Series of labels
        groups : np.array of file_source group ids
    """
    print(f"\n  Loading {config_name} from: {data_root}", flush=True)
    config = Config()
    config.data_root = data_root
    config.random_state = RANDOM_STATE_OFFSET

    detector = DefenseDetector(config)
    tall_df = detector.load_simulation_data_enhanced()
    X, y, groups = detector.preprocess_data_enhanced(tall_df)

    missing = [m for m in METRICS_33 if m not in X.columns]
    if missing:
        raise ValueError(f"Missing metrics in {config_name}: {missing}")

    if "_measurement_duration" not in X.columns:
        raise ValueError(
            f"_measurement_duration missing in {config_name} after "
            "preprocess_data_enhanced; needed for CDR computation."
        )

    keep_cols = METRICS_33 + ["_measurement_duration"]
    X = X[keep_cols].copy()

    print(f"    Loaded: X={X.shape}, classes={dict(y.value_counts())}, "
          f"groups={len(set(groups))}", flush=True)
    return X, y, np.array(groups)


# ============================================================================
# Feature builders (float32 to halve memory; matches v1)
# ============================================================================

def _base33(X: pd.DataFrame) -> np.ndarray:
    return X[METRICS_33].values.astype(np.float32)


def build_standard_33(X: pd.DataFrame) -> np.ndarray:
    return _base33(X)


def _cdr_tdr(X: pd.DataFrame):
    """CDR and TDR per defense_detection_v2.py."""
    md = X["_measurement_duration"].values.astype(np.float32)
    data_packet_rate = (
        X["AvgTxPacketsPerFlow"].values.astype(np.float32) *
        X["FlowCount"].values.astype(np.float32)
    ) / md
    cdr = X["TcMessageRate"].values.astype(np.float32) / (data_packet_rate + EPS)
    tdr = (X["TcMessageRate"].values.astype(np.float32) *
           X["AverageAdvertisedLinksPerTCMessage"].values.astype(np.float32))
    return cdr, tdr


def build_plus_engineered(X: pd.DataFrame) -> np.ndarray:
    """33 + CDR + TDR (=35)."""
    X_base = _base33(X)
    cdr, tdr = _cdr_tdr(X)
    return np.column_stack([X_base, cdr, tdr]).astype(np.float32)


def build_plus_log(X: pd.DataFrame, skew_mask: np.ndarray) -> np.ndarray:
    """33 + log1p of skewed non-negative features."""
    X_base = _base33(X)
    if not skew_mask.any():
        return X_base
    return np.column_stack(
        [X_base, np.log1p(X_base[:, skew_mask])]
    ).astype(np.float32)


def build_plus_poly2(X: pd.DataFrame) -> np.ndarray:
    """PolynomialFeatures(degree=2, include_bias=False) -> 33 + 33 + 528 = 594."""
    X_base = _base33(X)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    return poly.fit_transform(X_base).astype(np.float32)


def build_plus_all(X: pd.DataFrame, skew_mask: np.ndarray) -> np.ndarray:
    """Standard + Engineered (CDR, TDR) + Log + Poly2-extras (no duplicates)."""
    X_base = _base33(X)

    cdr, tdr = _cdr_tdr(X)
    X_eng = np.column_stack([cdr, tdr]).astype(np.float32)

    if skew_mask.any():
        X_log = np.log1p(X_base[:, skew_mask]).astype(np.float32)
    else:
        X_log = np.empty((X_base.shape[0], 0), dtype=np.float32)

    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X_base).astype(np.float32)
    # Drop the first 33 columns (linear features == X_base) to avoid duplication.
    X_poly_extra = X_poly[:, 33:]

    return np.column_stack([X_base, X_eng, X_log, X_poly_extra]).astype(np.float32)


def build_ablation_dj_27(X: pd.DataFrame) -> np.ndarray:
    """33 minus the 6 delay/jitter features (=27)."""
    keep = [m for m in METRICS_33 if m not in DJ_FEATURES]
    assert len(keep) == 27
    return X[keep].values.astype(np.float32)


def build_features(config_name: str, X: pd.DataFrame,
                   skew_mask: np.ndarray) -> np.ndarray:
    if config_name == "Standard_33":
        return build_standard_33(X)
    if config_name == "Plus_Engineered":
        return build_plus_engineered(X)
    if config_name == "Plus_Log":
        return build_plus_log(X, skew_mask)
    if config_name == "Plus_Poly2":
        return build_plus_poly2(X)
    if config_name == "Plus_All":
        return build_plus_all(X, skew_mask)
    if config_name == "Ablation_DJ_27":
        return build_ablation_dj_27(X)
    raise ValueError(f"Unknown config: {config_name}")


def compute_skew_mask(X_s: pd.DataFrame, X_m: pd.DataFrame,
                       skew_threshold: float = 1.0) -> np.ndarray:
    """Mask over 33 metrics: skewed (>threshold) in either config AND
    non-negative in BOTH configs. Matches v1 logic.
    """
    Xs = _base33(X_s)
    Xm = _base33(X_m)
    skew_s = np.asarray(stats.skew(Xs, axis=0, nan_policy="omit")) > skew_threshold
    skew_m = np.asarray(stats.skew(Xm, axis=0, nan_policy="omit")) > skew_threshold
    nonneg_both = np.all(Xs >= 0, axis=0) & np.all(Xm >= 0, axis=0)
    return (skew_s | skew_m) & nonneg_both


# ============================================================================
# Evaluation pipeline (matches confusion_matrices_universal4_v2.py)
# ============================================================================

def split_source_60_20_20(X, y, groups, seed):
    """Group-aware 60/20/20 split. Returns (train_idx, val_idx, test_idx)."""
    outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_val_idx, test_idx = next(outer.split(X, y, groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    # inner split: 0.25 of 0.80 -> 0.20 val, 0.60 train (overall).
    sub_groups = groups[train_val_idx]
    sub_X = X[train_val_idx] if isinstance(X, np.ndarray) else X.iloc[train_val_idx]
    sub_y = y[train_val_idx] if isinstance(y, np.ndarray) else y.iloc[train_val_idx]
    train_idx, val_idx = next(inner.split(sub_X, sub_y, sub_groups))
    return train_val_idx[train_idx], train_val_idx[val_idx], test_idx


def select_threshold(y_val, p_val) -> float:
    """Threshold in [0.10, 0.90] step 0.01 maximizing val accuracy."""
    thresholds = np.arange(0.1, 0.91, 0.01)
    best_thr, best_acc = 0.5, -1.0
    for t in thresholds:
        y_pred = (p_val >= t).astype(int)
        acc = accuracy_score(y_val, y_pred)
        if acc > best_acc:
            best_acc, best_thr = acc, t
    return best_thr


def evaluate_seed_config_clf(
    F_src: np.ndarray, y_src: np.ndarray, g_src: np.ndarray,
    F_tgt: np.ndarray, y_tgt: np.ndarray,
    best_params: dict, clf_name: str, seed: int,
):
    """One (config, classifier) evaluation for one seed.

    Returns dict with: te_acc, te_auc, tgt_acc, tgt_auc.
    """
    tr, va, te = split_source_60_20_20(F_src, y_src, g_src, seed)

    sc = StandardScaler().fit(F_src[tr])
    X_tr = sc.transform(F_src[tr])
    X_va = sc.transform(F_src[va])
    X_te = sc.transform(F_src[te])
    X_tg = sc.transform(F_tgt)

    if clf_name == "CatBoost":
        clf = build_catboost(best_params, seed=seed)
    elif clf_name == "LogReg":
        clf = build_logreg(seed=seed)
    else:
        raise ValueError(f"Unknown clf: {clf_name}")

    clf.fit(X_tr, y_src[tr])

    p_va = clf.predict_proba(X_va)[:, 1]
    thr = select_threshold(y_src[va], p_va)

    p_te = clf.predict_proba(X_te)[:, 1]
    p_tg = clf.predict_proba(X_tg)[:, 1]

    yhat_te = (p_te >= thr).astype(int)
    yhat_tg = (p_tg >= thr).astype(int)

    return {
        "te_acc":  accuracy_score(y_src[te], yhat_te),
        "te_auc":  roc_auc_score(y_src[te], p_te),
        "tgt_acc": accuracy_score(y_tgt, yhat_tg),
        "tgt_auc": roc_auc_score(y_tgt, p_tg),
        "thr":     thr,
    }


def run_one_seed(
    X_s: pd.DataFrame, y_s: pd.Series, g_s: np.ndarray,
    X_m: pd.DataFrame, y_m: pd.Series, g_m: np.ndarray,
    skew_mask: np.ndarray,
    best_params: dict,
    seed: int,
    verbose: bool = False,
):
    """One seed: build all 6 configs for both sources, run 2 classifiers
    in both directions. Returns one row (dict) with all metrics.
    """
    t0 = time.time()
    row = {"seed": seed}

    # Build feature matrices once per config (per direction).
    feats_s = {cfg: build_features(cfg, X_s, skew_mask) for cfg in CONFIGS}
    feats_m = {cfg: build_features(cfg, X_m, skew_mask) for cfg in CONFIGS}

    y_s_arr = y_s.to_numpy()
    y_m_arr = y_m.to_numpy()

    for cfg in CONFIGS:
        F_s = feats_s[cfg]
        F_m = feats_m[cfg]

        for clf_name in CLASSIFIERS:
            # S -> M : train on static, evaluate on mobile
            res_sm = evaluate_seed_config_clf(
                F_src=F_s, y_src=y_s_arr, g_src=g_s,
                F_tgt=F_m, y_tgt=y_m_arr,
                best_params=best_params["static"],
                clf_name=clf_name, seed=seed,
            )
            row[f"{cfg}_{clf_name}_SM_te_acc"]  = res_sm["te_acc"]
            row[f"{cfg}_{clf_name}_SM_te_auc"]  = res_sm["te_auc"]
            row[f"{cfg}_{clf_name}_SM_tgt_acc"] = res_sm["tgt_acc"]
            row[f"{cfg}_{clf_name}_SM_tgt_auc"] = res_sm["tgt_auc"]

            # M -> S : train on mobile, evaluate on static
            res_ms = evaluate_seed_config_clf(
                F_src=F_m, y_src=y_m_arr, g_src=g_m,
                F_tgt=F_s, y_tgt=y_s_arr,
                best_params=best_params["mobile"],
                clf_name=clf_name, seed=seed,
            )
            row[f"{cfg}_{clf_name}_MS_te_acc"]  = res_ms["te_acc"]
            row[f"{cfg}_{clf_name}_MS_te_auc"]  = res_ms["te_auc"]
            row[f"{cfg}_{clf_name}_MS_tgt_acc"] = res_ms["tgt_acc"]
            row[f"{cfg}_{clf_name}_MS_tgt_auc"] = res_ms["tgt_auc"]

            if verbose:
                print(f"  seed={seed} cfg={cfg:<16} clf={clf_name:<8} "
                      f"SM tgt_acc={res_sm['tgt_acc']:.4f}  "
                      f"MS tgt_acc={res_ms['tgt_acc']:.4f}", flush=True)

    row["seed_elapsed_s"] = time.time() - t0
    return row


# ============================================================================
# Statistics
# ============================================================================

def paired_analysis(baseline_vals: np.ndarray, exp_vals: np.ndarray) -> dict:
    """Paired stats: t-test, percentile CI, Cohen's d on per-seed deltas."""
    deltas = np.asarray(exp_vals) - np.asarray(baseline_vals)
    n = len(deltas)
    if n < 2 or deltas.std(ddof=1) == 0:
        t_stat, p_val = float("nan"), float("nan")
        d_paired = float("nan")
    else:
        t_stat, p_val = stats.ttest_rel(exp_vals, baseline_vals,
                                        alternative="two-sided")
        d_paired = float(deltas.mean() / deltas.std(ddof=1))
    return {
        "delta_mean":         float(deltas.mean()),
        "delta_std":          float(deltas.std(ddof=1)) if n > 1 else 0.0,
        "delta_ci_lo":        float(np.percentile(deltas, 2.5)),
        "delta_ci_hi":        float(np.percentile(deltas, 97.5)),
        "paired_t":           float(t_stat) if not np.isnan(t_stat) else float("nan"),
        "paired_p_twosided":  float(p_val) if not np.isnan(p_val) else float("nan"),
        "cohens_d_paired":    d_paired,
        "n_seeds":            n,
    }


def write_summary_csv(df: pd.DataFrame, out_path: Path):
    rows = []
    for cfg in CONFIGS:
        for clf in CLASSIFIERS:
            for (_, _, dlabel) in DIRECTIONS:
                for partition in ["te", "tgt"]:
                    for metric in ["acc", "auc"]:
                        col = f"{cfg}_{clf}_{dlabel}_{partition}_{metric}"
                        vals = df[col].values
                        rows.append({
                            "config":     cfg,
                            "classifier": clf,
                            "direction":  dlabel,
                            "partition":  partition,  # 'te' in-domain | 'tgt' cross-domain
                            "metric":     metric,
                            "mean":       float(vals.mean()),
                            "std":        float(vals.std(ddof=1)),
                            "ci_lo":      float(np.percentile(vals, 2.5)),
                            "ci_hi":      float(np.percentile(vals, 97.5)),
                        })
    pd.DataFrame(rows).to_csv(out_path, index=False)


def write_paired_csv(df: pd.DataFrame, out_path: Path):
    rows = []
    for cfg in CONFIGS:
        if cfg == "Standard_33":
            continue  # baseline; nothing to pair against itself
        for clf in CLASSIFIERS:
            for (_, _, dlabel) in DIRECTIONS:
                for partition in ["te", "tgt"]:
                    for metric in ["acc", "auc"]:
                        base_col = f"Standard_33_{clf}_{dlabel}_{partition}_{metric}"
                        exp_col  = f"{cfg}_{clf}_{dlabel}_{partition}_{metric}"
                        res = paired_analysis(df[base_col].values,
                                              df[exp_col].values)
                        res.update({
                            "config":     cfg,
                            "classifier": clf,
                            "direction":  dlabel,
                            "partition":  partition,
                            "metric":     metric,
                        })
                        rows.append(res)
    pd.DataFrame(rows).to_csv(out_path, index=False)


# ============================================================================
# Sanity check
# ============================================================================

def sanity_check_standard_33(df: pd.DataFrame) -> dict:
    """Compare Standard_33 CatBoost cross-domain means to the k_sweep K=21 reference.

    Returns dict with the comparison; also prints to stdout.
    Warning printed if |delta| > SANITY_TOL.
    """
    sm = df["Standard_33_CatBoost_SM_tgt_acc"].values
    ms = df["Standard_33_CatBoost_MS_tgt_acc"].values
    sm_mean = float(sm.mean())
    ms_mean = float(ms.mean())
    sm_delta = sm_mean - SWEEP_K21_REF["SM"]
    ms_delta = ms_mean - SWEEP_K21_REF["MS"]

    lines = [
        "",
        "=" * 70,
        "SANITY CHECK: Standard_33 (CatBoost) vs k_sweep K=21 reference",
        "=" * 70,
        f"  S->M:  Standard_33 = {sm_mean:.4f}    "
            f"k_sweep K=21 = {SWEEP_K21_REF['SM']:.4f}    "
            f"delta = {sm_delta:+.4f}",
        f"  M->S:  Standard_33 = {ms_mean:.4f}    "
            f"k_sweep K=21 = {SWEEP_K21_REF['MS']:.4f}    "
            f"delta = {ms_delta:+.4f}",
    ]
    warns = []
    if abs(sm_delta) > SANITY_TOL:
        warns.append(f"  WARNING: |delta(S->M)| = {abs(sm_delta):.4f} > {SANITY_TOL}; "
                      "pipeline may diverge from the k_sweep reference.")
    if abs(ms_delta) > SANITY_TOL:
        warns.append(f"  WARNING: |delta(M->S)| = {abs(ms_delta):.4f} > {SANITY_TOL}; "
                      "pipeline may diverge from the k_sweep reference.")
    if not warns:
        lines.append(f"  OK: both deltas within +/-{SANITY_TOL}.")
    else:
        lines.extend(warns)
    lines.append("=" * 70)

    for line in lines:
        print(line, flush=True)

    return {
        "sm_mean": sm_mean, "sm_ref": SWEEP_K21_REF["SM"], "sm_delta": sm_delta,
        "ms_mean": ms_mean, "ms_ref": SWEEP_K21_REF["MS"], "ms_delta": ms_delta,
        "report_lines": lines,
    }


def print_dj_note(df: pd.DataFrame):
    """Say plainly whether Ablation_DJ_27 measured anything (STATE 25.77.3).

    Checked against the per-seed values rather than asserted, so this stays
    correct if the bundle ever does carry the delay/jitter columns.
    """
    rows = []
    for clf in CLASSIFIERS:
        for _, _, dlabel in DIRECTIONS:
            a = f"Standard_33_{clf}_{dlabel}_tgt_acc"
            b = f"Ablation_DJ_27_{clf}_{dlabel}_tgt_acc"
            if a in df.columns and b in df.columns:
                rows.append((f"{clf}/{dlabel}", bool((df[a] == df[b]).all())))
    if not rows:
        return
    n_ident = sum(1 for _, ok in rows if ok)

    print("", flush=True)
    print("=" * 70, flush=True)
    print("NOTE ON Ablation_DJ_27 — is it measuring anything on this arm?", flush=True)
    print("=" * 70, flush=True)
    for name, ok in rows:
        print(f"  {name:<18} identical to Standard_33 across all "
              f"{len(df)} seeds: {ok}", flush=True)
    if n_ident:
        print("", flush=True)
        print("  The six DJ_FEATURES are all among the 12 columns this bundle has no", flush=True)
        print("  analogue for; they arrive zero-filled, so this arm removes constant-zero", flush=True)
        print("  columns. A delta of 0.0000 here is STRUCTURAL, not a null result — do", flush=True)
        print("  not read it as an ablation finding. Any non-identical row differs only", flush=True)
        print("  by solver conditioning on 33 vs 27 dimensions, not by information.", flush=True)
        print("  It remains evidence that the listener carries no delay/jitter signal.", flush=True)
        print("  STATE 25.77.3; manuscript: r2_1_answer.tex:63, r2_10_answer.tex:27.", flush=True)
    print("=" * 70, flush=True)


# ============================================================================
# Brief stdout report (cross-domain primary metric)
# ============================================================================

def print_primary_report(df: pd.DataFrame):
    """Cross-domain (tgt) accuracy: baseline + paired deltas, both directions."""
    print("\n" + "=" * 80, flush=True)
    print("PRIMARY: CROSS-DOMAIN (tgt) ACCURACY — Standard_33 (baseline) "
          "and paired deltas", flush=True)
    print("=" * 80, flush=True)
    for clf in CLASSIFIERS:
        for (_, _, dlabel) in DIRECTIONS:
            print(f"\n  Classifier: {clf}    Direction: {dlabel}", flush=True)
            print(f"  {'Configuration':<18} {'Accuracy':>20} "
                  f"{'Delta vs Std_33':>16} {'p':>12} {'d':>8}", flush=True)
            print(f"  {'-'*18} {'-'*20} {'-'*16} {'-'*12} {'-'*8}", flush=True)
            base_col = f"Standard_33_{clf}_{dlabel}_tgt_acc"
            base = df[base_col].values
            print(f"  {'Standard_33':<18} "
                  f"{base.mean():>10.4f} +/- {base.std(ddof=1):.4f}   "
                  f"{'---':>16} {'---':>12} {'---':>8}", flush=True)
            for cfg in CONFIGS[1:]:
                exp = df[f"{cfg}_{clf}_{dlabel}_tgt_acc"].values
                stat = paired_analysis(base, exp)
                print(f"  {cfg:<18} "
                      f"{exp.mean():>10.4f} +/- {exp.std(ddof=1):.4f}   "
                      f"{stat['delta_mean']:>+16.4f} "
                      f"{stat['paired_p_twosided']:>12.2e} "
                      f"{stat['cohens_d_paired']:>+8.2f}",
                      flush=True)


# ============================================================================
# main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--static-root",
                        default="../simulations/features_static")
    parser.add_argument("--mobile-root",
                        default="../simulations/features_mobile")
    parser.add_argument("--hp-results-dir",
                        default="./results/hp_search_extended",
                        help="Dir with {static,mobile}/best_models.pkl")
    parser.add_argument("--out-dir",
                        default="./results/feature_eng_ablation_v2")
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=RANDOM_STATE_OFFSET,
                        help="First seed; seeds range base..base+n-1")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72, flush=True)
    print("FEATURE ENGINEERING ABLATION v2 (strict observability, VI-G)",
          flush=True)
    print("=" * 72, flush=True)
    print(f"  Static root:    {args.static_root}", flush=True)
    print(f"  Mobile root:    {args.mobile_root}", flush=True)
    print(f"  HP results dir: {args.hp_results_dir}", flush=True)
    print(f"  Out dir:        {out_dir}", flush=True)
    print(f"  N seeds:        {args.n_seeds}  "
          f"(range {args.base_seed}..{args.base_seed+args.n_seeds-1})", flush=True)
    print(f"  Configs:        {CONFIGS}", flush=True)
    print(f"  Classifiers:    {CLASSIFIERS}", flush=True)
    print(f"  Pipeline:       60/20/20 group-aware + StandardScaler + "
          "threshold tuning (matches VI-F)", flush=True)

    # ---- Load best CatBoost hyperparameters ----
    print("\n[1/5] Loading best CatBoost hyperparameters...", flush=True)
    best_params = load_best_params(args.hp_results_dir)
    print(f"  Static CatBoost params: "
          f"{filter_catboost_params(best_params['static'])}", flush=True)
    print(f"  Mobile CatBoost params: "
          f"{filter_catboost_params(best_params['mobile'])}", flush=True)

    # ---- Load data ----
    print("\n[2/5] Loading static and mobile datasets...", flush=True)
    X_s, y_s, g_s = load_raw_data(args.static_root, "static")
    X_m, y_m, g_m = load_raw_data(args.mobile_root, "mobile")

    # ---- Skew mask (shared across both configs, computed once) ----
    skew_mask = compute_skew_mask(X_s, X_m, skew_threshold=1.0)
    n_skew = int(skew_mask.sum())

    n_feat = {
        "Standard_33":    33,
        "Plus_Engineered": 35,
        "Plus_Log":       33 + n_skew,
        "Plus_Poly2":     594,  # 33 + 33 + C(33,2)
        "Plus_All":       33 + 2 + n_skew + 561,  # 561 = 594 - 33 (poly extras)
        "Ablation_DJ_27": 27,
    }
    print(f"\n  Skew mask (skewed in either config, non-neg in both): "
          f"{n_skew} features", flush=True)
    print(f"  Feature counts per config:", flush=True)
    for name, count in n_feat.items():
        print(f"    {name:<18} : {count}", flush=True)
    total_fits = args.n_seeds * len(CONFIGS) * len(CLASSIFIERS) * len(DIRECTIONS)
    print(f"  Total fits: {total_fits}", flush=True)

    # ---- Run seeds sequentially ----
    # CatBoost itself uses 16 threads per fit, so parallelizing across seeds
    # would oversubscribe the CPU. Sequential is the right choice.
    print(f"\n[3/5] Running {args.n_seeds} seeds (sequential; "
          f"CatBoost uses {N_JOBS_CATBOOST} threads internally)...", flush=True)
    seeds = list(range(args.base_seed, args.base_seed + args.n_seeds))

    # Incremental CSV save after each seed (recovery if process is killed).
    per_seed_path = out_dir / "feature_eng_per_seed.csv"
    rows = []
    t_global = time.time()
    for i, seed in enumerate(seeds):
        t_seed = time.time()
        row = run_one_seed(X_s, y_s, g_s, X_m, y_m, g_m,
                           skew_mask=skew_mask,
                           best_params=best_params,
                           seed=seed, verbose=False)
        rows.append(row)
        # Write incremental CSV every seed.
        pd.DataFrame(rows).to_csv(per_seed_path, index=False)
        elapsed_seed = time.time() - t_seed
        elapsed_total = time.time() - t_global
        eta = elapsed_total / (i + 1) * (args.n_seeds - i - 1)
        print(f"  seed {seed} ({i+1}/{args.n_seeds}) done in "
              f"{elapsed_seed:.1f}s  elapsed={elapsed_total/60:.1f}min  "
              f"eta={eta/60:.1f}min", flush=True)

    df = pd.DataFrame(rows)

    # ---- Summary + paired CSVs ----
    print("\n[4/5] Writing summary CSVs...", flush=True)
    summary_path = out_dir / "feature_eng_summary.csv"
    paired_path  = out_dir / "feature_eng_paired.csv"
    write_summary_csv(df, summary_path)
    write_paired_csv(df, paired_path)
    print(f"  Wrote: {per_seed_path}", flush=True)
    print(f"  Wrote: {summary_path}", flush=True)
    print(f"  Wrote: {paired_path}", flush=True)

    # ---- Sanity check vs the k_sweep K=21 reference ----
    print("\n[5/5] Sanity check + brief report...", flush=True)
    sanity = sanity_check_standard_33(df)
    print_primary_report(df)
    print_dj_note(df)

    # Append sanity-check block to summary CSV header for permanent record.
    with open(out_dir / "sanity_check_vs_VI_F.txt", "w") as f:
        f.write("\n".join(sanity["report_lines"]) + "\n")

    print(f"\nTotal elapsed: {(time.time()-t_global)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
