#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
catboost_retune.py -- a CONTROL, not a replacement: re-tune CatBoost on the
reported campaign and measure how far the inherited configuration is from one
selected on this data.

WHY THIS EXISTS
    The appendix's hyperparameters come from the earlier round -- N = 2,000, the
    33-metric space, SMOTE -- and five analyses still read them:
        k_sweep_universal4_v2.py:386          (Sec. 6.6, Tables 7-8)
        feature_importance_sensitivity_v2.py:1154  (Sec. 6.5, the universal set)
        feature_eng_ablation_v2.py:775        (Sec. 6.7, Table 9)
        threshold_decomposition_v2.py:581     (Sec. 6.8, Table 10)
        instability_ablation_v2.py:405        (the instability ablation)
    All five take only the CatBoost entry (`filter_catboost_params`), so CatBoost
    is the whole dependency.

    The standing defence of the reuse -- extend_hp_grids moved results by
    < 0.002 -- was itself measured in the superseded round, which makes it
    circular.  This script measures the same thing on the campaign the paper
    reports.

WHAT IT DOES NOT DO
    ⛔ It does not replace anything.  The five analyses above keep the
    configuration they ran with; nothing downstream is recomputed, and no
    campaign number moves.  The deliverable is one difference, reported as a
    control.
    ⛔ It does not touch results.csv, the pipeline, or the HP pickle.
    Everything written goes to <results>/<mode>/catboost_retune/.

TWO DEFECTS OF unified_hp_search_v2.py THAT THIS SCRIPT DOES NOT INHERIT
    1. Its inner CV is `StratifiedKFold` (:588-589), NOT grouped, so the four
       windows of one run can fall in different folds and the selection
       criterion is optimistic at the run level -- the leakage class Comments
       3.7 and 5.5 are about.  Here the inner CV is StratifiedGroupKFold over
       `file_source`.
    2. It searches on SMOTE-augmented data (:1047) while the reported campaign
       runs --no-augmentation.  Here the search sees the campaign's own
       unaugmented training split.
    ⛔ Consequently this is NOT a reproduction of the appendix search.  It is a
    cleaner search, and the comparison is "inherited configuration versus one
    selected properly on this data".

PROTOCOL
    The pipeline's own methods produce the split, the scaler and the feature
    selection (identical to lda_classifier.py, same gate).  Then:
      1. RandomizedSearchCV over the CatBoost grid IMPORTED from
         unified_hp_search_v2.SEARCH_SPACES -- not retyped -- with grouped
         3-fold CV and accuracy scoring, on the training split only.
      2. Two CatBoost models are fitted on that same training split: one from
         the INHERITED params (the pickle the five analyses read) and one from
         the SELECTED params.
      3. Both go through the campaign protocol -- calibrate on validation
         half A, threshold on half B -- and are scored once each on test.
      4. The reported quantity is the difference between them, with a PAIRED
         CLUSTER BOOTSTRAP interval at the simulation-run level: B = 2,000
         resamples, the same resampled runs applied to both configurations,
         each scored at its OWN locked threshold (the one its validation half B
         selected -- neither threshold is re-chosen inside the bootstrap).
         Two-sided 95 % percentile interval and the bootstrap significance
         level 2*min{P(d<=0), P(d>=0)}, matching the definition the rest of the
         revision uses.
    No test label informs the search.  The test partition is read at step 3 and
    nowhere else.

    ⚠️ A single split gives a point difference and no idea whether it is noise.
    The interval is what makes the comparison reportable; without it a ~1-point
    difference cannot be distinguished from run-to-run variation at this scale.

USAGE
    AN=/path/to/ns-3.47/analysis
    ARM=$AN/arms_r34_17feat_frozencal67
    cd $AN
    # timing probe first -- two sampled configurations, prints the per-fit rate
    MAX_JOBS=8 OMP_NUM_THREADS=2 \
    DCFM_PIPELINE=$AN/frozencal67_pipeline_17 \
    DCFM_DATA_BUNDLE=$ARM/listener/colab_data \
        ~/miniconda3/envs/manet/bin/python -u catboost_retune.py \
            --mode static --results-root $ARM/listener/results --n-iter 2

    # the real run: --n-iter 80, the paper's setting

    # --skip-search: reuse the configuration a previous run selected, refit
    # both, and produce the bootstrap. ~3 minutes instead of ~20, because the
    # 240 search fits are what cost the time and their answer is already on
    # disk. The 'search' block of the old summary is carried forward verbatim.
        ... catboost_retune.py --mode mobile \
            --results-root $ARM/listener/results --skip-search
"""
import runtime_guard  # noqa: F401  MUST be first: thread/MAX_JOBS caps pre-numpy

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PIPELINE = os.environ.get("DCFM_PIPELINE",
                          os.path.join(HERE, "frozencal67_pipeline_17"))
sys.path.insert(0, PIPELINE)

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import (GroupShuffleSplit, RandomizedSearchCV,
                                     StratifiedGroupKFold)
from sklearn.preprocessing import RobustScaler

import defense_detection_v2 as ddv2
from k_sweep_universal4_v2 import (build_catboost, filter_catboost_params,
                                   load_best_params)
# ⚠️ feature_importance_sensitivity_v2 is the only one of the five that reads
# XGBoost as well as CatBoost (:160 EVAL_CLASSIFIERS, :1159), and it owns the
# XGBoost builder and filter. Imported rather than retyped so the control uses
# the same construction the analysis does.
from feature_importance_sensitivity_v2 import (build_xgboost as _fis_build_xgb,
                                               _filter_xgb_params,
                                               load_best_params as _fis_load)
sys.path.insert(0, HERE)
from hp_search_spaces import SEARCH_SPACES      # the grids, as data (analysis/)


# ⛔ The two builders have DIFFERENT calling conventions and mixing them is the
# easy mistake here: k_sweep's build_catboost takes the params dict itself,
# fis's build_xgboost takes the whole {"xgboost": {...}} mapping.
MODELS = {
    "catboost": {
        "build": lambda params, seed, n_jobs: build_catboost(params, seed, n_jobs),
        "filter": filter_catboost_params,
        "grid_key": "catboost",
        "inherited": lambda hp_dir, mode: load_best_params(hp_dir)[mode],
    },
    "xgboost": {
        "build": lambda params, seed, n_jobs: _fis_build_xgb(
            {"xgboost": params}, seed, n_jobs=n_jobs),
        "filter": _filter_xgb_params,
        "grid_key": "xgboost",
        "inherited": lambda hp_dir, mode: _fis_load(hp_dir)[mode].get("xgboost", {}),
    },
}

# Where the five analyses read their CatBoost configuration from. Same default
# as run_arm.py:51, so "inherited" here means exactly what they used.
HP_RESULTS_DEFAULT = os.path.join(HERE, "hp_selected")   # {static,mobile}/best_params.json

# Read off the reported campaign's own logs,
# frozencal_marathon_logs/arms_r34_17feat_frozencal67_{static,mobile}_20260906_021443.log.
GATE = {
    "static": {"rows": 40000, "cols": 18, "engineered": 67, "selected": 28},
    "mobile": {"rows": 40000, "cols": 18, "engineered": 67, "selected": 32},
}
GATE_SPLIT = {"train": 24000, "cal": 4000, "thr": 4000, "test": 8000}

CV_FOLDS = 3          # the paper's setting
SCORING = "accuracy"  # the paper's setting
N_BOOT = 2000         # the revision's standard, r2_3's Methods definition


def build_config(mode, results_dir):
    """The campaign's flags: --no-augmentation --split-validation
    --add-linearsvc --calibrate-stacking."""
    cfg = ddv2.Config()
    cfg.data_root = "./features_%s/" % mode
    cfg.results_dir = results_dir
    cfg.group_split_by_file_source = True
    cfg.use_aggressive_augmentation = False
    cfg.split_validation = True
    cfg.add_linearsvc = True
    cfg.calibrate_stacking = True
    return cfg


def prepare(det, mode):
    """Replay run_full_pipeline up to model training with the pipeline's own
    methods, gated against the campaign log. Returns the training groups too,
    because the inner CV needs them."""
    g = GATE[mode]

    tall = det.load_simulation_data_enhanced()
    X, y, groups = det.preprocess_data_enhanced(tall)
    if X.shape != (g["rows"], g["cols"]):
        raise SystemExit("GATE FAIL: preprocessed shape %s, campaign log says (%d, %d)"
                         % (X.shape, g["rows"], g["cols"]))

    X_eng = det.engineer_advanced_features(X)
    if X_eng.shape[1] != g["engineered"]:
        raise SystemExit("GATE FAIL: %d engineered features, campaign log says %d"
                         % (X_eng.shape[1], g["engineered"]))

    groups_arr = np.array(groups)
    splitter = GroupShuffleSplit(n_splits=1, test_size=det.config.test_size,
                                 random_state=det.random_state)
    train_val_idx, test_idx = next(splitter.split(X_eng, y, groups_arr))
    X_temp, y_temp = X_eng.iloc[train_val_idx], y.iloc[train_val_idx]
    X_test, y_test = X_eng.iloc[test_idx], y.iloc[test_idx]
    groups_test = groups_arr[test_idx]   # the cluster the bootstrap resamples

    groups_temp = groups_arr[train_val_idx]
    splitter_val = GroupShuffleSplit(n_splits=1, test_size=0.25,
                                     random_state=det.random_state)
    train_idx, val_idx = next(splitter_val.split(X_temp, y_temp, groups_temp))
    X_train, y_train = X_temp.iloc[train_idx], y_temp.iloc[train_idx]
    X_val, y_val = X_temp.iloc[val_idx], y_temp.iloc[val_idx]
    groups_train = groups_temp[train_idx]
    val_groups = groups_temp[val_idx]

    det.scaler = RobustScaler()
    X_train_s = pd.DataFrame(det.scaler.fit_transform(X_train),
                             columns=X_train.columns, index=X_train.index)
    X_val_s = pd.DataFrame(det.scaler.transform(X_val),
                           columns=X_val.columns, index=X_val.index)
    X_test_s = pd.DataFrame(det.scaler.transform(X_test),
                            columns=X_test.columns, index=X_test.index)

    X_train_sel, X_val_sel = det.select_features_intelligent(
        X_train_s, X_val_s, y_train)
    if len(det.feature_names) != g["selected"]:
        raise SystemExit("GATE FAIL: %d features selected, campaign log says %d"
                         % (len(det.feature_names), g["selected"]))
    X_test_sel = X_test_s[det.feature_names]

    # ⛔ No augmentation call at all -- see the header. The campaign runs
    # --no-augmentation, so the search must see the same rows the models do.
    half = GroupShuffleSplit(n_splits=1, test_size=0.5,
                             random_state=det.random_state)
    a_idx, b_idx = next(half.split(X_val_sel, y_val, np.array(val_groups)))
    X_cal, y_cal = X_val_sel.iloc[a_idx], y_val.iloc[a_idx]
    X_thr, y_thr = X_val_sel.iloc[b_idx], y_val.iloc[b_idx]

    got = {"train": len(X_train_sel), "cal": len(X_cal),
           "thr": len(X_thr), "test": len(X_test_sel)}
    if got != GATE_SPLIT:
        raise SystemExit("GATE FAIL: split sizes %s, campaign log says %s"
                         % (got, GATE_SPLIT))

    det.log("  [retune] GATE OK: %d x %d -> %d engineered -> %d selected; "
            "train %d, cal half %d, thr half %d, test %d"
            % (g["rows"], g["cols"], g["engineered"], g["selected"],
               got["train"], got["cal"], got["thr"], got["test"]))
    return (X_train_sel, y_train, groups_train, X_cal, y_cal, X_thr, y_thr,
            X_test_sel, y_test, groups_test)


def paired_cluster_bootstrap(y_true, groups, p_a, thr_a, p_b, thr_b,
                             n_boot=N_BOOT, seed=42):
    """Difference (b - a) in test accuracy and AUC, resampling SIMULATION RUNS
    with replacement and applying the same drawn runs to both configurations.

    ⛔ Each configuration keeps its OWN threshold, the one its validation half B
    selected. Re-selecting a threshold inside the bootstrap would measure a
    different quantity -- the two operating points are part of what is being
    compared.
    """
    y_true = np.asarray(y_true)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    idx_of = {g: np.flatnonzero(groups == g) for g in uniq}

    pred_a = (p_a >= thr_a).astype(int)
    pred_b = (p_b >= thr_b).astype(int)
    obs_acc = accuracy_score(y_true, pred_b) - accuracy_score(y_true, pred_a)
    obs_auc = roc_auc_score(y_true, p_b) - roc_auc_score(y_true, p_a)

    rng = np.random.RandomState(seed)
    d_acc, d_auc = [], []
    for _ in range(n_boot):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_of[g] for g in drawn])
        yy = y_true[rows]
        d_acc.append(accuracy_score(yy, pred_b[rows])
                     - accuracy_score(yy, pred_a[rows]))
        if len(np.unique(yy)) > 1:
            d_auc.append(roc_auc_score(yy, p_b[rows])
                         - roc_auc_score(yy, p_a[rows]))
    d_acc = np.asarray(d_acc)
    d_auc = np.asarray(d_auc)

    def summarise(observed, draws):
        lo, hi = np.percentile(draws, [2.5, 97.5])
        p_two = 2 * min((draws <= 0).mean(), (draws >= 0).mean())
        return {"observed": float(observed),
                "boot_mean": float(draws.mean()),
                "ci_lo": float(lo), "ci_hi": float(hi),
                "excludes_zero": bool(lo > 0 or hi < 0),
                "p_two_sided": float(min(p_two, 1.0))}

    return {"n_clusters": int(len(uniq)), "n_windows": int(len(y_true)),
            "n_boot": int(n_boot), "unit": "file_source",
            "statistic": "retuned minus inherited, each at its own locked "
                         "threshold",
            "Accuracy": summarise(obs_acc, d_acc),
            "AUC": summarise(obs_auc, d_auc)}


def score_config(det, label, params, X_train, y_train, X_cal, y_cal,
                 X_thr, y_thr, X_test, y_test, spec):
    """Fit the model from `params` and put it through the campaign protocol."""
    model = spec["build"](params, det.random_state, ddv2.MAX_JOBS)
    t0 = time.time()
    model.fit(X_train, y_train)
    secs = time.time() - t0

    models = det.calibrate_models({label: model}, X_cal, y_cal)
    if models[label] is model:
        raise SystemExit("GATE FAIL (%s): calibration fell back to the raw model"
                         % label)
    thresholds = det.optimize_thresholds(models, X_thr, y_thr)
    res = det.evaluate_model(label, models[label], X_test, y_test,
                             thresholds[label])
    y_prob = models[label].predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(
        y_test, (y_prob >= thresholds[label]).astype(int)).ravel()

    det.log("  [retune] %-10s threshold=%.4f  Accuracy=%.4f  AUC=%.4f  (%.0fs)"
            % (label, thresholds[label], res["Accuracy"], res["AUC"], secs))
    return {"params": spec["filter"](params),
            "threshold": float(thresholds[label]),
            "Accuracy": float(res["Accuracy"]), "AUC": float(res["AUC"]),
            "F1": float(res["F1"]), "MCC": float(res["MCC"]),
            "confusion": {"TN": int(tn), "FP": int(fp),
                          "FN": int(fn), "TP": int(tp)},
            "fit_seconds": round(secs, 1)}, y_prob


def run_mode(mode, results_root, hp_dir, n_iter, grid_mode, skip_search,
             model_name):
    spec = MODELS[model_name]
    out_dir = os.path.join(results_root, mode, "%s_retune" % model_name)
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "summary.json")

    cfg = build_config(mode, os.path.join(results_root, mode))
    det = ddv2.DefenseDetector(cfg)
    (X_train, y_train, groups_train, X_cal, y_cal, X_thr, y_thr,
     X_test, y_test, groups_test) = prepare(det, mode)

    inherited = spec["inherited"](hp_dir, mode)
    det.log("  [retune] model=%s; inherited params (%s): %s"
            % (model_name, hp_dir, spec["filter"](inherited)))

    if skip_search:
        # The 240 search fits are the whole cost, and their answer is already
        # on disk. Refit both configurations, keep the old search block.
        if not os.path.isfile(summary_path):
            raise SystemExit("--skip-search: no %s to read the selected "
                             "configuration from" % summary_path)
        with open(summary_path) as fh:
            prev = json.load(fh)
        selected = prev["retuned"]["params"]
        search_block = dict(prev["search"])
        search_block["reused_from"] = "previous run of this script; --skip-search"
        det.log("  [retune] --skip-search: reusing selected params %s" % selected)
    else:
        space = SEARCH_SPACES[spec["grid_key"]][grid_mode]
        n_combos = int(np.prod([len(v) for v in space.values()]))
        det.log("  [retune] grid '%s': %d combinations, sampling n_iter=%d, "
                "cv=%d grouped by run, scoring=%s"
                % (grid_mode, n_combos, n_iter, CV_FOLDS, SCORING))

        # ⛔ Grouped, unlike unified_hp_search_v2's StratifiedKFold: the four
        # windows of a run must not be split across folds or the criterion is
        # optimistic.
        cv = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True,
                                  random_state=det.random_state)
        search = RandomizedSearchCV(
            estimator=spec["build"]({}, det.random_state, ddv2.MAX_JOBS),
            param_distributions=space, n_iter=min(n_iter, n_combos),
            scoring=SCORING, cv=cv, n_jobs=1, refit=False,
            random_state=det.random_state, verbose=2)

        t0 = time.time()
        search.fit(X_train, y_train, groups=groups_train)
        search_seconds = time.time() - t0
        n_fits = min(n_iter, n_combos) * CV_FOLDS
        det.log("  [retune] search: %d fits in %.1f s  (%.1f s per fit)"
                % (n_fits, search_seconds, search_seconds / n_fits))
        det.log("  [retune] selected: %s  (cv %s = %.4f)"
                % (search.best_params_, SCORING, search.best_score_))

        pd.DataFrame(search.cv_results_).to_csv(
            os.path.join(out_dir, "cv_results.csv"), index=False)

        selected = dict(search.best_params_)
        # Selection on a grid boundary means the grid, not the data, may be
        # setting the answer -- the reason extend_hp_grids.py exists. Recorded,
        # never silently accepted.
        at_edge = {k: v for k, v in selected.items()
                   if v == min(space[k]) or v == max(space[k])}
        if at_edge:
            det.log("  [retune] ⚠ %d of %d selected values sit at a grid "
                    "boundary: %s" % (len(at_edge), len(selected), at_edge))
        search_block = {"grid": grid_mode, "n_combinations": n_combos,
                        "n_iter": min(n_iter, n_combos), "cv_folds": CV_FOLDS,
                        "cv": "StratifiedGroupKFold over file_source",
                        "scoring": SCORING, "augmentation": "none",
                        "fitted_on": "training split only, %d rows, %d features"
                                     % (len(X_train), X_train.shape[1]),
                        "seconds": round(search_seconds, 1),
                        "seconds_per_fit": round(search_seconds / n_fits, 2),
                        "best_cv_score": float(search.best_score_),
                        "selected_at_grid_boundary": at_edge}

    a, p_a = score_config(det, "inherited", inherited, X_train, y_train,
                          X_cal, y_cal, X_thr, y_thr, X_test, y_test, spec)
    b, p_b = score_config(det, "retuned", selected, X_train, y_train,
                          X_cal, y_cal, X_thr, y_thr, X_test, y_test, spec)

    # The per-window probabilities are what makes the comparison re-checkable
    # without another fit.
    pd.DataFrame({"file_source": groups_test, "y_true": np.asarray(y_test),
                  "p_inherited": p_a, "p_retuned": p_b}).to_csv(
        os.path.join(out_dir, "test_probs.csv"), index=False)

    boot = paired_cluster_bootstrap(y_test, groups_test,
                                    p_a, a["threshold"], p_b, b["threshold"],
                                    n_boot=N_BOOT, seed=det.random_state)

    summary = {
        "mode": mode,
        "model": model_name,
        "purpose": "control: how far the inherited configuration is "
                   "from one selected on this campaign. Replaces nothing.",
        "search": search_block,
        "hp_results_dir": hp_dir,
        "inherited": a,
        "retuned": b,
        "difference_retuned_minus_inherited": {
            "Accuracy": round(b["Accuracy"] - a["Accuracy"], 6),
            "AUC": round(b["AUC"] - a["AUC"], 6),
        },
        "paired_cluster_bootstrap": boot,
        "does_not_touch": "results.csv, the five analyses that read the pickle, "
                          "the pipeline, the HP pickle itself",
    }
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    for metric in ("Accuracy", "AUC"):
        s = boot[metric]
        det.log("  [retune] %s %s: %+.4f  95%% CI [%+.4f, %+.4f]  "
                "p=%.3f  excludes zero: %s"
                % (mode, metric, s["observed"], s["ci_lo"], s["ci_hi"],
                   s["p_two_sided"], s["excludes_zero"]))
    det.log("  [retune] %d test runs, %d windows, B=%d"
            % (boot["n_clusters"], boot["n_windows"], boot["n_boot"]))
    det.log("  [retune] wrote %s" % out_dir)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["static", "mobile"], required=True)
    ap.add_argument("--results-root", required=True, help="<arm>/listener/results")
    ap.add_argument("--hp-results-dir", default=HP_RESULTS_DEFAULT,
                    help="dir with {static,mobile}/best_models.pkl — the one "
                         "the five analyses read")
    ap.add_argument("--n-iter", type=int, default=80,
                    help="RandomizedSearchCV samples; 80 is the paper's setting. "
                         "Use a small value for a timing probe.")
    ap.add_argument("--grid", choices=["narrow", "extended"], default="extended",
                    help="extended is the grid extend_hp_grids.py used")
    ap.add_argument("--model", choices=sorted(MODELS), default="catboost",
                    help="which entry of the pickle to control. catboost is "
                         "read by all five analyses; xgboost only by "
                         "feature_importance_sensitivity_v2, which also builds "
                         "the stacking model from the two.")
    ap.add_argument("--skip-search", action="store_true",
                    help="reuse the configuration the previous run of this "
                         "script selected (read from summary.json), refit both "
                         "and produce the bootstrap. ~3 min instead of ~20.")
    args = ap.parse_args()
    if not os.environ.get("DCFM_DATA_BUNDLE"):
        raise SystemExit("DCFM_DATA_BUNDLE is not set; see the usage block")
    run_mode(args.mode, args.results_root, args.hp_results_dir,
             args.n_iter, args.grid, args.skip_search, args.model)


if __name__ == "__main__":
    main()
