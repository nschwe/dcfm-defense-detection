#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lda_classifier.py -- fit Linear Discriminant Analysis as a CLASSIFIER on the
reported campaign, through the campaign's own protocol.

WHY THIS EXISTS
    LDA appears in the submitted manuscript twice, and they are different
    things.  The LDA SEPARATION RATIO (compute_separability_v2.py) is a measure
    on the feature space and is already computed for the revision.  LDA as a
    CLASSIFIER -- Sec. 6.1 (0.8929) and Sec. 7.1 -- is not: the revised
    pipeline imports LinearDiscriminantAnalysis at :86 but never instantiates
    it, so results.csv carries 17 rows and no discriminant-analysis row.
    Comments 2.5 and 3.9 are both about the classifier, and neither asks for it
    to be dropped.  This script supplies the missing row so the keep-or-remove
    decision rests on a measured number rather than on its absence.

WHAT IT DOES NOT DO
    It does not retrain the 16 campaign models, does not build the stacking
    ensemble, and does not write results.csv.  Adding LDA to the pipeline's
    model dict would be the WRONG way to get this number: create_super_ensemble
    takes list(models.items())[:10], so an insertion before position 10 would
    silently change the ensemble and every headline figure with it.
    Everything written goes to <results>/<mode>/lda_classifier/.

PROTOCOL (identical to the campaign -- STATE 23.5, and the frozen-calibration
fix of 1/9)
    train  ->  calibration half A  ->  threshold half B  ->  test
      1. The pipeline's own methods produce the bundle, the 18 -> 67
         engineering, the run-grouped 60/20/20 split, the RobustScaler and the
         feature selection.  Nothing here is a reimplementation.
      2. LinearDiscriminantAnalysis() at library defaults, fitted on the
         training split.  No hyperparameter search: the campaign runs every
         model at fixed settings, and a search here would not be comparable.
      3. det.calibrate_models -- CalibratedClassifierCV(FrozenEstimator(lda),
         isotonic) on half A, exactly as the 16 models are calibrated.
      4. det.optimize_thresholds on half B, from the calibrated probabilities.
      5. det.evaluate_model on the test split.  Read once, at this step.

    There is no selection step anywhere, so no test label informs anything.

USAGE  (one mode per invocation; run them one at a time)
    AN=/path/to/ns-3.47/analysis
    ARM=$AN/arms_r34_17feat_frozencal67
    cd $AN
    MAX_JOBS=8 OMP_NUM_THREADS=2 \
    DCFM_PIPELINE=$AN/frozencal67_pipeline_17 \
    DCFM_DATA_BUNDLE=$ARM/listener/colab_data \
        ~/miniconda3/envs/manet/bin/python -u lda_classifier.py \
            --mode static --results-root $ARM/listener/results
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import RobustScaler

import defense_detection_v2 as ddv2

MODEL_NAME = "LDA"

# Read off the reported campaign's own logs,
# frozencal_marathon_logs/arms_r34_17feat_frozencal67_{static,mobile}_20260906_021443.log.
# If any of these does not reproduce, this script is not standing on the
# campaign's splits and its number would not be comparable with results.csv.
GATE = {
    "static": {"rows": 40000, "cols": 18, "engineered": 67, "selected": 28},
    "mobile": {"rows": 40000, "cols": 18, "engineered": 67, "selected": 32},
}
GATE_SPLIT = {"train": 24000, "cal": 4000, "thr": 4000, "test": 8000}


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
    methods, and gate every structural number against the campaign log."""
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

    groups_temp = groups_arr[train_val_idx]
    splitter_val = GroupShuffleSplit(n_splits=1, test_size=0.25,
                                     random_state=det.random_state)
    train_idx, val_idx = next(splitter_val.split(X_temp, y_temp, groups_temp))
    X_train, y_train = X_temp.iloc[train_idx], y_temp.iloc[train_idx]
    X_val, y_val = X_temp.iloc[val_idx], y_temp.iloc[val_idx]
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

    X_train_aug, y_train_aug = det.augment_data_aggressive(X_train_sel, y_train)

    half = GroupShuffleSplit(n_splits=1, test_size=0.5,
                             random_state=det.random_state)
    a_idx, b_idx = next(half.split(X_val_sel, y_val, np.array(val_groups)))
    X_cal, y_cal = X_val_sel.iloc[a_idx], y_val.iloc[a_idx]
    X_thr, y_thr = X_val_sel.iloc[b_idx], y_val.iloc[b_idx]

    got = {"train": len(X_train_aug), "cal": len(X_cal),
           "thr": len(X_thr), "test": len(X_test_sel)}
    if got != GATE_SPLIT:
        raise SystemExit("GATE FAIL: split sizes %s, campaign log says %s"
                         % (got, GATE_SPLIT))

    det.log("  [lda] GATE OK: %d x %d -> %d engineered -> %d selected; "
            "train %d, cal half %d, thr half %d, test %d"
            % (g["rows"], g["cols"], g["engineered"], g["selected"],
               got["train"], got["cal"], got["thr"], got["test"]))
    return (X_train_aug, y_train_aug, X_cal, y_cal, X_thr, y_thr,
            X_test_sel, y_test)


def campaign_context(results_root, mode, accuracy):
    """Read-only: where this accuracy would sit among the campaign's 17 rows.
    results.csv is opened for reading and never written."""
    path = os.path.join(results_root, mode, "results.csv")
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path)
    better = int((df["Accuracy"] > accuracy).sum())
    return {"results_csv": path,
            "n_campaign_models": int(len(df)),
            "rank_if_inserted": better + 1,
            "campaign_accuracy_range": [float(df["Accuracy"].min()),
                                        float(df["Accuracy"].max())]}


def run_mode(mode, results_root):
    out_dir = os.path.join(results_root, mode, "lda_classifier")
    os.makedirs(out_dir, exist_ok=True)

    cfg = build_config(mode, os.path.join(results_root, mode))
    det = ddv2.DefenseDetector(cfg)
    (X_train, y_train, X_cal, y_cal, X_thr, y_thr,
     X_test, y_test) = prepare(det, mode)

    t0 = time.time()
    lda = LDA()
    lda.fit(X_train, y_train)
    fit_seconds = time.time() - t0
    det.log("  [lda] fitted in %.1fs on %d rows, %d features"
            % (fit_seconds, len(X_train), X_train.shape[1]))

    models = det.calibrate_models({MODEL_NAME: lda}, X_cal, y_cal)
    if models[MODEL_NAME] is lda:
        raise SystemExit("GATE FAIL: calibration fell back to the raw model")

    thresholds = det.optimize_thresholds(models, X_thr, y_thr)
    result = det.evaluate_model(MODEL_NAME, models[MODEL_NAME],
                                X_test, y_test, thresholds[MODEL_NAME])

    y_prob = models[MODEL_NAME].predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(
        y_test, (y_prob >= thresholds[MODEL_NAME]).astype(int)).ravel()

    # Same columns as results.csv, in the same order, so the row can be read
    # beside the campaign's without reshaping anything.
    row = pd.DataFrame([{
        "Model": MODEL_NAME,
        "Threshold": thresholds[MODEL_NAME],
        "AUC": result["AUC"], "Accuracy": result["Accuracy"],
        "Precision": result["Precision"], "Recall": result["Recall"],
        "F1": result["F1"], "MCC": result["MCC"], "Kappa": result["Kappa"],
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
    }])
    row.to_csv(os.path.join(out_dir, "lda_result.csv"), index=False)

    summary = {
        "mode": mode,
        "model": "sklearn LinearDiscriminantAnalysis, library defaults",
        "calibration": "CalibratedClassifierCV(FrozenEstimator(lda), isotonic) "
                       "on validation half A",
        "threshold": {"value": float(thresholds[MODEL_NAME]),
                      "selected_on": "validation half B, calibrated probabilities",
                      "grid": "0.10..0.89 step 0.01, accuracy-maximising"},
        "n_features": int(X_train.shape[1]),
        "fit_seconds": round(fit_seconds, 1),
        "test": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                 for k, v in result.items()},
        "confusion": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "campaign_context": campaign_context(results_root, mode,
                                             result["Accuracy"]),
        "does_not_touch": "results.csv, the 16 campaign models, the stacking ensemble",
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    det.log("\n  [lda] %s: threshold=%.4f  test Accuracy=%.4f  test AUC=%.4f"
            % (mode, thresholds[MODEL_NAME], result["Accuracy"], result["AUC"]))
    det.log("  [lda] wrote %s" % out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["static", "mobile"], required=True)
    ap.add_argument("--results-root", required=True, help="<arm>/listener/results")
    args = ap.parse_args()
    if not os.environ.get("DCFM_DATA_BUNDLE"):
        raise SystemExit("DCFM_DATA_BUNDLE is not set; see the usage block")
    run_mode(args.mode, args.results_root)


if __name__ == "__main__":
    main()
