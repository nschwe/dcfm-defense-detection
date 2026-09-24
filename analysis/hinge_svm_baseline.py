#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hinge_svm_baseline.py -- fit the SUBMITTED manuscript's linear SVM with HINGE
loss on the reported campaign, through the campaign's own protocol.

WHY THIS EXISTS
    The submitted manuscript reports two diagnostic linear baselines beside the
    tuned models: Linear Discriminant Analysis (Sec. 6.1, 0.8929) and a linear
    SVM with HINGE loss (Sec. 6.1, 0.5475).  Neither is computed in the revised
    campaign.  LDA is supplied by lda_classifier.py (STATE 25.291); this script
    supplies the other one, on the same principle: a result that was in the
    submitted paper is not dropped merely because it was not recomputed.

    ⛔ IT IS NOT THE CAMPAIGN'S `SVM_linear`.  That model is
    LinearSVC(loss='squared_hinge'), sklearn's default, and r2_5:1497-1500 is
    explicit that the two are different estimators and that the campaign's row
    must not be presented as a recovery of the submitted one.  The difference
    is exactly the loss function, which is why this script sets it explicitly.

WHAT IT DOES NOT DO
    It does not retrain the 16 campaign models, does not build the stacking
    ensemble, and does not write results.csv.  Output goes to
    <results>/<mode>/hinge_svm/.

ESTIMATOR, AND WHY IT IS WRAPPED
    LinearSVC(loss='hinge', C=1.0, dual=True, max_iter=10000) -- liblinear
    supports hinge only in the dual formulation, and C=1.0 is the campaign's
    setting for its own linear SVM, so the two rows differ in the loss and in
    nothing else.
    LinearSVC exposes no predict_proba, so it is wrapped in
    CalibratedClassifierCV(cv=3, isotonic) exactly as the campaign wraps its
    SVM_linear (defense_detection_v2.py:882-885).  The pipeline's own frozen
    calibration on validation half A then applies on top, again exactly as it
    does for SVM_linear.  ⚠️ The double wrapping is the campaign's arrangement,
    not this script's invention; mirroring it is what makes the rows comparable.
    Convergence is recorded, not suppressed: hinge loss can hit the iteration
    cap, and if it does the summary says so.

PROTOCOL
    Identical to lda_classifier.py and to the campaign: the pipeline's own
    methods produce the split, the RobustScaler and the feature selection
    (same gate); then train -> calibrate on validation half A -> threshold on
    half B -> score once on test.  No selection step, so no test label informs
    anything.

USAGE  (one mode per invocation; run them one at a time)
    AN=/path/to/ns-3.47/analysis
    ARM=$AN/arms_r34_17feat_frozencal67
    cd $AN
    MAX_JOBS=8 OMP_NUM_THREADS=2 \
    DCFM_PIPELINE=$AN/frozencal67_pipeline_17 \
    DCFM_DATA_BUNDLE=$ARM/listener/colab_data \
        ~/miniconda3/envs/manet/bin/python -u hinge_svm_baseline.py \
            --mode static --results-root $ARM/listener/results
"""
import runtime_guard  # noqa: F401  MUST be first: thread/MAX_JOBS caps pre-numpy

import argparse
import json
import os
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
PIPELINE = os.environ.get("DCFM_PIPELINE",
                          os.path.join(HERE, "frozencal67_pipeline_17"))
sys.path.insert(0, PIPELINE)

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import RobustScaler
from sklearn.svm import LinearSVC

import defense_detection_v2 as ddv2

MODEL_NAME = "SVM_linear_hinge"
C_VALUE = 1.0             # the campaign's setting for its own linear SVM
MAX_ITER_DEFAULT = 10000  # the campaign's setting
# linearsvc_csweep.py raised the cap tenfold for exactly this reason; --max-iter
# 100000 reproduces that. ⛔ Raising the cap is legitimate; raising `tol` is not
# -- that would manufacture convergence instead of achieving it.

# Read off the reported campaign's own logs,
# frozencal_marathon_logs/arms_r34_17feat_frozencal67_{static,mobile}_20260906_021443.log.
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

    det.log("  [hinge] GATE OK: %d x %d -> %d engineered -> %d selected; "
            "train %d, cal half %d, thr half %d, test %d"
            % (g["rows"], g["cols"], g["engineered"], g["selected"],
               got["train"], got["cal"], got["thr"], got["test"]))
    return (X_train_aug, y_train_aug, X_cal, y_cal, X_thr, y_thr,
            X_test_sel, y_test)


def campaign_context(results_root, mode, accuracy):
    """Read-only: where this accuracy would sit among the campaign's rows, and
    what the campaign's own squared-hinge linear SVM scored. results.csv is
    opened for reading and never written."""
    path = os.path.join(results_root, mode, "results.csv")
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path)
    out = {"results_csv": path,
           "n_campaign_models": int(len(df)),
           "rank_if_inserted": int((df["Accuracy"] > accuracy).sum()) + 1,
           "campaign_accuracy_range": [float(df["Accuracy"].min()),
                                       float(df["Accuracy"].max())]}
    row = df[df["Model"] == "SVM_linear"]
    if len(row) == 1:
        out["campaign_SVM_linear_squared_hinge"] = {
            "Accuracy": float(row["Accuracy"].iloc[0]),
            "AUC": float(row["AUC"].iloc[0]),
            "note": "different estimator (squared hinge); r2_5:1497-1500 "
                    "forbids presenting either as a recovery of the other"}
    return out


def run_mode(mode, results_root, max_iter):
    # A non-default cap writes to its own directory so the campaign-setting run
    # is never overwritten: both are measurements.
    sub = "hinge_svm" if max_iter == MAX_ITER_DEFAULT         else "hinge_svm_maxiter%d" % max_iter
    out_dir = os.path.join(results_root, mode, sub)
    os.makedirs(out_dir, exist_ok=True)

    cfg = build_config(mode, os.path.join(results_root, mode))
    det = ddv2.DefenseDetector(cfg)
    (X_train, y_train, X_cal, y_cal, X_thr, y_thr,
     X_test, y_test) = prepare(det, mode)

    base = LinearSVC(loss="hinge", C=C_VALUE, dual=True, max_iter=max_iter,
                     random_state=det.random_state)
    model = CalibratedClassifierCV(base, cv=3, method="isotonic")

    t0 = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X_train, y_train)
        converged = not any(issubclass(w.category, ConvergenceWarning)
                            for w in caught)
    fit_seconds = time.time() - t0
    det.log("  [hinge] fitted in %.1fs on %d rows, %d features; converged=%s"
            % (fit_seconds, len(X_train), X_train.shape[1], converged))
    if not converged:
        det.log("  [hinge] ⚠️ liblinear did not converge within max_iter=%d; "
                "the summary records this and the number must be reported "
                "with it" % max_iter)

    models = det.calibrate_models({MODEL_NAME: model}, X_cal, y_cal)
    if models[MODEL_NAME] is model:
        raise SystemExit("GATE FAIL: calibration fell back to the raw model")

    thresholds = det.optimize_thresholds(models, X_thr, y_thr)
    result = det.evaluate_model(MODEL_NAME, models[MODEL_NAME],
                                X_test, y_test, thresholds[MODEL_NAME])

    y_prob = models[MODEL_NAME].predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(
        y_test, (y_prob >= thresholds[MODEL_NAME]).astype(int)).ravel()

    # Same columns as results.csv, in the same order.
    pd.DataFrame([{
        "Model": MODEL_NAME,
        "Threshold": thresholds[MODEL_NAME],
        "AUC": result["AUC"], "Accuracy": result["Accuracy"],
        "Precision": result["Precision"], "Recall": result["Recall"],
        "F1": result["F1"], "MCC": result["MCC"], "Kappa": result["Kappa"],
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
    }]).to_csv(os.path.join(out_dir, "hinge_svm_result.csv"), index=False)

    summary = {
        "mode": mode,
        "model": "CalibratedClassifierCV(LinearSVC(loss='hinge', C=%g, "
                 "dual=True, max_iter=%d), cv=3, isotonic)" % (C_VALUE, max_iter),
        "max_iter": int(max_iter),
        "max_iter_is_campaign_setting": bool(max_iter == MAX_ITER_DEFAULT),
        "is_not": "the campaign's SVM_linear, which uses squared hinge",
        "converged": bool(converged),
        "calibration": "CalibratedClassifierCV(FrozenEstimator(model), isotonic) "
                       "on validation half A, on top of the model's own wrapper",
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

    det.log("\n  [hinge] %s: threshold=%.4f  test Accuracy=%.4f  test AUC=%.4f"
            % (mode, thresholds[MODEL_NAME], result["Accuracy"], result["AUC"]))
    det.log("  [hinge] wrote %s" % out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["static", "mobile"], required=True)
    ap.add_argument("--results-root", required=True, help="<arm>/listener/results")
    ap.add_argument("--max-iter", type=int, default=MAX_ITER_DEFAULT,
                    help="liblinear iteration cap. %d is the campaign's "
                         "setting; 100000 is the tenfold cap linearsvc_csweep.py "
                         "uses. A non-default value writes to its own output "
                         "directory." % MAX_ITER_DEFAULT)
    args = ap.parse_args()
    if not os.environ.get("DCFM_DATA_BUNDLE"):
        raise SystemExit("DCFM_DATA_BUNDLE is not set; see the usage block")
    run_mode(args.mode, args.results_root, args.max_iter)


if __name__ == "__main__":
    main()
