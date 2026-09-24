#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
linearsvc_csweep.py -- R#3.12: sweep C for LinearSVC on the reported campaign's
own splits, and report convergence.

WHY THIS EXISTS
    The reported campaign trains LinearSVC at the sklearn default C=1 and never
    sweeps it. The reviewer's question is whether an SVM implementation that does
    not suffer the RBF/libsvm convergence failure keeps working -- and keeps
    performing -- at the high C values where SVC failed. Answering that needs a
    sweep, not a single point.

WHAT IT DOES NOT DO
    It does not retrain the other 16 models, and it does not touch results.csv.
    Everything it writes goes to <results>/<mode>/linearsvc_csweep/.

PROTOCOL (deliberately identical to the campaign, see STATE 23.5 / 23.6)
    The pipeline's own DefenseDetector methods are called for every step up to
    model training, so the bundle, the 17->77 engineering, the run-grouped
    60/20/20 split, the RobustScaler and the feature selection are the campaign's
    and not a reimplementation. Then:

      1. For each C: fit a RAW LinearSVC (no calibration wrapper) on the training
         split. Record convergence status and n_iter_.
      2. Select C* by AUC on validation half A. AUC is threshold-free and
         invariant under the monotone isotonic calibration that half A is later
         used for, so this selection cannot import a threshold-selection bias --
         and half B, which decides the threshold, is never touched by selection.
      3. Refit at C*, then hand it to the pipeline's OWN calibrate_models (half A)
         and optimize_thresholds (half B).
      4. Score once on the test split via the pipeline's own evaluate_model.

    The test split is read exactly once, at step 4.

    tol is left at its default for every C. Raising it would manufacture
    convergence and destroy the very measurement the reviewer asked for.

USAGE
    cd ~/ns3/nv347/ns-3.47/analysis/arms_r34_17feat
    MAX_JOBS=8 OMP_NUM_THREADS=2 ~/miniconda3/envs/manet/bin/python -u \
        linearsvc_csweep.py --mode static
"""
import runtime_guard  # noqa: F401  MUST be first: thread/MAX_JOBS caps pre-numpy

import argparse
import json
import os
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
PIPELINE = os.environ.get("DCFM_PIPELINE", os.path.join(HERE, "pipeline_17"))
sys.path.insert(0, PIPELINE)

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import RobustScaler
from sklearn.svm import LinearSVC

import defense_detection_v2 as ddv2


# The reviewer's question is specifically about C above the 1e5 cap, so the grid
# runs two decades past it. C=1 is included because it is what the reported
# campaign used, and its row is the anchor the answer already quotes.
C_GRID = [1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
MAX_ITER = 100000          # 10x the campaign's 1e4; the reviewer asked for generous
DUAL = True                # the campaign's setting -- kept for comparability


def build_config(mode, results_dir):
    """The campaign's Config: --group-by-file --no-augmentation
    --split-validation --add-linearsvc --calibrate-stacking."""
    cfg = ddv2.Config()
    cfg.data_root = "./features_%s/" % mode
    cfg.results_dir = results_dir
    cfg.group_split_by_file_source = True
    cfg.use_aggressive_augmentation = False    # --no-augmentation
    cfg.split_validation = True                # --split-validation
    cfg.add_linearsvc = True                   # --add-linearsvc
    cfg.calibrate_stacking = True              # --calibrate-stacking (no ensemble here)
    return cfg


def prepare(det):
    """Replay run_full_pipeline up to model training, using the pipeline's own
    methods. Returns the training split, the two validation halves and the test
    split, all in the campaign's feature space."""
    tall = det.load_simulation_data_enhanced()
    X, y, groups = det.preprocess_data_enhanced(tall)
    X_eng = det.engineer_advanced_features(X)

    groups_arr = np.array(groups)
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=det.config.test_size,
        random_state=det.random_state)
    train_val_idx, test_idx = next(splitter.split(X_eng, y, groups_arr))
    X_temp, y_temp = X_eng.iloc[train_val_idx], y.iloc[train_val_idx]
    X_test, y_test = X_eng.iloc[test_idx], y.iloc[test_idx]

    groups_temp = groups_arr[train_val_idx]
    splitter_val = GroupShuffleSplit(
        n_splits=1, test_size=0.25, random_state=det.random_state)
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
    X_test_sel = X_test_s[det.feature_names]

    X_train_aug, y_train_aug = det.augment_data_aggressive(X_train_sel, y_train)

    half = GroupShuffleSplit(n_splits=1, test_size=0.5,
                             random_state=det.random_state)
    a_idx, b_idx = next(half.split(X_val_sel, y_val, np.array(val_groups)))
    X_cal, y_cal = X_val_sel.iloc[a_idx], y_val.iloc[a_idx]
    X_thr, y_thr = X_val_sel.iloc[b_idx], y_val.iloc[b_idx]

    det.log("  [csweep] train %d rows, cal half %d, thr half %d, test %d, "
            "%d features" % (len(X_train_aug), len(X_cal), len(X_thr),
                             len(X_test_sel), X_train_aug.shape[1]))
    return (X_train_aug, y_train_aug, X_cal, y_cal, X_thr, y_thr,
            X_test_sel, y_test)


def fit_one(C, X_train, y_train, random_state):
    """Fit a raw LinearSVC at this C and record whether it converged."""
    model = LinearSVC(C=C, dual=DUAL, max_iter=MAX_ITER,
                      random_state=random_state)
    t0 = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X_train, y_train)
        converged = not any(issubclass(w.category, ConvergenceWarning)
                            for w in caught)
    n_iter = getattr(model, "n_iter_", None)
    if isinstance(n_iter, np.ndarray):
        n_iter = int(np.max(n_iter))
    elif n_iter is not None:
        n_iter = int(n_iter)
    return model, converged, n_iter, time.time() - t0


def best_threshold_accuracy(scores, y):
    """The pipeline's threshold rule (0.10..0.89 step 0.01) applied to a
    decision_function, rescaled to [0,1] so the grid is meaningful."""
    lo, hi = scores.min(), scores.max()
    p = (scores - lo) / (hi - lo) if hi > lo else np.full_like(scores, 0.5)
    best_acc, best_t = 0.0, 0.5
    for t in np.arange(0.1, 0.9, 0.01):
        acc = accuracy_score(y, (p >= t).astype(int))
        if acc > best_acc:
            best_acc, best_t = acc, t
    return best_acc, best_t


def run_mode(mode, out_root):
    out_dir = os.path.join(out_root, mode, "linearsvc_csweep")
    os.makedirs(out_dir, exist_ok=True)

    cfg = build_config(mode, os.path.join(out_root, mode))
    det = ddv2.DefenseDetector(cfg)
    (X_train, y_train, X_cal, y_cal, X_thr, y_thr,
     X_test, y_test) = prepare(det)

    rows = []
    for C in C_GRID:
        model, converged, n_iter, secs = fit_one(C, X_train, y_train,
                                                 det.random_state)
        s_cal = model.decision_function(X_cal)
        s_thr = model.decision_function(X_thr)
        auc_cal = roc_auc_score(y_cal, s_cal)
        acc_cal, _ = best_threshold_accuracy(s_cal, y_cal)
        rows.append({
            "C": C,
            "converged": bool(converged),
            "n_iter": n_iter,
            "max_iter": MAX_ITER,
            "fit_seconds": round(secs, 1),
            "sel_AUC_calhalf": auc_cal,
            "sel_Acc_calhalf": acc_cal,
            "AUC_thrhalf_reference": roc_auc_score(y_thr, s_thr),
        })
        det.log("  [csweep] C=%-8g converged=%-5s n_iter=%-7s "
                "AUC(A)=%.4f acc(A)=%.4f  %.0fs"
                % (C, converged, n_iter, auc_cal, acc_cal, secs))

    sweep = pd.DataFrame(rows)
    sweep.to_csv(os.path.join(out_dir, "csweep.csv"), index=False)

    # C* on half A only -- half B has decided nothing yet -- and ONLY among fits
    # that converged. A non-converged fit is not a configuration we can report as
    # selected: its parameters are wherever the iteration cap stopped it, so the
    # "peak" it shows is a property of the budget, not of C.
    conv = sweep[sweep["converged"]]
    if conv.empty:
        raise SystemExit("no C in the grid converged; nothing selectable")
    best = conv.loc[conv["sel_AUC_calhalf"].idxmax()]
    c_star = float(best["C"])
    unrestricted = sweep.loc[sweep["sel_AUC_calhalf"].idxmax()]
    det.log("  [csweep] C* = %g (converged, AUC on calibration half %.4f)"
            % (c_star, best["sel_AUC_calhalf"]))
    if float(unrestricted["C"]) != c_star:
        det.log("  [csweep] note: unrestricted argmax was C=%g (AUC %.4f) but it "
                "did NOT converge, so it is not selectable"
                % (float(unrestricted["C"]), unrestricted["sel_AUC_calhalf"]))

    # The chosen model now goes through the campaign's OWN protocol: one
    # isotonic calibration on half A, threshold on half B, test read once.
    final, conv_star, n_iter_star, _ = fit_one(c_star, X_train, y_train,
                                               det.random_state)
    models = det.calibrate_models({"SVM_linear_Cstar": final}, X_cal, y_cal)
    thresholds = det.optimize_thresholds(models, X_thr, y_thr)
    result = det.evaluate_model("SVM_linear_Cstar", models["SVM_linear_Cstar"],
                                X_test, y_test,
                                thresholds["SVM_linear_Cstar"])

    summary = {
        "mode": mode,
        "C_star": c_star,
        "C_star_converged": bool(conv_star),
        "C_star_n_iter": n_iter_star,
        "max_iter": MAX_ITER,
        "dual": DUAL,
        "n_converged": int(sweep["converged"].sum()),
        "n_grid": len(C_GRID),
        "highest_converged_C": (float(sweep.loc[sweep["converged"], "C"].max())
                                if sweep["converged"].any() else None),
        "selection": "best AUC on validation half A (calibration half) AMONG "
                     "CONVERGED FITS ONLY; threshold half B untouched by selection",
        "unrestricted_argmax_C": float(unrestricted["C"]),
        "unrestricted_argmax_converged": bool(unrestricted["converged"]),
        "test": {k: (float(v) if isinstance(v, (int, float, np.floating))
                     else v) for k, v in result.items()},
        "campaign_reference": {
            "note": "reported campaign, SVM_linear at C=1, same splits",
            "source": "listener/results/%s/results.csv" % mode,
        },
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    det.log("\n  [csweep] %s: C*=%g  test Accuracy=%.4f  test AUC=%.4f"
            % (mode, c_star, result["Accuracy"], result["AUC"]))
    det.log("  [csweep] wrote %s" % out_dir)
    return summary


# --- re-score mode -------------------------------------------------------
# The sweep above fits a BARE LinearSVC, so its test numbers are not
# interchangeable with the campaign's SVM_linear row, which is a LinearSVC inside
# an extra isotonic wrapper (defense_detection_v2.py:848) at max_iter=1e4. To
# replace a number in the reported table we must re-score the CAMPAIGN's
# estimator at the selected C -- same estimator, same downstream protocol, one C.
CAMPAIGN_MAX_ITER = 10000


def campaign_estimator(C, random_state, max_iter=None):
    """Exactly models['SVM_linear'] from the campaign, at an arbitrary C.

    max_iter defaults to the campaign's 1e4. Raising it tests whether the
    reported C=1 row is a converged fit given a larger budget; if it converges
    AND the test figures hold, the reported row needs no change at all.
    """
    return CalibratedClassifierCV(
        LinearSVC(C=C, dual=DUAL,
                  max_iter=CAMPAIGN_MAX_ITER if max_iter is None else max_iter,
                  random_state=random_state),
        cv=3, method="isotonic")


def run_rescore(mode, out_root, c_values, max_iter=None):
    out_dir = os.path.join(out_root, mode, "linearsvc_csweep")
    os.makedirs(out_dir, exist_ok=True)

    cfg = build_config(mode, os.path.join(out_root, mode))
    det = ddv2.DefenseDetector(cfg)
    (X_train, y_train, X_cal, y_cal, X_thr, y_thr,
     X_test, y_test) = prepare(det)

    # Stage 1 -- selection. Fit the campaign estimator at each C and score it on
    # validation half A. The test split is NOT touched here: scoring test at every
    # C and then picking the best would be selection on the test set.
    fitted, rows = {}, []
    for C in c_values:
        est = campaign_estimator(C, det.random_state, max_iter)
        t0 = time.time()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            est.fit(X_train, y_train)
            converged = not any(issubclass(w.category, ConvergenceWarning)
                                for w in caught)
        auc_cal = roc_auc_score(y_cal, est.predict_proba(X_cal)[:, 1])
        fitted[C] = est
        rows.append({"C": C, "converged": bool(converged),
                     "sel_AUC_calhalf": auc_cal,
                     "seconds": round(time.time() - t0, 1)})
        det.log("  [rescore] %s C=%-8g converged=%-5s AUC(A)=%.5f  %.0fs"
                % (mode, C, converged, auc_cal, time.time() - t0))

    sel = pd.DataFrame(rows)
    sel.to_csv(os.path.join(out_dir, "rescore_selection.csv"), index=False)

    conv = sel[sel["converged"]]
    if conv.empty:
        det.log("  [rescore] ⚠️ NO C converged for the campaign estimator; "
                "falling back to the unrestricted argmax and saying so")
        pick = sel.loc[sel["sel_AUC_calhalf"].idxmax()]
    else:
        pick = conv.loc[conv["sel_AUC_calhalf"].idxmax()]
    c_star = float(pick["C"])
    det.log("  [rescore] C* = %g (converged=%s, AUC on half A %.5f)"
            % (c_star, bool(pick["converged"]), pick["sel_AUC_calhalf"]))

    # Stage 2 -- score C*, and C=1 as the reproduction gate, on the test split.
    finals = []
    for C in sorted({c_star, 1.0}):
        if C not in fitted:
            continue
        models = det.calibrate_models({"SVM_linear": fitted[C]}, X_cal, y_cal)
        thresholds = det.optimize_thresholds(models, X_thr, y_thr)
        res = det.evaluate_model("SVM_linear", models["SVM_linear"],
                                 X_test, y_test, thresholds["SVM_linear"])
        res["C"] = C
        res["is_selected"] = (C == c_star)
        res["is_reproduction_gate"] = (C == 1.0)
        finals.append(res)
        det.log("  [rescore] %s C=%-8g TEST Accuracy=%.5f AUC=%.5f%s"
                % (mode, C, res["Accuracy"], res["AUC"],
                   "   <- selected" if C == c_star else ""))

    df = pd.DataFrame(finals)
    df.to_csv(os.path.join(out_dir, "rescore_campaign_estimator.csv"),
              index=False)
    det.log("  [rescore] wrote %s" % out_dir)
    return df


def main():
    ap = argparse.ArgumentParser(
        description="R#3.12: LinearSVC C sweep on the reported campaign's splits.")
    ap.add_argument("--mode", choices=["static", "mobile", "both"],
                    default="both")
    ap.add_argument("--arm", default="listener")
    ap.add_argument("--rescore", default=None,
                    help="comma-separated C values to re-score using the "
                         "CAMPAIGN's estimator (inner isotonic wrapper, "
                         "max_iter=1e4) instead of running the sweep. Include "
                         "1.0 as a reproduction gate against results.csv.")
    ap.add_argument("--max-iter", type=int, default=None,
                    help="override LinearSVC max_iter in --rescore mode "
                         "(default: the campaign's 10000). Use to test whether "
                         "the reported C=1 row converges given a larger budget.")
    args = ap.parse_args()

    arms_root = os.environ.get("DCFM_ARMS_ROOT", HERE)
    bundle = os.path.join(arms_root, args.arm, "colab_data")
    if not os.path.isdir(bundle):
        sys.exit("bundle not found: %s" % bundle)
    os.environ["DCFM_DATA_BUNDLE"] = bundle
    out_root = os.path.join(arms_root, args.arm, "results")

    modes = ["static", "mobile"] if args.mode == "both" else [args.mode]

    if args.rescore:
        c_values = [float(c) for c in args.rescore.split(",")]
        for m in modes:
            print("\n" + "=" * 70)
            print("=== re-score, campaign estimator -- %s (max_iter=%s)"
                  % (m, args.max_iter or CAMPAIGN_MAX_ITER))
            print("=" * 70, flush=True)
            run_rescore(m, out_root, c_values, args.max_iter)
        return

    out = {}
    for m in modes:
        print("\n" + "=" * 70)
        print("=== LinearSVC C sweep -- %s" % m)
        print("=" * 70, flush=True)
        out[m] = run_mode(m, out_root)

    print("\n" + "=" * 70)
    for m, s in out.items():
        print("%-7s C*=%-8g converged %d/%d  highest converged C=%s  "
              "test acc=%.4f auc=%.4f"
              % (m, s["C_star"], s["n_converged"], s["n_grid"],
                 s["highest_converged_C"], s["test"]["Accuracy"],
                 s["test"]["AUC"]))
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
