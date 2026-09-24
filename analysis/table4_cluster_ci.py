#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
table4_cluster_ci.py -- the run-level cluster-bootstrap accuracy intervals that
Table 4 of the manuscript needs, for all nineteen rows of the reported campaign.

WHY THIS EXISTS
    Table 4 prints an accuracy with a 95% cluster-bootstrap interval for every
    model.  No such interval exists for the reported campaign:
    pipeline_17/cluster_bootstrap_ci.py produced no output in either arm and
    hardcodes cfg.data_root to the pre-revision 33-metric space (STATE 25.190.0,
    correcting 25.189.0 item 7).  build_roc_cache.py rebuilds the test split but
    discards the group labels, so its cache cannot support a cluster bootstrap.

    This script supplies the missing column, and nothing else.

WHAT IT COVERS
    17 campaign rows -- the sixteen classifier configurations and the stacking
    ensemble -- scored from the arm's STORED estimators, scaler, feature list
    and thresholds.  Nothing is refitted for these; the stored objects are the
    campaign.
     2 baseline rows -- LDA and the hinge-loss linear SVM.  Neither has a stored
    estimator and neither writes per-window scores, so both are refitted through
    the same protocol their own scripts use (lda_classifier.py,
    hinge_svm_baseline.py).  ⛔ A refit is only admissible because it is gated:
    the reproduced accuracy and AUC must equal the stored summary.json to 1e-9,
    or the run aborts.  These two are reference baselines and are NOT part of
    the seventeen (r2_5:1601-1610).

WHAT IT DOES NOT DO
    It does not write results.csv, does not touch the stored pickle, does not
    rebuild the ensemble, and does not select anything on test labels.  All
    output goes to <results>/<mode>/table4_ci/, a new directory.

THE INTERVAL
    Clusters are simulation runs (file_source), which is the unit the campaign
    partitions on.  For B = 2000 resamples, len(unique) runs are drawn with
    replacement and the windows of the drawn runs concatenated; the statistic is
    the accuracy of that resample at the model's own stored threshold.  The
    interval is the 2.5/97.5 percentile pair.  This is the scheme of
    r23_paired_bootstrap.paired_bootstrap, unpaired: Table 4 reports a level per
    model, not a difference between arms, so nothing here is paired.
    ⚠️ This resamples the TEST SAMPLE.  It is not a distribution over refits of
    the model, and it is not the 20-repetition scheme of Section 5.5.

USAGE  (one mode per invocation; run them one at a time)
    AN=/path/to/ns-3.47/analysis
    ARM=$AN/arms_r34_17feat_frozencal67
    cd $AN
    MAX_JOBS=8 OMP_NUM_THREADS=2 \
    DCFM_PIPELINE=$AN/frozencal67_pipeline_17 \
    DCFM_DATA_BUNDLE=$ARM/listener/colab_data \
        ~/miniconda3/envs/manet/bin/python -u table4_cluster_ci.py \
            --mode static --results-root $ARM/listener/results
"""
import runtime_guard  # noqa: F401  MUST be first: thread/MAX_JOBS caps pre-numpy

import argparse
import glob
import json
import os
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
PIPELINE = os.environ.get("DCFM_PIPELINE",
                          os.path.join(HERE, "frozencal67_pipeline_17"))
sys.path.insert(0, PIPELINE)

import joblib
import numpy as np
from decimal import Decimal, ROUND_HALF_UP
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import RobustScaler
from sklearn.svm import LinearSVC

import defense_detection_v2 as ddv2

# The two baselines, exactly as their own scripts construct them.
LDA_NAME = "LDA"
HINGE_NAME = "SVM_linear_hinge"
HINGE_C = 1.0             # hinge_svm_baseline.py:83
HINGE_MAX_ITER = 10000    # hinge_svm_baseline.py:84, the campaign's setting

# The C-sweep's SELECTED static configuration, which r3_12:5653 requires the
# manuscript to print instead of the campaign's C = 1 row.  ⛔ The 3/9 re-score
# that produced 0.8935 / 0.924301 ran on the SEVENTY-SEVEN-column arm -- its log
# reads "Created 77 features" and its artefact lives under arms_r34_17feat --
# while the reported campaign is the 67-column arm (STATE 25.284.5.1, 6/9).
# Nobody re-measured it there.  This refit does, on the reported arm, and reports
# whether the published pair carries across.  Static only: the mobile sweep
# selected C = 1, which IS the campaign configuration, so the campaign row stands.
RESCORE_NAME = "SVM_linear_C0.01"
RESCORE_C = 0.01
RESCORE_REF = os.path.join(
    HERE, "arms_r34_17feat", "listener", "results", "static",
    "linearsvc_csweep", "rescore_campaign_estimator.csv")

# Read off the reported campaign's own logs, frozencal_marathon_logs/
# arms_r34_17feat_frozencal67_{static,mobile}_20260906_021443.log.
GATE = {
    "static": {"rows": 40000, "cols": 18, "engineered": 67, "selected": 28},
    "mobile": {"rows": 40000, "cols": 18, "engineered": 67, "selected": 32},
}
GATE_SPLIT = {"train": 24000, "cal": 4000, "thr": 4000, "test": 8000}

TOL = 1e-9


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
    methods, gate every structural number against the campaign log, and return
    both the freshly selected frames (for the two refits) and the UNSCALED
    engineered test frame (for the stored estimators, which carry their own
    scaler).  Identical to lda_classifier.prepare, plus the extra returns."""
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
    X_test_eng, y_test = X_eng.iloc[test_idx], y.iloc[test_idx]
    groups_test = groups_arr[test_idx]

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
    X_test_s = pd.DataFrame(det.scaler.transform(X_test_eng),
                            columns=X_test_eng.columns, index=X_test_eng.index)

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

    det.log("  [ci] GATE OK: %d x %d -> %d engineered -> %d selected; "
            "train %d, cal half %d, thr half %d, test %d; %d test runs"
            % (g["rows"], g["cols"], g["engineered"], g["selected"],
               got["train"], got["cal"], got["thr"], got["test"],
               len(np.unique(groups_test))))
    return {
        "X_train": X_train_aug, "y_train": y_train_aug,
        "X_cal": X_cal, "y_cal": y_cal,
        "X_thr": X_thr, "y_thr": y_thr,
        "X_test_sel": X_test_sel,          # fresh scaler + fresh selection
        "X_test_eng": X_test_eng,          # unscaled; the stored scaler wants this
        "y_test": y_test, "groups_test": groups_test,
    }


def draw_clusters(groups, n_boot, seed):
    """One (n_boot x n_clusters) matrix of cluster positions, drawn once and
    reused by every model.  Drawing per model would give each row its own
    resamples and the columns of the table would not be comparable."""
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    return uniq, rng.integers(0, len(uniq), size=(n_boot, len(uniq)))


def cluster_ci(correct, groups, uniq, draws):
    """Resample CLUSTERS (simulation runs) with replacement; the statistic is
    the accuracy of the resample.  Same scheme as
    r23_paired_bootstrap.paired_bootstrap, without the pairing.

    ⭐ Computed on per-cluster (sum, count) rather than by concatenating the
    drawn rows.  It is the same number: the accuracy of a resample is the total
    number of correct windows over the total number of windows in the drawn
    clusters, and a cluster drawn twice contributes both to the numerator and to
    the denominator twice.  Clusters here hold four windows each, so the
    denominator is constant in practice -- but it is carried anyway, because
    nothing in this script may assume that."""
    order = np.argsort(groups, kind="stable")
    g_sorted = groups[order]
    starts = np.searchsorted(g_sorted, uniq, side="left")
    ends = np.searchsorted(g_sorted, uniq, side="right")
    sums = np.array([correct[order[a:b]].sum() for a, b in zip(starts, ends)])
    counts = (ends - starts).astype(float)

    accs = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return (float(np.percentile(accs, 2.5)),
            float(np.percentile(accs, 97.5)),
            float(accs.std(ddof=1)))


def campaign_rows(data, results_root, mode):
    """The seventeen, from the arm's STORED objects.  Nothing is fitted."""
    res_dir = os.path.join(results_root, mode)
    results = pd.read_csv(os.path.join(res_dir, "results.csv"))

    hits = sorted(glob.glob(os.path.join(res_dir, "best_model_*.pkl")))
    if len(hits) != 1:
        raise SystemExit("GATE FAIL: %d model pickles under %s" % (len(hits), res_dir))
    md = joblib.load(hits[0])
    all_models = md.get("all_models") or {}
    all_thr = md.get("all_thresholds") or {}
    if not all_models:
        raise SystemExit("GATE FAIL: %s carries no 'all_models'" % os.path.basename(hits[0]))

    scaler, cols = md["scaler"], list(md["feature_names"])
    n_seen = getattr(scaler, "n_features_in_", None)
    X_eng = data["X_test_eng"]
    if n_seen is not None and n_seen != X_eng.shape[1]:
        raise SystemExit("GATE FAIL: stored scaler expects %d columns, rebuilt "
                         "frame has %d" % (n_seen, X_eng.shape[1]))
    X_scaled = pd.DataFrame(scaler.transform(X_eng),
                            columns=X_eng.columns, index=X_eng.index)
    X_sel = X_scaled[cols]
    y_arr = data["y_test"].to_numpy()

    print("\n=== the seventeen (stored estimators, stored thresholds) ===")
    rows = []
    for _, r in results.iterrows():
        name = str(r["Model"])
        est = all_models.get(name)
        if est is None:
            raise SystemExit("GATE FAIL: %s is in results.csv but not in "
                             "all_models" % name)
        thr = float(all_thr.get(name, r["Threshold"]))
        proba = est.predict_proba(X_sel)[:, 1]
        correct = ((proba >= thr).astype(int) == y_arr).astype(float)
        acc, auc = float(correct.mean()), float(roc_auc_score(y_arr, proba))

        d_acc, d_auc = acc - float(r["Accuracy"]), auc - float(r["AUC"])
        if abs(d_acc) > TOL or abs(d_auc) > TOL:
            raise SystemExit("GATE FAIL [reproduction] %s: acc %+.1e, auc %+.1e "
                             "against results.csv. No interval is computed from "
                             "a reconstruction that does not match."
                             % (name, d_acc, d_auc))
        rows.append({"Model": name, "block": "campaign", "Threshold": thr,
                     "Accuracy": acc, "AUC": auc, "correct": correct})
        print("  ok  %-22s acc %.6f  auc %.6f" % (name, acc, auc))
    if len(rows) != 17:
        raise SystemExit("GATE FAIL: %d campaign rows, expected 17" % len(rows))
    return rows


def _fit_baseline(det, data, name, estimator):
    """train -> calibrate on validation half A -> threshold on half B -> test.
    The protocol of lda_classifier.py and hinge_svm_baseline.py, unchanged."""
    models = det.calibrate_models({name: estimator}, data["X_cal"], data["y_cal"])
    if models[name] is estimator:
        raise SystemExit("GATE FAIL: calibration fell back to the raw model (%s)" % name)
    thresholds = det.optimize_thresholds(models, data["X_thr"], data["y_thr"])
    result = det.evaluate_model(name, models[name], data["X_test_sel"],
                                data["y_test"], thresholds[name])
    proba = models[name].predict_proba(data["X_test_sel"])[:, 1]
    return float(thresholds[name]), float(result["Accuracy"]), float(result["AUC"]), proba


def _gate_against_summary(results_root, mode, sub, name, acc, auc):
    path = os.path.join(results_root, mode, sub, "summary.json")
    if not os.path.isfile(path):
        raise SystemExit("GATE FAIL: no %s to reproduce against. The baseline "
                         "row must exist before its interval does." % path)
    with open(path) as fh:
        stored = json.load(fh)["test"]
    d_acc, d_auc = acc - float(stored["Accuracy"]), auc - float(stored["AUC"])
    if abs(d_acc) > TOL or abs(d_auc) > TOL:
        raise SystemExit("GATE FAIL [reproduction] %s: acc %+.1e, auc %+.1e "
                         "against %s. The refit is not the stored row and no "
                         "interval may be attached to it." % (name, d_acc, d_auc, path))
    print("  ok  %-22s acc %.6f  auc %.6f  (reproduces %s)"
          % (name, acc, auc, os.path.join(sub, "summary.json")))


def baseline_rows(det, data, results_root, mode):
    """LDA and the hinge-loss linear SVM, refitted and gated.  ⛔ Outside the
    seventeen: r2_5:1601-1610."""
    print("\n=== the two additional baselines (refitted, gated) ===")
    rows = []
    y_arr = data["y_test"].to_numpy()

    t0 = time.time()
    lda = LDA()
    lda.fit(data["X_train"], data["y_train"])
    thr, acc, auc, proba = _fit_baseline(det, data, LDA_NAME, lda)
    _gate_against_summary(results_root, mode, "lda_classifier", LDA_NAME, acc, auc)
    rows.append({"Model": LDA_NAME, "block": "baseline", "Threshold": thr,
                 "Accuracy": acc, "AUC": auc, "converged": True,
                 "correct": ((proba >= thr).astype(int) == y_arr).astype(float)})
    det.log("  [ci] LDA refitted in %.1fs" % (time.time() - t0))

    t0 = time.time()
    base = LinearSVC(loss="hinge", C=HINGE_C, dual=True,
                     max_iter=HINGE_MAX_ITER, random_state=det.random_state)
    hinge = CalibratedClassifierCV(base, cv=3, method="isotonic")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        hinge.fit(data["X_train"], data["y_train"])
        converged = not any(issubclass(w.category, ConvergenceWarning)
                            for w in caught)
    thr, acc, auc, proba = _fit_baseline(det, data, HINGE_NAME, hinge)
    _gate_against_summary(results_root, mode, "hinge_svm", HINGE_NAME, acc, auc)
    if converged:
        # The stored summary records converged=false in both modes. A refit that
        # converges is a DIFFERENT fit, and its interval would not belong to the
        # reported row even if the accuracy happened to match.
        raise SystemExit("GATE FAIL: the hinge refit converged, while "
                         "hinge_svm/summary.json records converged=false.")
    rows.append({"Model": HINGE_NAME, "block": "baseline", "Threshold": thr,
                 "Accuracy": acc, "AUC": auc, "converged": converged,
                 "correct": ((proba >= thr).astype(int) == y_arr).astype(float)})
    det.log("  [ci] hinge SVM refitted in %.1fs; converged=%s"
            % (time.time() - t0, converged))

    if mode == "static":
        rows.append(rescore_row(det, data, y_arr))
    return rows


def rescore_row(det, data, y_arr):
    """The C-sweep's selected static configuration, refitted on the REPORTED
    arm.  Built exactly as the campaign builds SVM_linear
    (defense_detection_v2.py:880-885), with C = 0.01 in place of C = 1.

    ⛔ The comparison against the 3/9 re-score is REPORTED, not enforced.  That
    measurement was made on the 77-column arm, so an abort on inequality would
    destroy the very fact this row exists to establish -- whether the published
    0.8935 / 0.9243 belongs to the campaign the manuscript names."""
    t0 = time.time()
    base = LinearSVC(C=RESCORE_C, dual=True, max_iter=HINGE_MAX_ITER,
                     random_state=det.random_state)
    model = CalibratedClassifierCV(base, cv=3, method="isotonic")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(data["X_train"], data["y_train"])
        converged = not any(issubclass(w.category, ConvergenceWarning)
                            for w in caught)
    thr, acc, auc, proba = _fit_baseline(det, data, RESCORE_NAME, model)

    if not os.path.isfile(RESCORE_REF):
        raise SystemExit("GATE FAIL: no %s to compare against." % RESCORE_REF)
    ref = pd.read_csv(RESCORE_REF).iloc[0]
    d_acc, d_auc = acc - float(ref["Accuracy"]), auc - float(ref["AUC"])
    carries = abs(d_acc) <= TOL and abs(d_auc) <= TOL

    print("  %s  %-22s acc %.6f  auc %.6f  thr %.2f  converged=%s"
          % ("ok " if carries else "⚠️ ", RESCORE_NAME, acc, auc, thr, converged))
    print("      77-column re-score (3/9): acc %.6f  auc %.6f  thr %.2f"
          % (float(ref["Accuracy"]), float(ref["AUC"]), float(ref["Threshold"])))
    print("      delta on the reported arm: acc %+.2e  auc %+.2e  ->  %s"
          % (d_acc, d_auc,
             "the published pair CARRIES to the 67-column arm"
             if carries else
             "⛔ THE PUBLISHED PAIR DOES NOT CARRY. r3_12 and the 0.0063 margin move."))
    det.log("  [ci] C=%g re-score refitted in %.1fs" % (RESCORE_C, time.time() - t0))
    return {"Model": RESCORE_NAME, "block": "rescore", "Threshold": thr,
            "Accuracy": acc, "AUC": auc, "converged": converged,
            "carries_from_77": bool(carries),
            "ref_accuracy": float(ref["Accuracy"]), "ref_auc": float(ref["AUC"]),
            "correct": ((proba >= thr).astype(int) == y_arr).astype(float)}


def run_mode(mode, results_root, n_boot, seed):
    out_dir = os.path.join(results_root, mode, "table4_ci")
    os.makedirs(out_dir, exist_ok=True)

    cfg = build_config(mode, os.path.join(results_root, mode))
    det = ddv2.DefenseDetector(cfg)
    data = prepare(det, mode)

    rows = campaign_rows(data, results_root, mode)
    rows += baseline_rows(det, data, results_root, mode)

    groups = data["groups_test"]
    n_clusters, n_windows = int(len(np.unique(groups))), int(len(groups))
    print("\n=== cluster bootstrap: B = %d, seed = %d, %d clusters, %d windows ==="
          % (n_boot, seed, n_clusters, n_windows))

    uniq, draws = draw_clusters(groups, n_boot, seed)
    out = []
    for r in rows:
        lo, hi, sd = cluster_ci(r["correct"], groups, uniq, draws)
        rec = {"Model": r["Model"], "block": r["block"],
               "Threshold": r["Threshold"], "Accuracy": r["Accuracy"],
               "acc_ci_low": lo, "acc_ci_high": hi, "acc_boot_sd": sd,
               "AUC": r["AUC"]}
        for k in ("converged", "carries_from_77", "ref_accuracy", "ref_auc"):
            if k in r:
                rec[k] = r[k]
        out.append(rec)
        print("  %-22s %.4f [%.4f, %.4f]" % (r["Model"], r["Accuracy"], lo, hi))

    df = pd.DataFrame(out)
    df.to_csv(os.path.join(out_dir, "table4_ci.csv"), index=False)
    with open(os.path.join(out_dir, "table4_ci.json"), "w") as fh:
        json.dump({"mode": mode, "arm": os.path.abspath(results_root),
                   "pipeline": PIPELINE, "n_boot": n_boot, "seed": seed,
                   "n_clusters": n_clusters, "n_windows": n_windows,
                   "statistic": "accuracy at the model's own stored threshold",
                   "cluster": "simulation run (file_source)",
                   "interval": "two-sided 95% percentile, 2.5/97.5",
                   "rows": out}, fh, indent=2)

    # LaTeX, so the table is copied from a computation rather than typed.
    # ⛔ NOT "%.4f": that rounds half to even on the binary value, and four cells
    # are exact halves where it disagrees with the response letter --
    # 0.92975 -> 0.9297 against the letter's 0.9298, 0.88425 -> 0.8842 against
    # 0.8843, 0.89625 -> 0.8962 against 0.8963. The letter is already with the
    # reviewers, so the paper rounds half UP and the two agree.
    def r4(x):
        return Decimal(str(x)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)

    print("\n=== %s column, for tab:in_domain_performance ===" % mode)
    for r in out:
        print("  %-22s & %s [%s, %s] & %s \\\\"
              % (r["Model"], r4(r["Accuracy"]), r4(r["acc_ci_low"]),
                 r4(r["acc_ci_high"]), r4(r["AUC"])))
    print("\n  wrote %s" % out_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["static", "mobile"], required=True)
    ap.add_argument("--results-root", required=True, help="<arm>/listener/results")
    ap.add_argument("--n-boot", type=int, default=2000,
                    help="2000, the value Section 5.5 defines for the whole paper")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if not os.environ.get("DCFM_DATA_BUNDLE"):
        raise SystemExit("DCFM_DATA_BUNDLE is not set; see the usage block")
    run_mode(args.mode, args.results_root, args.n_boot, args.seed)


if __name__ == "__main__":
    main()
