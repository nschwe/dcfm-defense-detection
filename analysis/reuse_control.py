#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""reuse_control.py -- R#3.13: the ISOLATED effect of validation-data reuse.

The submitted-vs-revised reconstruction (arms_r34_17feat_nosplit vs
_frozencal) changes three things at once: whether the tuning steps share data,
how many rows the calibration step sees, and whether the threshold comes from
calibrated or uncalibrated scores. It therefore measures a net protocol change,
not the reviewer's quantity.

This script changes exactly ONE thing. Everything is held fixed -- the same
trained classifier, the same calibrator already fitted on validation half A,
the same threshold grid, the same test split -- and the only difference is:

    reuse arm    : threshold chosen on half A, the rows the calibrator SAW
    disjoint arm : threshold chosen on half B, rows it did not

Delta = test accuracy(reuse) - test accuracy(disjoint). That is the isolated
effect of the dependence R#3.13 asks about.

No training happens here: the models come from the arm's stored pickle, which
holds every fitted estimator (`all_models`) and the scaler, so the splits and
the estimators are the run's own.

  ⛔ Stacking_Ensemble is reported separately and is NOT a reuse control. With
  the 2/9 stacking fix the ensemble is built from the RAW models, so half A
  never entered its construction; for it, A and B are simply two held-out
  halves and the comparison measures threshold-selection noise, not reuse.

Reproduction gate -- TWO checks, both in-script since 2/9:
  * accuracy : the disjoint arm must reproduce the stored test accuracy;
  * threshold: the disjoint arm must reproduce the stored threshold exactly.
  ⛔ The second is not redundant. The accuracy-vs-threshold curve is full of
  plateaus, so a wrong threshold can still land on the right test accuracy --
  which is how the max() tie-break bug passed an accuracy-only gate and had to
  be caught by a separate script (STATE 25.188.10). Never remove it.

Usage:
  python reuse_control.py --arm arms_r34_17feat_frozencal --mode both
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

AN = os.path.dirname(os.path.abspath(__file__))
PIPELINE = os.environ.get("REUSE_PIPELINE", os.path.join(AN, "frozencal", "pipeline_17"))
sys.path.insert(0, PIPELINE)

import joblib
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import RobustScaler

import defense_detection_v2 as ddv2

GRID = np.arange(0.1, 0.9, 0.01)      # the pipeline's own threshold grid


def build_config(mode, arm_dir):
    cfg = ddv2.Config()
    cfg.data_root = "./features_%s/" % mode
    cfg.results_dir = os.path.join(arm_dir, "listener", "results", mode)
    cfg.group_split_by_file_source = True
    cfg.use_aggressive_augmentation = False
    cfg.split_validation = True
    cfg.add_linearsvc = True
    cfg.calibrate_stacking = True
    return cfg


def rebuild_splits(det):
    """Reproduce the run's own partitions, using the pipeline's own methods."""
    tall = det.load_simulation_data_enhanced()
    X, y, groups = det.preprocess_data_enhanced(tall)
    X_eng = det.engineer_advanced_features(X)

    g = np.array(groups)
    sp = GroupShuffleSplit(n_splits=1, test_size=det.config.test_size,
                           random_state=det.random_state)
    tv_idx, te_idx = next(sp.split(X_eng, y, g))
    X_tv, y_tv = X_eng.iloc[tv_idx], y.iloc[tv_idx]
    X_te, y_te = X_eng.iloc[te_idx], y.iloc[te_idx]

    g_tv = g[tv_idx]
    sv = GroupShuffleSplit(n_splits=1, test_size=0.25,
                           random_state=det.random_state)
    tr_idx, va_idx = next(sv.split(X_tv, y_tv, g_tv))
    X_tr, y_tr = X_tv.iloc[tr_idx], y_tv.iloc[tr_idx]
    X_va, y_va = X_tv.iloc[va_idx], y_tv.iloc[va_idx]
    g_va = g_tv[va_idx]

    det.scaler = RobustScaler()
    X_tr_s = pd.DataFrame(det.scaler.fit_transform(X_tr),
                          columns=X_tr.columns, index=X_tr.index)
    X_va_s = pd.DataFrame(det.scaler.transform(X_va),
                          columns=X_va.columns, index=X_va.index)
    X_te_s = pd.DataFrame(det.scaler.transform(X_te),
                          columns=X_te.columns, index=X_te.index)

    _, X_va_sel = det.select_features_intelligent(X_tr_s, X_va_s, y_tr)
    X_te_sel = X_te_s[det.feature_names]

    half = GroupShuffleSplit(n_splits=1, test_size=0.5,
                             random_state=det.random_state)
    a_idx, b_idx = next(half.split(X_va_sel, y_va, np.array(g_va)))
    return (X_va_sel.iloc[a_idx], y_va.iloc[a_idx],      # half A: calibration
            X_va_sel.iloc[b_idx], y_va.iloc[b_idx],      # half B: threshold
            X_te_sel, y_te)


def best_threshold(model, X, y):
    """Byte-for-byte the pipeline's rule (defense_detection_v2.optimize_thresholds).

    ⛔ The tie-break matters and is not cosmetic. The accuracy-vs-threshold curve
    is full of plateaus; the pipeline scans ascending and keeps a candidate only
    on STRICTLY greater accuracy, so among tied thresholds it returns the LOWEST.
    A max() over (acc, t) pairs returns the HIGHEST instead, which moved 12 of 17
    thresholds and broke the reproduction gate on 2-3 models per domain.
    """
    p = model.predict_proba(X)[:, 1]
    best_acc, best_t = 0.0, 0.5
    for t in GRID:
        acc = accuracy_score(y, (p >= t).astype(int))
        if acc > best_acc:
            best_acc, best_t = acc, t
    return float(best_t), float(best_acc)


def run_mode(arm_dir, mode):
    cfg = build_config(mode, arm_dir)
    det = ddv2.DefenseDetector(cfg)
    X_a, y_a, X_b, y_b, X_te, y_te = rebuild_splits(det)
    det.log("  [reuse] half A %d rows, half B %d rows, test %d rows, %d features"
            % (len(X_a), len(X_b), len(X_te), X_te.shape[1]))

    res_dir = os.path.join(arm_dir, "listener", "results", mode)
    hits = glob.glob(os.path.join(res_dir, "best_model_*.pkl"))
    if len(hits) != 1:
        sys.exit("ABORT: %d pickles in %s, expected 1" % (len(hits), res_dir))
    md = joblib.load(hits[0])
    models = md.get("all_models") or {}
    stored_thr = md.get("all_thresholds") or {}
    stored = pd.read_csv(os.path.join(res_dir, "results.csv")).set_index("Model")

    rows = []
    for name, model in models.items():
        try:
            t_a, sel_a = best_threshold(model, X_a, y_a)   # REUSE: calibrator saw A
            t_b, sel_b = best_threshold(model, X_b, y_b)   # DISJOINT
            p_te = model.predict_proba(X_te)[:, 1]
            acc_a = accuracy_score(y_te, (p_te >= t_a).astype(int))
            acc_b = accuracy_score(y_te, (p_te >= t_b).astype(int))
            auc = roc_auc_score(y_te, p_te)
        except Exception as exc:
            det.log("  [reuse] %s skipped (%s)" % (name, exc))
            continue

        # reproduction gate: half B is what the run itself used.
        # TWO checks, not one. Accuracy agreement alone is NOT sufficient: the
        # accuracy-vs-threshold curve has plateaus, so a wrong threshold can
        # still give the right test accuracy. That is exactly how the max()
        # tie-break bug survived an accuracy-only gate and had to be caught
        # separately by gate_diag.sh (STATE 25.188.10). Both are now in-script.
        ref_thr = stored_thr.get(name)
        ref_acc = float(stored.loc[name, "Accuracy"]) if name in stored.index else np.nan
        gate = (abs(acc_b - ref_acc) < 5e-4) if ref_acc == ref_acc else None
        gate_thr = (abs(t_b - float(ref_thr)) < 1e-9) if ref_thr is not None else None

        rows.append({
            "Model": name,
            "thr_reuse_A": t_a, "thr_disjoint_B": t_b,
            "sel_acc_A": sel_a, "sel_acc_B": sel_b,
            "test_acc_reuse": acc_a, "test_acc_disjoint": acc_b,
            "delta_reuse_minus_disjoint": acc_a - acc_b,
            "optimism_selA_minus_testA": sel_a - acc_a,
            "test_AUC": auc,
            "stored_threshold": ref_thr, "stored_test_acc": ref_acc,
            "reproduces_stored": gate,
            "reproduces_stored_threshold": gate_thr,
            "is_reuse_control": name != "Stacking_Ensemble",
        })

    df = pd.DataFrame(rows).sort_values("Model")
    out = os.path.join(res_dir, "reuse_control.csv")
    df.to_csv(out, index=False)

    base = df[df["is_reuse_control"]]
    det.log("\n  ===== %s =====" % mode)
    det.log("  base classifiers (n=%d), threshold on the calibration half vs the disjoint half:"
            % len(base))
    det.log("    mean test-accuracy delta (reuse - disjoint) : %+.5f"
            % base["delta_reuse_minus_disjoint"].mean())
    det.log("    median                                      : %+.5f"
            % base["delta_reuse_minus_disjoint"].median())
    det.log("    models hurt by reuse / helped / tied        : %d / %d / %d"
            % ((base["delta_reuse_minus_disjoint"] < 0).sum(),
               (base["delta_reuse_minus_disjoint"] > 0).sum(),
               (base["delta_reuse_minus_disjoint"] == 0).sum()))
    det.log("    mean selection optimism on the reused half  : %+.5f"
            % base["optimism_selA_minus_testA"].mean())
    ok = base["reproduces_stored"].sum()
    det.log("    reproduction gate, accuracy  (disjoint == stored) : %d/%d"
            % (ok, len(base)))
    # The threshold gate covers ALL models, stacking included: the stacking row
    # is not a reuse control but its threshold must still reproduce, and a
    # mismatch there is the same defect.
    thr_col = df["reproduces_stored_threshold"]
    n_thr = int(thr_col.notna().sum())
    ok_thr = int(thr_col.fillna(False).sum())
    det.log("    reproduction gate, threshold (disjoint == stored) : %d/%d"
            % (ok_thr, n_thr))
    if n_thr and ok_thr < n_thr:
        bad = df.loc[thr_col.fillna(False) == False, ["Model", "thr_disjoint_B",
                                                      "stored_threshold"]]
        det.log("    ⛔ THRESHOLD MISMATCH -- the reconstruction is NOT exact. Check the")
        det.log("       tie-break before reading any delta below (STATE 25.188.10):")
        for _, r in bad.iterrows():
            det.log("         %-22s reconstructed %.3f vs stored %s"
                    % (r["Model"], r["thr_disjoint_B"], r["stored_threshold"]))
    if "Stacking_Ensemble" in df["Model"].values:
        r = df[df["Model"] == "Stacking_Ensemble"].iloc[0]
        det.log("  Stacking_Ensemble (NOT a reuse control -- half A never entered "
                "its construction): delta %+.5f"
                % r["delta_reuse_minus_disjoint"])
    det.log("  wrote %s" % out)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="arms_r34_17feat_frozencal")
    ap.add_argument("--mode", choices=["static", "mobile", "both"], default="both")
    args = ap.parse_args()

    arm_dir = os.path.join(AN, args.arm)
    os.environ["DCFM_DATA_BUNDLE"] = os.path.join(arm_dir, "listener", "colab_data")
    modes = ["static", "mobile"] if args.mode == "both" else [args.mode]
    summary = {}
    for m in modes:
        print("\n" + "=" * 70)
        print("=== isolated reuse control -- %s" % m)
        print("=" * 70, flush=True)
        df = run_mode(arm_dir, m)
        b = df[df["is_reuse_control"]]
        summary[m] = {
            "mean_delta": float(b["delta_reuse_minus_disjoint"].mean()),
            "median_delta": float(b["delta_reuse_minus_disjoint"].median()),
            "n_models": int(len(b)),
            "mean_selection_optimism": float(b["optimism_selA_minus_testA"].mean()),
        }
    print("\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
