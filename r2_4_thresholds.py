#!/usr/bin/env python
"""
R#2.4 — redo the whole threshold / cost analysis on the REPORTED campaign
(arms_r34_17feat, 17 observables), instead of arms_c10k_stackcal.

Read-only. Writes nothing except an optional CSV under --out.
Nothing is retrained: the saved stacking model is loaded and applied.

    python r2_4_thresholds.py static
    python r2_4_thresholds.py mobile
    python r2_4_thresholds.py both            # runs each in turn

Gates (the script exits non-zero rather than printing a number it cannot back):
  GATE 1  the reproduced TEST split must give results.csv's Accuracy, AUC and
          all four confusion-matrix cells exactly.
  GATE 2  the engineered space must be 77 features from 18 columns, 8,000 test
          rows, as recorded in run17.log.
"""
import os, sys
import numpy as np
import pandas as pd
import joblib

ARM = os.environ.get("R24_ARM",
                     os.path.expanduser("~/ns3/nv347/ns-3.47/analysis/arms_r34_17feat"))
PIPE = os.environ.get("R24_PIPE", os.path.join(ARM, "pipeline_17"))
COSTS = [0.25, 0.5, 1, 2, 4, 10]
B = 2000
GRID = np.round(np.arange(0.01, 1.00, 0.01), 2)

os.environ["DCFM_DATA_BUNDLE"] = os.path.join(ARM, "listener", "colab_data")
sys.path.insert(0, PIPE)
from defense_detection_v2 import DefenseDetector, Config              # noqa: E402
from sklearn.model_selection import GroupShuffleSplit                 # noqa: E402
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score  # noqa: E402


def acc_at(p, y, t):
    return accuracy_score(y, (p >= t).astype(int))


def cost_at(p, y, t, C):
    yh = (p >= t).astype(int)
    fp = int(((yh == 1) & (y == 0)).sum())
    fn = int(((yh == 0) & (y == 1)).sum())
    return (fp + C * fn) / len(y)


def run(mode):
    print("\n" + "=" * 72)
    print("  MODE: %s" % mode)
    print("=" * 72)

    cfg = Config(
        data_root="./features_%s/" % mode,
        results_dir="/tmp/_r2_4_unused",
        use_aggressive_augmentation=False,   # --no-augmentation
        split_validation=True,               # --split-validation
        add_linearsvc=True,                  # --add-linearsvc
        calibrate_stacking=True,             # --calibrate-stacking
        verbose=0,
    )
    det = DefenseDetector(cfg)
    tall = det.load_simulation_data_enhanced()
    X, y, groups = det.preprocess_data_enhanced(tall)
    X_eng = det.engineer_advanced_features(X)
    g = np.array(groups)

    if X.shape[1] != 18 or X_eng.shape[1] != 77:
        sys.exit("GATE 2 FAILED: shape %s / %d engineered, expected (N,18) / 77"
                 % (X.shape, X_eng.shape[1]))

    # --- split exactly as run_full_pipeline does ---------------------------
    sp = GroupShuffleSplit(n_splits=1, test_size=cfg.test_size,
                           random_state=cfg.random_state)
    tv_idx, te_idx = next(sp.split(X_eng, y, g))
    X_te, y_te = X_eng.iloc[te_idx], y.iloc[te_idx].to_numpy()
    runs_te = g[te_idx]

    X_tv, y_tv, g_tv = X_eng.iloc[tv_idx], y.iloc[tv_idx], g[tv_idx]
    sp2 = GroupShuffleSplit(n_splits=1, test_size=0.25,
                            random_state=cfg.random_state)
    tr_idx, va_idx = next(sp2.split(X_tv, y_tv, g_tv))
    X_va, y_va, g_va = X_tv.iloc[va_idx], y_tv.iloc[va_idx].to_numpy(), g_tv[va_idx]

    if len(y_te) != 8000:
        sys.exit("GATE 2 FAILED: %d test rows, expected 8000" % len(y_te))

    # --- the --split-validation halves, grouped by file_source -------------
    sp3 = GroupShuffleSplit(n_splits=1, test_size=0.5,
                            random_state=cfg.random_state)
    cal_idx, thr_idx = next(sp3.split(X_va, y_va, g_va))
    X_thr, y_thr = X_va.iloc[thr_idx], y_va[thr_idx]
    print("  validation halves: calibration %d rows, threshold %d rows"
          % (len(cal_idx), len(thr_idx)))

    # --- saved model --------------------------------------------------------
    res = os.path.join(ARM, "listener", "results", mode)
    md = joblib.load(os.path.join(res, "best_model_Stacking_Ensemble.pkl"))
    scaler, feat = md["scaler"], md["feature_names"]

    def proba(Xd):
        S = pd.DataFrame(scaler.transform(Xd), columns=X_eng.columns, index=Xd.index)
        return md["model"].predict_proba(S[feat])[:, 1]

    p_te, p_thr = proba(X_te), proba(X_thr)

    # --- GATE 1 -------------------------------------------------------------
    ref = pd.read_csv(os.path.join(res, "results.csv"))
    row = ref[ref["Model"] == "Stacking_Ensemble"].iloc[0]
    a = acc_at(p_te, y_te, md["threshold"])
    u = roc_auc_score(y_te, p_te)
    cm = confusion_matrix(y_te, (p_te >= md["threshold"]).astype(int)).ravel().tolist()
    ok = (abs(a - row["Accuracy"]) < 1e-9 and abs(u - row["AUC"]) < 1e-9
          and cm == [int(row[k]) for k in ("TN", "FP", "FN", "TP")])
    print("  GATE 1  acc %.6f vs %.6f | auc %.6f vs %.6f | cm %s vs %s -> %s"
          % (a, row["Accuracy"], u, row["AUC"], cm,
             [int(row[k]) for k in ("TN", "FP", "FN", "TP")],
             "PASS" if ok else "FAIL"))
    if not ok:
        sys.exit("GATE 1 FAILED for %s" % mode)

    # --- (1) three thresholds ----------------------------------------------
    t_tuned = GRID[np.argmax([acc_at(p_thr, y_thr, t) for t in GRID])]
    t_post = GRID[np.argmax([acc_at(p_te, y_te, t) for t in GRID])]
    a_fix, a_tun, a_post = (acc_at(p_te, y_te, 0.5), acc_at(p_te, y_te, t_tuned),
                            acc_at(p_te, y_te, t_post))
    print("\n  --- Table 1: threshold variants (test accuracy) ---")
    print("    fixed                 0.500  %.4f   vs post-hoc %+.4f" % (a_fix, a_fix - a_post))
    print("    selected on tuning half %.3f  %.4f   vs post-hoc %+.4f" % (t_tuned, a_tun, a_tun - a_post))
    print("    post-hoc test optimum  %.3f  %.4f   ---" % (t_post, a_post))
    print("    stored model threshold %.3f  (from the pickle)" % md["threshold"])

    # --- flatness -----------------------------------------------------------
    accs = np.array([acc_at(p_te, y_te, t) for t in GRID])
    within = GRID[accs >= accs.max() - 0.001]
    band = (GRID >= 0.30) & (GRID <= 0.70)
    print("    within 0.1pp of optimum: [%.3f, %.3f]" % (within.min(), within.max()))
    print("    spread over 0.30-0.70  : %.2f percentage points"
          % ((accs[band].max() - accs[band].min()) * 100))

    # --- paired cluster bootstrap on the retuning delta ---------------------
    rng = np.random.RandomState(cfg.random_state)
    uniq = np.unique(runs_te)
    idx_by_run = {r: np.where(runs_te == r)[0] for r in uniq}
    d = []
    for _ in range(B):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        ii = np.concatenate([idx_by_run[r] for r in pick])
        d.append(acc_at(p_te[ii], y_te[ii], t_tuned) - acc_at(p_te[ii], y_te[ii], 0.5))
    d = np.array(d)
    print("\n  --- retuning cost, %d cluster bootstrap resamples of %d runs ---"
          % (B, len(uniq)))
    print("    point delta            : %+.2f percentage points" % ((a_tun - a_fix) * 100))
    print("    one-sided 95%% lower    : %+.2f percentage points" % (np.percentile(d, 5) * 100))
    print("    resamples where lower  : %.0f%%" % ((d < 0).mean() * 100))
    print("    worst resample         : %+.2f percentage points" % (d.min() * 100))

    # --- (3) cost-sensitive sweep ------------------------------------------
    print("\n  --- Table 2: cost-sensitive rule (threshold picked on tuning half) ---")
    print("    %5s %9s %20s %10s %10s %s" %
          ("C", "threshold", "test cost", "acc there", "cost@op", "saving"))
    out = []
    for C in COSTS:
        tC = GRID[np.argmin([cost_at(p_thr, y_thr, t, C) for t in GRID])]
        cT = cost_at(p_te, y_te, tC, C)
        t_op = md["threshold"]          # the operating point the paper reports
        c5 = cost_at(p_te, y_te, t_op, C)
        aT = acc_at(p_te, y_te, tC)
        sc, ss = [], []
        for _ in range(B):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            ii = np.concatenate([idx_by_run[r] for r in pick])
            sc.append(cost_at(p_te[ii], y_te[ii], tC, C))
            ss.append(cost_at(p_te[ii], y_te[ii], t_op, C) - cost_at(p_te[ii], y_te[ii], tC, C))
        print("    %5s %9.2f  %.4f +/- %.4f %10.4f %10.4f  %+.4f +/- %.4f"
              % (C, tC, cT, np.std(sc), aT, c5, c5 - cT, np.std(ss)))
        out.append(dict(mode=mode, C=C, threshold=tC, test_cost=cT,
                        test_cost_sd=np.std(sc), acc_there=aT, cost_at_op=c5,
                        saving=c5 - cT, saving_sd=np.std(ss)))
    return out


if __name__ == "__main__":
    modes = sys.argv[1:] or ["static"]
    if modes == ["both"]:
        modes = ["static", "mobile"]
    rows = []
    for m in modes:
        rows += run(m)
    print("\nAll gates passed. Numbers above are on %s" % os.path.basename(ARM))
    print("  pipeline: %s" % PIPE)
