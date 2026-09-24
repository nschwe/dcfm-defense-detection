#!/usr/bin/env python3
"""windoworder_by_slot_test.py — per-slot accuracy on the HELD-OUT TEST SPLIT.

Why this exists (STATE §25.33, §25.143). `windoworder_by_slot.py` predicts over
ALL rows, train and test — it says so at its own lines 106-109 — so its absolute
accuracies are inflated and only the relative profile is interpretable. R#4.5,
from the same reviewer as R#4.1, is a confirmed defect about scoring on training
data. Offering an in-sample decomposition in the answer to R#4.1 is therefore a
liability, and this script removes it.

What changes, and nothing else:
  1. The pipeline's outer split is reproduced exactly, as
     windoworder_paired_ci.py:95-99 does — GroupShuffleSplit(n_splits=1,
     test_size=cfg.test_size, random_state=cfg.random_state) over `groups` —
     and only test rows are scored.
  2. The decision threshold is the model's OWN tuned threshold from
     `all_thresholds`, applied to predict_proba, so the overall figure here
     reconciles with results.csv. The original used model.predict(), i.e. an
     implicit 0.5, which is not the operating point results.csv reports.

⛔ Writes ONLY under --out-dir. It never touches accuracy_by_slot.csv, the arm
   results, or any bundle. Read-only on everything else.

Usage:
    python windoworder_by_slot_test.py --arms-root analysis/arms_windoworder \\
        --mode static --arm shuffled --out-dir analysis/arms_windoworder/by_slot_test
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GroupShuffleSplit

HERE = os.path.dirname(os.path.abspath(__file__))
# ⛔ The pipeline decides the feature space; the arm alone does not. Default is
#    unchanged (analysis/pipeline, 33 metrics) so every earlier invocation still
#    reproduces. Set WO_PIPELINE=.../cleanfeat/pipeline_17 for the 17-observable
#    arms. Same lesson as STATE §25.112.1 / r23_paired_bootstrap.py's R23_PIPELINE.
PIPELINE = os.environ.get("WO_PIPELINE", os.path.join(HERE, "pipeline"))
if not os.path.isfile(os.path.join(PIPELINE, "defense_detection_v2.py")):
    sys.exit("ABORT: no defense_detection_v2.py under %s" % PIPELINE)
sys.path.insert(0, PIPELINE)


def start_to_slot(t):
    """slot from _StartTime under --formationLeadIn=60: 120->0 220->1 320->2 420->3."""
    return int(round((float(t) - 120.0) / 100.0))


def run_arm(arms_root, arm, mode, model_name):
    import defense_detection_v2 as dd

    arm_root = os.path.join(arms_root, arm, "listener")
    res_dir = os.path.join(arm_root, "results", mode)
    bundle_dir = os.path.join(arm_root, "colab_data")
    bundle = os.path.join(bundle_dir, "wide_%s.csv.gz" % mode)

    pkls = [f for f in os.listdir(res_dir)
            if f.startswith("best_model_") and f.endswith(".pkl")]
    if not pkls:
        sys.exit("ABORT: no best_model_*.pkl under %s" % res_dir)
    md = joblib.load(os.path.join(res_dir, pkls[0]))

    all_models = md.get("all_models") or {}
    if model_name not in all_models:
        sys.exit("ABORT: %r not in %s (has: %s)"
                 % (model_name, pkls[0], sorted(all_models)[:8]))
    model = all_models[model_name]
    thr = (md.get("all_thresholds") or {}).get(model_name, 0.5)

    os.environ["DCFM_DATA_BUNDLE"] = bundle_dir
    sub = "features_static" if mode == "static" else "features_mobile"
    cfg = dd.Config(data_root="./%s/" % sub, results_dir="/tmp/_wo_slot_test", verbose=0)
    cfg.group_split_by_file_source = True
    pipe = dd.DefenseDetector(cfg)

    tall = pipe.load_simulation_data_enhanced()
    X, y, groups = pipe.preprocess_data_enhanced(tall)
    X_eng = pipe.engineer_advanced_features(X)

    g = np.array(groups)
    sp = GroupShuffleSplit(n_splits=1, test_size=cfg.test_size,
                           random_state=cfg.random_state)
    _, test_idx = next(sp.split(X_eng, y, g))

    feats = list(md["feature_names"])
    scaler = md["scaler"]
    sf = list(scaler.feature_names_in_) if hasattr(scaler, "feature_names_in_") else feats
    missing = [f for f in sf if f not in X_eng.columns]
    if missing:
        sys.exit("ABORT: %d features missing for the scaler: %s"
                 % (len(missing), missing[:5]))

    Xs = pd.DataFrame(scaler.transform(X_eng[sf]), columns=sf, index=X_eng.index)
    X_test = Xs.iloc[test_idx][feats]
    y_test = np.asarray(y.iloc[test_idx])

    raw = pd.read_csv(bundle)
    if "_StartTime" not in raw.columns:
        sys.exit("ABORT: bundle has no _StartTime column")
    if len(raw) != len(X_eng):
        sys.exit("ABORT: bundle rows %d != engineered rows %d — positional slot "
                 "alignment is unsafe" % (len(raw), len(X_eng)))
    slots_all = pd.Series(raw["_StartTime"].to_numpy()).map(start_to_slot)
    slots = slots_all.iloc[test_idx].to_numpy()

    try:
        score = model.predict_proba(X_test)[:, 1]
    except Exception:
        score = model.decision_function(X_test)
    pred = (score >= thr).astype(int)

    rows = []
    for s in sorted(set(slots.tolist())):
        m = slots == s
        rows.append({"arm": arm, "mode": mode, "slot": int(s),
                     "window_start": 120 + int(s) * 100,
                     "n": int(m.sum()),
                     "accuracy": float((pred[m] == y_test[m]).mean())})
    overall = float((pred == y_test).mean())
    accs = [r["accuracy"] for r in rows]
    return rows, overall, max(accs) - min(accs), thr, len(set(g[test_idx].tolist()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms-root", default=os.path.join(HERE, "arms_windoworder"))
    ap.add_argument("--mode", choices=["static", "mobile"], required=True)
    # No choices=: the 17-observable re-learn lands in canonical_17 / shuffled_17
    # (run_prop17.sh SUFFIX). The directory either exists or run_arm aborts.
    ap.add_argument("--arm", required=True,
                    help="canonical | shuffled | canonical_17 | shuffled_17")
    ap.add_argument("--model", default="Stacking_Ensemble")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    # ⛔ Gate: never write into an arm's results tree.
    ap_out = os.path.abspath(args.out_dir)
    for bad in ("results/static", "results/mobile"):
        if ap_out.replace("\\", "/").endswith(bad):
            sys.exit("ABORT: --out-dir must not be an arm results directory")
    os.makedirs(ap_out, exist_ok=True)

    rows, overall, spread, thr, n_runs = run_arm(
        args.arms_root, args.arm, args.mode, args.model)

    print("  arm=%s mode=%s model=%s  threshold=%.4f  test runs=%d"
          % (args.arm, args.mode, args.model, thr, n_runs))
    print("  %-6s%7s%11s   window-start" % ("slot", "n", "accuracy"))
    for r in rows:
        print("  %-6d%7d%11.4f   t=%d" % (r["slot"], r["n"], r["accuracy"],
                                          r["window_start"]))
    print("  %-6s%7d%11.4f" % ("all", sum(r["n"] for r in rows), overall))
    print("  spread across slots: %.4f" % spread)

    out = os.path.join(ap_out, "accuracy_by_slot_test_%s_%s.csv" % (args.arm, args.mode))
    df = pd.DataFrame(rows)
    df["overall_test_accuracy"] = overall
    df["spread"] = spread
    df["threshold"] = thr
    df["n_test_runs"] = n_runs
    df.to_csv(out, index=False)
    print("  saved: %s" % out)


if __name__ == "__main__":
    main()
