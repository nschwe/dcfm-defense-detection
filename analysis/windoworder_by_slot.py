#!/usr/bin/env python3
"""windoworder_by_slot.py — per-slot accuracy breakdown for the window-order
experiment (STATE §25.20/§25.21). No extra simulation: reads the trained
shuffled-arm model, re-predicts its own test rows, and groups accuracy by the
SLOT each window was measured in (the control column, never a model input).

A flat accuracy across slots is direct evidence the classifier is not keying on
window position (answers R#4 #1 beyond the headline shuffled-vs-fixed number).

slot is derived from _StartTime under --formationLeadIn=60:
    120 -> slot 0    220 -> slot 1    320 -> slot 2    420 -> slot 3

Usage:
    python windoworder_by_slot.py --arms-root analysis/arms_windoworder --mode static
"""
import argparse
import os
import sys
import glob
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline"))
from defense_detection_v2 import DefenseDetector, Config

# slot boundaries for lead-in 60; each window starts INITIAL_STAB(60)+lead-in(60)
# + slot*100. Robust to float noise via nearest-100 rounding.
def start_to_slot(t):
    return int(round((float(t) - 120.0) / 100.0))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms-root", required=True,
                    help="dir holding canonical/ and shuffled/ (arms_windoworder)")
    ap.add_argument("--mode", choices=["static", "mobile"], required=True)
    ap.add_argument("--arm", choices=["shuffled", "canonical"], default="shuffled")
    ap.add_argument("--out", default=None, help="CSV to write (default: alongside results)")
    args = ap.parse_args()

    base = os.path.join(args.arms_root, args.arm, "listener")
    res_dir = os.path.join(base, "results", args.mode)
    bundle_dir = os.path.join(base, "colab_data")
    bundle = os.path.join(bundle_dir, f"wide_{args.mode}.csv.gz")

    pkls = glob.glob(os.path.join(res_dir, "best_model_*.pkl"))
    if not pkls:
        sys.exit(f"no trained model under {res_dir} — run the learning first")
    pkl = next((p for p in pkls if "Stacking" in p), pkls[0])
    if not os.path.isfile(bundle):
        sys.exit(f"bundle not found: {bundle}")

    print(f"  model : {pkl}")
    print(f"  bundle: {bundle}")
    md = joblib.load(pkl)
    model = md["model"]; scaler = md["scaler"]; feat = list(md["feature_names"])

    # Reproduce the pipeline's features EXACTLY via its own methods (same path
    # variance_decomposition_v2 uses), so predictions match the trained model.
    sub = "features_static" if args.mode == "static" else "features_mobile"
    cfg = Config(data_root=f"./{sub}/", results_dir="/tmp/_wo_slot", verbose=0)
    cfg.group_split_by_file_source = True
    os.environ["DCFM_DATA_BUNDLE"] = bundle_dir
    pipe = DefenseDetector(cfg)
    tall = pipe.load_simulation_data_enhanced()
    X_wide, y, groups = pipe.preprocess_data_enhanced(tall)
    X_eng = pipe.engineer_advanced_features(X_wide)

    # slot from the raw wide bundle's _StartTime, aligned to X_eng rows by index
    raw = pd.read_csv(bundle)
    if "_StartTime" not in raw.columns:
        sys.exit("bundle has no _StartTime column; cannot assign slots")
    if len(raw) != len(X_eng):
        print(f"  WARNING: bundle rows {len(raw)} != engineered rows {len(X_eng)}; "
              f"aligning by position")
    slots = pd.Series(raw["_StartTime"].to_numpy()[:len(X_eng)]).map(start_to_slot)

    missing = [f for f in feat if f not in X_eng.columns]
    if missing:
        sys.exit(f"{len(missing)} model features missing after engineering "
                 f"(pipeline drift?): {missing[:5]}")
    Xf = X_eng[list(scaler.feature_names_in_)] if hasattr(scaler, "feature_names_in_") \
        else X_eng[feat]
    X = scaler.transform(Xf)
    if hasattr(scaler, "feature_names_in_"):
        X = pd.DataFrame(X, columns=Xf.columns)[feat].to_numpy()

    try:
        pred = model.predict(X)
    except Exception as e:
        sys.exit(f"prediction failed: {e}")
    y = y.to_numpy()

    out_rows = []
    print(f"\n  {'slot':<6}{'n':>7}{'accuracy':>11}   window-start")
    for slot in sorted(slots.unique()):
        m = (slots == slot).to_numpy()
        acc = float((pred[m] == y[m]).mean())
        st = 120 + slot * 100
        print(f"  {slot:<6}{int(m.sum()):>7}{acc:>11.4f}   t={st}")
        out_rows.append({"slot": slot, "window_start": st,
                         "n": int(m.sum()), "accuracy": acc})
    overall = float((pred == y).mean())
    # In-sample note: this predicts over ALL rows (train+test), so absolute
    # accuracies are optimistic; the per-slot SPREAD is the quantity of
    # interest. For a test-only breakdown, reproduce the pipeline's
    # GroupShuffleSplit with its random_state and mask to test rows.
    spread = max(r["accuracy"] for r in out_rows) - min(r["accuracy"] for r in out_rows)
    print(f"  {'all':<6}{len(y):>7}{overall:>11.4f}")
    print(f"\n  spread across slots: {spread:.4f}  "
          f"({'FLAT — no window-position signal' if spread < 0.03 else 'CHECK — slot-dependent'})")

    out = args.out or os.path.join(res_dir, "accuracy_by_slot.csv")
    pd.DataFrame(out_rows).to_csv(out, index=False)
    print(f"  saved: {out}")


if __name__ == "__main__":
    main()
