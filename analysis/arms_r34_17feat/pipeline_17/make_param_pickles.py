#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
make_param_pickles.py
============================================================================
The cross-domain analysis scripts (k_sweep, confusion_matrices,
feature_eng_ablation, flowthroughputstd, feature_importance_sensitivity,
threshold_decomposition, dj_ablation_*) load
results/hp_search_extended/<cfg>/best_models.pkl ONLY to call .get_params()
on the tuned estimators and rebuild fresh classifiers. They never use the
fitted weights of that pickle.

Each real pickle is ~1.5 GB (the fitted ensembles). This script writes a
PARAMS-ONLY replacement: an sklearn-cloned (unfitted) copy of every
estimator, which preserves get_params() exactly but is a few KB. Shipping
these lets the entire Tier-1 (no-retrain) cross-domain pipeline run on Colab
without the 18.7 GB of fitted models, and WITHOUT editing any analysis script.

In RETRAIN mode the notebook regenerates the real fitted pickles via
unified_hp_search_v2.py, overwriting these stubs.

Output: colab_data/cache/hp_search_extended/<cfg>/best_models.pkl
        colab_data/cache/hp_search_focused/<cfg>/best_models.pkl  (already tiny; copied)
============================================================================
"""
import os
import shutil
import joblib
from sklearn.base import clone


def stub_pickle(src: str, dst: str) -> None:
    d = joblib.load(src)
    if not isinstance(d, dict):
        raise TypeError(f"{src}: expected dict, got {type(d)}")
    stub = {}
    for name, est in d.items():
        try:
            stub[name] = clone(est)            # unfitted, same hyperparameters
        except Exception as e:
            # Non-estimator entry (e.g. a plain params dict) — keep verbatim.
            print(f"    [keep-as-is] {name}: clone failed ({e})")
            stub[name] = est
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    joblib.dump(stub, dst)
    print(f"    {src}  ->  {dst}  ({os.path.getsize(dst)/1024:.1f} KB, "
          f"from {os.path.getsize(src)/1e6:.0f} MB)")


def main():
    extended = "results/hp_search_extended"
    focused = "results/hp_search_focused"
    out_ext = "colab_data/cache/hp_search_extended"
    out_foc = "colab_data/cache/hp_search_focused"

    for cfg in ("static", "mobile"):
        print(f"[extended] {cfg}")
        stub_pickle(os.path.join(extended, cfg, "best_models.pkl"),
                    os.path.join(out_ext, cfg, "best_models.pkl"))
        # full_log.json carries the selected-feature list used by generate_scaler_v2
        src_log = os.path.join(extended, cfg, "full_log.json")
        if os.path.exists(src_log):
            os.makedirs(os.path.join(out_ext, cfg), exist_ok=True)
            shutil.copy2(src_log, os.path.join(out_ext, cfg, "full_log.json"))
            print(f"    copied full_log.json")

    # focused pickles are SVM-only and already tiny — copy verbatim (needed by
    # rebuild_stacking_v2 in the RETRAIN path).
    for cfg in ("static", "mobile"):
        src = os.path.join(focused, cfg, "best_models.pkl")
        if os.path.exists(src):
            dst = os.path.join(out_foc, cfg, "best_models.pkl")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            print(f"[focused]  {cfg}  copied ({os.path.getsize(src)/1e6:.1f} MB)")

    print("done.")


if __name__ == "__main__":
    main()
