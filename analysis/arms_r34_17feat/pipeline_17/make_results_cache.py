#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
make_results_cache.py
============================================================================
Mirror the small, git-shippable outputs of results/ into
colab_data/results_cache/ so the notebook's DEFAULT (no-retrain) mode can
display the real paper tables/figures instantly, and so scripts that read a
prior stage's output find it on Colab.

What is shipped:
  * every results/ file <= MAX_MB (CSV/PNG/TXT/JSON/NPZ + the small useful
    pickles: scaler_v2.pkl, test_predictions_cache.pkl, importance_cache/*,
    hp_search_focused/*/best_models.pkl, best_model_Stacking_Ensemble.pkl)
What is NOT shipped (regenerated only in RETRAIN mode):
  * the ~1.4 GB fitted ensembles (hp_search_{extended,final}/*/best_models.pkl)
    and the edge-extension model pickles -> excluded by size.
  * redundant backup / alternate-seed directories.

After mirroring, the huge extended best_models.pkl (excluded above) is
replaced by the params-only stub from make_param_pickles.py so the
cross-domain scripts' load_best_params() still works (get_params only).

On Colab the notebook copies results_cache/* into ./results/.
============================================================================
"""
import os
import shutil
import joblib
from sklearn.base import clone

MAX_MB = 20.0
SRC = "results"
DST = "colab_data/results_cache"

# Directories we do not need to reproduce the paper (backups / alt-seed dups).
EXCLUDE_DIR_SUBSTR = (
    ".backup", "_10seeds", "separability_100k", "separability.backup",
    "k_sweep_low_k", "hp_search_extended.backup", "hp_search_narrow",
    "hp_search_edge_extension", "catboost_info",
)


def excluded_dir(relpath: str) -> bool:
    return any(s in relpath for s in EXCLUDE_DIR_SUBSTR)


def main():
    if os.path.exists(DST):
        shutil.rmtree(DST)
    os.makedirs(DST, exist_ok=True)

    kept, skipped_big, skipped_dir, total_bytes = 0, 0, 0, 0
    for root, _dirs, files in os.walk(SRC):
        rel = os.path.relpath(root, SRC)
        if rel != "." and excluded_dir(rel):
            skipped_dir += 1
            continue
        for fn in files:
            sp = os.path.join(root, fn)
            relf = os.path.relpath(sp, SRC)
            if excluded_dir(relf):
                continue
            try:
                sz = os.path.getsize(sp)
            except OSError:
                continue
            if sz > MAX_MB * 1e6:
                skipped_big += 1
                continue
            dp = os.path.join(DST, relf)
            os.makedirs(os.path.dirname(dp), exist_ok=True)
            shutil.copy2(sp, dp)
            kept += 1
            total_bytes += sz

    # Overlay params-only stubs for the excluded 1.4 GB extended ensembles.
    for cfg in ("static", "mobile"):
        real = os.path.join(SRC, "hp_search_extended", cfg, "best_models.pkl")
        if not os.path.exists(real):
            continue
        d = joblib.load(real)
        stub = {}
        for name, est in d.items():
            try:
                stub[name] = clone(est)
            except Exception:
                stub[name] = est
        dp = os.path.join(DST, "hp_search_extended", cfg, "best_models.pkl")
        os.makedirs(os.path.dirname(dp), exist_ok=True)
        joblib.dump(stub, dp)
        print(f"  [stub] hp_search_extended/{cfg}/best_models.pkl "
              f"({os.path.getsize(dp)/1024:.1f} KB)")

    # Slim stub of the 2.4 GB best_model_Stacking_Ensemble.pkl: variance_decomposition_v2
    # only reads its "feature_names" + fitted "scaler" (both small).
    for cfg in ("static", "mobile"):
        real = os.path.join(SRC, cfg, "best_model_Stacking_Ensemble.pkl")
        if not os.path.exists(real):
            continue
        md = joblib.load(real)
        slim = {"feature_names": md.get("feature_names"), "scaler": md.get("scaler")}
        dp = os.path.join(DST, cfg, "best_model_Stacking_Ensemble.pkl")
        os.makedirs(os.path.dirname(dp), exist_ok=True)
        joblib.dump(slim, dp)
        print(f"  [slim] {cfg}/best_model_Stacking_Ensemble.pkl "
              f"({os.path.getsize(dp)/1024:.1f} KB, from {os.path.getsize(real)/1e9:.1f} GB)")

    print(f"\nMirrored {kept} files ({total_bytes/1e6:.1f} MB) -> {DST}")
    print(f"Skipped {skipped_big} oversized (> {MAX_MB:.0f} MB) and "
          f"{skipped_dir} backup/alt dirs.")


if __name__ == "__main__":
    main()
