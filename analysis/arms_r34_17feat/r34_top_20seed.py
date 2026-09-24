#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
r34_top_20seed.py -- 20-seed re-evaluation of NAMED 3-feature subsets.

WHY THIS EXISTS
    A reported comparison put 0.8654 +- 0.0056 for the selected subset
    against 0.8088 +- 0.0051 for "the runner-up identified by the
    enumeration". Two problems:

    1. NO ARTEFACT. That pair has no artefact behind it, while every other
       figure of the same comparison is re-derivable from
       r34_subsets/subsets_k1_2_3_4_5_6_7_8_s3.csv.

    2. WRONG SUBSET. The runner-up as quoted is
       {AvgAdvLinks, AverageMprCount, AvgTxPacketsPerFlow}. In the enumeration
       that subset is rank 14 of 680 (cross_mean 0.80706 at 3 seeds). The
       actual rank 2 is {AvgAdvLinks, AverageMprCount, AvgTxPacketSize} at
       0.83607. 0.8088 at 20 seeds is within noise of 0.80706 and 0.027 below
       0.83607, so the 20-seed run evaluated the rank-14 subset.

    This script re-measures all three named subsets under exactly the
    enumeration's protocol and writes the artefact that was missing.

PROTOCOL
    ksw.evaluate_seed, unchanged, identical to r34_subsets.py: scaler and
    CatBoost fitted on the source TRAIN split, threshold from source
    VALIDATION, the fitted model applied to the target. No calibration
    anywhere in this path, so the isotonic defect of the main campaign
    cannot reach these numbers.

SANITY CHECK ON THE OUTPUT
    The selected subset's 20-seed cross_mean must land near 0.8681 -- the LIVE
    K=3 cross-domain score of this same arm's K sweep
    (listener/results/k_sweep_universal4_v2/optimal_k.json, score_at_optimal
    0.86809 = (0.8730 + 0.8632)/2, 25/8 22:00). NOT 0.8670: that is the
    superseded 21/8 sweep, the same table that carries the forbidden
    0.8720 / 0.8619 pair.

COST
    3 subsets x 20 seeds x 2 directions = 120 fits. Minutes.

RUN (from this directory)
    PYTHONIOENCODING=utf-8 FIT_THREADS=8 \
      ~/miniconda3/envs/manet/bin/python -u r34_top_20seed.py \
      2>&1 | tee r34_top_20seed_$(date +%Y%m%d_%H%M%S).log
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

HP_DIR = os.environ.get(
    "DCFM_HP_RESULTS",
    os.path.expanduser("~/ns3/Final_Project_NS3-master/strict_observable_v2/"
                       "colab_data/results_cache/hp_search_extended"))

N_SEEDS = int(os.environ.get("N_SEEDS", "20"))
FIT_THREADS = int(os.environ.get("FIT_THREADS", "8"))

# name -> (features, 3-seed cross_mean from the enumeration, 3-seed rank)
SUBSETS = {
    "selected": (
        ["TcMessageRate", "AverageAdvertisedLinksPerTCMessage",
         "AverageMprCount"], 0.8618833333333333, 1),
    "runner_up_rank2": (
        ["AverageAdvertisedLinksPerTCMessage", "AverageMprCount",
         "AvgTxPacketSize"], 0.8360749999999999, 2),
    "rank14_previously_quoted": (
        ["AverageAdvertisedLinksPerTCMessage", "AverageMprCount",
         "AvgTxPacketsPerFlow"], 0.8070583333333332, 14),
}


def _imports():
    """Import the pipeline modules with the isolated tree first on the path."""
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(HERE / "pipeline_17"))
    os.environ["DCFM_DATA_BUNDLE"] = str(HERE / "listener" / "colab_data")
    import feature_importance_sensitivity_v2 as fis   # noqa: E402
    import k_sweep_universal4_v2 as ksw               # noqa: E402
    return fis, ksw


def main() -> None:
    fis, ksw = _imports()

    # evaluate_seed hardcodes n_jobs=16; cap it exactly as r34_subsets.py does.
    orig_build = ksw.build_catboost
    ksw.build_catboost = (
        lambda best_params, seed, n_jobs=16:
        orig_build(best_params, seed=seed, n_jobs=FIT_THREADS))

    Xs, ys, gs = fis.to_wide(fis.load_dataset("./features_static"))
    Xm, ym, gm = fis.to_wide(fis.load_dataset("./features_mobile"))
    bp = ksw.load_best_params(HP_DIR)

    t0 = time.time()
    out = {
        "n_seeds": N_SEEDS,
        "protocol": "ksw.evaluate_seed, identical to r34_subsets.py",
        "hp_dir": HP_DIR,
        "bundle": str(HERE / "listener" / "colab_data"),
        "subsets": {},
    }

    for name, (feats, screen_cross, screen_rank) in SUBSETS.items():
        s2m, m2s, in_s, in_m, per_seed_cross = [], [], [], [], []
        for seed in range(N_SEEDS):
            rs = ksw.evaluate_seed(Xs, ys, gs, Xm, ym, feats, bp["static"], seed)
            rm = ksw.evaluate_seed(Xm, ym, gm, Xs, ys, feats, bp["mobile"], seed)
            s2m.append(rs["acc_cross"]); in_s.append(rs["acc_in"])
            m2s.append(rm["acc_cross"]); in_m.append(rm["acc_in"])
            per_seed_cross.append((rs["acc_cross"] + rm["acc_cross"]) / 2.0)

        rec = {
            "features": feats,
            "screen_3seed_cross_mean": screen_cross,
            "screen_3seed_rank_of_680": screen_rank,
            # cross_mean defined exactly as in r34_subsets.py:
            #   (mean(s2m) + mean(m2s)) / 2
            "cross_mean": float((np.mean(s2m) + np.mean(m2s)) / 2.0),
            # sd ACROSS SEEDS of the per-seed (s2m+m2s)/2, sample sd (ddof=1)
            "cross_sd_across_seeds": float(np.std(per_seed_cross, ddof=1)),
            "s2m_mean": float(np.mean(s2m)),
            "s2m_sd": float(np.std(s2m, ddof=1)),
            "m2s_mean": float(np.mean(m2s)),
            "m2s_sd": float(np.std(m2s, ddof=1)),
            "in_static_mean": float(np.mean(in_s)),
            "in_mobile_mean": float(np.mean(in_m)),
            "per_seed_cross": [float(v) for v in per_seed_cross],
            "per_seed_s2m": [float(v) for v in s2m],
            "per_seed_m2s": [float(v) for v in m2s],
        }
        out["subsets"][name] = rec
        print("%-26s cross=%.4f sd=%.4f  (3-seed screen %.4f, rank %d)"
              % (name, rec["cross_mean"], rec["cross_sd_across_seeds"],
                 screen_cross, screen_rank), flush=True)

    sel = out["subsets"]["selected"]
    r2 = out["subsets"]["runner_up_rank2"]
    r14 = out["subsets"]["rank14_previously_quoted"]

    # Paired over shared seeds -- the correct comparison, per r3_4's OURS note.
    d = np.array(sel["per_seed_cross"]) - np.array(r2["per_seed_cross"])
    out["margin_selected_minus_rank2"] = float(sel["cross_mean"] - r2["cross_mean"])
    out["paired_diff_mean"] = float(np.mean(d))
    out["paired_diff_sd"] = float(np.std(d, ddof=1))
    out["margin_selected_minus_rank14"] = float(
        sel["cross_mean"] - r14["cross_mean"])
    out["elapsed_min"] = (time.time() - t0) / 60.0

    dest = HERE / "r34_subsets" / "top_subsets_20seed.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=2)

    print("", flush=True)
    print("margin selected - rank2  : %.4f  (paired mean %.4f, sd %.4f)"
          % (out["margin_selected_minus_rank2"], out["paired_diff_mean"],
             out["paired_diff_sd"]), flush=True)
    print("margin selected - rank14 : %.4f"
          % out["margin_selected_minus_rank14"], flush=True)
    print("wrote %s in %.1f min" % (dest, out["elapsed_min"]), flush=True)


if __name__ == "__main__":
    main()
