#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
runtime_guard.py — import this FIRST, before numpy/sklearn/xgboost/lightgbm/catboost,
in every arm runner. It caps the two resources that crashed earlier campaigns.

Two DISTINCT historical failures, both confirmed from logs, both guarded here:

1. OOM (April campaign).  feature_sensitivity_wrapper.log and
   unified_wrapper_extended.log both end in
       <pid> Killed  ... MAX_JOBS=16 ...
   i.e. the Linux OOM killer terminated the process. 16 parallel workers each
   held a copy of the data and its fitted models until RAM was exhausted. The
   team's own fix is visible in the logs:
       [info] bagging_et uses reduced parallelism: n_jobs=8 ... to prevent OOM
   GUARD: MAX_JOBS defaults to 2 (the value the published Colab notebook uses),
   overridable upward only by an explicit environment variable.

2. SIGSEGV (2026-08-08 run of the now-discarded run_full_v4.py).
       TerminatedWorkerError ... exit codes of the workers are {SIGSEGV(-11)}
   StackingClassifier(cv=5, n_jobs=MAX_JOBS) over 10 base estimators, each itself
   built with n_jobs=MAX_JOBS, nested to 12x12 = 144 threads on 24 cores. This is
   a THREAD-count fault, independent of #1.
   GUARD: pin every BLAS/OpenMP pool to a single thread so the nesting collapses.
   Outer process parallelism (loky) is unaffected, so throughput barely changes:
   the stacking layer already fits its base models 2-at-a-time via MAX_JOBS.

Both caps are set with setdefault, so an operator who has measured headroom can
still raise them from the shell (e.g. MAX_JOBS=4 python run_arm.py ...). The
current machine has 48 GiB free and 24 GiB swap, so MAX_JOBS=2 is very safe; the
value is left low deliberately because the arms may be launched back-to-back.
"""
import os

# --- thread caps: MUST precede numpy/BLAS/OpenMP import (they read pool size once) ---
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

# --- process-parallelism cap: read by defense_detection_v2.py as MAX_JOBS ---
os.environ.setdefault("MAX_JOBS", "2")

# --- silence the sklearn/joblib UserWarning flood seen in the 8/8 run ---
os.environ.setdefault("PYTHONWARNINGS", "ignore")


def report():
    """One-line confirmation, printed by each runner so the log records the caps."""
    return (f"[runtime_guard] MAX_JOBS={os.environ['MAX_JOBS']} "
            f"OMP_NUM_THREADS={os.environ['OMP_NUM_THREADS']} "
            f"(OOM + SIGSEGV guards active)")


if __name__ == "__main__":
    print(report())
