#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inference_cost_report.py -- model size, inference latency, memory.

R#3.19: "No training time, inference latency, or memory usage is reported
anywhere. For a stacking ensemble evaluated on potentially resource-constrained
MANET nodes, this information seems relevant to practical applicability."

Training time is not measured here: the pipeline prints "Total Runtime" into
its own stage-1 log, so take it from the log that says it saved into this arm
and this mode. This script supplies the two figures no log carries -- the size
of the deployed model and the cost of scoring one measurement window.

WHAT IS TIMED, AND WHAT IS NOT
------------------------------
Latency is measured on a synthetic matrix drawn in the post-scaler space, with
the same column count as the fitted model. For tree ensembles the cost of
`predict_proba` is governed by the number of estimators, tree depth and batch
shape, not by the values, so this is a faithful timing measurement and it
avoids reloading and re-engineering the whole bundle -- which matters here,
because this script is meant to be safe to run beside other work.

⛔ It therefore times INFERENCE ONLY. On a real node the dominant cost is
likely feature extraction from captured frames, which this does not measure
and which the paper should not silently omit. Pass --with-feature-eng to also
time the pipeline's feature engineering on the real bundle; that path is much
heavier and should only be used on an idle machine.

WHAT CLOSES THE COMMENT
-----------------------
    training   not here -- the pipeline logs "Total Runtime" itself
    latency    single window and batch, at full width and, with --threads 1,
               on one pinned core: the bound for a slower platform
    memory     --all-models gives every stored model's serialised size beside
               its accuracy, so a node's budget maps to a measured cost

GATES (hard aborts, all printed)
-------------------------------
    SPACE       the pipeline used for feature engineering has METRICS == 17
    CALIBRATION that pipeline contains FrozenEstimator
    VANTAGE     --arm-root is a "listener" directory, and it is the only arm
    BUNDLE      wide_<mode>.csv.gz has 27 columns
    MODEL       the requested model has a row in the arm's results.csv

The 20/8 run had none of these and was made against arms_c10k_stackcal with the
33-metric generator; its numbers describe that arm and must not be quoted for
the reported campaign.

SAFETY
------
Read-only. Loads one pkl, allocates its own arrays, writes one CSV to
--out-dir, which defaults to a per-arm subdirectory so that one arm's numbers
can never overwrite another's. Touches no campaign directory.
"""

import argparse
import glob
import gzip
import io
import os
import sys

# ⛔ Thread limits must be in the environment BEFORE numpy/BLAS is imported, so
# --threads is read from argv here rather than after argparse. Pinning happens
# later, in main(), where it can be printed.
_THREADS = None
for _i, _a in enumerate(sys.argv):
    if _a == "--threads" and _i + 1 < len(sys.argv):
        _THREADS = sys.argv[_i + 1]
    elif _a.startswith("--threads="):
        _THREADS = _a.split("=", 1)[1]
if _THREADS:
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[_v] = _THREADS
import resource
import time

import numpy as np
import pandas as pd
import joblib

LOAD_LIMIT = 4.0
BATCHES = [1, 10, 100, 1000, 8000]


def rss_mb():
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kilobytes
    return ru / 1024.0


def stage1_log(arm_root, mode):
    """The arm's own stage-1 log: training time and the feature counts.

    A log counts only if it says it saved into THIS arm and THIS mode, so the
    numbers cannot drift onto another campaign's run.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    arm_root = os.path.abspath(arm_root)
    want = os.path.join(arm_root, "results", mode).replace(os.sep, "/")
    found = []
    for root, _dirs, files in os.walk(here):
        for fn in files:
            if not fn.endswith(".log"):
                continue
            path = os.path.join(root, fn)
            try:
                with io.open(path, encoding="utf-8", errors="replace") as fh:
                    text = fh.read()
            except OSError:
                continue
            if "Results saved to: " + want not in text:
                continue
            rec = {"log": path, "runtime": None, "created": None,
                   "selected": None}
            for line in text.splitlines():
                line = line.strip()
                if line.startswith("Total Runtime:"):
                    rec["runtime"] = line.split(":", 1)[1].strip()
                elif line.startswith("Created ") and "features" in line:
                    try:
                        rec["created"] = int(line.split()[1])
                    except ValueError:
                        pass
                elif line.startswith("Selected ") and "features" in line:
                    try:
                        rec["selected"] = int(line.split()[1])
                    except ValueError:
                        pass
            found.append(rec)
    return found


def gate(arm_root, mode, model):
    """Refuse to measure anything but the reported campaign. Every check prints."""
    here = os.path.dirname(os.path.abspath(__file__))
    pipeline = os.environ.get("R23_PIPELINE",
                              os.path.join(here, "frozencal", "pipeline_17"))
    dd_path = os.path.join(pipeline, "defense_detection_v2.py")
    if not os.path.isfile(dd_path):
        sys.exit(f"ABORT: no defense_detection_v2.py under {pipeline}")
    with io.open(dd_path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    marker = chr(10) + "    METRICS = ["
    start = text.find(marker)
    end = text.find(chr(10) + "    ]", start)
    n_metrics = text.count('"', start, end) // 2 if start >= 0 else -1
    n_frozen = text.count("FrozenEstimator")
    print(f"  gate SPACE   : METRICS = {n_metrics}   ({pipeline})")
    print(f"  gate CALIB   : FrozenEstimator = {n_frozen}")
    if n_metrics != 17:
        sys.exit(f"ABORT: {n_metrics} metrics, expected the 17 listener observables")
    if n_frozen < 1:
        sys.exit("ABORT: pre-fix pipeline -- no FrozenEstimator")

    arm_root = os.path.abspath(arm_root)
    parent = os.path.dirname(arm_root)
    arms = sorted(d for d in os.listdir(parent)
                  if os.path.isdir(os.path.join(parent, d, "colab_data")))
    print(f"  gate VANTAGE : {arms}  (measuring {os.path.basename(arm_root)})")
    if arms != ["listener"] or os.path.basename(arm_root) != "listener":
        sys.exit(f"ABORT: expected one arm 'listener' under {parent}, found {arms}")

    bundle = os.path.join(arm_root, "colab_data", f"wide_{mode}.csv.gz")
    with gzip.open(bundle, "rt") as fh:
        ncol = len(fh.readline().rstrip(chr(10)).split(","))
    print(f"  gate BUNDLE  : wide_{mode} has {ncol} columns")
    if ncol != 27:
        sys.exit(f"ABORT: bundle has {ncol} columns, expected 27")

    res_csv = os.path.join(arm_root, "results", mode, "results.csv")
    res = pd.read_csv(res_csv)
    hit = res[res["Model"] == model]
    if hit.empty:
        sys.exit(f"ABORT: {model!r} has no row in {res_csv}")
    row = hit.iloc[0]
    logs = stage1_log(arm_root, mode)
    if not logs:
        print("  gate LOG     : no stage-1 log claims this arm/mode -- the "
              "space check below cannot run and training time is unknown")
    for rec in logs:
        print(f"  gate LOG     : {os.path.basename(rec['log'])} -- "
              f"created {rec['created']}, selected {rec['selected']}, "
              f"Total Runtime {rec['runtime']}  <- TRAINING TIME")

    print(f"  gate MODEL   : {model} acc {float(row['Accuracy']):.5f} "
          f"auc {float(row['AUC']):.5f} (from results.csv)")
    return pipeline, logs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-root", required=True)
    ap.add_argument("--mode", required=True, choices=["static", "mobile"])
    ap.add_argument("--model", default="Stacking_Ensemble")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--with-feature-eng", action="store_true",
                    help="also time feature engineering on the real bundle "
                         "(heavy; idle machine only)")
    ap.add_argument("--threads", type=int, default=0,
                    help="BLAS/OMP threads; 1 also pins the process to one "
                         "CPU, which is the constrained-platform bound")
    ap.add_argument("--all-models", action="store_true",
                    help="size and time every model in the pickle, giving a "
                         "measured size/accuracy frontier")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if hasattr(os, "getloadavg"):
        load1 = os.getloadavg()[0]
        if load1 > LOAD_LIMIT and not args.force:
            sys.exit(f"load average is {load1:.2f} (> {LOAD_LIMIT}); a campaign "
                     f"or learning job is probably running. Timing measured "
                     f"under contention is meaningless. Re-run with --force "
                     f"only if you accept that.")

    print("=" * 92)
    print("GATES")
    print("=" * 92)
    pipeline, stage1 = gate(args.arm_root, args.mode, args.model)
    print()

    if args.threads:
        try:
            os.sched_setaffinity(0, {0})
            cpus = sorted(os.sched_getaffinity(0))
        except (AttributeError, OSError) as exc:
            cpus = f"not pinned ({exc})"
        print(f"  PLATFORM     : {args.threads} thread(s), cpus {cpus}")
    else:
        print(f"  PLATFORM     : default, {os.cpu_count()} cpus visible")
    print()

    res_dir = os.path.join(args.arm_root, "results", args.mode)
    pkls = sorted(glob.glob(os.path.join(res_dir, "best_model_*.pkl")))
    if not pkls:
        sys.exit(f"no best_model_*.pkl in {res_dir}")

    rss_before = rss_mb()
    t0 = time.perf_counter()
    md = joblib.load(pkls[0])
    load_s = time.perf_counter() - t0
    rss_after = rss_mb()

    models = md.get("all_models") or {}
    if args.model not in models:
        sys.exit(f"{args.model!r} not in {os.path.basename(pkls[0])}")
    clf = models[args.model]
    feats = list(md["feature_names"])
    n_feat = len(feats)

    pkl_mb = os.path.getsize(pkls[0]) / (1024.0 * 1024.0)

    # What the PICKLE says about calibration, beside what the source said.
    marks = sorted({w for w in ("FrozenEstimator", "CalibratedClassifierCV",
                                "IsotonicRegression", "_SigmoidCalibration")
                    if w in repr(clf)[:200000]})
    import datetime as _dt
    mtime = _dt.datetime.fromtimestamp(os.path.getmtime(pkls[0]))
    print(f"  gate CALIB(pkl): {marks or 'nothing calibration-shaped found'} "
          f"-- fitted {mtime:%Y-%m-%d %H:%M}")

    # Size of just this model. ⛔ Do NOT joblib.dump() it to measure: the
    # results pkl here is 2.3-2.5 GB, and serialising the ensemble again would
    # write multiple GB to /tmp for the sake of one number. Serialise to an
    # in-memory counter instead, which allocates nothing beyond a small buffer.
    class _Counter(object):
        """A write-only sink that counts bytes. joblib needs tell()/seekable()
        because it writes a header and then reports the offset."""

        def __init__(self):
            self.n = 0

        def write(self, b):
            self.n += len(b)
            return len(b)

        def tell(self):
            return self.n

        def seekable(self):
            return False

        def writable(self):
            return True

        def flush(self):
            pass

        def close(self):
            pass

    try:
        cnt = _Counter()
        joblib.dump(clf, cnt)
        model_mb = cnt.n / (1024.0 * 1024.0)
    except Exception as exc:                       # pragma: no cover
        print(f"  [warn] could not size the model in memory ({exc}); "
              f"reporting the full pkl only")
        model_mb = float("nan")

    print("=" * 92)
    print(f"INFERENCE COST -- {args.model}, {args.mode}")
    print(f"  arm      : {args.arm_root}")
    print(f"  pkl      : {os.path.basename(pkls[0])}")
    print("=" * 92)
    print(f"  features consumed by the model : {n_feat}")
    print(f"  full results pkl (all models)  : {pkl_mb:9.2f} MB")
    print(f"  this model alone, serialised   : {model_mb:9.2f} MB")
    print(f"  peak RSS after load            : {rss_after:9.2f} MB "
          f"(delta {rss_after - rss_before:+.2f} MB)")
    print(f"  deserialisation time           : {load_s:9.3f} s")
    print()

    rng = np.random.default_rng(42)
    rows = []
    hdr = (f"{'batch (windows)':>18}{'total ms':>12}{'ms / window':>14}"
           f"{'windows / s':>14}")
    print(hdr)
    print("-" * len(hdr))
    for n in BATCHES:
        X = pd.DataFrame(rng.standard_normal((n, n_feat)), columns=feats)
        clf.predict_proba(X.iloc[:1])           # warm up
        best = float("inf")
        for _ in range(args.repeats):
            t = time.perf_counter()
            clf.predict_proba(X)
            best = min(best, time.perf_counter() - t)
        per = best / n
        print(f"{n:>18,}{best * 1e3:>12.2f}{per * 1e3:>14.4f}{1.0 / per:>14,.0f}")
        rows.append({"mode": args.mode, "model": args.model, "batch": n,
                     "total_ms": best * 1e3, "ms_per_window": per * 1e3,
                     "windows_per_s": 1.0 / per, "n_features": n_feat,
                     "model_mb": model_mb, "results_pkl_mb": pkl_mb,
                     "peak_rss_mb": rss_after, "load_s": load_s})
    print("-" * len(hdr))
    print("  Best of %d repeats per batch size; synthetic post-scaler input."
          % args.repeats)

    if args.with_feature_eng:
        print("\n  timing feature engineering on the real bundle "
              "(this is the heavy path)...")
        # ⛔ Not analysis/pipeline: that is the 33-metric generator with no
        # FrozenEstimator, and using it here would time the engineering of a
        # different feature space than the one the model was fitted in.
        sys.path.insert(0, pipeline)
        import defense_detection_v2 as dd
        os.environ["DCFM_DATA_BUNDLE"] = os.path.join(args.arm_root, "colab_data")
        cfg = dd.Config(data_root=f"./features_{args.mode}/",
                        results_dir=os.path.join(os.sep, "tmp", "_icr_scratch"),
                        verbose=0)
        pipe = dd.DefenseDetector(cfg)
        raw = pipe.load_simulation_data_enhanced()
        X, y, _ = pipe.preprocess_data_enhanced(raw)
        t = time.perf_counter()
        Xe = pipe.engineer_advanced_features(X)
        eng_s = time.perf_counter() - t
        print(f"  feature engineering: {eng_s:.2f} s for {len(X):,} windows "
              f"= {eng_s / len(X) * 1e3:.4f} ms/window "
              f"({X.shape[1]} -> {Xe.shape[1]} columns)")

        # ⛔ SPACE, exactly: the generator just used must reproduce the columns
        # the stored model was fitted on, and the counts the arm's own log
        # recorded. A 33-metric fit cannot satisfy all three.
        missing = [c for c in feats if c not in Xe.columns]
        if missing:
            sys.exit(f"⛔ ABORT [space]: {len(missing)} of the model's "
                     f"{len(feats)} features are not produced by {pipeline}, "
                     f"e.g. {missing[:5]}. The model was fitted in a different "
                     f"feature space than the one just timed.")
        print(f"  gate SPACE+  : all {len(feats)} stored features are produced "
              f"by this generator")
        for rec in stage1:
            if rec["created"] is not None and rec["created"] != Xe.shape[1]:
                sys.exit(f"⛔ ABORT [space]: the arm's log recorded "
                         f"{rec['created']} engineered columns, this "
                         f"generator produced {Xe.shape[1]}")
            if rec["selected"] is not None and rec["selected"] != len(feats):
                sys.exit(f"⛔ ABORT [space]: the arm's log recorded "
                         f"{rec['selected']} selected features, the pickle "
                         f"carries {len(feats)}")
            print(f"  gate SPACE+  : matches {os.path.basename(rec['log'])} "
                  f"-- created {rec['created']}, selected {rec['selected']}")
        for r in rows:
            r["feature_eng_ms_per_window"] = eng_s / len(X) * 1e3

    # Per arm. ⛔ The flat default would overwrite the 20/8 CSVs, which are the
    # record of arms_c10k_stackcal and are cited in STATE §25.71.1.
    arm_name = os.path.basename(os.path.dirname(os.path.abspath(args.arm_root)))
    if args.all_models:
        print()
        print("=" * 92)
        print("FRONTIER -- every stored model: what it costs and what it buys")
        print("=" * 92)
        res_all = pd.read_csv(os.path.join(res_dir, "results.csv"))
        frontier = []
        hdr2 = (f"{'model':<24}{'acc':>9}{'auc':>9}{'MB':>10}"
                f"{'ms/window':>12}{'ms batch':>11}{'scorer':>16}")
        print(hdr2)
        print("-" * len(hdr2))
        for _, r in res_all.iterrows():
            name = r["Model"]
            est = models.get(name)
            if est is None:
                print(f"{name:<24}  not in the pickle -- skipped")
                continue
            if hasattr(est, "predict_proba"):
                scorer, call = "predict_proba", est.predict_proba
            elif hasattr(est, "decision_function"):
                scorer, call = "decision_function", est.decision_function
            else:
                print(f"{name:<24}  no scoring method -- skipped")
                continue
            cnt2 = _Counter()
            try:
                joblib.dump(est, cnt2)
                mb2 = cnt2.n / (1024.0 * 1024.0)
            except Exception:
                mb2 = float("nan")
            timings = {}
            for n in (1, 8000):
                X = pd.DataFrame(rng.standard_normal((n, n_feat)), columns=feats)
                call(X.iloc[:1])
                best = float("inf")
                for _ in range(args.repeats):
                    t = time.perf_counter()
                    call(X)
                    best = min(best, time.perf_counter() - t)
                timings[n] = best / n * 1e3
            print(f"{name:<24}{float(r['Accuracy']):>9.5f}{float(r['AUC']):>9.5f}"
                  f"{mb2:>10.2f}{timings[1]:>12.3f}{timings[8000]:>11.4f}"
                  f"{scorer:>16}")
            frontier.append({"mode": args.mode, "model": name,
                             "accuracy": float(r["Accuracy"]),
                             "auc": float(r["AUC"]), "model_mb": mb2,
                             "ms_per_window_batch1": timings[1],
                             "ms_per_window_batch8000": timings[8000],
                             "scorer": scorer, "n_features": n_feat,
                             "threads": args.threads or os.cpu_count()})
        print("-" * len(hdr2))
        print("  Sizes are the model alone, serialised; latency is best of "
              f"{args.repeats}, synthetic post-scaler input.")

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "inference_cost_out", arm_name)
    os.makedirs(out_dir, exist_ok=True)
    # The platform is part of the identity of these numbers: a --threads run
    # measures a different machine than the default one, so it gets its own
    # file rather than overwriting it.
    suffix = f"_t{args.threads}" if args.threads else ""

    if args.all_models:
        os.makedirs(out_dir, exist_ok=True)
        fpath = os.path.join(
            out_dir, f"inference_cost_frontier_{args.mode}{suffix}.csv")
        pd.DataFrame(frontier).to_csv(fpath, index=False)
        print(f"  frontier -> {fpath}")

    path = os.path.join(
        out_dir, f"inference_cost_{args.mode}_{args.model}{suffix}.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"\nwritten: {path}")


if __name__ == "__main__":
    main()
