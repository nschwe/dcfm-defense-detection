#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_arm.py — run a paper analysis for an arm, using the UNMODIFIED pipeline scripts.

Design: arms differ ONLY in their data bundle (the listener bundle of one campaign
or another). So instead of copying the pipeline scripts per arm (and risking
drift), each arm keeps just its colab_data/ bundle, and this thin runner:

  1. imports runtime_guard FIRST (thread + MAX_JOBS caps, before numpy loads),
  2. points DCFM_DATA_BUNDLE at the arm's bundle,
  3. invokes the requested pipeline script (DCFM_PIPELINE) with the arm's
     results dir, so every arm runs byte-identical pipeline code.

This is the property every comparison between arms rests on: identical code,
different data. If a script were copied per arm, a later edit could silently
diverge one arm from another; here that is impossible.

Usage:
    # one analysis, one arm:
    python run_arm.py --arm listener --script defense_detection_v2.py --mode static
    # the full set for an arm (in notebook order):
    python run_arm.py --arm listener --all

Results land in <DCFM_ARMS_ROOT>/<arm>/results/.
"""
import runtime_guard  # noqa: F401  MUST be first: sets thread/MAX_JOBS caps pre-import
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))          # analysis/
# Arms root holds <arm>/{colab_data,results}/; the pipeline holds the imported
# v2 stage scripts. Both default to the analysis/ layout, overridable via env.
V4_ROOT = os.environ.get("DCFM_ARMS_ROOT", os.path.join(HERE, "arms"))
V2 = os.environ.get("DCFM_PIPELINE", os.path.join(HERE, "pipeline"))
ARMS = ["listener"]

# Where {static,mobile}/best_models.pkl lives. Decision STATE §22.0.2/§23.1(7):
# no new HP search — the paper's grids are reused. The four stage scripts that
# need it take --hp-results-dir; this is the value run_arm hands them.
# 17/8 (smoke run): the FULL results/hp_search_extended pickles embed fitted
# models from sklearn 1.7.2 and fail to unpickle under the manet env
# (ModuleNotFoundError: _loss). The reproduction cache in colab_data holds the
# same tuned estimators as light params-only pickles that DO load (with a
# version warning); load_best_params only calls .get_params() on them.
# 24/9: the package ships the selected configurations as analysis/hp_selected/
# {static,mobile}/best_params.json (dump_hp_params.py); the loaders read that
# file first and fall back to best_models.pkl.
_HP_CANDIDATES = [os.path.join(HERE, "hp_selected"),
                  os.path.join(os.path.dirname(HERE), "hp_selected")]
HP_RESULTS = os.environ.get(
    "DCFM_HP_RESULTS",
    next((c for c in _HP_CANDIDATES if os.path.isdir(c)), _HP_CANDIDATES[0]))

# The paper's analyses in notebook order, mapped to their v2 scripts. Each entry is
# (script, needs_mode) — mode-parametrised scripts run once per {static,mobile};
# cross-domain scripts read both configs themselves and run once.
PIPELINE = [
    ("defense_detection_v2.py",            True),   # Part 2  §5-6.1 Table 4 (in-domain acc)
    ("compute_cohens_d_v2.py",             False),  # Part 3  §6.2  Table 4 (Cohen's d)
    ("compute_separability_v2.py",         False),  # Part 3  §6.3  Table 5
    ("feature_importance_sensitivity_v2.py", False),# Part 4  §6.5  Universal-4 selection
    ("k_sweep_universal4_v2.py",           False),  # Part 4  §6.6  Table 7-8 (K sweep)
    ("feature_eng_ablation_v2.py",         False),  # Part 5  §6.7  Table 9 (ablation)
    ("threshold_decomposition_v2.py",      False),  # Part 6  §6.8  Table 10
    ("variance_decomposition_v2.py",       False),  # Part 1  §4    leakage R-ratio
]


def bundle_dir(arm):
    return os.path.join(V4_ROOT, arm, "colab_data")


def results_dir(arm):
    d = os.path.join(V4_ROOT, arm, "results")
    os.makedirs(d, exist_ok=True)
    return d


def already_done(arm, script, mode):
    """True if this exact (arm, script, mode) already produced results.

    Each in-domain run costs ~40 min and writes a 2.5 GB model, so re-running one
    by accident is expensive and silently overwrites a result we may have already
    reasoned about. results.csv is the pipeline's final artefact, so its presence
    means the run reached the end.
    """
    if script != "defense_detection_v2.py":
        return False
    marker = os.path.join(results_dir(arm), mode, "results.csv")
    legacy = os.path.join(results_dir(arm), "results.csv")   # the first run, before per-mode dirs
    return os.path.isfile(marker) or (mode == "static" and os.path.isfile(legacy))


def run_script(arm, script, mode, extra=None, force=False):
    """Invoke one v2 script for one arm (one mode if the script is mode-parametrised)."""
    if not force and already_done(arm, script, mode):
        print(f"[skip] {arm}/{script}/{mode}: results already present "
              f"(use --force to recompute)", flush=True)
        return 0
    env = dict(os.environ)
    env["DCFM_DATA_BUNDLE"] = bundle_dir(arm)
    env["PYTHONUNBUFFERED"] = "1"
    # The loader infers static/mobile from the 'static'/'mobile' substring of
    # data_root, so we still pass a data_root even though the bundle is what is read.
    sub = "features_static" if mode == "static" else "features_mobile"
    # -u on the child too: without it the v2 script's stdout is block-buffered
    # when redirected to a file, so a long run looks frozen for many minutes and
    # then dumps everything at once. PYTHONUNBUFFERED covers libraries that
    # re-open stdout themselves.
    cmd = [sys.executable, "-u", os.path.join(V2, script)]

    # Scripts vary in their arg names; the pipeline ones we use accept these.
    if script == "defense_detection_v2.py":
        cmd += ["--data-root", f"./{sub}/",
                "--results-dir", os.path.join(results_dir(arm), mode),
                "--group-by-file"]
    elif script == "cluster_bootstrap_ci.py":
        # Resamples the TEST RUNS of an already-trained model; it does not read
        # the raw feature roots at all, so it takes --results-root (where the
        # trained pipeline wrote its artefacts) rather than --static-root.
        # Verified against its argparse at cluster_bootstrap_ci.py:377-381.
        cmd += ["--results-root", results_dir(arm),
                "--out-dir", os.path.join(results_dir(arm), "cluster_bootstrap")]
    elif script in ("compute_cohens_d_v2.py", "compute_separability_v2.py"):
        # These two predate the -root convention: they take --static/--mobile/
        # --out (verified against their argparse 17/8). Each writes into its own
        # subdir so cross-script outputs never collide.
        sub_out = os.path.join(results_dir(arm), os.path.splitext(script)[0])
        cmd += ["--static", "./features_static",
                "--mobile", "./features_mobile",
                "--out", sub_out]
    elif script in ("k_sweep_universal4_v2.py", "threshold_decomposition_v2.py",
                    "feature_importance_sensitivity_v2.py",
                    "feature_eng_ablation_v2.py",
                    "instability_ablation_v2.py"):
        # -root convention plus best_models.pkl (STATE §22.0: --hp-results-dir is
        # required by the first two and defaults to a nonexistent relative path
        # in the other two; run_arm always passes it explicitly).
        sub_out = os.path.join(results_dir(arm), os.path.splitext(script)[0])
        cmd += ["--static-root", "./features_static",
                "--mobile-root", "./features_mobile",
                "--hp-results-dir", HP_RESULTS,
                "--out-dir", sub_out]
        if script == "instability_ablation_v2.py":
            # Same reason as the k_sweep path below: its Cohen's d input lives
            # in a sibling stage's subdir, which no relative default can find.
            cmd += ["--cohens-d-csv",
                    os.path.join(results_dir(arm), "compute_cohens_d_v2",
                                 "cohens_d_all_features.csv")]
        if script == "k_sweep_universal4_v2.py":
            # --universal-set became REQUIRED on 10/9. Its built-in default
            # was the SUBMITTED Universal-4, so a bare invocation used to
            # sweep the wrong anchor silently. The anchor is stage 4's
            # K_rank=4 intersection, which runs earlier in SCRIPTS.
            cmd += ["--universal-set",
                    os.path.join(results_dir(arm),
                                 "feature_importance_sensitivity_v2",
                                 "universal_set.txt")]
        if script == "threshold_decomposition_v2.py":
            # STATE §25.11 blocker 7(i): its default k_sweep path is relative
            # to cwd and never matches the per-stage subdirs; point it at this
            # arm's k_sweep output explicitly.
            cmd += ["--k-sweep-csv",
                    os.path.join(results_dir(arm), "k_sweep_universal4_v2",
                                 "k_sweep_results.csv")]
            # --universal-set became REQUIRED on 16/9, for the same reason it
            # did for the k_sweep above: the built-in default is the SUBMITTED
            # four-feature set, so a bare invocation decomposed against the
            # wrong anchor in silence. The anchor here is the SELECTED K = 3
            # subset, which stage 4 writes beside its K_rank = 4 intersection.
            cmd += ["--universal-set",
                    os.path.join(results_dir(arm),
                                 "feature_importance_sensitivity_v2",
                                 "universal_set_k3", "universal_set.txt")]
    elif script in ("dj_ablation_breakdown.py", "variance_decomposition_v2.py"):
        # Had no argparse at all (STATE §22.0); it was added 17/8 with defaults
        # equal to their old hardcoded paths, so bare invocation is unchanged.
        sub_out = os.path.join(results_dir(arm), os.path.splitext(script)[0])
        cmd += ["--static-root", "./features_static",
                "--mobile-root", "./features_mobile",
                "--hp-results-dir", HP_RESULTS,
                "--results-root", results_dir(arm),
                "--out-dir", sub_out]
    else:
        # unknown script: the old generic guess, kept for anything ad-hoc.
        cmd += ["--static-root", "./features_static",
                "--mobile-root", "./features_mobile",
                "--out-dir", results_dir(arm)]

    if extra:
        cmd += list(extra)

    print(f"\n{'='*70}\n[{arm}] {script}"
          f"{' ('+mode+')' if mode else ''}\n{'='*70}", flush=True)
    print(runtime_guard.report(), flush=True)
    r = subprocess.run(cmd, env=env, cwd=V2)
    return r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--all-arms", action="store_true")
    ap.add_argument("--script")
    ap.add_argument("--all", action="store_true", help="run the full pipeline for the arm")
    ap.add_argument("--mode", choices=["static", "mobile"], default=None)
    ap.add_argument("--force", action="store_true",
                    help="recompute even if results already exist")
    ap.add_argument("extra", nargs=argparse.REMAINDER,
                    help="args after -- are forwarded verbatim to the v2 script")
    args = ap.parse_args()
    extra = [a for a in args.extra if a != "--"]

    arms = ARMS if args.all_arms else ([args.arm] if args.arm else None)
    if not arms:
        ap.error("give --arm or --all-arms")

    for arm in arms:
        # The guard asks "is the bundle this run will read present?", so it must
        # follow --mode. It used to test wide_static.csv.gz unconditionally,
        # which silently skipped every mobile-only bundle dir; the static
        # campaign never hit it because make_arm_bundles without --mode builds
        # both modes. Scripts that take no mode read both configs, so there the
        # guard still requires both. (14/8: this is why the first mobile
        # hop-ablation run produced no results at all.)
        needed = ([f"wide_{args.mode}.csv.gz"] if args.mode
                  else ["wide_static.csv.gz", "wide_mobile.csv.gz"])
        missing = [b for b in needed
                   if not os.path.isfile(os.path.join(bundle_dir(arm), b))]
        if missing:
            print(f"[skip] {arm}: bundle not built ({', '.join(missing)})"
                  f" — run make_arm_bundles.py")
            continue
        if args.all:
            jobs = PIPELINE
        elif args.script:
            needs_mode = args.script == "defense_detection_v2.py"
            jobs = [(args.script, needs_mode)]
        else:
            ap.error("give --script or --all")
        for script, needs_mode in jobs:
            modes = [args.mode] if args.mode else (["static", "mobile"] if needs_mode else [None])
            for m in modes:
                rc = run_script(arm, script, m, extra, args.force)
                if rc != 0:
                    print(f"[warn] {arm}/{script}/{m} exited {rc}")


if __name__ == "__main__":
    main()
