#!/usr/bin/env python3
"""Build the hop-filter ablation populations (static).

Three populations of exactly N runs each, differing ONLY in the admission
criterion applied to the same seed scan:

    A_ge3   source >= 3 hops from the victim   (the current campaign's rule)
    B_ge2   source >= 2 hops                   (rejects direct neighbours only)
    C_all   connectivity only                  (no distance condition)

Each population is the first N connected seeds, in seed order, that satisfy its
criterion -- exactly the set a campaign run under that criterion would have
accepted. Far runs come from the main campaign (simulations_v347/, ids 1..10000);
runs closer than three hops come from the no-hop-filter rerun
(simulations_v347_nohopfilter/, ids seed+1,000,000), whose hop_metadata.csv
records each run's hop category (1 = direct neighbour, 2 = two hops).

The build stages SYMLINKS only: nothing is copied, nothing existing is touched.
Layout per population matches what make_arm_bundles.py globs
(features_static/<scenario>/{metrics_output,observer_metrics,observer_detail}-<id>.csv),
so bundles are built by pointing DCFM_SIM_ROOT at a staging dir.

Hop category is deliberately absent from every staged file: it exists only in
hop_metadata.csv and in the manifests written here. Keeping it out of
metrics_output keeps it out of the feature space; any per-distance breakdown is
computed after training by joining predictions on file_source.

Usage:
    python3 build_hopablation.py            # stage + manifests
    (bundles are then built by the caller; see run_hopablation_learning.sh)
"""
import csv
import os
import sys

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # ns-3.47/
MAIN = os.path.join(R, "simulations_v347")
RERUN = os.path.join(R, "simulations_v347_nohopfilter")
# STAGE/N/POPS are overridable so a LARGER staging can be built without
# touching an existing one. This matters: the default tree's manifests are the
# seed pool every derived experiment draws from (STATE §25.23), and rebuilding
# it at a different N would silently redefine them. Defaults reproduce the
# original invocation exactly.
STAGE = os.environ.get("STAGE", os.path.join(R, "simulations_v347_hopablation"))
SCEN = ["baseline", "attack_only", "defense_only", "defense_vs_attack"]
KINDS = ["metrics_output", "observer_metrics", "observer_detail"]
N = int(os.environ.get("N", "2000"))
WANT = os.environ.get("POPS", "A_ge3,B_ge2,C_all").split(",")

# ---- inputs -----------------------------------------------------------------
main_runs = []            # (seed, id, source, hop_cat) — hop_cat 3 by construction
with open(os.path.join(MAIN, "accepted_seeds_static.csv"), newline="") as f:
    for row in csv.DictReader(f):
        main_runs.append((int(row["seed"]), int(row["run_id"]), "main", 3))

rerun = {1: [], 2: []}
with open(os.path.join(RERUN, "hop_metadata.csv"), newline="") as f:
    for row in csv.DictReader(f):
        if row["result"] != "OK":
            continue
        cat = int(row["hop_category"])
        rerun[cat].append((int(row["seed"]), int(row["run_id"]), "rerun", cat))

print(f"inputs: main={len(main_runs)}  rerun cat1={len(rerun[1])}  cat2={len(rerun[2])}")

# sanity: the two id spaces must be disjoint and seeds must not repeat
all_seeds = [s for s, *_ in main_runs] + [s for c in (1, 2) for s, *_ in rerun[c]]
if len(all_seeds) != len(set(all_seeds)):
    sys.exit("FATAL: duplicate seed across sources")

POPS = {
    "A_ge3": sorted(main_runs),
    "B_ge2": sorted(main_runs + rerun[2]),
    "C_all": sorted(main_runs + rerun[1] + rerun[2]),
}

# ---- stage ------------------------------------------------------------------
os.makedirs(os.path.join(STAGE, "manifests"), exist_ok=True)

for pop, pool in POPS.items():
    if pop not in WANT:
        continue
    take = pool[:N]
    if len(take) < N:
        sys.exit(f"FATAL: {pop} has only {len(take)} runs (asked for N={N})")

    # manifest first: it documents exactly what was selected and from where
    mpath = os.path.join(STAGE, "manifests", f"{pop}.csv")
    with open(mpath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "run_id", "source", "hop_category"])
        w.writerows(take)

    linked = 0
    for seed, rid, source, cat in take:
        src_root = MAIN if source == "main" else RERUN
        for scen in SCEN:
            ddir = os.path.join(STAGE, pop, "features_static", scen)
            os.makedirs(ddir, exist_ok=True)
            for kind in KINDS:
                src = os.path.join(src_root, "features_static", scen, f"{kind}-{rid}.csv")
                dst = os.path.join(ddir, f"{kind}-{rid}.csv")
                if not os.path.isfile(src):
                    sys.exit(f"FATAL: missing source file {src}")
                if os.path.islink(dst) or os.path.exists(dst):
                    os.remove(dst)
                os.symlink(src, dst)
                linked += 1

    n_main = sum(1 for t in take if t[2] == "main")
    n_rerun = N - n_main
    cats = {c: sum(1 for t in take if t[3] == c) for c in (1, 2, 3)}
    print(f"{pop}: staged {linked} links | main={n_main} rerun={n_rerun} "
          f"| cat1={cats[1]} cat2={cats[2]} cat3={cats[3]} "
          f"| seed range {take[0][0]}..{take[-1][0]} | manifest={os.path.basename(mpath)}")

print("staging complete — nothing copied, nothing deleted.")
