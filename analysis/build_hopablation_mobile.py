#!/usr/bin/env python3
"""Build the hop-filter ablation populations (MOBILE).

Mirror of build_hopablation.py. That file is left untouched as the provenance
record of the static staging; this is its mobile sibling. Everything differs
only in the mode-specific paths (features_mobile, accepted_seeds_mobile.csv,
hop_metadata_mobile.csv) and in the staging root, which is kept separate so the
static staging is never touched.

    A_ge3   source >= 3 hops from the victim   (what the 3.47 campaign ran)
    B_ge2   source >= 2 hops                   (THE PAPER'S CRITERION -- 3.19
                                                rejected only direct neighbours,
                                                verified in all three 3.19
                                                scenarios: `if (isItNeighbor())`)
    C_all   connectivity only                  (no distance condition)

Each population is the first N connected seeds, in seed order, satisfying its
criterion -- exactly the set a campaign run under that criterion would have
accepted. Far runs come from simulations_v347/ (ids 1..10000); near runs from
simulations_v347_nohopfilter/ (ids seed+1,000,000), whose hop_metadata_mobile.csv
records each run's category (1 = direct neighbour, 2 = two hops).

SIZE. N defaults to 10000 -- the paper's dataset size, so the populations are
size-comparable to the published result. The 2,000-run ablation used N=2000 as
a cheap diagnostic; do not compare across the two sizes (STATE §20.6).
Measured requirement for mobile: N=2000 needs 668 rerun runs, N=10000 needs
3355, the whole pool needs all 5055.

The build stages SYMLINKS only: nothing is copied, nothing existing is touched.

Hop category is deliberately absent from every staged file: it lives only in
hop_metadata_mobile.csv and in the manifests written here. Keeping it out of
metrics_output keeps it out of the feature space; any per-distance breakdown is
computed after training by joining predictions on file_source.

Usage:
    python3 build_hopablation_mobile.py             # N=10000
    N=2000 python3 build_hopablation_mobile.py      # the diagnostic size
    POPS=B_ge2,C_all python3 build_hopablation_mobile.py   # subset
"""
import csv
import os
import sys

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # ns-3.47/
MAIN = os.path.join(R, "simulations_v347")
RERUN = os.path.join(R, "simulations_v347_nohopfilter")
# STAGE is overridable so a larger staging can be built without touching an
# existing one. ⚠️ N defaults to 10000 here while the tree on disk was built at
# N=2000, so a bare re-run would REDEFINE manifests/*.csv — the seed pool every
# derived experiment draws from (STATE §25.23). Always pass STAGE when building
# a different size.
STAGE = os.environ.get("STAGE", os.path.join(R, "simulations_v347_hopablation_mobile"))
SCEN = ["baseline", "attack_only", "defense_only", "defense_vs_attack"]
KINDS = ["metrics_output", "observer_metrics", "observer_detail"]
N = int(os.environ.get("N", "10000"))
WANT = os.environ.get("POPS", "A_ge3,B_ge2,C_all").split(",")

# ---- inputs -----------------------------------------------------------------
main_runs = []            # (seed, id, source, hop_cat) — cat 3 by construction
with open(os.path.join(MAIN, "accepted_seeds_mobile.csv"), newline="") as f:
    for row in csv.DictReader(f):
        main_runs.append((int(row["seed"]), int(row["run_id"]), "main", 3))

meta = os.path.join(RERUN, "hop_metadata_mobile.csv")
if not os.path.isfile(meta):
    sys.exit(f"FATAL: {meta} not found — run run_nohopfilter_mobile.sh first")

rerun = {1: [], 2: []}
bad_cat = 0
with open(meta, newline="") as f:
    for row in csv.DictReader(f):
        if row["result"] != "OK":
            continue
        cat = int(row["hop_category"])
        if cat not in (1, 2):
            # Category 3 here means the seed recovery pulled a run that was
            # never distance-rejected. That invalidates the population split,
            # so refuse rather than silently mislabel.
            bad_cat += 1
            continue
        rerun[cat].append((int(row["seed"]), int(row["run_id"]), "rerun", cat))

if bad_cat:
    sys.exit(f"FATAL: {bad_cat} rerun runs report hop_category=3. Every seed in "
             f"the rerun list was rejected on distance, so none may be >=3 hops. "
             f"The seed recovery or the mobility flag is wrong — do not build.")

print(f"inputs: main={len(main_runs)}  rerun cat1={len(rerun[1])}  cat2={len(rerun[2])}")

# sanity: seeds must not repeat across sources
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

for pop in WANT:
    pop = pop.strip()
    if pop not in POPS:
        sys.exit(f"FATAL: unknown population {pop}")
    pool = POPS[pop]
    take = pool[:N]
    if len(take) < N:
        sys.exit(f"FATAL: {pop} has only {len(take)} runs, need {N}. "
                 f"More rerun seeds must be simulated — widen N_SEEDS in "
                 f"run_nohopfilter_mobile.sh.")

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
            ddir = os.path.join(STAGE, pop, "features_mobile", scen)
            os.makedirs(ddir, exist_ok=True)
            for kind in KINDS:
                src = os.path.join(src_root, "features_mobile", scen, f"{kind}-{rid}.csv")
                dst = os.path.join(ddir, f"{kind}-{rid}.csv")
                if not os.path.isfile(src):
                    sys.exit(f"FATAL: missing source file {src}")
                if os.path.islink(dst) or os.path.exists(dst):
                    os.remove(dst)
                os.symlink(src, dst)
                linked += 1

    n_main = sum(1 for t in take if t[2] == "main")
    cats = {c: sum(1 for t in take if t[3] == c) for c in (1, 2, 3)}
    print(f"{pop}: staged {linked} links | main={n_main} rerun={N - n_main} "
          f"| cat1={cats[1]} cat2={cats[2]} cat3={cats[3]} "
          f"| seed range {take[0][0]}..{take[-1][0]} | manifest={os.path.basename(mpath)}")

print("staging complete — nothing copied, nothing deleted.")
