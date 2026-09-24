#!/usr/bin/env python3
"""Reproduce the admission figures of MS Section 4.2 from the recorded campaign.

Three quantities, all read from files the campaign runners wrote; nothing is
simulated and nothing is written.

  Table 1   attempts, accepted, rejected and accept rate under the reported
            admission rule (connectivity only).
  Table 2   Cohen's d between accepted and rejected seeds on three topology
            metrics that the admission rule does not use.
  L289      the share of admitted runs whose traffic source is within two hops
            of the victim in at least one measurement window.

Why a replay. The campaign search (run_campaign_v347.sh) admitted a seed only
if the network was connected AND the source was at least three hops from the
victim. The reported population keeps the first condition only. No search was
run under that rule, so its admission record is reconstructed: a seed passes
connectivity-only iff the search accepted it OR it is one of the seeds the
search rejected for distance alone (run_nohopfilter*.sh re-ran exactly those,
and their lists are rejected_seeds_<mode>.txt). Seeds were probed in order
1, 2, 3, ..., so walking them in that order and stopping at the 10,000th pass
gives the number of attempts the connectivity-only rule needs.

Table 2 uses the per-seed topology probe written at t = 59 s for every probed
seed, accepted or rejected. Under the reported rule a seed is accepted iff
fully_converged_nodes == 50, the connectivity verdict itself. The probes are
restricted to the seeds up to the Table 1 stop index, so the population is the
one Table 1 describes. d = (mean accepted - mean rejected) / pooled SD.

Inputs, resolved against this script's parent directory (the ns-3.47 tree);
override the two directories with ADM_SIM and ADM_NOHOP:
    simulations_v347/run_status_<mode>.csv
    simulations_v347/topology_probes_<mode>.csv
    simulations_v347_nohopfilter/rejected_seeds_<mode>.txt
    simulations_v347_hopablation_10k/manifests/C_all.csv          (static)
    simulations_v347_hopablation_mobile_10k/manifests/C_all.csv   (mobile)

Usage:
    python admission_tables.py            # both configurations
    python admission_tables.py static

Each figure is compared with the value printed in the manuscript and marked
OK or DIFF; the exit status is non-zero if any figure differs.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SIM = os.environ.get("ADM_SIM", os.path.join(ROOT, "simulations_v347"))
NOHOP = os.environ.get("ADM_NOHOP", os.path.join(ROOT, "simulations_v347_nohopfilter"))
MANIFEST = {
    "static": os.path.join(ROOT, "simulations_v347_hopablation_10k", "manifests", "C_all.csv"),
    "mobile": os.path.join(ROOT, "simulations_v347_hopablation_mobile_10k", "manifests", "C_all.csv"),
}
TARGET = 10_000
N_NODES = 50

# The manuscript's values (MS Section 4.2, Tables 1 and 2, line 289).
PUBLISHED = {
    "static": {"attempts": 19_932, "rejected": 9_932, "accept_pct": 50.2,
               "d": {"avg_neighbor_count": -0.140, "avg_two_hop_count": -0.265,
                     "avg_min_euclidean_dist": +0.075},
               "within2_pct": 27.3},
    "mobile": {"attempts": 28_241, "rejected": 18_241, "accept_pct": 35.4,
               "d": {"avg_neighbor_count": -0.066, "avg_two_hop_count": -0.171,
                     "avg_min_euclidean_dist": +0.126},
               "within2_pct": 33.5},
}
LABELS = {
    "avg_neighbor_count": "Avg. neighbor count",
    "avg_two_hop_count": "Avg. two-hop neighborhood size",
    "avg_min_euclidean_dist": "Avg. min. Euclidean dist. to neighbor",
}


def cohens_d(a, b):
    """Pooled-SD Cohen's d; the sign is a.mean() - b.mean()."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / pooled


def check(label, got, want, fmt):
    ok = fmt.format(got) == fmt.format(want)
    print(f"  {label:<40s} {fmt.format(got):>10s}   manuscript {fmt.format(want):>10s}   "
          f"{'OK' if ok else 'DIFF'}")
    return ok


def run(mode):
    pub = PUBLISHED[mode]
    print(f"\n=== {mode} ===")
    ok = True

    # ---- Table 1: replay of the connectivity-only rule ----
    status = pd.read_csv(os.path.join(SIM, f"run_status_{mode}.csv"))
    with open(os.path.join(NOHOP, f"rejected_seeds_{mode}.txt")) as f:
        dist_rejected = {int(x) for x in f if x.strip()}
    accepted = set(status.loc[status["result"] == "ACCEPT", "seed"].astype(int))
    seeds = np.sort(status["seed"].astype(int).unique())
    print(f"  probed seeds {seeds.min()}..{seeds.max()}, {len(seeds)} distinct; "
          f"ACCEPT {len(accepted)}, distance-rejected {len(dist_rejected)}, "
          f"overlap {len(accepted & dist_rejected)} (want 0)")
    if not (len(seeds) == seeds.max() - seeds.min() + 1):
        print("  ⛔ seeds are not contiguous; the replay assumes probe order = seed order")
        ok = False
    passing = 0
    stop = None
    for s in seeds:
        if s in accepted or s in dist_rejected:
            passing += 1
            if passing == TARGET:
                stop = int(s)
                break
    if stop is None:
        print(f"  ⛔ fewer than {TARGET} seeds pass the connectivity-only rule")
        return False
    surplus_below = int(((status["result"] == "SURPLUS") & (status["seed"] <= stop)).sum())
    print(f"  SURPLUS seeds at or below the stop index: {surplus_below} (want 0)")
    ok &= surplus_below == 0
    print("  Table 1")
    ok &= check("Attempts", stop, pub["attempts"], "{:,}")
    ok &= check("Rejected", stop - TARGET, pub["rejected"], "{:,}")
    ok &= check("Accept rate (%)", 100.0 * TARGET / stop, pub["accept_pct"], "{:.1f}")

    # ---- Table 2: accepted vs rejected, on the same seed range ----
    probe = pd.read_csv(os.path.join(SIM, f"topology_probes_{mode}.csv"))
    probe = probe[probe["seed"] <= stop]
    acc = probe[probe["fully_converged_nodes"] == N_NODES]
    rej = probe[probe["fully_converged_nodes"] != N_NODES]
    print(f"  Table 2   (probes up to seed {stop:,}: {len(acc):,} accepted, {len(rej):,} rejected)")
    for col, want in pub["d"].items():
        ok &= check(LABELS[col], cohens_d(acc[col].dropna(), rej[col].dropna()), want, "{:+.3f}")

    # ---- MS line 289: within two hops in at least one window ----
    man = pd.read_csv(MANIFEST[mode])
    within2 = 100.0 * man["hop_category"].isin([1, 2]).sum() / len(man)
    print(f"  C_all manifest: {len(man):,} runs")
    ok &= check("Source within two hops (%)", within2, pub["within2_pct"], "{:.1f}")
    return ok


def main():
    modes = sys.argv[1:] or ["static", "mobile"]
    results = [run(m) for m in modes]
    print("\nALL OK" if all(results) else "\nDIFFERENCES FOUND")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
