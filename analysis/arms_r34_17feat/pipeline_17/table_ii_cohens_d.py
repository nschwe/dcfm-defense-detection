"""
Compute Cohen's d for the three topology metrics in Table II of the paper:
  1. Avg. neighbor count
  2. Avg. two-hop neighborhood size
  3. Avg. min. Euclidean distance to neighbor

Effect size is computed between accepted and rejected runs.
The probe was written at t=59s (before acceptance decision at t=60s),
so topology_probes_*.csv contains rows for ALL attempts (accepted + rejected).
Acceptance is determined by joining with run_status_*.csv on seed.

Usage (from project root, ~/ns3/Final_Project_NS3-master):
    python3 -u table_ii_cohens_d.py 2>&1 | tee table_ii_cohens_d.log
"""

import os
import numpy as np
import pandas as pd


def cohens_d(a, b):
    """Pooled-SD Cohen's d. Sign indicates a.mean() - b.mean()."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    sa2 = a.var(ddof=1)
    sb2 = b.var(ddof=1)
    pooled_sd = np.sqrt(((na - 1) * sa2 + (nb - 1) * sb2) / (na + nb - 2))
    return (a.mean() - b.mean()) / pooled_sd


def analyze(cfg, probe_path, status_path):
    print(f"\n{'=' * 60}\n  {cfg.upper()}\n{'=' * 60}")

    probe = pd.read_csv(probe_path)
    status = pd.read_csv(status_path)
    print(f"Probe rows : {len(probe)}")
    print(f"Status rows: {len(status)}")

    # Acceptance label per seed (from run_status)
    status["accept"] = (status["result"] == "ACCEPT")
    seed_accept = status[["seed", "accept"]].drop_duplicates(subset="seed")

    merged = probe.merge(seed_accept, on="seed", how="left")
    n_unmatched = merged["accept"].isna().sum()
    print(f"Merged rows: {len(merged)}  (unmatched: {n_unmatched})")
    merged = merged.dropna(subset=["accept"])

    accepted = merged[merged["accept"]]
    rejected = merged[~merged["accept"]]
    print(f"Accepted   : {len(accepted)}")
    print(f"Rejected   : {len(rejected)}")

    metrics = [
        ("Avg. neighbor count",                   "avg_neighbor_count"),
        ("Avg. two-hop neighborhood size",        "avg_two_hop_count"),
        ("Avg. min. Euclidean dist. to neighbor", "avg_min_euclidean_dist"),
    ]

    print()
    print(f"{'Metric':<42s}  {cfg+' d':>12s}")
    print("-" * 60)
    for label, col in metrics:
        if col not in merged.columns:
            print(f"  {label:<40s}  MISSING column: {col}")
            continue
        a = accepted[col].dropna().values
        r = rejected[col].dropna().values
        d = cohens_d(a, r)
        print(f"  {label:<40s}  {d:+.4f}")


if __name__ == "__main__":
    analyze(
        "static",
        "simulations/topology_probes_static.csv",
        "simulations/run_status_static.csv",
    )
    analyze(
        "mobile",
        "simulations/topology_probes_mobile.csv",
        "simulations/run_status_mobile.csv",
    )