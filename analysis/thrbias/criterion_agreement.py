#!/usr/bin/env python3
"""
R#5.10 --- does Universal-3 depend on which importance criterion was chosen?

The comment, on the submitted version:
    "Universal-4 was selected by optimizing the criterion that yields the best
     result ... the subset was chosen based on the metric it is then evaluated
     against. The authors should validate Universal-4 on an independent holdout
     set or demonstrate that the same four features consistently appear under
     other criteria without picking the best-performing one."

The second remedy is answerable from data already on disk. Stage 4 of the
reported campaign records, for every (importance method x stability variant x
K x threshold), which features passed the bootstrap stability test. This script
reports how often the revised subset appears WITHOUT reference to which
criterion performs best.

⛔ It does NOT re-derive the subset and it does not evaluate anything. It counts
   agreement. Reading it as validation of the subset's performance would repeat
   the circularity the reviewer is objecting to.

USAGE
    criterion_agreement.py [K]      (default 3)
"""
import os
import sys
from collections import Counter

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))          # analysis/thrbias
SRC = os.path.join(HERE, "..", "arms_r34_17feat", "listener", "results",
                   "feature_importance_sensitivity_v2", "universal_set_members.csv")
U3 = {"TcMessageRate", "AverageAdvertisedLinksPerTCMessage", "AverageMprCount"}


def main(K=3):
    df = pd.read_csv(SRC)
    df["members"] = df.members.fillna("")
    df["set"] = df.members.apply(lambda s: set(x for x in s.split(";") if x))

    print(f"source : {SRC.split('results/')[-1]}")
    print(f"rows   : {len(df)}   methods: {sorted(df.method.unique())}")
    print(f"variants: {sorted(df.variant.unique())}   "
          f"thresholds: {sorted(df.threshold.unique())}   "
          f"K values: {sorted(df.K.unique())[:12]}...")
    print()

    at = df[df.K == K]
    print("=" * 78)
    print(f"AT K = {K}   ({len(at)} method x variant x threshold cells)")
    print("=" * 78)
    print(f"  {'method':<16s}{'variant':<9s}{'thr':>5s}  {'passing':>7s}  members")
    exact = 0
    for _, r in at.sort_values(["method", "variant", "threshold"]).iterrows():
        mark = "  <= same set" if r["set"] == U3 else ""
        exact += (r["set"] == U3)
        print(f"  {r.method:<16s}{r.variant:<9s}{r.threshold:>5}  "
              f"{r.n_passing:>7}  {';'.join(sorted(r['set']))}{mark}")
    print()
    print(f"  ⭐ cells returning EXACTLY the revised subset : {exact} / {len(at)}")

    # per-feature retention, which is the honest way to report partial agreement
    print()
    print(f"  per-feature retention at K = {K}:")
    for f in sorted(U3):
        n = sum(f in s for s in at["set"])
        print(f"    {f:<38s} {n}/{len(at)}")

    # what appears that is NOT in the subset
    extra = Counter()
    for s in at["set"]:
        for f in s - U3:
            extra[f] += 1
    if extra:
        print()
        print(f"  features appearing at K = {K} that are NOT in the subset:")
        for f, n in extra.most_common():
            print(f"    {f:<38s} {n}/{len(at)}")

    # and the same question across every K, to show it is not a K=3 artefact
    print()
    print("=" * 78)
    print("ACROSS ALL K --- how often does each subset member survive?")
    print("=" * 78)
    for f in sorted(U3):
        n = sum(f in s for s in df["set"])
        print(f"  {f:<38s} {n}/{len(df)} cells")
    same = sum(s == U3 for s in df["set"])
    print(f"  cells whose passing set is exactly the subset: {same}/{len(df)}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
