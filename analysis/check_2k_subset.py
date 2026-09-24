#!/usr/bin/env python3
"""
READ-ONLY. Can the reported 17-observable campaign be cut to the same 2,000 runs
the density and window sweeps use, directly from its own bundle?

The sweeps take "the first N of the C_all manifest" (run_traffic_all.sh,
run_windowsweep.sh), which is also copy_subset.py's rule for arms_gm_control.
This checks that the same ids are present in arms_r34_17feat's bundle, and that
they agree with gm_control's set.

Writes nothing.
"""
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = 2000
MODES = {
    "static": f"{ROOT}/simulations_v347_hopablation_10k/manifests/C_all.csv",
    "mobile": f"{ROOT}/simulations_v347_hopablation_mobile_10k/manifests/C_all.csv",
}
SCEN = {"baseline", "attack_only", "defense_only", "defense_vs_attack"}


def bundle_ids(path):
    df = pd.read_csv(path)
    df["run_id"] = df.file_source.str.extract(r"metrics_output-(\d+)\.csv",
                                              expand=False).astype(int)
    return df


for mode, man in MODES.items():
    print("=" * 74)
    print(f"MODE = {mode}")
    print("=" * 74)

    m = pd.read_csv(man)
    print(f"  manifest columns: {list(m.columns)}  rows={len(m)}")
    want = list(m.iloc[:N, 1])          # column 2 = run_id, per the runners
    want_set = set(int(x) for x in want)
    print(f"  first {N} manifest run_ids: n={len(want_set)} unique, "
          f"min={min(want_set)} max={max(want_set)}")

    big = bundle_ids(f"{ROOT}/analysis/arms_r34_17feat/listener/colab_data/wide_{mode}.csv.gz")
    have = set(big.run_id)
    print(f"  arms_r34_17feat bundle: {len(big)} rows, {len(have)} unique run_ids")

    missing = want_set - have
    print(f"  wanted ids NOT in the 10k bundle : {len(missing)}")
    if missing:
        print(f"    e.g. {sorted(missing)[:10]}")

    sub = big[big.run_id.isin(want_set)]
    print(f"  rows after the cut               : {len(sub)}  (expect {4*len(want_set)})")
    per = sub.groupby("run_id").scenario.nunique()
    print(f"  run_ids with all four scenarios  : {(per == 4).sum()} / {len(per)}")
    bad = sub.groupby("run_id").scenario.apply(lambda s: set(s) != SCEN)
    print(f"  run_ids with a wrong scenario set: {int(bad.sum())}")

    gm = bundle_ids(f"{ROOT}/analysis/arms_gm_control/listener/colab_data/wide_{mode}.csv.gz")
    gset = set(gm.run_id)
    print(f"  gm_control ids                   : {len(gset)}")
    print(f"  identical to the manifest cut    : {gset == want_set}")
    if gset != want_set:
        print(f"    in gm_control only : {sorted(gset - want_set)[:10]}")
        print(f"    in manifest only   : {sorted(want_set - gset)[:10]}")

    print(f"  columns preserved by the cut     : {list(big.columns) == list(sub.columns)} "
          f"({len(big.columns)} cols)")
    print()
