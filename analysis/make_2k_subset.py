#!/usr/bin/env python3
"""
Cut the reported 17-observable campaign's listener bundle down to the same 2,000
runs the density and window sweeps use, so every point in those series is a
2,000-run fit of the SAME campaign.

Selection rule: the first 2,000 run_ids of the C_all manifest, i.e. exactly what
run_traffic_all.sh / run_windowsweep.sh feed the simulator and what
copy_subset.py gave arms_gm_control. Verified identical to gm_control's id set
in both modes by analysis/check_2k_subset.py, 26/8.

⛔ NOT --limit 2000. That cuts by file-discovery order, which is lexicographic
(-1, -10, -100, -1000, ...), not manifest order. The ids here span 1..1,004,049.

Writes arms_r34_17feat_2k/listener/colab_data/wide_{mode}.csv.gz with the SAME
columns, in the same order, as the source bundle. Nothing else is touched, and
the source campaign is opened read-only.
"""
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = 2000
SRC_ARM = f"{ROOT}/analysis/arms_r34_17feat"
DST_ARM = f"{ROOT}/analysis/arms_r34_17feat_2k"
MANIFEST = {
    "static": f"{ROOT}/simulations_v347_hopablation_10k/manifests/C_all.csv",
    "mobile": f"{ROOT}/simulations_v347_hopablation_mobile_10k/manifests/C_all.csv",
}
SCEN = {"baseline", "attack_only", "defense_only", "defense_vs_attack"}

out_dir = f"{DST_ARM}/listener/colab_data"
os.makedirs(out_dir, exist_ok=True)

for mode, man in MANIFEST.items():
    src = f"{SRC_ARM}/listener/colab_data/wide_{mode}.csv.gz"
    dst = f"{out_dir}/wide_{mode}.csv.gz"
    if os.path.exists(dst):
        print(f"[skip] {dst} exists")
        continue

    want = set(int(x) for x in pd.read_csv(man).iloc[:N, 1])
    df = pd.read_csv(src)
    cols = list(df.columns)

    rid = df.file_source.str.extract(r"metrics_output-(\d+)\.csv",
                                     expand=False).astype(int)
    sub = df[rid.isin(want)].copy()

    # gates -- refuse to write anything that is not exactly the intended cut
    got = set(rid[rid.isin(want)])
    assert got == want, f"{mode}: id set mismatch ({len(want - got)} missing)"
    assert len(sub) == 4 * N, f"{mode}: {len(sub)} rows, expected {4 * N}"
    per = sub.groupby(rid[rid.isin(want)]).scenario.apply(set)
    assert all(s == SCEN for s in per), f"{mode}: a run is missing a scenario"
    assert list(sub.columns) == cols, f"{mode}: column set changed"

    sub.to_csv(dst, index=False, compression="gzip")
    print(f"[ok] {mode}: {len(sub)} rows x {len(cols)} cols -> {dst}")

print("\nsource campaign untouched:", SRC_ARM)
