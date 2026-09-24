#!/usr/bin/env python3
"""
R#5.7 -- paired seed bootstrap over the traffic-density series.

The three intervals were fitted on the SAME 2,000 manifest ids, so the series is
paired and the difference between two intervals can be bootstrapped over seeds,
exactly as §25.109 did for the activation-fraction series.

Why this is worth running even though §25.118 already showed the spread is inside
split noise: those are different questions. §25.118 measured how much a single
stage-1 accuracy moves when the train/test split changes. This measures whether
the DATA at two densities are separable to different degrees at all -- a
classifier-free, split-free quantity. If the paired CI on the difference contains
zero, "denser traffic does not change detectability" becomes a measured statement
rather than a failure to detect a change.

  2.0 s  arms_r34_17feat_2k          (the reported campaign, cut to 2,000 ids)
  1.0 s  arms_trafficsweep/i1.0_{m}
  0.5 s  arms_trafficsweep/i0.5_{m}

Mahalanobis distance between the defended and undefended classes under the pooled
covariance of the Universal-3, resampled over seeds with the SAME seed vector
applied to every interval in each iteration.
"""
import os

import numpy as np
import pandas as pd

R = os.path.dirname(os.path.abspath(__file__))   # analysis/
U3 = ["TcMessageRate", "AverageAdvertisedLinksPerTCMessage", "AverageMprCount"]
DEFENDED = ["defense_only", "defense_vs_attack"]
UNDEFENDED = ["baseline", "attack_only"]
B = 2000
STAGE1 = {"static": {2.0: 0.8956, 1.0: 0.8950, 0.5: 0.8931},
          "mobile": {2.0: 0.9138, 1.0: 0.9075, 0.5: 0.9100}}


# the arm directories are literally i1.0_* and i0.5_*, so %g (which renders 1.0
# as "1") does not address them
DIRNAME = {1.0: "i1.0", 0.5: "i0.5"}


def path(interval, mode):
    if interval == 2.0:
        return f"{R}/arms_r34_17feat_2k/listener/colab_data/wide_{mode}.csv.gz"
    return f"{R}/arms_trafficsweep/{DIRNAME[interval]}_{mode}/listener/colab_data/wide_{mode}.csv.gz"


def load(interval, mode):
    d = pd.read_csv(path(interval, mode))
    d["run_id"] = d.file_source.str.extract(r"metrics_output-(\d+)\.csv",
                                            expand=False).astype(int)
    return d


def maha(pos, neg):
    mp, mn = pos.mean(0), neg.mean(0)
    a, b = len(pos), len(neg)
    sp = ((a - 1) * np.cov(pos, rowvar=False) + (b - 1) * np.cov(neg, rowvar=False)) / (a + b - 2)
    diff = mp - mn
    return float(np.sqrt(max(diff @ np.linalg.pinv(sp) @ diff, 0.0)))


def stacked(df, ids, scenarios):
    parts = [df[df.scenario == s].set_index("run_id")[U3].reindex(ids).to_numpy(float)
             for s in scenarios]
    return np.stack(parts)          # (n_scen, n_ids, 3)


rng = np.random.default_rng(20260826)
INTERVALS = (2.0, 1.0, 0.5)

for mode in ("static", "mobile"):
    print("=" * 78)
    print(f"MODE = {mode}")
    print("=" * 78)

    data = {i: load(i, mode) for i in INTERVALS}
    idsets = {i: set(data[i].run_id) for i in INTERVALS}
    for i in INTERVALS:
        print(f"  i={i:<4g} rows={len(data[i]):<6d} ids={len(idsets[i])}")
    common = sorted(set.intersection(*idsets.values()))
    ids = np.array(common)
    print(f"  paired ids in common: {len(ids)}")
    for i in INTERVALS:
        extra = sorted(idsets[i] - set(common))
        if extra:
            print(f"    ⚠️ i={i:g} carries {len(extra)} id(s) not shared: {extra[:5]}")
    print()

    pos = {i: stacked(data[i], ids, DEFENDED) for i in INTERVALS}
    neg = {i: stacked(data[i], ids, UNDEFENDED) for i in INTERVALS}
    point = {i: maha(pos[i].reshape(-1, 3), neg[i].reshape(-1, 3)) for i in INTERVALS}

    boot = {i: np.empty(B) for i in INTERVALS}
    for b in range(B):
        take = rng.integers(0, len(ids), len(ids))     # one seed vector for all
        for i in INTERVALS:
            boot[i][b] = maha(pos[i][:, take].reshape(-1, 3),
                              neg[i][:, take].reshape(-1, 3))

    print(f"  paired seed bootstrap, B={B}, {len(ids)} seeds")
    print(f"    {'interval':<12s}{'Mahalanobis':>13s}{'boot SD':>10s}{'95% CI':>22s}{'stage-1 acc':>13s}")
    for i in INTERVALS:
        lo, hi = np.percentile(boot[i], [2.5, 97.5])
        print(f"    i={i:<10g}{point[i]:>13.4f}{boot[i].std():>10.4f}"
              f"{f'[{lo:.4f}, {hi:.4f}]':>22s}{STAGE1[mode][i]:>13.4f}")
    print()
    for a, b_ in ((2.0, 1.0), (2.0, 0.5), (1.0, 0.5)):
        d = boot[a] - boot[b_]
        lo, hi = np.percentile(d, [2.5, 97.5])
        zero = "contains 0" if lo <= 0 <= hi else "EXCLUDES 0"
        print(f"    paired diff  i={a:g} - i={b_:g} : {point[a]-point[b_]:+.4f}  "
              f"95% CI [{lo:+.4f}, {hi:+.4f}]  -> {zero}")
    print()
