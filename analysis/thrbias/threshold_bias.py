#!/usr/bin/env python3
"""
R#5.8 --- a cross-validated estimate of threshold-selection bias.

WHAT THE REVIEWER ASKED FOR
    "...or providing a cross-validated estimate of threshold bias, would resolve
     this issue."

WHAT BIAS MEANS HERE
    tau* is chosen to maximize accuracy on a set of rows, so the accuracy AT tau*
    ON THOSE SAME ROWS is optimistically biased. The estimate:

        for each of K grouped folds of the threshold rows:
            pick tau_k on the in-fold part      (same grid the pipeline uses)
            acc_in  = accuracy at tau_k on the in-fold part
            acc_out = accuracy at tau_k on the held-out part
            bias_k  = acc_in - acc_out

    Report mean(bias), a bootstrap CI, and the per-model spread.

⛔ WHAT THIS IS NOT
    * Not the difference between running with and without --split-validation.
      That is a separate question (how much the fix moved the numbers) and is
      still unmeasured -- see STATE 25.124.3. Do not report one as the other.
    * Not the bias of the SUBMITTED configuration. These dumps come from runs
      that already use --split-validation, so this is the bias of the pipeline
      as the revision performs it.

INPUT
    The .npz files written by analysis/thrbias/pipeline_17 under THRBIAS_DUMP.
    call0 = the calibrated base models; call1 = the stacking ensemble. They are
    scored on the same rows but by different models and are never pooled.

USAGE
    threshold_bias.py <dump_dir> [<dump_dir> ...]
"""
import os
import sys
from collections import defaultdict

import numpy as np

GRID = np.arange(0.1, 0.9, 0.01)      # identical to optimize_thresholds
K_FOLDS = 10
B_BOOT = 2000
SEED = 20260826


def pick_tau(y_prob, y_true):
    """The pipeline's rule: first threshold on the grid attaining the best
    accuracy (strict >, so ties keep the earliest)."""
    best_t, best_a = 0.5, -1.0
    for t in GRID:
        a = ((y_prob >= t).astype(int) == y_true).mean()
        if a > best_a:
            best_a, best_t = a, t
    return float(best_t), float(best_a)


def acc_at(y_prob, y_true, t):
    return float(((y_prob >= t).astype(int) == y_true).mean())


def grouped_folds(groups, k, rng):
    """k folds over distinct groups, so no simulation run straddles a fold."""
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    chunks = np.array_split(uniq, k)
    return [np.isin(groups, c) for c in chunks]


def analyse(path):
    d = np.load(path, allow_pickle=True)
    y_prob = d["y_prob"]
    y_true = d["y_true"]
    groups = d["groups"]
    model = str(d["model"][0])
    call = int(d["call"][0])
    tau_run = float(d["threshold"][0])
    acc_run = float(d["best_accuracy"][0])

    if groups.size != y_prob.size:
        groups = np.arange(y_prob.size).astype(str)   # ungrouped fallback
        grouped = False
    else:
        grouped = True

    rng = np.random.default_rng(SEED)
    folds = grouped_folds(groups, K_FOLDS, rng)

    rows = []
    for held in folds:
        infold = ~held
        if held.sum() < 10 or infold.sum() < 10:
            continue
        t_k, a_in = pick_tau(y_prob[infold], y_true[infold])
        a_out = acc_at(y_prob[held], y_true[held], t_k)
        rows.append((t_k, a_in, a_out, a_in - a_out))

    if not rows:
        return None
    arr = np.array(rows)
    bias = arr[:, 3]

    # bootstrap over folds for a CI on the mean bias
    bs = np.empty(B_BOOT)
    for i in range(B_BOOT):
        bs[i] = bias[rng.integers(0, len(bias), len(bias))].mean()

    return dict(
        model=model, call=call, n=int(y_prob.size),
        n_groups=int(np.unique(groups).size), grouped=grouped,
        tau_run=tau_run, acc_run=acc_run,
        tau_mean=float(arr[:, 0].mean()), tau_sd=float(arr[:, 0].std(ddof=1)),
        acc_in=float(arr[:, 1].mean()), acc_out=float(arr[:, 2].mean()),
        bias=float(bias.mean()), bias_sd=float(bias.std(ddof=1)),
        lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)),
        folds=len(rows),
    )


def main(dirs):
    for dd in dirs:
        if not os.path.isdir(dd):
            print(f"!! missing {dd}"); continue
        print("=" * 100)
        print(f"DUMP = {dd}")
        print("=" * 100)
        by_call = defaultdict(list)
        for f in sorted(os.listdir(dd)):
            if not f.endswith(".npz"):
                continue
            r = analyse(os.path.join(dd, f))
            if r:
                by_call[r["call"]].append(r)

        for call in sorted(by_call):
            label = {0: "call0 — calibrated base models",
                     1: "call1 — stacking ensemble"}.get(call, f"call{call}")
            rs = sorted(by_call[call], key=lambda r: -r["bias"])
            print(f"\n  {label}   ({len(rs)} model(s), "
                  f"n={rs[0]['n']} rows / {rs[0]['n_groups']} groups, "
                  f"grouped folds={rs[0]['grouped']}, K={rs[0]['folds']})")
            print(f"    {'model':<22s}{'tau* (run)':>11s}{'acc_in':>9s}"
                  f"{'acc_out':>9s}{'BIAS':>9s}{'sd':>8s}{'95% CI':>22s}")
            for r in rs:
                ci = f"[{r['lo']:+.4f}, {r['hi']:+.4f}]"
                print(f"    {r['model']:<22s}{r['tau_run']:>11.3f}"
                      f"{r['acc_in']:>9.4f}{r['acc_out']:>9.4f}"
                      f"{r['bias']:>+9.4f}{r['bias_sd']:>8.4f}{ci:>22s}")
            b = np.array([r["bias"] for r in rs])
            excl = sum(1 for r in rs if r["lo"] > 0)
            print(f"    {'--- across models':<22s}"
                  f"mean bias {b.mean():+.4f}   median {np.median(b):+.4f}   "
                  f"range [{b.min():+.4f}, {b.max():+.4f}]")
            print(f"    {'':22s}CIs excluding zero: {excl}/{len(rs)}")

            # ⭐ The per-model CI is the wrong instrument when the models are not
            # independent replicates of one measurement but different views of
            # the same data. Agreement of SIGN across models is the evidence that
            # survives fold noise -- the same argument as STATE 25.108's
            # "17 of 17 models" reading. A two-sided exact sign test:
            pos = int((b > 0).sum())
            neg = int((b < 0).sum())
            n_eff = pos + neg
            if n_eff:
                from math import comb
                k = max(pos, neg)
                tail = sum(comb(n_eff, i) for i in range(k, n_eff + 1)) / (2 ** n_eff)
                p = min(1.0, 2 * tail)
                print(f"    {'':22s}sign test: {pos} positive / {neg} negative "
                      f"of {n_eff}   two-sided p = {p:.3g}")
                if p < 0.05 and pos > neg:
                    print(f"    {'':22s}-> the bias is positive consistently across "
                          f"models; the magnitude is the mean above, not any single CI")
                elif p >= 0.05:
                    print(f"    {'':22s}-> sign is not consistent; do not claim a "
                          f"direction")
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
