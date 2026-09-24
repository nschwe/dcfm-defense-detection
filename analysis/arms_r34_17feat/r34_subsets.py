#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
r34_subsets.py -- R#3.4: is it the features, or just how few of them there are?

The reviewer's charge: the selected set may work because using few features
reduces overfitting, not because those particular features carry the signal.
Two controls are asked for -- random same-size subsets, and a set chosen on
in-domain performance alone.

WHAT THIS SCRIPT DOES
    Enumerates EVERY subset of the listener's 17 observables at each requested
    size (and samples the larger sizes), evaluates each one cross-domain under
    the published protocol, and reports where the selected set falls in that
    distribution. Exhaustive enumeration rather than N random draws: there is
    no N to justify, and the answer is a percentile rather than a mean.

THE POOL IS 17
    arm_spec.LISTENER holds 21 canonical columns; paper_names.py maps exactly 4
    of them to None as duplicates sharing one obs: expression. The remaining 17
    are the space (STATE 25.89.17, 25.89.23).

    Two of the 17 -- MidMessageRate, HnaMessageRate -- are constant zero in both
    configurations (no MID/HNA traffic in these scenarios). They are NOT removed
    from the pool: 17 is the declared space and curating it would be exactly the
    kind of unjustified narrowing this experiment exists to rule out. Instead
    both populations are reported:

        all      -- every subset of the 17
        nodead   -- only subsets containing neither constant

    At k=3 that is 680 and 455. The 225 subsets in between contain a constant
    column, which no classifier can use, so they behave as (k-1)-subsets and
    drag the baseline down. Reporting only "all" would flatter the selected set;
    reporting only "nodead" would be a curated pool. Hence both.

PROTOCOL
    ksw.evaluate_seed, unchanged, so every number is comparable with the
    published K sweep: scaler and classifier fitted on the source TRAIN split,
    threshold from source VALIDATION, the same fitted model applied to the
    target. No refit, no target adaptation.

PARALLELISM
    --workers processes, each fitting with --fit-threads CatBoost threads, so
    total threads = workers x fit-threads and is chosen deliberately instead of
    inherited from evaluate_seed's hardcoded n_jobs=16. Each worker loads the
    bundles once (initializer), then receives only (k, coverage, combo) tasks.
    Partial results are flushed to partial.csv every --flush rows, so a crash
    loses minutes, not the run.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

# The selected set under test.
SELECTED = ["AverageAdvertisedLinksPerTCMessage", "AverageMprCount",
            "TcMessageRate"]
# Constant in both modes; kept in the pool, tracked separately. See the header.
DEAD = ["MidMessageRate", "HnaMessageRate"]

HP_DIR = os.environ.get(
    "DCFM_HP_RESULTS",
    os.path.expanduser("~/ns3/Final_Project_NS3-master/strict_observable_v2/"
                       "colab_data/results_cache/hp_search_extended"))

# Worker-process state, set once by _init_worker.
_G: dict = {}


def _imports():
    """Import the pipeline modules with the isolated tree first on the path."""
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(HERE / "pipeline_17"))
    os.environ["DCFM_DATA_BUNDLE"] = str(HERE / "listener" / "colab_data")
    import feature_importance_sensitivity_v2 as fis   # noqa: E402
    import k_sweep_universal4_v2 as ksw               # noqa: E402
    from paper_names import PAPER_NAMES               # noqa: E402
    return fis, ksw, PAPER_NAMES


def pool_features(paper_names) -> list[str]:
    """The 17, derived -- never hand-written."""
    p = [c for c in paper_names if paper_names[c][0] is not None]
    if len(p) != 17:
        sys.exit(f"expected 17 observables, derived {len(p)}")
    return p


def _init_worker(fit_threads: int) -> None:
    """Load data and params once per worker; cap CatBoost's threads.

    evaluate_seed hardcodes n_jobs=16 in its build_catboost calls; with several
    workers that would be workers x 16 threads. Wrapping build_catboost here
    forces every fit in this process to fit_threads instead.
    """
    fis, ksw, PAPER_NAMES = _imports()
    orig_build = ksw.build_catboost
    ksw.build_catboost = (
        lambda best_params, seed, n_jobs=16:
        orig_build(best_params, seed=seed, n_jobs=fit_threads))

    Xs, ys, gs = fis.to_wide(fis.load_dataset("./features_static"))
    Xm, ym, gm = fis.to_wide(fis.load_dataset("./features_mobile"))
    bp = ksw.load_best_params(HP_DIR)
    _G.update(ksw=ksw, Xs=Xs, ys=ys, gs=gs, Xm=Xm, ym=ym, gm=gm,
              bp_s=bp["static"], bp_m=bp["mobile"])


def _eval_one(task) -> dict:
    """Evaluate one subset: n_seeds seeds, both directions.

    Subsets consisting ONLY of constant columns ({Mid}, {Hna} at k=1 and
    {Mid,Hna} at k=2 -- three in total) cannot be fitted at all: CatBoost
    raises "All features are either constant or ignored". A classifier over
    constants IS the majority-class predictor, so these are recorded at 0.5
    (the classes are balanced 4000/4000 in every test split) and flagged
    degenerate, rather than skipped -- skipping would silently curate the
    population.
    """
    from _catboost import CatBoostError
    k, coverage, combo, n_seeds = task
    ksw = _G["ksw"]
    f = list(combo)
    degenerate = False
    s2m, m2s, in_s, in_m = [], [], [], []
    try:
        for seed in range(n_seeds):
            rs = ksw.evaluate_seed(_G["Xs"], _G["ys"], _G["gs"],
                                   _G["Xm"], _G["ym"], f, _G["bp_s"], seed)
            rm = ksw.evaluate_seed(_G["Xm"], _G["ym"], _G["gm"],
                                   _G["Xs"], _G["ys"], f, _G["bp_m"], seed)
            s2m.append(rs["acc_cross"]); in_s.append(rs["acc_in"])
            m2s.append(rm["acc_cross"]); in_m.append(rm["acc_in"])
    except CatBoostError:
        degenerate = True
        s2m, m2s, in_s, in_m = [0.5], [0.5], [0.5], [0.5]
    return {
        "k": k,
        "features": ";".join(f),
        "s2m": float(np.mean(s2m)),
        "m2s": float(np.mean(m2s)),
        "cross_mean": float((np.mean(s2m) + np.mean(m2s)) / 2),
        # in-domain accuracy on the SOURCE TEST split -- recorded so the
        # reviewer's baseline (ii), the subset chosen purely on in-domain
        # performance, falls out of this same enumeration with zero extra fits.
        "in_static": float(np.mean(in_s)),
        "in_mobile": float(np.mean(in_m)),
        "coverage": coverage,
        "has_dead": any(d in f for d in DEAD),
        "degenerate": degenerate,
        "is_selected": sorted(f) == sorted(SELECTED),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-list", default="1,2,3,4,5",
                    help="subset sizes to enumerate exhaustively")
    ap.add_argument("--k-sample", default="6,7,8",
                    help="sizes covered by random sampling instead of full "
                         "enumeration (cost only; the small-k end carries the "
                         "information -- at large k almost every subset contains "
                         "the strong features and the distribution collapses)")
    ap.add_argument("--sample-n", type=int, default=500,
                    help="subsets drawn per sampled size")
    ap.add_argument("--sample-rng", type=int, default=20250824,
                    help="RNG seed for the sampled sizes, fixed for "
                         "reproducibility")
    ap.add_argument("--n-seeds", type=int, default=3,
                    help="3 to screen, 20 to report")
    ap.add_argument("--workers", type=int, default=6,
                    help="worker processes over subsets")
    ap.add_argument("--fit-threads", type=int, default=2,
                    help="CatBoost threads per fit, per worker "
                         "(total threads = workers x fit-threads)")
    ap.add_argument("--flush", type=int, default=500,
                    help="write partial.csv every N completed subsets")
    ap.add_argument("--out", default=str(HERE / "r34_subsets"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    k_list = [int(k) for k in args.k_list.split(",")] if args.k_list else []
    k_sample = [int(k) for k in args.k_sample.split(",")] if args.k_sample else []

    _fis, _ksw, PAPER_NAMES = _imports()
    feats = pool_features(PAPER_NAMES)

    # (size, coverage, subsets): exhaustive for k_list, sampled for k_sample
    rng = np.random.default_rng(args.sample_rng)
    jobs = []
    for k in k_list:
        jobs.append((k, "exhaustive", list(itertools.combinations(feats, k))))
    for k in k_sample:
        n = min(args.sample_n, math.comb(len(feats), k))
        seen = set()
        while len(seen) < n:
            seen.add(tuple(sorted(rng.choice(len(feats), size=k, replace=False))))
        jobs.append((k, "sampled",
                     [tuple(feats[i] for i in idx) for idx in sorted(seen)]))

    tasks = [(k, cov, combo, args.n_seeds)
             for k, cov, combos in jobs for combo in combos]
    total = len(tasks)
    print(f"pool      : {len(feats)} observables")
    print(f"exhaustive: {k_list}   sampled: {k_sample} (n={args.sample_n})")
    print(f"subsets   : {total}")
    print(f"seeds     : {args.n_seeds}")
    print(f"fits      : {total * args.n_seeds * 2} (both directions)")
    print(f"workers   : {args.workers} x {args.fit_threads} threads "
          f"= {args.workers * args.fit_threads} total\n", flush=True)

    rows = []
    partial = out / "partial.csv"
    if args.workers <= 1:
        _init_worker(args.fit_threads)
        for i, t in enumerate(tasks, 1):
            rows.append(_eval_one(t))
            if i % 50 == 0:
                print(f"  {i}/{total}  ({100*i/total:.1f}%)", flush=True)
            if i % args.flush == 0:
                pd.DataFrame(rows).to_csv(partial, index=False)
    else:
        with ProcessPoolExecutor(max_workers=args.workers,
                                 initializer=_init_worker,
                                 initargs=(args.fit_threads,)) as ex:
            futures = [ex.submit(_eval_one, t) for t in tasks]
            for i, fut in enumerate(as_completed(futures), 1):
                rows.append(fut.result())
                if i % 50 == 0:
                    print(f"  {i}/{total}  ({100*i/total:.1f}%)", flush=True)
                if i % args.flush == 0:
                    pd.DataFrame(rows).to_csv(partial, index=False)

    df = pd.DataFrame(rows).sort_values(["k", "features"]).reset_index(drop=True)
    tag = "_".join(map(str, k_list + k_sample))
    csv = out / f"subsets_k{tag}_s{args.n_seeds}.csv"
    df.to_csv(csv, index=False)
    print(f"\nwrote {csv}\n")

    # ---- where does the selected set fall -----------------------------------
    summary = {}
    for label, sub in (("all", df), ("nodead", df[~df.has_dead])):
        print("=" * 70)
        print(f"POPULATION: {label}   ({len(sub)} subsets)")
        for k in sorted(sub.k.unique()):
            d = sub[sub.k == k]
            if d.empty:
                continue
            sel = d[d.is_selected]
            line = (f"  k={k:<3} n={len(d):<6} "
                    f"best={d.cross_mean.max():.4f} "
                    f"mean={d.cross_mean.mean():.4f} "
                    f"sd={d.cross_mean.std():.4f}")
            if not sel.empty:
                v = float(sel.cross_mean.iloc[0])
                rank = int((d.cross_mean > v).sum()) + 1
                pct = 100.0 * (d.cross_mean < v).mean()
                line += (f"  |  SELECTED={v:.4f} "
                         f"rank {rank}/{len(d)}  percentile {pct:.1f}")
                summary[f"{label}_k{k}"] = {
                    "selected": v, "rank": rank, "n": len(d),
                    "percentile": pct, "best": float(d.cross_mean.max()),
                    "mean": float(d.cross_mean.mean()),
                    "sd": float(d.cross_mean.std()),
                }
            print(line)
        print()

    # ---- reviewer baseline (ii): best subset by IN-DOMAIN accuracy ---------
    print("=" * 70)
    print("BASELINE (ii): subset chosen purely on in-domain performance,")
    print("then evaluated cross-domain (per size, per direction; nodead pop.)")
    nod = df[~df.has_dead]
    for k in sorted(nod.k.unique()):
        d = nod[nod.k == k]
        bs = d.loc[d.in_static.idxmax()]   # chosen on static in-domain
        bm = d.loc[d.in_mobile.idxmax()]   # chosen on mobile in-domain
        print(f"  k={k}:")
        print(f"    static-chosen : in={bs.in_static:.4f}  s2m={bs.s2m:.4f}"
              f"  [{bs.features}]")
        print(f"    mobile-chosen : in={bm.in_mobile:.4f}  m2s={bm.m2s:.4f}"
              f"  [{bm.features}]")
        summary[f"indomain_k{k}"] = {
            "static_chosen": bs.features, "static_in": float(bs.in_static),
            "static_s2m": float(bs.s2m),
            "mobile_chosen": bm.features, "mobile_in": float(bm.in_mobile),
            "mobile_m2s": float(bm.m2s),
        }
    print()
    (out / f"summary_s{args.n_seeds}.json").write_text(json.dumps(summary, indent=2))
    print("NOTE: 'all' includes subsets carrying a constant column, which behave")
    print("as (k-1)-subsets. 'nodead' excludes them. The selected set's standing")
    print("in BOTH is the honest report -- neither alone.")


if __name__ == "__main__":
    main()
