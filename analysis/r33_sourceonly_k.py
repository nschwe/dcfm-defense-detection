#!/usr/bin/env python3
"""
r33_sourceonly_k.py -- closing the second half of Reviewer 3, Comment 3.3.

WHAT THE FIRST CONTROL LEFT OPEN. r33_sourceonly.py showed that the FEATURE
subset does not depend on the cross-domain intersection: derived from one
domain alone it comes out the same. It held K fixed, so it said nothing about
the other channel -- K itself is chosen on cross-domain accuracy over the full
target (k_sweep_universal4_v2.py has no held-out target fold), which is the same
objection one level up.

WHAT THIS SCRIPT DOES. It derives BOTH the subset and K from the source domain
alone, then evaluates on the target with everything frozen:

    1. split the SOURCE 60/20/20, grouped by file_source (the campaign's own
       split function, imported, not reimplemented);
    2. rank features on the TRAINING split of each seed (--rank-on-train, the
       primary arm) or from the campaign cache (--no flag, the diagnostic arm;
       that cache was computed over the full source and so saw the validation
       rows -- a source-internal limitation, not target leakage);
    3. for each candidate K, build the source-only subset, train on
       source-train, and score on source-VALIDATION -- the target is not read;
    4. pick K by the paper's own rule: the smallest K whose mean validation
       accuracy is within one standard deviation of the best;
    5. freeze (subset, K, threshold) and evaluate on the target through the
       EXACT protocol of k_sweep_universal4_v2.evaluate_seed: scaler and
       classifier fitted on train only, threshold from validation, the same
       fitted model applied to the target. No refit on train+val -- adding one
       (an early draft did) makes the numbers incomparable with the published
       0.8730 / 0.8632.

Per seed, the locked source-derived model, subset and threshold are applied to
the target with no target-domain adaptation and no model selection. With N
seeds the target is scored N times; what matters is that no target result ever
feeds back into a decision. Accuracy AND AUC are reported: AUC is independent
of the transferred threshold, so if it holds up the transfer is of the
separability itself and not of a well-travelled cut.

⛔ IT DOES NOT REPLACE THE PUBLISHED PIPELINE. It is a control run beside it.
The published selection is unchanged; this shows what happens without it.

Classifier: CatBoost with the campaign's tuned parameters -- the same classifier
stage 5 used for the published K sweep, so the comparison is like-for-like.

Reads: the arm's bundle and importance caches. Writes: only its own out-dir.

    python3 r33_sourceonly_k.py --direction s2m
    python3 r33_sourceonly_k.py --direction m2s
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib as jl
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
# 28/8: this line pinned the 33-metric modules --
#   analysis/pipeline           -> METRICS = 33
#   <arm>/pipeline_17           -> METRICS = 17
# The arm alone is not enough: the pipeline decides the space. Overridable
# now; unset reproduces the 24/8 run exactly.
R33_PIPELINE = os.environ.get("R33_PIPELINE", str(HERE / "pipeline"))
sys.path.insert(0, R33_PIPELINE)
import feature_importance_sensitivity_v2 as fis  # noqa: E402
import k_sweep_universal4_v2 as ksw              # noqa: E402

# ⛔ RULE: any difference from k_sweep_universal4_v2.py beyond the data
# boundary of feature selection is a BUG in this control, not an improvement.
# That is why the evaluation below imports ksw's own split, threshold,
# classifier-construction and params-loading functions instead of using the
# same-named ones in feature_importance_sensitivity_v2 -- the two modules
# differ in interface (ksw.load_best_params returns the catboost dict directly,
# fis wraps it per model and would silently fall back to default parameters),
# and mixing them is exactly the silent divergence the rule forbids.
# fis is used ONLY for: load_dataset/to_wide (data), importance_one_seed
# (the ranking whose data boundary IS the experiment).

# 28/8: the arm was hardcoded to arms_c10k_stackcal, which is
# the superseded 33-metric campaign -- analysis/r33_full.log records
# "source (40000, 33)". The control is needed on the reported campaign.
# Overridable now; the default reproduces the 24/8 run exactly.
ARM_NAME = os.environ.get("R33_ARM", "arms_c10k_stackcal")
ARM = HERE / ARM_NAME / "listener"
# Refuse to run in a space other than the one asked for. 0 = no gate
# (the default, i.e. the old behaviour). A file-exists test is not a gate.
EXPECT_COLS = int(os.environ.get("R33_EXPECT_COLS", "0"))
CAMPAIGN = ARM / "results" / "feature_importance_sensitivity_v2"
CACHE_DIR = CAMPAIGN / "importance_cache"
BUNDLE = ARM / "colab_data"
# Same default run_arm.py uses (run_arm.py:50-53), so this control is fitted
# with the campaign's own tuned hyperparameters and not with library defaults.
HP_DIR = os.environ.get(
    "DCFM_HP_RESULTS",
    os.path.join(HERE, "hp_selected"))   # 24/9: {static,mobile}/best_params.json

METHODS = ["rf_native", "xgb_gain", "catboost_pvc",
           "perm_rf", "perm_xgb", "perm_catboost"]
THRESHOLD = 0.8
# The revised analysis's cross-domain transfer at K=3, read from
# arms_r34_17feat/listener/results/k_sweep_universal4_v2/k_sweep_summary.txt
# (25/8 22:00), row "3 | 0.9032 | 0.9291 | 0.8730 | 0.8632 | 0.0098".
# 3/9 CORRECTION: this constant read 0.8720 / 0.8619 and had gone stale against
# that artefact. Nothing computed here depends on it -- it is reported as
# `published_reference_acc` and differenced -- but r3_3 quoted the field, so the
# letter carried the stale pair.
# WARNING: a hard-coded reference cannot go stale loudly. Every consistency
# sweep we run compares the letter against artefacts, and this value is not one.
# If the k-sweep is ever re-run, update this line in the same commit.
PUBLISHED = {"s2m": 0.8730, "m2s": 0.8632}


def build_single(k, all_imp, threshold, variant="soft"):
    """Single-domain twin of fis.build_feature_set (see r33_sourceonly.py)."""
    n = len(all_imp)
    counts: dict[str, int] = {}
    for imp in all_imp:
        for f in imp.nlargest(k).index:
            counts[f] = counts.get(f, 0) + 1
    stable = [f for f, c in counts.items() if c / n >= threshold]
    if variant == "strict":
        return sorted(stable)
    avg = sum(all_imp) / n
    if len(stable) >= k:
        return sorted(avg.loc[stable].sort_values(ascending=False).head(k).index)
    feats = list(stable)
    for f in avg.sort_values(ascending=False).index:
        if f not in feats:
            feats.append(f)
        if len(feats) >= k:
            break
    return sorted(feats)


def load_domain(mode: str):
    os.environ["DCFM_DATA_BUNDLE"] = str(BUNDLE)
    tall = fis.load_dataset(f"./features_{mode}")
    X, y, g = fis.to_wide(tall)
    if EXPECT_COLS and X.shape[1] != EXPECT_COLS:
        raise SystemExit(
            f"ABORT: {ARM_NAME}/{mode} loaded {X.shape[1]} columns, "
            f"expected {EXPECT_COLS}. Wrong arm or wrong space.")
    return X, y, g


def evaluate_seed(features, X_src, y_src, g_src, X_tgt, y_tgt, bp, seed):
    """One seed, through k_sweep_universal4_v2's OWN functions.

    The body mirrors ksw.evaluate_seed (lines 286-320) line for line, and every
    protocol component -- the 60/20/20 grouped split, the scaler fitted on the
    TRAIN split only, the classifier construction (tuned catboost params,
    n_jobs=16), the threshold rule on VALIDATION -- is ksw's own code, imported.
    The very same fitted model is then applied to the target. There is NO refit
    on train+val: an earlier draft added one, which silently made its numbers
    incomparable with the published 0.8730 / 0.8632.

    The one extension over ksw.evaluate_seed is acc_val in the return value,
    which the K-selection step needs; ksw computes the same quantity internally
    for its threshold search and simply does not return it.

    The source TEST partition is used for the in-domain figures only; it takes
    no part in any selection.
    """
    tr, va, te = ksw.split_source_60_20_20(X_src, y_src, g_src, seed)

    Xs = X_src[features]
    Xt = X_tgt[features]

    sc = StandardScaler().fit(Xs.iloc[tr])
    X_tr_s = sc.transform(Xs.iloc[tr])
    X_va_s = sc.transform(Xs.iloc[va])
    X_te_s = sc.transform(Xs.iloc[te])
    X_tgt_s = sc.transform(Xt)

    clf = ksw.build_catboost(bp, seed=seed, n_jobs=16)
    clf.fit(X_tr_s, y_src.iloc[tr])

    p_va = clf.predict_proba(X_va_s)[:, 1]
    thr = ksw.select_threshold(y_src.iloc[va], p_va)

    p_te = clf.predict_proba(X_te_s)[:, 1]
    p_tgt = clf.predict_proba(X_tgt_s)[:, 1]

    return {
        "acc_val": float(accuracy_score(y_src.iloc[va], (p_va >= thr).astype(int))),
        "acc_in": float(accuracy_score(y_src.iloc[te], (p_te >= thr).astype(int))),
        "auc_in": float(roc_auc_score(y_src.iloc[te], p_te)),
        "acc_cross": float(accuracy_score(y_tgt, (p_tgt >= thr).astype(int))),
        "auc_cross": float(roc_auc_score(y_tgt, p_tgt)),
        "thr": float(thr),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", choices=["s2m", "m2s"], required=True)
    ap.add_argument("--method", default="perm_catboost",
                    help="importance method for the source-only subset")
    # FULL range 1..21, not the published sweep's sparse grid
    # {1,2,3,4,5,7,9,13,17,21}. That grid is residue from an earlier round
    # rather than a deliberate design choice, so there is nothing to be faithful
    # to; preserving it would only carry an arbitrary constraint into the
    # control. Sweeping every value removes it. Ranking cost is unaffected --
    # it is computed once per seed, independently of K.
    #
    # ⚠️ The published K=3 was obtained WITH the sparse grid. If the full sweep
    # returns a different K, that is a gap against the paper caused by grid
    # granularity and NOT by the circularity question this control is about.
    # Report it as such; do not let the two be conflated.
    #
    # Restricting the candidates around the known answer would in any case make
    # "K was selected" vacuous -- the control must be free to return another K.
    ap.add_argument("--k-list",
                    default=",".join(str(k) for k in range(1, 22)))
    # 20, matching the published K sweep, which used 20 seeds per K.
    # An earlier default of 10 was an unforced deviation.
    ap.add_argument("--n-seeds", type=int, default=20)
    ap.add_argument("--out-dir", default=str(HERE / "r33_sourceonly"))
    ap.add_argument("--rank-on-train", action="store_true",
                    help="compute the importance ranking from the TRAINING "
                         "split of each seed instead of reading the campaign "
                         "cache (which was computed over the full source and "
                         "therefore saw the validation rows). This is the only "
                         "difference from the first run: no selection rule, "
                         "no K rule, no threshold rule changes -- only the data "
                         "boundary the ranking is computed on. Changing "
                         "anything else after the first result was seen would "
                         "reintroduce information through the protocol.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    if str(out_dir).startswith(str(CAMPAIGN)):
        sys.exit(f"ABORT: refusing to write inside the campaign tree: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    src_mode, tgt_mode = ("static", "mobile") if args.direction == "s2m" \
        else ("mobile", "static")
    k_list = [int(x) for x in args.k_list.split(",")]

    print(f"arm       : {ARM_NAME}  (expect_cols={EXPECT_COLS})")
    print(f"pipeline  : {R33_PIPELINE}")
    print(f"direction : {args.direction}  ({src_mode} -> {tgt_mode})")
    print(f"method    : {args.method}")
    print(f"seeds     : {args.n_seeds}\n")

    if args.rank_on_train:
        imp_src = None          # computed per seed, inside the K loop
        print("ranking: computed per seed on the TRAINING split only\n")
    else:
        cache = jl.load(CACHE_DIR / f"importance_{args.method}.pkl")
        imp_src = cache[src_mode]
        print("ranking: campaign cache (computed over the full source)\n")

    print(f"loading {src_mode} ...")
    Xs, ys, gs = load_domain(src_mode)
    print(f"loading {tgt_mode} ...")
    Xt, yt, gt = load_domain(tgt_mode)
    print(f"  source {Xs.shape}, target {Xt.shape}\n")

    # Each module gets parameters in ITS OWN shape, exactly as its own campaign
    # run consumed them: the ranking is fis machinery (like the campaign cache),
    # so it takes fis.load_best_params' per-model dict; the evaluation is ksw
    # machinery, so it takes ksw.load_best_params' already-extracted catboost
    # dict. Bridging one into the other is how default-parameter fallbacks
    # happen silently.
    bp_rank = fis.load_best_params(HP_DIR)[src_mode]
    bp_eval = ksw.load_best_params(HP_DIR)[src_mode]

    if args.rank_on_train:
        # One ranking per seed, from that seed's TRAINING rows only -- the same
        # split function the evaluation uses, so "training rows" means the same
        # rows in both places. The list is then consumed exactly as the cached
        # list is: build_single counts, over seeds, how often a feature is in
        # that seed's top-K. Nothing else about the procedure changes.
        print("computing per-seed importance on training splits ...")
        imp_src = []
        for seed in range(args.n_seeds):
            tr, _va, _te = ksw.split_source_60_20_20(Xs, ys, gs, seed)
            imp_src.append(fis.importance_one_seed(
                args.method, Xs.iloc[tr], ys.iloc[tr], bp_rank, seed))
            print(f"    seed {seed} done")
        print()

    # ---- step 2: K chosen on a held-out SOURCE split; target never read -----
    print("K selection on the source domain only")
    print(f"  {'K':<4} {'features':<62} {'val acc':<9} sd")
    per_k = {}
    for k in k_list:
        feats = build_single(k, imp_src, THRESHOLD)
        accs = []
        for seed in range(args.n_seeds):
            # Only acc_val is read here. The same call also computes target
            # figures, which are DISCARDED at this stage -- K must not be
            # chosen with any knowledge of them.
            r = evaluate_seed(feats, Xs, ys, gs, Xt, yt, bp_eval, seed)
            accs.append(r["acc_val"])
        per_k[k] = {"features": feats,
                    "val_mean": float(np.mean(accs)),
                    "val_sd": float(np.std(accs))}
        print(f"  {k:<4} {';'.join(feats)[:60]:<62} "
              f"{per_k[k]['val_mean']:.4f}    {per_k[k]['val_sd']:.4f}")

    # the paper's rule: smallest K within one SD of the best
    best_k = max(per_k, key=lambda k: per_k[k]["val_mean"])
    tol = per_k[best_k]["val_mean"] - per_k[best_k]["val_sd"]
    chosen_k = min(k for k in k_list if per_k[k]["val_mean"] >= tol)
    print(f"\n  best K={best_k} ({per_k[best_k]['val_mean']:.4f}), "
          f"tolerance {tol:.4f} -> chosen K={chosen_k}")
    chosen_feats = per_k[chosen_k]["features"]
    print(f"  locked subset: {chosen_feats}\n")

    # ---- evaluation: the locked model is applied to the target -------------
    # Per seed, the locked subset, K, scaler, classifier and threshold -- all
    # fixed on the source -- are applied to the target with no target-domain
    # adaptation and no model selection. With 10 seeds the target is scored 10
    # times, by 10 source-derived models; that is not "reading it once", and
    # saying so would invite a correction. What matters is that no target
    # result feeds back into any decision.
    print("evaluation on the target: locked source-derived models applied "
          "with no target adaptation")
    acc_cross, auc_cross, acc_in, auc_in = [], [], [], []
    for seed in range(args.n_seeds):
        r = evaluate_seed(chosen_feats, Xs, ys, gs, Xt, yt, bp_eval, seed)
        acc_cross.append(r["acc_cross"])
        auc_cross.append(r["auc_cross"])
        acc_in.append(r["acc_in"])
        auc_in.append(r["auc_in"])

    def ms(v):
        return float(np.mean(v)), float(np.std(v))

    a_mean, a_sd = ms(acc_cross)
    u_mean, u_sd = ms(auc_cross)
    ai_mean, ai_sd = ms(acc_in)
    ui_mean, ui_sd = ms(auc_in)
    pub = PUBLISHED[args.direction]

    print(f"  zero-shot accuracy : {a_mean:.4f} +/- {a_sd:.4f}   "
          f"(published {pub:.4f}, difference {a_mean - pub:+.4f})")
    # AUC does not depend on the transferred threshold, so if it holds up the
    # transfer is of the separability itself and not of a well-travelled cut.
    print(f"  zero-shot AUC      : {u_mean:.4f} +/- {u_sd:.4f}")
    print(f"  in-domain accuracy : {ai_mean:.4f} +/- {ai_sd:.4f}  "
          f"(source test split, reporting only)")
    print(f"  in-domain AUC      : {ui_mean:.4f} +/- {ui_sd:.4f}")

    res = {"direction": args.direction, "method": args.method,
           "rank_on_train": bool(args.rank_on_train),
           "n_seeds": args.n_seeds, "k_candidates": k_list,
           "chosen_k": chosen_k, "chosen_features": chosen_feats,
           "source_val_by_k": {str(k): per_k[k] for k in per_k},
           "zero_shot_acc_mean": a_mean, "zero_shot_acc_sd": a_sd,
           "zero_shot_auc_mean": u_mean, "zero_shot_auc_sd": u_sd,
           "in_domain_acc_mean": ai_mean, "in_domain_acc_sd": ai_sd,
           "in_domain_auc_mean": ui_mean, "in_domain_auc_sd": ui_sd,
           "published_reference_acc": pub, "difference": a_mean - pub}
    tag = "trainrank" if args.rank_on_train else "cacherank"
    p = out_dir / f"sourceonly_k_{args.direction}_{tag}.json"
    p.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {p}")
    # The methodological requirement is NOT "we read the target once" -- one may
    # legitimately compute accuracy, AUC and F1 on it. What matters is that no
    # result obtained on the target ever fed back into a decision.
    print("\nNOTE: the target domain was used exclusively for final evaluation.")
    print("It did not enter feature ranking, feature identity, the choice of K,")
    print("model fitting, or thresholding -- all of which were fixed on the")
    print("source domain and frozen before the target was touched.")
    print("Within the source, K and the threshold were chosen on the VALIDATION")
    print("split (the test split is never used here).")
    if args.rank_on_train:
        print("RANKING: computed per seed on that seed's TRAINING rows only, so")
        print("the source side is separated as well. This is the primary arm.")
    else:
        print("KNOWN LIMITATION: the importance ranking is inherited from the")
        print("campaign cache, which computes it over the FULL source domain, so")
        print("it saw the rows later used for validation. This is a property of")
        print("the published pipeline and is source-internal; it does not affect")
        print("the target-side separation this control is about. This is the")
        print("diagnostic arm -- the primary arm is --rank-on-train.")


if __name__ == "__main__":
    main()
