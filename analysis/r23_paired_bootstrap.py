#!/usr/bin/env python3
"""
r23_paired_bootstrap.py -- paired cluster bootstrap on detection accuracy, for R#2.3.

WHY THIS EXISTS. r2_3_answer.tex used to report a two-sided p and a one-sided
95% lower bound on the paired accuracy difference (-0.38 pp static, -0.44 pp
mobile). Those were computed on the pre-17-observable campaign, whose paired
differences were +1.69 and +1.56 points. The 17-observable re-learn of 29/8
(STATE 25.166) gives +2.38 and +1.00, so the old interval does not describe the
new estimate and was removed from the answer. This recomputes it.

WHAT IT DOES NOT DO. It does not re-learn anything. It reconstructs each arm's
test split exactly as defense_detection_v2.py made it, loads that arm's stored
model, and bootstraps over test-set CLUSTERS (file_source = one simulation run =
one seed), which is the unit the two arms are paired on.

⛔ TWO HARD CHECKS, AND THE SCRIPT ABORTS ON EITHER.
  1. Reconstruction fidelity: the accuracy recomputed from the reconstructed
     test split must equal the accuracy stored in that arm's results.csv, to
     1e-9. If the reconstruction has drifted from the pipeline, the number is
     not the campaign's number and nothing downstream is meaningful.
  2. Pairing validity: both arms' test splits must cover the SAME set of
     file_source groups. The whole design is "paired on identical seeds"; if the
     two splits disagree, the difference is not paired and no interval computed
     from it is valid.

Reads:  <arm>/listener/colab_data/wide_{mode}.csv.gz
        <arm>/listener/results/{mode}/results.csv
        <arm>/listener/results/{mode}/best_model_*.pkl
Writes: only its own --out-dir.

  python3 r23_paired_bootstrap.py --realistic arms_propmodel_rl42_17 \
      --baseline arms_propmodel_rl42_baseline_17 --mode static
"""
import argparse
import glob
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

HERE = Path(__file__).resolve().parent

# The pipeline decides the feature space; the arm alone is not enough. Same
# lesson as STATE 25.161 -- analysis/pipeline is 33 metrics, pipeline_17 is 17.
PIPELINE = os.environ.get("R23_PIPELINE", str(HERE / "cleanfeat" / "pipeline_17"))


def load_pipeline_module():
    path = os.path.join(PIPELINE, "defense_detection_v2.py")
    if not os.path.isfile(path):
        sys.exit(f"ABORT: no defense_detection_v2.py under {PIPELINE}")
    spec = importlib.util.spec_from_file_location("dd", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["dd"] = mod
    spec.loader.exec_module(mod)
    return mod


def rebuild_test_split(dd, arm_dir, mode):
    """Reconstruct X_test, y_test and the test groups exactly as the pipeline did.

    Uses the pipeline's OWN load/preprocess/engineer functions and its own
    splitter settings, so this cannot drift from the campaign by reimplementation.
    """
    # The campaign loaded the compact wide bundle, not raw CSVs -- prop17.log
    # reads "[1/14] Loading data from bundle: wide_<mode>.csv.gz". That path is
    # selected by DCFM_DATA_BUNDLE, and _load_bundle_tall picks static vs mobile
    # by looking for one of those words inside data_root. Both are required, and
    # without them the loader silently takes the raw-CSV branch instead.
    bundle = arm_dir / "colab_data"
    if not bundle.is_dir():
        sys.exit(f"ABORT: no bundle dir {bundle}")
    os.environ["DCFM_DATA_BUNDLE"] = str(bundle)

    cfg = dd.Config()
    cfg.data_root = str(bundle / mode)   # mode-bearing; only parsed, never opened
    cfg.results_dir = str(arm_dir / "results" / mode)

    det = dd.DefenseDetector(cfg)
    det.verbosity = 0

    tall = det.load_simulation_data_enhanced()
    X, y, groups = det.preprocess_data_enhanced(tall)
    X_eng = det.engineer_advanced_features(X)

    groups_arr = np.array(groups)
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=cfg.test_size, random_state=cfg.random_state
    )
    _, test_idx = next(splitter.split(X_eng, y, groups_arr))
    return X_eng.iloc[test_idx], y.iloc[test_idx], groups_arr[test_idx]


def stored_row(arm_dir, mode, model=None):
    """The arm's headline row, or the row for a named model.

    ⛔ Default is unchanged: the arm's own best row by accuracy. --model exists
    because a paired comparison needs the SAME estimator in both arms, and two
    arms can select different winners; comparing across estimators measures the
    estimator, not the intervention.
    """
    res = pd.read_csv(arm_dir / "listener" / "results" / mode / "results.csv")
    if model:
        hit = res[res["Model"] == model]
        if hit.empty:
            sys.exit(f"ABORT: {arm_dir.name}/{mode} results.csv has no row for "
                     f"model '{model}'")
        return hit.iloc[0]
    return res.sort_values("Accuracy", ascending=False).iloc[0]


def load_model_bundle(arm_dir, mode, model_name):
    """The pickle is a dict, not a bare estimator (defense_detection_v2.py:1307).

    It carries 'model', 'scaler', 'feature_names' and 'threshold' -- the fitted
    scaler and the selected column list, in order. That is what makes this script
    a reconstruction rather than a reimplementation: neither the scaling nor the
    feature selection is re-derived here.
    """
    import joblib

    pattern = str(arm_dir / "listener" / "results" / mode / "best_model_*.pkl")
    hits = glob.glob(pattern)
    if not hits:
        sys.exit(f"ABORT: no best_model_*.pkl under {pattern}")
    if len(hits) > 1:
        sys.exit(f"ABORT: {len(hits)} model pickles under {pattern}; expected 1")
    stem = os.path.basename(hits[0])[len("best_model_"):-len(".pkl")]

    md = joblib.load(hits[0])
    if not isinstance(md, dict):
        sys.exit(f"ABORT: expected a dict in {hits[0]}, got {type(md).__name__}")
    for key in ("model", "scaler", "feature_names", "threshold"):
        if key not in md:
            sys.exit(f"ABORT: {hits[0]} has no '{key}'")

    if stem == model_name:
        return md

    # The requested model is not this arm's winner. The pickle carries every
    # fitted estimator and its tuned threshold, so the comparison is still a
    # reconstruction and not a refit -- but scaler and feature_names are shared,
    # so only 'model' and 'threshold' are swapped.
    all_m = md.get("all_models") or {}
    all_t = md.get("all_thresholds") or {}
    if model_name not in all_m:
        sys.exit(
            f"ABORT: the stored pickle is '{stem}' and carries no fitted "
            f"'{model_name}' in all_models ({len(all_m)} present). Refusing to "
            f"score a model the campaign did not fit.")
    if model_name not in all_t:
        sys.exit(f"ABORT: {hits[0]} has a fitted '{model_name}' but no tuned "
                 f"threshold for it in all_thresholds.")
    md = dict(md)
    md["model"] = all_m[model_name]
    md["threshold"] = all_t[model_name]
    return md


def arm_predictions(dd, arm_name, mode, model=None):
    """Per-window correctness for one arm, with its cluster ids."""
    arm_dir = HERE / arm_name
    if not arm_dir.is_dir():
        sys.exit(f"ABORT: no such arm: {arm_dir}")

    row = stored_row(arm_dir, mode, model)
    md = load_model_bundle(arm_dir, mode, row["Model"])
    model, scaler, cols = md["model"], md["scaler"], list(md["feature_names"])
    X_test, y_test, g_test = rebuild_test_split(dd, arm_dir / "listener", mode)

    # Same two steps the pipeline applies at :1194-1203 -- scale on all engineered
    # columns, then subset to the stored selection.
    n_seen = getattr(scaler, "n_features_in_", None)
    if n_seen is not None and n_seen != X_test.shape[1]:
        sys.exit(f"ABORT: {arm_name}/{mode} scaler was fitted on {n_seen} "
                 f"columns, the rebuilt test frame has {X_test.shape[1]}.")
    X_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )
    missing = [c for c in cols if c not in X_scaled.columns]
    if missing:
        sys.exit(f"ABORT: {arm_name}/{mode} reconstruction is missing "
                 f"{len(missing)} selected columns, e.g. {missing[:5]}")
    X_sel = X_scaled[cols]

    thr = float(md["threshold"])
    proba = model.predict_proba(X_sel)[:, 1]
    pred = (proba >= thr).astype(int)
    correct = (pred == y_test.to_numpy()).astype(float)

    acc = float(correct.mean())
    stored_acc = float(row["Accuracy"])
    if abs(acc - stored_acc) > 1e-9:
        sys.exit(
            f"⛔ ABORT [check 1: reconstruction fidelity] {arm_name}/{mode}: "
            f"recomputed accuracy {acc:.10f} != stored {stored_acc:.10f} "
            f"(diff {acc - stored_acc:+.2e}, n={len(correct)}). The rebuilt "
            f"test split is not the campaign's test split. Do not interpret "
            f"any interval computed from it.")

    return correct, g_test, row


def paired_bootstrap(c_a, c_b, groups, n_boot, seed):
    """Resample CLUSTERS with replacement, the same clusters for both arms."""
    uniq = np.unique(groups)
    idx_by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)

    deltas = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_by_group[g] for g in drawn])
        deltas[b] = c_a[rows].mean() - c_b[rows].mean()
    return deltas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--realistic", required=True, help="realistic-channel arm")
    ap.add_argument("--baseline", required=True, help="its OWN staged baseline")
    ap.add_argument("--mode", choices=["static", "mobile"], required=True)
    ap.add_argument("--n-boot", type=int, default=2000,
                    help="2000, matching the interval this replaces (STATE 25.46)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default=None,
                    help="score BOTH arms with this estimator instead of each "
                         "arm's own winner. Required when the two arms select "
                         "different winners, or the delta mixes intervention "
                         "with estimator choice.")
    ap.add_argument("--out-dir", default=str(HERE / "r23_paired_bootstrap"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dd = load_pipeline_module()
    print(f"pipeline  : {PIPELINE}")
    print(f"pair      : {args.realistic}  vs  {args.baseline}   [{args.mode}]")

    c_r, g_r, row_r = arm_predictions(dd, args.realistic, args.mode, args.model)
    c_b, g_b, row_b = arm_predictions(dd, args.baseline, args.mode, args.model)
    if str(row_r["Model"]) != str(row_b["Model"]):
        sys.exit(
            f"⛔ ABORT [model mismatch] the two arms are scored with different "
            f"estimators: {row_r['Model']} vs {row_b['Model']}. A paired delta "
            f"across estimators measures the estimator, not the intervention. "
            f"Pass --model to fix one for both arms.")
    print(f"  realistic : {row_r['Model']:<20} acc={float(row_r['Accuracy']):.6f} "
          f"thr={float(row_r['Threshold']):.2f}  n={len(c_r)}")
    print(f"  baseline  : {row_b['Model']:<20} acc={float(row_b['Accuracy']):.6f} "
          f"thr={float(row_b['Threshold']):.2f}  n={len(c_b)}")

    # ---- check 2: the pairing is real ------------------------------------
    if set(g_r) != set(g_b):
        only_r, only_b = set(g_r) - set(g_b), set(g_b) - set(g_r)
        sys.exit(
            f"⛔ ABORT [check 2: pairing validity] the two arms' test splits "
            f"cover different runs: {len(only_r)} only in {args.realistic}, "
            f"{len(only_b)} only in {args.baseline}. This comparison is not "
            f"paired and no interval from it is valid.")
    if len(c_r) != len(c_b):
        sys.exit(f"⛔ ABORT [check 2]: same groups but {len(c_r)} vs {len(c_b)} "
                 f"windows. Refusing to align them by assumption.")

    order_r = np.argsort(g_r, kind="stable")
    order_b = np.argsort(g_b, kind="stable")
    c_r, c_b, groups = c_r[order_r], c_b[order_b], g_r[order_r]

    delta = float(c_r.mean() - c_b.mean())
    deltas = paired_bootstrap(c_r, c_b, groups, args.n_boot, args.seed)

    lo95_1s = float(np.percentile(deltas, 5))          # one-sided 95% lower bound
    ci = [float(np.percentile(deltas, 2.5)),
          float(np.percentile(deltas, 97.5))]
    # two-sided bootstrap p for H0: delta = 0, by the interval-inversion rule
    p_two = float(2 * min((deltas <= 0).mean(), (deltas >= 0).mean()))
    p_two = min(1.0, p_two)

    res = {
        "realistic": args.realistic, "baseline": args.baseline, "mode": args.mode,
        "n_windows": int(len(c_r)), "n_clusters": int(len(np.unique(groups))),
        "n_boot": args.n_boot, "seed": args.seed,
        "acc_realistic": float(c_r.mean()), "acc_baseline": float(c_b.mean()),
        "delta": delta, "delta_pp": delta * 100,
        "one_sided_95_lower_pp": lo95_1s * 100,
        "ci95_pp": [ci[0] * 100, ci[1] * 100],
        "p_two_sided": p_two,
        "model_realistic": str(row_r["Model"]), "model_baseline": str(row_b["Model"]),
    }

    print(f"\n  delta            : {delta*100:+.2f} pp")
    print(f"  one-sided 95% LB : {lo95_1s*100:+.2f} pp")
    print(f"  95% CI           : [{ci[0]*100:+.2f}, {ci[1]*100:+.2f}] pp")
    print(f"  two-sided p      : {p_two:.4f}")
    print(f"  clusters         : {res['n_clusters']}  windows: {res['n_windows']}")
    print("\n  ⚠️  A delta inside the +/-1.4 pp paired noise floor (STATE 25.33) is "
          "INDISTINGUISHABLE,\n      not 'restored' or 'improved'.")

    # 2/9: use the arm NAME, not the raw argument -- an absolute --realistic
    # path was interpolated whole, producing a nested path that does not exist,
    # losing the result after the bootstrap had already finished.
    p = out_dir / f"r23_paired_{Path(args.realistic).name}_{args.mode}.json"
    p.write_text(json.dumps(res, indent=2))
    print(f"\n  wrote {p}")


if __name__ == "__main__":
    main()
