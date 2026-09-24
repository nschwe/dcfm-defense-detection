#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dump_hp_params.py -- the selected hyperparameters, as a parameter file.

The analyses that need tuned hyperparameters (k_sweep_universal4_v2,
feature_importance_sensitivity_v2, feature_eng_ablation_v2,
instability_ablation_v2, r45_confusion_universal3_t16, catboost_retune) read
them with --hp-results-dir <dir>, expecting <dir>/{static,mobile}/. Until
24/9 that directory held best_models.pkl, a pickle of fitted estimators from
the search; the loaders only ever called get_params() on them. This script
writes what they use and nothing else:

    <out>/static/best_params.json     {model_name: {parameter: value}}
    <out>/mobile/best_params.json

for every model in the pickle, with every get_params() entry whose value is a
JSON value (int, float, str, bool, None, or a list of those). Entries whose
value is an object (a base estimator, for instance) are listed in
<out>/<config>/best_params.dropped.txt and not written; no analysis reads
them -- each loader keeps only the numeric parameters of its own model.
The loaders read best_params.json when it is present, and the pickle
otherwise (see load_best_params in each script).

Usage:
    python dump_hp_params.py --hp-results-dir <dir with static/ mobile/ best_models.pkl> \
                             --out analysis/hp_selected
The output directory must not already contain a best_params.json. After
writing, the file is read back and every kept parameter is compared with the
pickle again; the exit status is non-zero on any difference.
"""
import argparse
import json
import os
import sys
import warnings

import joblib

JSON_SCALARS = (int, float, str, bool, type(None))


def _jsonable(v):
    """Return (True, value) when v is representable exactly in JSON."""
    try:
        import numpy as np
        if isinstance(v, np.generic):
            v = v.item()
    except ImportError:
        pass
    if isinstance(v, JSON_SCALARS):
        return True, v
    if isinstance(v, (list, tuple)):
        out = []
        for x in v:
            ok, y = _jsonable(x)
            if not ok:
                return False, None
            out.append(y)
        return True, out
    return False, None


def params_of(entry):
    if hasattr(entry, "get_params"):
        return entry.get_params()
    if isinstance(entry, dict):
        for k in ("best_params", "params"):
            if k in entry:
                return entry[k]
        return entry
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hp-results-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    bad = 0
    for cfg in ("static", "mobile"):
        pkl = os.path.join(args.hp_results_dir, cfg, "best_models.pkl")
        if not os.path.exists(pkl):
            sys.exit(f"missing: {pkl}")
        outdir = os.path.join(args.out, cfg)
        js = os.path.join(outdir, "best_params.json")
        if os.path.exists(js):
            sys.exit(f"refusing to overwrite {js}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")      # sklearn version warnings on unpickle
            d = joblib.load(pkl)
        kept, dropped = {}, {}
        for name, entry in d.items():
            p = params_of(entry)
            kept[name] = {}
            for k, v in p.items():
                ok, jv = _jsonable(v)
                if ok:
                    kept[name][k] = jv
                else:
                    dropped.setdefault(name, []).append(f"{k} = {type(v).__name__}")
        os.makedirs(outdir, exist_ok=True)
        with open(js, "w") as fh:
            json.dump(kept, fh, indent=2, sort_keys=True)
        with open(os.path.join(outdir, "best_params.dropped.txt"), "w") as fh:
            for name in sorted(dropped):
                for item in dropped[name]:
                    fh.write(f"{name}: {item}\n")
        # read back and compare with the pickle
        with open(js) as fh:
            back = json.load(fh)
        for name, entry in d.items():
            p = params_of(entry)
            for k, v in back[name].items():
                ok, jv = _jsonable(p[k])
                same = (jv == v) or (isinstance(jv, float) and isinstance(v, float)
                                     and jv != jv and v != v)   # NaN == NaN
                if not same:
                    print(f"  MISMATCH {cfg}/{name}/{k}: json {v!r} pickle {jv!r}")
                    bad += 1
        n_par = sum(len(x) for x in kept.values())
        n_drop = sum(len(x) for x in dropped.values())
        print(f"{cfg}: {len(kept)} models, {n_par} parameters written, {n_drop} dropped -> {js}")
        for name in sorted(kept):
            print(f"    {name:<20s} {len(kept[name])} params"
                  + (f", dropped: {', '.join(x.split(' = ')[0] for x in dropped[name])}" if name in dropped else ""))
    print("round-trip OK" if bad == 0 else f"round-trip FAILED: {bad} mismatches")
    sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
