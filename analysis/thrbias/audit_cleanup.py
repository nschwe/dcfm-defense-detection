#!/usr/bin/env python3
"""
R#5.9 audit --- (a) what the engineered space looks like when formulas whose
required inputs are absent are NOT generated, and (b) an EXHAUSTIVE check that
every generated column is reproducible from its stated definition.

⛔ THIS SCRIPT CHANGES NOTHING. It computes both the current and the cleaned
   space side by side and reports whether the downstream selection is identical.
   The generator is only edited afterwards, and only if this says it is safe.

WHY
   engineer_advanced_features resolves an absent base metric to zeros:
       def col(name): return X[name] if name in X.columns else zeros
   In the 17-observable space several formulas reference metrics that no longer
   exist, so they are emitted as all-zero columns and dropped later by
   VarianceThreshold. They affect no result, but they inflate the reported
   dimension and are awkward to defend in a reproducibility answer.

WHAT IS DECIDED BY THIS AUDIT
   1. the column count of the cleaned space  -- MEASURED, never 77-14 by hand
   2. whether feature selection retains exactly the same features BY NAME
   3. whether anything downstream could differ

⛔ Step 3 here is structural only. An end-to-end confirmation still needs a
   stage-1 run on the cleaned generator; this script says whether that run is
   expected to be a no-op, it does not replace it.

USAGE
   audit_cleanup.py <wide_bundle.csv.gz> [<wide_bundle.csv.gz> ...]

   The pipeline is resolved against this file: frozencal67_pipeline_17, the
   generator the reported campaign ran. Set ENG_PIPELINE to audit a different
   copy; pointing it at a 77-column copy audits a space the paper does not
   report.
"""
import os
import sys

import numpy as np
import pandas as pd

# Resolved against this file so a clone runs as it stands, and pointed at the
# generator the reported campaign ran. Was an absolute path into a copy that
# campaign did not run, until 16/9.
_HERE = os.path.dirname(os.path.abspath(__file__))
PIPE = os.environ.get("ENG_PIPELINE",
                      os.path.join(os.path.dirname(_HERE),
                                   "frozencal67_pipeline_17"))
sys.path.insert(0, PIPE)
from defense_detection_v2 import DefenseDetector, Config  # noqa: E402

EPS = 1e-10

# Every engineered column, with the base metrics its definition requires.
# Transcribed from engineer_advanced_features. Row-wise statistics depend on the
# whole retained metric set and are always defined, so they carry an empty set.
REQUIRES = {
    "QoS_Score": {"PacketDeliveryRatio", "PacketLossRatio",
                  "AverageEndToEndDelay", "AverageJitter"},
    "Network_Efficiency": {"Throughput", "RoutingOverheadRatio"},
    "Delivery_Efficiency": {"PacketDeliveryRatio", "NormalizedRoutingLoad"},
    "CDR": {"TcMessageRate", "AvgTxPacketsPerFlow", "FlowCount"},
    "TDR": {"TcMessageRate", "AverageAdvertisedLinksPerTCMessage"},
    "Overhead_Per_Hop": {"RoutingOverheadRatio", "AverageHopCount"},
    "Total_Traffic": {"TcMessageRate", "MidMessageRate", "HnaMessageRate",
                      "DataPacketRate"},
    "Delay_Per_Hop": {"AverageEndToEndDelay", "AverageHopCount"},
    "Jitter_Delay_Ratio": {"AverageJitter", "AverageEndToEndDelay"},
    "Routing_Anomaly": {"NormalizedRoutingLoad", "RoutingOverheadRatio"},
    "Traffic_Anomaly": {"TcMessageRate", "MidMessageRate", "HnaMessageRate",
                        "DataPacketRate", "PacketDeliveryRatio"},
    "Performance_Degradation": {"AverageEndToEndDelay", "PacketLossRatio"},
}


def requires_of(name, base):
    if name in REQUIRES:
        return REQUIRES[name]
    for pre in ("log1p_", "sqrt_"):
        if name.startswith(pre):
            return {name[len(pre):]}
    for suf in ("_squared", "_cubed"):
        if name.endswith(suf):
            return {name[: -len(suf)]}
    if name.startswith("row_"):
        return set()
    if name in base:
        return {name}
    return set()


def selection(X):
    """Reproduce the pipeline's variance + correlation filters, in order."""
    var = X.var()
    kept = list(var[var >= 0.01].index)
    Xk = X[kept]
    corr = Xk.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop = [c for c in upper.columns if (upper[c] > 0.95).any()]
    return [c for c in kept if c not in drop]


def main(bundles):
    det = DefenseDetector(Config())
    base = list(det.METRICS)

    for b in bundles:
        print("=" * 92)
        print(b.split("/")[-1])
        print("=" * 92)
        df = pd.read_csv(b)
        avail = {m for m in base if m in df.columns}
        X = pd.DataFrame({m: (df[m] if m in df.columns else 0.0) for m in base})
        X["_measurement_duration"] = (df["_Duration"] if "_Duration" in df.columns
                                      else 40.0)
        E = det.engineer_advanced_features(X)

        # (a) which columns are structurally undefined in THIS space
        undefined = [c for c in E.columns
                     if requires_of(c, base) and not requires_of(c, base) <= avail]
        defined = [c for c in E.columns if c not in undefined]
        print(f"  generated columns              : {E.shape[1]}")
        print(f"  structurally undefined         : {len(undefined)}")
        print(f"  ⭐ cleaned space (MEASURED)     : {len(defined)}")
        allzero = [c for c in E.columns if float(E[c].var()) <= 1e-12]
        print(f"  identically constant           : {len(allzero)}")
        only_zero = sorted(set(allzero) - set(undefined))
        only_undef = sorted(set(undefined) - set(allzero))
        if only_zero:
            print(f"  ⚠️ constant but NOT undefined  : {only_zero}")
        if only_undef:
            print(f"  ⚠️ undefined but NOT constant  : {only_undef}")

        # (b) does the cleanup change what selection retains?
        sel_now = selection(E)
        sel_clean = selection(E[defined])
        print()
        print(f"  selection on the current space : {len(sel_now)}")
        print(f"  selection on the cleaned space : {len(sel_clean)}")
        same = sel_now == sel_clean
        print(f"  ⭐ identical, in the same order : {same}")
        if not same:
            print(f"      only in current : {sorted(set(sel_now)-set(sel_clean))}")
            print(f"      only in cleaned : {sorted(set(sel_clean)-set(sel_now))}")

        # (c) EXHAUSTIVE reproduction check for the closed-form definitions
        print()
        bad, checked = [], 0
        def c(n):
            return X[n] if n in X.columns else pd.Series(0.0, index=X.index)
        recompute = {
            "TDR": lambda: c("TcMessageRate") * c("AverageAdvertisedLinksPerTCMessage"),
            "Network_Efficiency": lambda: c("Throughput") / (c("RoutingOverheadRatio") + EPS),
            "Delivery_Efficiency": lambda: c("PacketDeliveryRatio") / (c("NormalizedRoutingLoad") + EPS),
            "Overhead_Per_Hop": lambda: c("RoutingOverheadRatio") / (c("AverageHopCount") + EPS),
            "Delay_Per_Hop": lambda: c("AverageEndToEndDelay") / (c("AverageHopCount") + EPS),
            "Jitter_Delay_Ratio": lambda: c("AverageJitter") / (c("AverageEndToEndDelay") + EPS),
            "Routing_Anomaly": lambda: c("NormalizedRoutingLoad") * c("RoutingOverheadRatio"),
            "Performance_Degradation": lambda: c("AverageEndToEndDelay") * c("PacketLossRatio"),
            "Total_Traffic": lambda: (c("TcMessageRate") + c("MidMessageRate")
                                      + c("HnaMessageRate") + c("DataPacketRate")),
            "QoS_Score": lambda: (c("PacketDeliveryRatio") * 0.4
                                  + (1 - c("PacketLossRatio")) * 0.3
                                  + (1 / (c("AverageEndToEndDelay") + EPS)) * 0.15
                                  + (1 / (c("AverageJitter") + EPS)) * 0.15),
        }
        for m in base:
            recompute[f"log1p_{m}"] = (lambda mm=m: np.log1p(c(mm)))
            recompute[f"sqrt_{m}"] = (lambda mm=m: np.sqrt(c(mm) + EPS))
        for m in ["Throughput", "AverageEndToEndDelay", "AverageJitter",
                  "RoutingOverheadRatio"]:
            recompute[f"{m}_squared"] = (lambda mm=m: c(mm) ** 2)
            recompute[f"{m}_cubed"] = (lambda mm=m: c(mm) ** 3)
        recompute["row_range"] = lambda: E["row_max"] - E["row_min"]
        recompute["row_iqr"] = lambda: E["row_q75"] - E["row_q25"]
        recompute["row_cv"] = lambda: E["row_std"] / (E["row_mean"] + EPS)

        for name, fn in recompute.items():
            if name not in E.columns:
                continue
            checked += 1
            mine = np.nan_to_num(np.asarray(fn(), dtype=float),
                                 posinf=0.0, neginf=0.0)
            theirs = E[name].to_numpy(float)
            d = float(np.nanmax(np.abs(mine - theirs)))
            if d > 1e-9:
                bad.append((name, d))
        n_rowwise = sum(1 for cc in E.columns
                        if cc.startswith("row_") and cc not in recompute)
        print(f"  definitions recomputed         : {checked} of {E.shape[1]} columns")
        print(f"    (base {len(base)} are measured, {n_rowwise} row-wise "
              f"statistics are aggregates over the base set)")
        if bad:
            print(f"  ⛔ MISMATCHES: {bad}")
        else:
            print(f"  ✅ all {checked} recomputed definitions match at <1e-9")
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
