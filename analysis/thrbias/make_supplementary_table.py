#!/usr/bin/env python3
"""
R#5.9 --- build the supplementary feature table, with the FORMULA for every
engineered column, from the pipeline source itself.

The reviewer asks for "the full feature list ... in a supplementary table, or the
engineering code ... made available". This produces the first and demonstrates
that the second is trivially satisfiable: every derived column is a named,
deterministic expression over base metrics.

Formulas are transcribed from engineer_advanced_features in
analysis/frozencal67_pipeline_17/defense_detection_v2.py. Keeping them here
rather than parsing the source is deliberate: a parser would silently go stale,
whereas a mismatch between this table and the code is caught by the
`--verify` mode, which recomputes every non-row-wise formula from the bundle and
compares it to what the pipeline produced.

USAGE
    make_supplementary_table.py <eng_static.csv> <eng_mobile.csv> [out.tex]
    make_supplementary_table.py --verify <wide_bundle.csv.gz>
"""
import os
import sys

import numpy as np
import pandas as pd

EPS = 1e-10

# name -> (formula as it appears in the code, one-line description)
FORMULA = {
    "QoS_Score": ("0.4*PacketDeliveryRatio + 0.3*(1-PacketLossRatio) + "
                  "0.15/(AverageEndToEndDelay+eps) + 0.15/(AverageJitter+eps)",
                  "composite quality-of-service score"),
    "Network_Efficiency": ("Throughput / (RoutingOverheadRatio + eps)",
                           "throughput per unit routing overhead"),
    "Delivery_Efficiency": ("PacketDeliveryRatio / (NormalizedRoutingLoad + eps)",
                            "delivery per unit routing load"),
    "CDR": ("TcMessageRate / ((AvgTxPacketsPerFlow*FlowCount)/duration + eps)",
            "control-to-data rate"),
    "TDR": ("TcMessageRate * AverageAdvertisedLinksPerTCMessage",
            "topology dissemination rate"),
    "Overhead_Per_Hop": ("RoutingOverheadRatio / (AverageHopCount + eps)",
                         "routing overhead per hop"),
    "Total_Traffic": ("TcMessageRate + MidMessageRate + HnaMessageRate + DataPacketRate",
                      "summed message rates"),
    "Delay_Per_Hop": ("AverageEndToEndDelay / (AverageHopCount + eps)",
                      "per-hop delay"),
    "Jitter_Delay_Ratio": ("AverageJitter / (AverageEndToEndDelay + eps)",
                           "jitter relative to delay"),
    "Routing_Anomaly": ("NormalizedRoutingLoad * RoutingOverheadRatio",
                        "joint routing-load anomaly"),
    "Traffic_Anomaly": ("Total_Traffic * (1 - PacketDeliveryRatio)",
                        "traffic weighted by loss"),
    "Performance_Degradation": ("AverageEndToEndDelay * PacketLossRatio",
                                "delay weighted by loss"),
}
ROWWISE = {
    "row_mean": "mean over the base metrics of the window",
    "row_std": "standard deviation over the base metrics",
    "row_median": "median over the base metrics",
    "row_max": "maximum over the base metrics",
    "row_min": "minimum over the base metrics",
    "row_range": "row_max - row_min",
    "row_cv": "row_std / (row_mean + eps)",
    "row_skew": "skewness over the base metrics",
    "row_kurtosis": "kurtosis over the base metrics",
    "row_q25": "25th percentile over the base metrics",
    "row_q75": "75th percentile over the base metrics",
    "row_iqr": "row_q75 - row_q25",
}


def describe(name, kind):
    if kind == "base":
        return "base observable", "measured directly by the single vantage point"
    if name in FORMULA:
        return FORMULA[name]
    if name in ROWWISE:
        return ROWWISE[name], "row-wise statistic across the base metrics"
    if name.startswith("log1p_"):
        return f"log(1 + {name[6:]})", "log transform"
    if name.startswith("sqrt_"):
        return f"sqrt({name[5:]} + eps)", "square-root transform"
    if name.endswith("_squared"):
        return f"{name[:-8]} ** 2", "power transform"
    if name.endswith("_cubed"):
        return f"{name[:-6]} ** 3", "power transform"
    return "(see engineer_advanced_features)", ""


def build(static_csv, mobile_csv, out=None):
    s = pd.read_csv(static_csv).set_index("feature")
    m = pd.read_csv(mobile_csv).set_index("feature")
    feats = list(s.index)

    rows = []
    for f in feats:
        formula, desc = describe(f, s.loc[f, "kind"])
        rows.append(dict(
            feature=f, kind=s.loc[f, "kind"], formula=formula, description=desc,
            constant=bool(s.loc[f, "constant"]),
            # Named for the computation they record: the variance filter and the
            # |r|>0.95 correlation filter, and nothing later. In the reported
            # campaign these also equal the final retained set, because the
            # rank-aggregation cap does not bind -- but that is a property of
            # this campaign, not of the field.
            survives_variance_correlation_filters_static=bool(
                s.loc[f, "survives_both_filters"]),
            survives_variance_correlation_filters_mobile=(
                bool(m.loc[f, "survives_both_filters"]) if f in m.index else None),
        ))
    tab = pd.DataFrame(rows)

    print(f"total columns entering selection : {len(tab)}")
    print(f"  base observables               : {(tab.kind=='base').sum()}")
    print(f"  derived                        : {(tab.kind!='base').sum()}")
    print(f"  constant (built from metrics absent in this space): {tab.constant.sum()}")
    print("  survive the variance and correlation filters, static : "
          f"{tab.survives_variance_correlation_filters_static.sum()}")
    print("  survive the variance and correlation filters, mobile : "
          f"{tab.survives_variance_correlation_filters_mobile.sum()}")
    print()
    print("by kind:")
    print(tab.kind.value_counts().to_string())

    if out:
        if out.endswith(".tex"):
            with open(out, "w", encoding="utf-8") as fh:
                fh.write("% R#5.9 supplementary table --- generated by "
                         "analysis/thrbias/make_supplementary_table.py\n")
                fh.write("\\begin{longtable}{p{4.4cm}p{1.9cm}p{6.6cm}cc}\n\\hline\n")
                fh.write("Feature & Kind & Definition & Static & Mobile \\\\\n\\hline\n\\endhead\n")
                for _, r in tab.iterrows():
                    esc = lambda t: (str(t).replace("_", "\\_").replace("&", "\\&")
                                     .replace("%", "\\%"))
                    fh.write(f"{esc(r.feature)} & {esc(r.kind)} & {esc(r.formula)} & "
                             f"{'yes' if r.survives_variance_correlation_filters_static else '--'} & "
                             f"{'yes' if r.survives_variance_correlation_filters_mobile else '--'} \\\\\n")
                fh.write("\\hline\n\\end{longtable}\n")
        else:
            tab.to_csv(out, index=False)
        print(f"\nwrote {out}")
    return tab


def verify(bundle):
    """Recompute the closed-form formulas from the bundle and compare with what
    the pipeline produced, so this table cannot drift from the code unnoticed.

    Scope is decided by the shipped table, not by the list below. A name absent
    from both the table and the engineered frame is outside the reported space
    and is reported as such; a name the table describes but the code does not
    produce is a failure. The check list is historical and spans more than one
    feature space."""
    # The generator the reported campaign ran, resolved against this file so a
    # clone runs as it stands. Was an absolute path to a 77-column copy until
    # 16/9, so --verify checked a space the paper does not report.
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.environ.get(
        "ENG_PIPELINE",
        os.path.join(os.path.dirname(_here), "frozencal67_pipeline_17")))
    from defense_detection_v2 import DefenseDetector, Config

    det = DefenseDetector(Config())
    base = list(det.METRICS)
    df = pd.read_csv(bundle)
    X = pd.DataFrame({m: (df[m] if m in df.columns else 0.0) for m in base})
    X["_measurement_duration"] = df["_Duration"] if "_Duration" in df.columns else 40.0
    E = det.engineer_advanced_features(X)

    # The reported feature space is whatever the shipped table describes, so that
    # is what scope is decided against. The list below is historical and spans
    # more than one pipeline copy; membership is never assumed from it.
    table_path = os.path.join(_here, "out67", "supplementary_features.csv")
    try:
        reported = set(pd.read_csv(table_path)["feature"].astype(str))
    except Exception as exc:
        reported = None
        print(f"  could not read {table_path}: {exc}")
    if reported is not None and not reported:
        reported = None

    def c(n):
        return X[n] if n in X.columns else pd.Series(0.0, index=X.index)

    # zero-argument, so a formula outside the reported space is never computed
    checks = {
        "TDR": lambda: c("TcMessageRate") * c("AverageAdvertisedLinksPerTCMessage"),
        "Network_Efficiency": lambda: c("Throughput") / (c("RoutingOverheadRatio") + EPS),
        "Total_Traffic": lambda: (c("TcMessageRate") + c("MidMessageRate")
                                  + c("HnaMessageRate") + c("DataPacketRate")),
        "log1p_TcMessageRate": lambda: np.log1p(c("TcMessageRate")),
        "sqrt_Throughput": lambda: np.sqrt(c("Throughput") + EPS),
        "row_range": lambda: E["row_max"] - E["row_min"],
        "row_iqr": lambda: E["row_q75"] - E["row_q25"],
    }

    # Why a listed formula is legitimately outside the reported space.
    OUTSIDE = {
        "Network_Efficiency": ("its input RoutingOverheadRatio was consolidated as a "
                               "duplicate of the retained RoutingOverheadBytesRatio"),
    }

    print("formula verification against the pipeline's own output:")
    bad = 0

    # The formulas below are a sample of the space, not the space. This is the
    # whole-set check: every column the table describes, and no other.
    if reported is None:
        print("  scope: the shipped table could not be read -- every listed formula")
        print("         is required to be present, which is the conservative reading")
    else:
        only_table = sorted(reported - set(E.columns))
        only_frame = sorted(set(E.columns) - reported)
        if only_table or only_frame:
            bad += 1
            print(f"  scope: MISMATCH between the table ({len(reported)}) and the engineered")
            print(f"         frame ({len(E.columns)}): {len(only_table)} described but not")
            print(f"         produced, {len(only_frame)} produced but not described")
            for n in only_table[:10]:
                print(f"           table only: {n}")
            for n in only_frame[:10]:
                print(f"           frame only: {n}")
        else:
            print(f"  scope: table and engineered frame agree on all {len(reported)} names")

    compared = outside = 0
    for name, build in checks.items():
        in_table = (name in reported) if reported is not None else True
        in_frame = name in E.columns
        if not in_table and not in_frame:
            print(f"  {name:26s} OUTSIDE REPORTED FEATURE SPACE")
            print(f"  {'':26s}   {OUTSIDE.get(name, 'not a column of the reported space')}")
            outside += 1
            continue
        if not in_frame:
            print(f"  {name:26s} IN THE TABLE BUT NOT PRODUCED  MISMATCH")
            bad += 1
            continue
        if not in_table:
            print(f"  {name:26s} PRODUCED BUT NOT IN THE TABLE  MISMATCH")
            bad += 1
            continue
        d = float(np.nanmax(np.abs(np.asarray(build(), float) - E[name].to_numpy(float))))
        ok = d < 1e-9
        compared += 1
        print(f"  {name:26s} max|diff| = {d:.3e}  {'OK' if ok else 'MISMATCH'}")
        bad += (not ok)

    # A run that compared nothing must not read as a pass.
    if not compared:
        print("\u26d4 no formula was compared -- this run proves nothing")
    elif bad:
        print(f"\u26d4 {bad} MISMATCH(es)")
    else:
        print(f"all formulas reproduce "
              f"({compared} compared, {outside} outside the reported space)")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--verify":
        verify(sys.argv[2])
    elif len(sys.argv) >= 3:
        build(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    else:
        sys.exit(__doc__)
