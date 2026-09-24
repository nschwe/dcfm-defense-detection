#!/usr/bin/env python3
"""
MS Section 6.3 -- protocol-level verification of the observable footprint.

For every accepted seed, the baseline window and the defense-only window come
from the same topology and neither contains an attack. For each seed this script
takes the single listener's row -- the same vantage point the detector's bundle
uses for that run -- and forms the defense-minus-baseline difference of three
raw, observer-received quantities:

  AvgAdvLinksPerTC   advertised links per received TC message
  TcCount            received TC messages
  ObsOlsrBytes       received OLSR control bytes

The differences are resampled at the seed level (paired cluster bootstrap,
B = 10,000, fixed seed), because observers and windows are nested in a run.

Listener row: ObserverIdx -1 is not a vantage point and is skipped; of the
remaining rows, row (run_id * 2654435761) % n is taken, exactly as
make_arm_bundles.py's _observer_row_for() does. That selection is made once by
r38_extract_listener.py, which writes the listener rows of the two windows to
    r38_listener_static.csv     r38_listener_mobile.csv
(one row per run and window, the six raw quantities, floats written exactly).
This script reads those two files from its own directory, or from R38_EXTRACT.

Read-only; writes nothing. An integer argument reads only that many runs per
scenario (a rehearsal; the comparison with the manuscript is then skipped). The
printed figures are compared with MS Section 6.3 and the exit status is
non-zero on any difference.
"""
import os, sys, csv
import numpy as np

EXTRACT = os.environ.get("R38_EXTRACT", os.path.dirname(os.path.abspath(__file__)))

COLS = {
    "SniffedBytes":            "SniffedBytes",
    "TcCount":                 "TcCount",
    "AvgAdvertisedLinksPerTC": "AvgAdvLinksPerTC",
    "SourcesHeard":            "SourcesHeard",
    "SrcRateStd":              "SrcRateStd",
    "ObsOlsrBytes":            "ObsOlsrBytes",
}
# The bootstrap draws are consumed in this order, then two further draws per
# configuration; keeping the sequence fixed is what makes the intervals exact.
ORDER = ["SrcRateStd", "TcCount", "AvgAdvLinksPerTC", "ObsOlsrBytes",
         "SniffedBytes", "SourcesHeard"]
REPORT = ["AvgAdvLinksPerTC", "TcCount", "ObsOlsrBytes"]
B = 10000
SEED = 12345

# MS Section 6.3 as printed: (difference, lower, upper, relative %).
PUBLISHED = {
    "static": {"AvgAdvLinksPerTC": ("0.88", "0.86", "0.89", "20.5"),
               "TcCount":          ("245.1", "239.1", "251.2", "39.7"),
               "ObsOlsrBytes":     ("12,885", "12,580", "13,197", "29.9")},
    "mobile": {"AvgAdvLinksPerTC": ("0.94", "0.92", "0.95", "22.9"),
               "TcCount":          ("403.4", "394.4", "412.6", "59.4"),
               "ObsOlsrBytes":     ("19,641", "19,170", "20,123", "43.3")},
}
FMT = {"AvgAdvLinksPerTC": "{:.2f}", "TcCount": "{:.1f}", "ObsOlsrBytes": "{:,.0f}"}


def load_scenario(mode, scen, limit=None):
    """run_id -> {quantity -> value} for the single listener of that run, read
    from the extract. run_id is kept as a string: seeds are ordered as strings,
    and the bootstrap indices refer to that order. The extract lists the runs of
    a window in the order the raw files sort, so `limit` takes the first runs
    exactly as reading the raw tree did."""
    path = os.path.join(EXTRACT, f"r38_listener_{mode}.csv")
    out = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["scenario"] != scen:
                continue
            if limit and len(out) >= limit:
                break
            out[row["run_id"]] = {COLS[k]: float(row[k]) for k in COLS}
    return out


def paired_bootstrap(d, rng):
    d = np.asarray(d, dtype=float)
    idx = rng.integers(0, len(d), size=(B, len(d)))
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return d.mean(), lo, hi


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    rng = np.random.default_rng(SEED)
    ok = True
    for mode in ("static", "mobile"):
        base = load_scenario(mode, "baseline", limit)
        dfns = load_scenario(mode, "defense_only", limit)
        seeds = sorted(set(base) & set(dfns))
        res = {}
        for name in ORDER:
            b = np.array([base[s][name] for s in seeds])
            f = np.array([dfns[s][name] for s in seeds])
            m, lo, hi = paired_bootstrap(f - b, rng)
            res[name] = (m, lo, hi, 100.0 * (f.mean() - b.mean()) / b.mean())
        for _ in range(2):                          # keep the draw sequence fixed
            rng.integers(0, len(seeds), size=(B, len(seeds)))

        print(f"\n{mode.upper()}   paired seeds: {len(seeds)}")
        print(f"{'quantity':<18}{'difference':>12}{'95% CI':>28}{'relative':>10}   manuscript")
        for name in REPORT:
            m, lo, hi, rel = res[name]
            fmt = FMT[name]
            got = (fmt.format(m), fmt.format(lo), fmt.format(hi), f"{rel:.1f}")
            pub = PUBLISHED[mode][name]
            if limit:
                verdict = "(rehearsal: not compared)"
            else:
                verdict = "OK" if got == pub else "DIFFERS"
                ok &= got == pub
            print(f"{name:<18}{got[0]:>12}{f'[{got[1]}, {got[2]}]':>28}{got[3] + '%':>10}   "
                  f"{pub[0]} [{pub[1]}, {pub[2]}] {pub[3]}%  {verdict}")
    if limit:
        print(f"\nREHEARSAL ({limit} runs per scenario)")
    else:
        print("\nALL OK" if ok else "\nDIFFERENCES FOUND")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
