#!/usr/bin/env python3
"""
Listener-only extract for MS Section 6.3.

For every accepted run of the reported campaign this takes the single listener's
row of observer_metrics-<id>.csv -- the same vantage point the detector's bundle
uses for that run (row (run_id * 2654435761) % n among the vantage-point rows;
the simulator's network-wide summary row, ObserverIdx -1, is not a vantage point
and is skipped) -- and writes the six raw quantities r38_paired_listener.py reads,
for the two windows that analysis compares (baseline and defense-only).

Output, one file per configuration, beside this script unless R38_OUT is set:
    r38_listener_static.csv     r38_listener_mobile.csv
Columns: scenario, run_id, ObserverIdx, ObserverNodeId, then the six quantities.
Floats are written with repr(), so they round-trip exactly; run_id is kept as
the string in the file name. Rows are in the order the raw files sort (as
strings), scenario by scenario, which is the order the analysis reads them in.

Inputs, overridable with R38_STATIC / R38_MOBILE:
  simulations_v347_hopablation_10k/C_all/features_static/<scenario>/observer_metrics-*.csv
  simulations_v347_hopablation_mobile_10k/C_all/features_mobile/<scenario>/observer_metrics-*.csv
An integer argument reads only that many runs per scenario (a rehearsal).
Refuses to overwrite an existing output file.
"""
import os, sys, glob, csv

STAT = os.environ.get("R38_STATIC", os.path.expanduser(
    "~/ns3/nv347/ns-3.47/simulations_v347_hopablation_10k/C_all/features_static"))
MOB = os.environ.get("R38_MOBILE", os.path.expanduser(
    "~/ns3/nv347/ns-3.47/simulations_v347_hopablation_mobile_10k/C_all/features_mobile"))
OUT = os.environ.get("R38_OUT", os.path.dirname(os.path.abspath(__file__)))
OBS_PICK = 2654435761
SCENARIOS = ["baseline", "defense_only"]
QUANTITIES = ["SniffedBytes", "TcCount", "AvgAdvertisedLinksPerTC",
              "SourcesHeard", "SrcRateStd", "ObsOlsrBytes"]
HEADER = ["scenario", "run_id", "ObserverIdx", "ObserverNodeId"] + QUANTITIES


def run_id_of(path):
    return os.path.basename(path).rsplit("-", 1)[1].rsplit(".", 1)[0]


def listener_row(path):
    """The chosen vantage-point row of one observer_metrics file, or None."""
    try:
        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh))
    except OSError:
        return None
    vantage = []
    for row in rows:
        try:
            vals = {k: float(row[k]) for k in QUANTITIES}
            oidx = int(row["ObserverIdx"])
        except (KeyError, ValueError, TypeError):
            continue
        if oidx != -1:
            vals["ObserverIdx"] = oidx
            vals["ObserverNodeId"] = row.get("ObserverNodeId", "")
            vantage.append(vals)
    if not vantage:
        return None
    rid = run_id_of(path)
    return rid, vantage[(int(rid) * OBS_PICK) % len(vantage)]


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    for mode, root in (("static", STAT), ("mobile", MOB)):
        out = os.path.join(OUT, f"r38_listener_{mode}.csv")
        if os.path.exists(out):
            sys.exit(f"refusing to overwrite {out}")
        n_rows = {}
        with open(out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(HEADER)
            for scen in SCENARIOS:
                files = sorted(glob.glob(os.path.join(root, scen, "observer_metrics-*.csv")))
                if limit:
                    files = files[:limit]
                n = 0
                for f in files:
                    got = listener_row(f)
                    if got is None:
                        continue
                    rid, v = got
                    w.writerow([scen, rid, v["ObserverIdx"], v["ObserverNodeId"]]
                               + [repr(v[k]) for k in QUANTITIES])
                    n += 1
                n_rows[scen] = n
        print(f"{mode}: " + ", ".join(f"{s} {n_rows[s]} runs" for s in SCENARIOS)
              + f" -> {out}", flush=True)


if __name__ == "__main__":
    main()
