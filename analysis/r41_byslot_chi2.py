#!/usr/bin/env python3
"""r41_byslot_chi2.py -- the 2x4 homogeneity test behind R#4.1's by-slot table.

WHY THIS EXISTS. windoworder_by_slot_test.py writes per-slot accuracies and the
spread, but not the test that says whether four subgroup accuracies of 400 runs
each differ by more than sampling noise. r4_1_answer.tex's OURS table prints a
chi2(3) and a p for each of the four cells; until now those two columns had no
generator of their own and were recomputed by hand each time the arms moved.

WHAT IT DOES. Reads every accuracy_by_slot_test_*.csv in --in-dir, turns each
slot's (n, accuracy) into a correct/incorrect pair, and runs a chi-square test of
independence on the resulting 2x4 table. Writes one row per cell.

⛔ It reads only. Nothing but --out is written.

⚠️ correct = round(n * accuracy). The consumer stores accuracy as a float and n
   as an integer, and n*accuracy is an integer up to float error, so the rounding
   is exact recovery and not an approximation. The script asserts that the
   distance to the nearest integer is below 1e-6 and aborts otherwise.

    python3 r41_byslot_chi2.py --in-dir analysis/r41_byslot_17_frozencal
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True,
                    help="directory of accuracy_by_slot_test_*.csv")
    ap.add_argument("--out", default=None,
                    help="default: <in-dir>/chi2_by_slot.csv")
    args = ap.parse_args()

    in_dir = os.path.abspath(args.in_dir)
    out = args.out or os.path.join(in_dir, "chi2_by_slot.csv")

    files = sorted(glob.glob(os.path.join(in_dir, "accuracy_by_slot_test_*.csv")))
    if not files:
        sys.exit("ABORT: no accuracy_by_slot_test_*.csv under %s" % in_dir)

    rows = []
    for f in files:
        df = pd.read_csv(f).sort_values("slot")
        if len(df) != 4:
            sys.exit("ABORT: %s has %d slots, expected 4" % (f, len(df)))

        exact = df["n"].to_numpy() * df["accuracy"].to_numpy()
        drift = np.abs(exact - np.rint(exact)).max()
        if drift > 1e-6:
            sys.exit("ABORT: %s -- n*accuracy is %.9f from an integer" % (f, drift))

        correct = np.rint(exact).astype(int)
        wrong = df["n"].to_numpy().astype(int) - correct
        chi2, p, dof, _ = chi2_contingency(np.vstack([correct, wrong]),
                                           correction=False)

        rows.append({
            "arm": df["arm"].iloc[0],
            "mode": df["mode"].iloc[0],
            "n_per_slot": int(df["n"].iloc[0]),
            "overall_test_accuracy": float(df["overall_test_accuracy"].iloc[0]),
            "spread": float(df["spread"].iloc[0]),
            "spread_pp": round(float(df["spread"].iloc[0]) * 100, 2),
            "chi2": round(float(chi2), 2),
            "dof": int(dof),
            "p": round(float(p), 4),
            "source": os.path.basename(f),
        })

    res = pd.DataFrame(rows)
    res.to_csv(out, index=False)

    print("  in : %s  (%d cells)" % (in_dir, len(res)))
    print()
    print(res[["arm", "mode", "spread_pp", "chi2", "dof", "p"]].to_string(index=False))
    print()
    print("  wrote %s" % out)
    print()
    print("  ⚠️ dof is 3 for a 2x4 table. correction=False: Yates' correction is")
    print("     defined for 2x2 only and scipy applies it there by default.")
    print("  ⛔ A large spread in the fixed-order arm is NOT evidence of a temporal")
    print("     signal -- there position and scenario are confounded by construction.")
    print("     Its role is as a positive control (r4_1_answer.tex).")


if __name__ == "__main__":
    main()
