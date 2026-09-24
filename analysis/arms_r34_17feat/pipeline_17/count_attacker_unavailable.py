"""
Scan all mobile attempt logs to count rejected runs caused by
'no attacker candidate available at attack-activation time'.

The C++ binary prints "Attacker not available." when ExecuteIsolationAttackByNeighbor
finds no neighbor of the victim able to mount the isolation attack. The simulation
is then stopped, which usually leaves outputs incomplete; in run_parallel_static_mobile.sh
this is logged as 'missing_or_empty_outputs', but the underlying cause is criterion (iii).

We classify each rejected attempt log into one of:
  - attacker_unavailable  (the message we are looking for)
  - connectivity_failed   (assert connectivity message)
  - neighbor_abort        (UDP source is a neighbor of the victim)
  - other                 (timeout, runtime error, etc.)

Results are reported per mode (static + mobile), even though the paper's 1.6%
claim is specifically about mobile.

Usage (from strict_observable_v2/):
    python3 -u count_attacker_unavailable.py 2>&1 | tee count_attacker_unavailable.log
"""

import os
import glob
import re

ATTACKER_PATTERN  = re.compile(r"Attacker not available\.\s*\*+\s*Terminated", re.IGNORECASE)
CONNECTIVITY_PAT  = re.compile(r"Assert connectivity failed", re.IGNORECASE)
NEIGHBOR_PAT      = re.compile(r"Sending node of udp packets is a neighbor", re.IGNORECASE)


def classify_log(path):
    try:
        with open(path, "r", errors="ignore") as f:
            text = f.read()
    except Exception:
        return "unreadable"

    if ATTACKER_PATTERN.search(text):
        return "attacker_unavailable"
    if CONNECTIVITY_PAT.search(text):
        return "connectivity_failed"
    if NEIGHBOR_PAT.search(text):
        return "neighbor_abort"
    return "other"


def analyze(mode):
    print(f"\n{'=' * 60}\n  {mode.upper()}\n{'=' * 60}")
    log_dir = f"../simulations/logs_parallel_{mode}/attempts"
    if not os.path.isdir(log_dir):
        print(f"Directory not found: {log_dir}")
        return

    pattern = os.path.join(log_dir, "*.log")
    log_files = glob.glob(pattern)
    print(f"Attempt log files: {len(log_files)}")

    counts = {"attacker_unavailable": 0,
              "connectivity_failed":  0,
              "neighbor_abort":       0,
              "other":                0,
              "unreadable":           0}

    for i, p in enumerate(log_files, 1):
        c = classify_log(p)
        counts[c] += 1
        if i % 5000 == 0:
            print(f"  scanned {i}/{len(log_files)}...")

    total = sum(counts.values())
    print()
    print(f"  {'Reason':<25s} {'Count':>8s}  {'% of all':>10s}")
    print("  " + "-" * 50)
    for k, v in counts.items():
        pct = 100.0 * v / total if total else 0.0
        print(f"  {k:<25s} {v:>8d}  {pct:>9.3f}%")
    print(f"  {'TOTAL':<25s} {total:>8d}")

    # Specific paper claim: % within "connectivity_failed" caused by attacker_unavailable.
    # In the run_parallel script, attacker_unavailable was classified as
    # 'missing_or_empty_outputs' in run_status.csv, but the paper sentence
    # references connectivity failures. Report both interpretations:
    if mode == "mobile":
        au = counts["attacker_unavailable"]
        cf = counts["connectivity_failed"]
        if cf > 0:
            print(f"\n  attacker_unavailable as % of connectivity_failed : "
                  f"{100*au/cf:.3f}%  ({au}/{cf})")
        if au + cf > 0:
            print(f"  attacker_unavailable as % of (CF + AU)            : "
                  f"{100*au/(au+cf):.3f}%  ({au}/{au+cf})")


if __name__ == "__main__":
    analyze("static")
    analyze("mobile")