#!/usr/bin/env python3
"""
Copy the first N runs of a staged hop-ablation population into a new,
self-contained tree.

⛔ WHY THIS DOES NOT SELECT BY FILE ID. A C_all staging tree is assembled from
two campaigns: runs whose source is >=3 hops from the victim come from
simulations_v347 with ids 1..10000, and the closer runs (hop categories 1 and 2)
come from the no-hop-filter rerun with ids seed+1,000,000. Selecting "the first
N ids" therefore takes only the far runs and silently reproduces the distance
filter -- 27% of the mobile population and 27% of the static one would vanish,
with no warning. `make_arm_bundles.py --limit` has exactly this behaviour
(it sorts by numeric id and slices), so it must not be used on these trees
either; copy the right runs here and bundle the copy in full.

Selection is by MANIFEST instead. manifests/<POP>.csv lists the population in
seed order, one row per admitted run, and its first N rows are precisely what a
campaign of N runs under that criterion would have accepted.

    python3 copy_subset.py --src STAGING --dst SIMROOT --mode mobile --limit 2000

Refuses to write into the source. Idempotent: existing files are skipped.
Staged entries are symlinks; copy2 follows them, so the destination holds real
files and does not depend on the source surviving.
"""
import argparse, csv, os, shutil, sys

SCENARIOS = ("baseline", "attack_only", "defense_only", "defense_vs_attack")
PREFIXES = ("metrics_output", "observer_metrics", "observer_detail")


def main():
    ap = argparse.ArgumentParser()
    # Layout of a staging tree:
    #     <src>/manifests/<population>.csv
    #     <src>/<population>/features_<mode>/<scenario>/*.csv
    # so --src is the staging ROOT, not the population subdirectory.
    ap.add_argument("--src", required=True,
                    help="staging root, e.g. simulations_v347_hopablation_mobile_10k")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--mode", choices=["static", "mobile"], required=True)
    ap.add_argument("--limit", type=int, required=True)
    ap.add_argument("--population", default="C_all")
    args = ap.parse_args()

    src, dst = os.path.abspath(args.src), os.path.abspath(args.dst)
    if src == dst or dst.startswith(src + os.sep):
        sys.exit(f"ABORT: destination sits inside the source\n  src={src}\n  dst={dst}")

    man = os.path.join(src, "manifests", f"{args.population}.csv")
    if not os.path.isfile(man):
        sys.exit(f"ABORT: no manifest at {man}. This tool selects by manifest, "
                 f"not by file id -- see the module docstring.")

    with open(man, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if "run_id" not in (rows[0] if rows else {}):
        sys.exit(f"ABORT: manifest has no run_id column: {man}")
    if len(rows) < args.limit:
        sys.exit(f"ABORT: manifest holds {len(rows)} runs, fewer than --limit {args.limit}")

    chosen = rows[:args.limit]
    ids = [r["run_id"] for r in chosen]

    # Report the composition we are about to copy. If this is not close to the
    # population's own mix, the selection is biased and the run is not a control.
    cats = {}
    for r in chosen:
        cats[r.get("hop_category", "?")] = cats.get(r.get("hop_category", "?"), 0) + 1
    full = {}
    for r in rows:
        full[r.get("hop_category", "?")] = full.get(r.get("hop_category", "?"), 0) + 1
    print(f"  population {args.population}/{args.mode}: taking {len(ids)} of {len(rows)}")
    for k in sorted(cats):
        pc_sel = 100.0 * cats[k] / len(chosen)
        pc_all = 100.0 * full.get(k, 0) / len(rows)
        flag = "" if abs(pc_sel - pc_all) < 5 else "   <-- SKEWED"
        print(f"    hop_category {k}: {cats[k]:5d} ({pc_sel:5.1f}%)  "
              f"population {pc_all:5.1f}%{flag}")

    src_feat = os.path.join(src, args.population, f"features_{args.mode}")
    dst_feat = os.path.join(dst, f"features_{args.mode}")
    if not os.path.isdir(src_feat):
        sys.exit(f"ABORT: no such source directory: {src_feat}")

    copied = skipped = missing = 0
    for scen in SCENARIOS:
        s, d = os.path.join(src_feat, scen), os.path.join(dst_feat, scen)
        if not os.path.isdir(s):
            sys.exit(f"ABORT: missing scenario directory in source: {s}")
        os.makedirs(d, exist_ok=True)
        for rid in ids:
            for pref in PREFIXES:
                name = f"{pref}-{rid}.csv"
                sp, dp = os.path.join(s, name), os.path.join(d, name)
                if not os.path.exists(sp):
                    missing += 1
                    continue
                if os.path.exists(dp):
                    skipped += 1
                else:
                    shutil.copy2(sp, dp)   # follows the staging symlink
                    copied += 1

    with open(os.path.join(dst, f"manifest_{args.mode}.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(chosen[0].keys()))
        w.writeheader()
        w.writerows(chosen)

    print(f"  files: {copied} copied, {skipped} already present, {missing} absent in source")
    n = len([f for f in os.listdir(os.path.join(dst_feat, "baseline"))
             if f.startswith("metrics_output-")])
    print(f"  destination baseline holds {n} metrics_output files (expected {args.limit})")
    if n != args.limit:
        sys.exit("ABORT: destination run count does not match --limit")


if __name__ == "__main__":
    main()
