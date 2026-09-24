#!/usr/bin/env bash
# ============================================================================
# run_nohopfilter.sh — the no-hop-filter experiment (STATE §17.14/§17.15).
#
# WHY THIS EXISTS
#   The main campaign rejects any run whose UDP source is closer than three
#   hops to the victim, so those runs were never written to disk (cleanup_id
#   removes rejected outputs). This script re-runs exactly those seeds with
#   --enforceHopFilter=0, producing the missing part of the unrestricted
#   population. Together with the existing simulations_v347/ data this yields:
#     - dataset B: the unconditioned population (73% far / 27% near, the
#       measured ratio: 10,000 accepted vs 3,714 distance-rejected)
#     - a per-hop-distance accuracy breakdown, computed AFTER training by
#       joining predictions on file_source against hop_metadata.csv.
#
# WHAT IT RUNS
#   The distance-rejected seeds of the STATIC campaign only. Static only: the
#   hop check fires at fixed times; under mobility the same seed can flip
#   between accept/reject as nodes move, which would break the pairing.
#
# WHERE THE SEED LIST COMES FROM
#   Recovered from the campaign worker logs (logs_parallel_static/): the seed
#   of every run whose log segment contains the distance-reject message. The
#   run_status reason column is NOT usable for this — its classification was
#   wrong before the 13/8 classifier fix (STATE §17.10); the worker logs are
#   the ground truth.
#
# GUARANTEES
#   - Writes ONLY under simulations_v347_nohopfilter/. Never deletes anything.
#   - The main campaign, its bundles and results are read-only inputs.
#   - Idempotent: a seed whose 12 output files exist is skipped on re-run.
#   - Per-run logs (the accumulated-log classifier bug of §17.10 does not
#     apply here — there is no reject classification at all; every run is
#     admitted by construction).
#
# HOP CATEGORY — THE LEAK RULE
#   The scenario prints HOP_CATEGORY=<1|2|3> to stdout. It is captured here
#   into hop_metadata.csv (seed,hop_category,result,timestamp), keyed by seed.
#   It must NEVER be written into metrics_output files: make_arm_bundles.py
#   would turn it into a feature column and hop distance correlates with task
#   difficulty — that would be label-adjacent leakage. Any per-distance
#   accuracy split is computed post-hoc on model PREDICTIONS via file_source.
#
# USAGE
#   ./run_nohopfilter.sh                          # all recovered seeds
#   N_SEEDS=2000 NUM_WORKERS=8 ./run_nohopfilter.sh
#
# Output layout:
#   simulations_v347_nohopfilter/
#     features_static/<scenario>/{metrics_output,...}-<1000000+seed>.csv
#         (run ids offset by RID_BASE so they can never collide with the main
#          campaign's ids 1..10,000 when the two are merged into dataset B)
#     hop_metadata.csv          seed,run_id,hop_category,result,timestamp
#     rejected_seeds_static.txt the recovered seed list (input)
#     logs/run_<seed>.log       one log per run
# ============================================================================
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$ROOT/build/scratch/ns3.47-iolsr-tests-corrected-default"
SRC="$ROOT/simulations_v347"                       # read-only input
OUT="$ROOT/simulations_v347_nohopfilter"
SCEN=(baseline attack_only defense_only defense_vs_attack)

NUM_WORKERS="${NUM_WORKERS:-8}"        # 8, not 22: learning shares this machine
RUN_TIMEOUT="${RUN_TIMEOUT:-900}"
N_SEEDS="${N_SEEDS:-0}"                # 0 = all recovered seeds

# Guard: never write anywhere but the experiment dir
case "$OUT" in */simulations_v347_nohopfilter) : ;; *) echo "REFUSING: bad OUT $OUT"; exit 1 ;; esac
[ -x "$BIN" ] || { echo "ERROR: binary not built: $BIN"; exit 1; }
export LD_LIBRARY_PATH="$ROOT/build/lib:${LD_LIBRARY_PATH:-}"
cd "$ROOT"

mkdir -p "$OUT/logs"
for s in "${SCEN[@]}"; do mkdir -p "$OUT/features_static/$s"; done

SEEDS="$OUT/rejected_seeds_static.txt"
META="$OUT/hop_metadata.csv"
LOCK="$OUT/.lock"; : > "$LOCK"
[ -f "$META" ] || echo "seed,run_id,hop_category,result,timestamp" > "$META"

# ---- 1. recover the distance-rejected seed list from the worker logs -------
if [ ! -s "$SEEDS" ]; then
    echo "recovering distance-rejected seeds from worker logs..."
    awk '
      /try seed=/       { seed=$0; sub(/.*try seed=/,"",seed) }
      /within two hops/ { print seed }
    ' "$SRC"/logs_parallel_static/worker_*.log | sort -n -u > "$SEEDS"
    echo "  recovered $(wc -l < "$SEEDS") seeds -> $SEEDS"
fi
TOTAL=$(wc -l < "$SEEDS")
LIMIT=$TOTAL
[ "$N_SEEDS" -gt 0 ] && [ "$N_SEEDS" -lt "$TOTAL" ] && LIMIT=$N_SEEDS
echo "running $LIMIT of $TOTAL recovered seeds, $NUM_WORKERS workers"

# ---- 2. run one seed --------------------------------------------------------
have_all() {  # $1 = run id
    local id="$1" s
    for s in baseline attack_only defense_only defense_vs_attack; do
        [ -s "$OUT/features_static/$s/metrics_output-$id.csv" ]   || return 1
        [ -s "$OUT/features_static/$s/observer_metrics-$id.csv" ] || return 1
        [ -s "$OUT/features_static/$s/observer_detail-$id.csv" ]  || return 1
    done
    return 0
}

# Run ids are OFFSET past the main campaign's id space.
#
# Dataset B merges these runs with runs 1..N of simulations_v347. Naming the
# output files after the raw seed would collide: seed 45 and accepted run id 45
# both produce metrics_output-45.csv, and file_source is the grouping key the
# split is built on — two different topologies under one group would silently
# corrupt the split. RID_BASE keeps the two id spaces provably disjoint
# (main: 1..10,000; here: 1,000,001+). Same reasoning as TMP_ID_BASE in
# run_campaign_v347.sh (STATE §17.5); g_currentRun only names files.
RID_BASE=1000000

one() {  # $1 = seed
    local seed="$1"
    local rid=$((seed + RID_BASE))
    local rlog="$OUT/logs/run_${seed}.log"
    if have_all "$rid"; then
        echo "[skip] seed=$seed complete"
        return 0
    fi
    timeout "$RUN_TIMEOUT" "$BIN" \
        --run="$rid" --RngRun="$seed" --bMobility=0 --enforceHopFilter=0 \
        --outputDir="$OUT/features_static/" \
        --topologyProbeFile="$OUT/topology_probes.csv" > "$rlog" 2>&1
    local rc=$?
    # LAST HOP_CATEGORY line, not the first: the scenario prints one per
    # measurement window as a running minimum, so the final line is the run's
    # classification (a run counts as "near" if it was ever within two hops,
    # which is exactly when the original filter would have rejected it).
    local cat
    cat=$(grep -oP "HOP_CATEGORY=\K[0-9]" "$rlog" | tail -1)
    [ -n "$cat" ] || cat="?"
    local res=OK
    { [ $rc -ne 0 ] || ! have_all "$rid"; } && res="FAIL rc=$rc"
    ( flock -x 9
      echo "$seed,$rid,$cat,$res,$(date '+%F %T')" >> "$META" ) 9>"$LOCK"
    echo "[$res] seed=$seed run_id=$rid hop_category=$cat"
}
export -f one have_all
export BIN OUT RUN_TIMEOUT META LOCK RID_BASE

echo "started $(date '+%F %T')"
head -"$LIMIT" "$SEEDS" | xargs -P "$NUM_WORKERS" -n 1 -I{} bash -c 'one {}'

echo
echo "==================== DONE ===================="
echo "results by outcome and hop category:"
tail -n +2 "$META" | awk -F, '{c[$4" cat"$3]++} END {for (k in c) printf "  %-12s %d\n", k, c[k]}' | sort
echo "output  : $OUT/features_static/"
echo "metadata: $META  (NOT part of any bundle — post-hoc join only)"
echo "finished $(date '+%F %T')"
