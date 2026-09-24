#!/usr/bin/env bash
# ============================================================================
# run_nohopfilter_mobile.sh — the MOBILE half of the no-hop-filter experiment.
#
# Mirror of run_nohopfilter.sh. That script is left byte-identical as the
# provenance record of what produced simulations_v347_nohopfilter/features_static/;
# this one is its mobile sibling. Only four things differ, all marked MOBILE
# below: the mobility flag, the worker-log directory, the seed-list file, and
# the output/metadata paths.
#
# WHY THIS EXISTS
#   The main campaign rejects any run whose UDP source is closer than three
#   hops to the victim, and cleanup_id deletes rejected outputs. The static
#   half of those runs has been recovered; the mobile half — 5,055 seeds — has
#   not, so C_all does not exist for the mobile configuration and the hop
#   filter can only be removed for static. This script produces the missing
#   part of the unrestricted mobile population.
#
# ---------------------------------------------------------------------------
# ⚠️  SUPERSEDES THE "STATIC ONLY" CAUTION IN run_nohopfilter.sh (lines 17-19)
#
#   That header says mobile was excluded because "the hop check fires at fixed
#   times; under mobility the same seed can flip between accept/reject as nodes
#   move, which would break the pairing." Decided 14/8 that this does not block
#   the experiment, for two reasons:
#
#   1. The paper defines the criterion as an instant-in-time check — Section
#      4.2(ii): "not a one-hop neighbor of the victim AT TRAFFIC INITIATION".
#      The code implements exactly that. There is no code/paper gap, and the
#      criterion never claimed to hold across the window.
#
#   2. Re-running is deterministic: same RngRun gives the same trajectory, so
#      the hop category recorded here is the one the original campaign saw.
#      --enforceHopFilter=0 only removes the Simulator::Stop(); everything up
#      to the check instant is identical.
#
#   What remains true, and belongs in the write-up rather than in the code:
#   under mobility the topology drifts after the check, so the hop label is
#   noisier than in static. Noise attenuates any real distance dependence
#   toward zero, therefore A FLAT RESULT IN MOBILE IS WEAKER EVIDENCE THAN A
#   FLAT RESULT IN STATIC. Do not conclude "accuracy is distance-independent"
#   from the mobile runs alone.
# ---------------------------------------------------------------------------
#
# CONNECTIVITY IS STILL ENFORCED — verified in the code 14/8
#   --enforceHopFilter=0 disables ONLY the distance gate. It is a single
#   condition inside AbortOnNeighbor (iolsr-tests-corrected.cc:1014):
#       if (g_enforceHopFilter && cat < 3) { ...; Simulator::Stop(); }
#
#   The connectivity gate is a separate function, AssertConnectivity (:956),
#   scheduled UNCONDITIONALLY at t = INITIAL_STABILIZATION = 60 s (:2759):
#       for every node: if (getRoutingTableSize() != N-1) -> Stop()
#
#   That is exactly the project's definition of full connectivity: every node
#   holds a route to every other node — reachable, not necessarily directly.
#   It is unaffected by any flag this script passes, so every run admitted here
#   is connectivity-verified.
#
#   Ordering also confirms the seeds are already known-connected: connectivity
#   fires at t=60, the hop check at (window start - 2) for each window, i.e.
#   afterwards. A seed rejected on distance had therefore already passed
#   connectivity in the original campaign. Re-running verifies it a second time.
#
#   A run that fails connectivity produces no complete output set, have_all()
#   returns false, and the run is recorded FAIL — it never enters the
#   population. (The static run recorded 4 such FAILs out of 3,714.)
#
# WHERE THE SEED LIST COMES FROM
#   Recovered from logs_parallel_mobile/ the same way as static. Verified 14/8:
#   the two markers ("try seed=", "within two hops") are present in all 22
#   worker logs and the recovery yields exactly 5,055 seeds, matching the
#   distance-rejection count in STATE §17.9.
#
#   ⛔ NOT from run_status_mobile.csv. Its reason column shows only 5
#   distance rejections — the known-bad classification of STATE §17.10. The
#   worker logs are the ground truth.
#
# GUARANTEES
#   - Writes ONLY under simulations_v347_nohopfilter/. Never deletes anything.
#   - features_static/ and the static metadata are untouched.
#   - The main campaign, its bundles and results are read-only inputs.
#   - Idempotent: a seed whose 12 output files exist is skipped on re-run.
#
# HOP CATEGORY — THE LEAK RULE (unchanged from static)
#   The scenario prints HOP_CATEGORY=<1|2|3> to stdout, captured here into
#   hop_metadata_mobile.csv keyed by seed. It must NEVER reach a metrics_output
#   file: make_arm_bundles.py turns every metric row into a feature column, and
#   hop distance correlates with task difficulty. Any per-distance accuracy
#   split is computed post-hoc on model PREDICTIONS via file_source.
#
# USAGE
#   ./run_nohopfilter_mobile.sh                       # all 5,055 seeds
#   N_SEEDS=500 NUM_WORKERS=4 ./run_nohopfilter_mobile.sh   # pilot
#
# Output layout:
#   simulations_v347_nohopfilter/
#     features_mobile/<scenario>/{metrics_output,...}-<1000000+seed>.csv
#     hop_metadata_mobile.csv    seed,run_id,hop_category,result,timestamp
#     rejected_seeds_mobile.txt  the recovered seed list (input)
#     logs_mobile/run_<seed>.log one log per run
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
[ -d "$SRC/logs_parallel_mobile" ] || { echo "ERROR: no $SRC/logs_parallel_mobile"; exit 1; }
export LD_LIBRARY_PATH="$ROOT/build/lib:${LD_LIBRARY_PATH:-}"
cd "$ROOT"

mkdir -p "$OUT/logs_mobile"
for s in "${SCEN[@]}"; do mkdir -p "$OUT/features_mobile/$s"; done   # MOBILE

SEEDS="$OUT/rejected_seeds_mobile.txt"                               # MOBILE
META="$OUT/hop_metadata_mobile.csv"                                  # MOBILE
LOCK="$OUT/.lock_mobile"; : > "$LOCK"
[ -f "$META" ] || echo "seed,run_id,hop_category,result,timestamp" > "$META"

# ---- 1. recover the distance-rejected seed list from the worker logs -------
if [ ! -s "$SEEDS" ]; then
    echo "recovering distance-rejected seeds from mobile worker logs..."
    awk '
      /try seed=/       { seed=$0; sub(/.*try seed=/,"",seed) }
      /within two hops/ { print seed }
    ' "$SRC"/logs_parallel_mobile/worker_*.log | sort -n -u > "$SEEDS"   # MOBILE
    echo "  recovered $(wc -l < "$SEEDS") seeds -> $SEEDS"
    echo "  (expected 5055 — STATE §17.9)"
fi
TOTAL=$(wc -l < "$SEEDS")
LIMIT=$TOTAL
[ "$N_SEEDS" -gt 0 ] && [ "$N_SEEDS" -lt "$TOTAL" ] && LIMIT=$N_SEEDS
echo "running $LIMIT of $TOTAL recovered seeds, $NUM_WORKERS workers"

# ---- 2. run one seed --------------------------------------------------------
have_all() {  # $1 = run id
    local id="$1" s
    for s in baseline attack_only defense_only defense_vs_attack; do
        [ -s "$OUT/features_mobile/$s/metrics_output-$id.csv" ]   || return 1
        [ -s "$OUT/features_mobile/$s/observer_metrics-$id.csv" ] || return 1
        [ -s "$OUT/features_mobile/$s/observer_detail-$id.csv" ]  || return 1
    done
    return 0
}

# Run ids are OFFSET past the main campaign's id space, exactly as in the
# static script. The same base is safe for mobile: features_static/ and
# features_mobile/ are never merged with each other — the main campaign
# already reuses ids 1..10,000 across both — so the only collision that
# matters is against the mobile ids of simulations_v347, and 1,000,001+
# clears them.
RID_BASE=1000000

one() {  # $1 = seed
    local seed="$1"
    local rid=$((seed + RID_BASE))
    local rlog="$OUT/logs_mobile/run_${seed}.log"
    if have_all "$rid"; then
        echo "[skip] seed=$seed complete"
        return 0
    fi
    timeout "$RUN_TIMEOUT" "$BIN" \
        --run="$rid" --RngRun="$seed" --bMobility=1 --enforceHopFilter=0 \
        --outputDir="$OUT/features_mobile/" \
        --topologyProbeFile="$OUT/topology_probes_mobile.csv" > "$rlog" 2>&1
    local rc=$?
    # LAST HOP_CATEGORY line, not the first: the scenario prints one per
    # measurement window as a running minimum, so the final line is the run's
    # classification (a run counts as "near" if it was ever within two hops,
    # which is exactly when the original filter would have rejected it).
    # This rule matters more under mobility than under static, because the
    # distance genuinely varies between windows.
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
echo "output  : $OUT/features_mobile/"
echo "metadata: $META  (NOT part of any bundle — post-hoc join only)"
echo "finished $(date '+%F %T')"
echo
echo "SANITY CHECK: category 3 should be 0 — every seed here was rejected on"
echo "distance, so none may be >=3 hops. A non-zero count means the recovery"
echo "pulled the wrong seeds. (The static run gave 1727/1983/0 + 4 FAIL.)"
