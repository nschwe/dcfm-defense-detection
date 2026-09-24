#!/usr/bin/env bash
# ============================================================================
# run_windowsweep.sh — window-length robustness sweep (reviewer #3, question 6).
#
# WHY THIS EXISTS (13/8/2026)
#   Reviewer #3 asked whether detection survives shorter measurement windows
#   (10-20 s). The main campaign fixed the window at 40 s. This sweep re-runs
#   a FIXED SUBSET of the already-accepted static seeds at other window
#   lengths, so accuracy can be reported as a function of window length with
#   the topology held constant per seed (paired comparison).
#
# HOW IT DIFFERS FROM run_campaign_v347.sh
#   - No seed search, no accept/reject: the seed list is taken verbatim from
#     simulations_v347/accepted_seeds_static.csv (first N_SEEDS entries).
#     Those seeds passed acceptance at w=40; under STATIC the topology is
#     frozen, so connectivity (t=60) and the 3-hop check (re-evaluated at each
#     window start) decide identically at any window length.
#   - No temp-id namespace: ids are pre-assigned (the original campaign ids),
#     each task writes only its own id, so no rename step and no collisions.
#   - STATIC ONLY. Under mobility the timing shift changes node positions at
#     the acceptance-check instants, which would silently change the sample
#     and break the pairing.
#
# WHAT IT RUNS
#   For each W in WINDOWS and each (id, seed): the scenario binary with
#   --windowSeconds=W. The probe rate is fixed in the scenario (one 512-byte
#   datagram / 2 s from +4 s), so the per-window datagram count is derived:
#   W=16->6, 28->12, 52->24, 64->30. W=40 is NOT run here — it already exists
#   as the main campaign (simulations_v347/), same seeds, same binary.
#
# OUTPUT LAYOUT (never touches simulations_v347/)
#   simulations_v347_windowsweep/
#     w<W>/features_static/{baseline,attack_only,defense_only,defense_vs_attack}/
#       metrics_output-<id>.csv  observer_metrics-<id>.csv  observer_detail-<id>.csv
#     w<W>/logs/run_<id>.log     one log per run (per-run, not accumulated:
#                                see the 12/8 reject-classifier bug, STATE §17.10)
#     w<W>/probe.csv             topology probes of the sweep runs only
#     sweep_status.csv           one row per (window,id): OK / FAILED rc
#
# IDEMPOTENT: a run whose 12 output files already exist is skipped, so the
# script can be re-launched after an interruption.
#
# USAGE
#   ./run_windowsweep.sh                     # all four windows, 2000 seeds
#   WINDOWS="16" N_SEEDS=50 ./run_windowsweep.sh    # pilot
# ============================================================================
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$ROOT/build/scratch/ns3.47-iolsr-tests-corrected-default"
# MODE=static (default) | mobile.  R#3.6 window sweep.
MODE="${MODE:-static}"
case "$MODE" in
  static) MOB=0; FEATDIR=features_static
          SRC="$ROOT/simulations_v347_hopablation_10k/manifests/C_all.csv" ;;
  mobile) MOB=1; FEATDIR=features_mobile
          SRC="$ROOT/simulations_v347_hopablation_mobile_10k/manifests/C_all.csv" ;;
  *) echo "REFUSING: MODE must be static or mobile, got $MODE"; exit 1 ;;
esac
OUT="$ROOT/simulations_v347_windowsweep"

WINDOWS="${WINDOWS:-16 28 52 64}"
N_SEEDS="${N_SEEDS:-2000}"
NUM_WORKERS="${NUM_WORKERS:-22}"
RUN_TIMEOUT="${RUN_TIMEOUT:-900}"
SCEN=(baseline attack_only defense_only defense_vs_attack)

[ -x "$BIN" ] || { echo "ERROR: binary not built: $BIN"; exit 1; }
[ -f "$SRC" ] || { echo "ERROR: seed list not found: $SRC"; exit 1; }
# Guard: never write anywhere but the sweep directory
case "$OUT" in */simulations_v347_windowsweep) : ;; *) echo "REFUSING: bad OUT $OUT"; exit 1 ;; esac

export LD_LIBRARY_PATH="$ROOT/build/lib:${LD_LIBRARY_PATH:-}"
cd "$ROOT"

STATUS="$OUT/sweep_status.csv"
mkdir -p "$OUT"
[ -f "$STATUS" ] || echo "window,run_id,seed,result,timestamp" > "$STATUS"
LOCK="$OUT/.lock"; : > "$LOCK"

for W in $WINDOWS; do
    for s in "${SCEN[@]}"; do mkdir -p "$OUT/w$W/$FEATDIR/$s"; done
    mkdir -p "$OUT/w$W/logs"
done

have_all() {  # $1=W  $2=id : all 12 outputs present and non-empty?
    local W="$1" id="$2" s
    for s in "${SCEN[@]}"; do
        [ -s "$OUT/w$W/$FEATDIR/$s/metrics_output-$id.csv" ]   || return 1
        [ -s "$OUT/w$W/$FEATDIR/$s/observer_metrics-$id.csv" ] || return 1
        [ -s "$OUT/w$W/$FEATDIR/$s/observer_detail-$id.csv" ]  || return 1
    done
    return 0
}

one() {  # $1=W  $2=id  $3=seed
    local W="$1" id="$2" seed="$3"
    if have_all "$W" "$id"; then return 0; fi
    local log="$OUT/w$W/logs/run_${id}.log"
    timeout "$RUN_TIMEOUT" "$BIN" \
        --run="$id" --RngRun="$seed" --bMobility="$MOB" --windowSeconds="$W" \
        --enforceHopFilter=0 \
        --outputDir="$OUT/w$W/$FEATDIR/" \
        --topologyProbeFile="$OUT/w$W/probe.csv" \
        > "$log" 2>&1
    local rc=$?
    local res=OK
    { [ $rc -eq 0 ] && have_all "$W" "$id"; } || res="FAILED rc=$rc"
    ( flock -x 9
      echo "$W,$id,$seed,$res,$(date '+%F %T')" >> "$STATUS" ) 9>"$LOCK"
    [ "$res" = OK ] || echo "[$W/$id] $res (log: $log)"
}
export -f one have_all
export BIN OUT RUN_TIMEOUT STATUS LOCK MOB FEATDIR

# bash -c below re-creates SCEN for the subshell (bash arrays don't export)
one_wrap() { SCEN=(baseline attack_only defense_only defense_vs_attack); one "$@"; }
export -f one_wrap

echo "============================================================"
echo "window sweep: windows [$WINDOWS], $N_SEEDS seeds, $NUM_WORKERS workers"
echo "mode:        $MODE  (bMobility=$MOB, $FEATDIR)"
echo "seed source: $SRC (first $N_SEEDS of the C_all manifest)"
echo "hop filter:  DISABLED (--enforceHopFilter=0), matching C_all"
echo "output:      $OUT/w<W>/  (simulations_v347/ is not touched)"
echo "started $(date '+%F %T')"
echo "============================================================"

for W in $WINDOWS; do
    # manifest columns are: seed,run_id,source,hop_category
    tail -n +2 "$SRC" | head -"$N_SEEDS" | cut -d, -f1,2 | tr ',' ' ' \
      | while read -r seed id; do echo "$W $id $seed"; done
done | xargs -P "$NUM_WORKERS" -n 3 bash -c 'one_wrap "$0" "$1" "$2"'

echo
echo "==================== DONE ===================="
echo "status by window:"
awk -F, 'NR>1{c[$1","$4]++} END{for(k in c) printf "  w%-4s %s\n", k, c[k]}' "$STATUS" | sort
echo "finished $(date '+%F %T')"
