#!/usr/bin/env bash
# ============================================================================
# run_fractionsweep.sh — intermittent-activation sweep (reviewer #3, q.11).
#
# WHY (26/8/2026)
#   R#3.11: "the defense is modeled as either fully active or fully inactive
#   throughout each measurement window... how does the classifier behave when
#   the defense is only intermittently active during the window?"
#   The reviewer is right about the experimental switch: activeDefence is set
#   at the stabilization start and never cleared mid-window. This sweep varies
#   --defenseActiveFraction so the defense is switched off partway through each
#   DEFENDED window, and detection accuracy is measured as a function of it.
#
#   0% is baseline (already on disk) and 100% is the reported campaign
#   (already on disk). Only the interior points are simulated here.
#
# WHY MOBILE IS FINE HERE (unlike run_windowsweep.sh)
#   The window boundaries do not move: the window stays 40 s, only the defense
#   is deactivated inside it. AbortOnNeighbor and AssertConnectivity therefore
#   fire at exactly the same instants as the main campaign, so the admitted
#   seed set is identical and the comparison is paired in both modes.
#
# SEEDS
#   First N_SEEDS (seed,run_id) pairs of the C_all manifest of the reported
#   campaign — the same population, with --enforceHopFilter=0 as C_all requires.
#
# OUTPUT (never touches any existing campaign)
#   simulations_v347_fractionsweep/f<F>/features_{static|mobile}/<scenario>/
#   simulations_v347_fractionsweep/f<F>/logs/run_<id>.log
#   simulations_v347_fractionsweep/fraction_status.csv
#
# IDEMPOTENT: a run whose 12 outputs exist is skipped.
#
# USAGE
#   MODE=static ./run_fractionsweep.sh                  # 0.25 0.5 0.75, 2000 seeds
#   MODE=mobile FRACTIONS="0.5" N_SEEDS=50 ./run_fractionsweep.sh   # pilot
# ============================================================================
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$ROOT/build/scratch/ns3.47-iolsr-tests-corrected-default"

MODE="${MODE:-static}"
case "$MODE" in
  static) MOB=0; FEATDIR=features_static
          SRC="$ROOT/simulations_v347_hopablation_10k/manifests/C_all.csv" ;;
  mobile) MOB=1; FEATDIR=features_mobile
          SRC="$ROOT/simulations_v347_hopablation_mobile_10k/manifests/C_all.csv" ;;
  *) echo "REFUSING: MODE must be static or mobile, got $MODE"; exit 1 ;;
esac
OUT="$ROOT/simulations_v347_fractionsweep"

FRACTIONS="${FRACTIONS:-0.25 0.5 0.75}"
N_SEEDS="${N_SEEDS:-2000}"
NUM_WORKERS="${NUM_WORKERS:-16}"
RUN_TIMEOUT="${RUN_TIMEOUT:-900}"
SCEN=(baseline attack_only defense_only defense_vs_attack)

[ -x "$BIN" ] || { echo "ERROR: binary not built: $BIN"; exit 1; }
[ -f "$SRC" ] || { echo "ERROR: manifest not found: $SRC"; exit 1; }
# Guard: never write anywhere but the sweep directory
case "$OUT" in */simulations_v347_fractionsweep) : ;; *) echo "REFUSING: bad OUT $OUT"; exit 1 ;; esac
# Guard: the binary must understand the flag
"$BIN" --PrintHelp 2>&1 | grep -q "defenseActiveFraction" || {
    echo "ERROR: binary has no --defenseActiveFraction. Rebuild first:"
    echo "  ./ns3 build iolsr-tests-corrected"; exit 1; }

export LD_LIBRARY_PATH="$ROOT/build/lib:${LD_LIBRARY_PATH:-}"
cd "$ROOT"

STATUS="$OUT/fraction_status.csv"
mkdir -p "$OUT"
[ -f "$STATUS" ] || echo "fraction,mode,run_id,seed,result,timestamp" > "$STATUS"
LOCK="$OUT/.lock"; : > "$LOCK"

for F in $FRACTIONS; do
    for s in "${SCEN[@]}"; do mkdir -p "$OUT/f$F/$FEATDIR/$s"; done
    mkdir -p "$OUT/f$F/logs"
done

have_all() {   # $1=F  $2=id
    local F="$1" id="$2" s
    for s in baseline attack_only defense_only defense_vs_attack; do
        [ -s "$OUT/f$F/$FEATDIR/$s/metrics_output-$id.csv" ]   || return 1
        [ -s "$OUT/f$F/$FEATDIR/$s/observer_metrics-$id.csv" ] || return 1
        [ -s "$OUT/f$F/$FEATDIR/$s/observer_detail-$id.csv" ]  || return 1
    done
    return 0
}

one() {   # $1=F  $2=id  $3=seed
    local F="$1" id="$2" seed="$3"
    if have_all "$F" "$id"; then return 0; fi
    local log="$OUT/f$F/logs/run_${id}.log"
    timeout "$RUN_TIMEOUT" "$BIN" \
        --run="$id" --RngRun="$seed" --bMobility="$MOB" \
        --defenseActiveFraction="$F" --enforceHopFilter=0 \
        --outputDir="$OUT/f$F/$FEATDIR/" \
        --topologyProbeFile="$OUT/f$F/probe.csv" \
        > "$log" 2>&1
    local rc=$?
    local res=OK
    { [ $rc -eq 0 ] && have_all "$F" "$id"; } || res="FAILED rc=$rc"
    ( flock -x 9
      echo "$F,$MODE,$id,$seed,$res,$(date '+%F %T')" >> "$STATUS" ) 9>"$LOCK"
    [ "$res" = OK ] || echo "[$F/$id] $res (log: $log)"
}
export -f one have_all
export BIN OUT RUN_TIMEOUT STATUS LOCK MOB FEATDIR MODE

echo "============================================================"
echo "fraction sweep: fractions [$FRACTIONS], $N_SEEDS seeds, $NUM_WORKERS workers"
echo "mode:        $MODE  (bMobility=$MOB, $FEATDIR)"
echo "seed source: $SRC (first $N_SEEDS of the C_all manifest)"
echo "hop filter:  DISABLED (--enforceHopFilter=0), matching C_all"
echo "output:      $OUT/f<F>/   (no existing campaign is touched)"
echo "started $(date '+%F %T')"
echo "============================================================"

for F in $FRACTIONS; do
    # manifest columns: seed,run_id,source,hop_category
    tail -n +2 "$SRC" | head -"$N_SEEDS" | cut -d, -f1,2 | tr ',' ' ' \
      | while read -r seed id; do echo "$F $id $seed"; done
done | xargs -P "$NUM_WORKERS" -n 3 bash -c 'one "$0" "$1" "$2"'

echo
echo "==================== DONE ===================="
awk -F, 'NR>1{c[$1","$2","$5]++} END{for(k in c) printf "  %-28s %s\n", k, c[k]}' "$STATUS" | sort
echo "finished $(date '+%F %T')"
