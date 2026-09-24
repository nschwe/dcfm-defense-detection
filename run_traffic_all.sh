#!/usr/bin/env bash
# ============================================================================
# run_traffic_all.sh — R#5.7 traffic-density sweep, end to end, one command.
#
#   [0] verify binary  ->  [1] smoke  ->  [2] campaigns  ->  [3] bundles + stage 1
#
# WHY (26/8/2026)
#   R#5.7: "eighteen packets per measurement window is very sparse... discuss
#   whether detection performance would hold under denser or bursty traffic."
#   The submitted campaign sends one 512-byte datagram every 2 s (18 per 40 s
#   window). This sweep shortens the interval so the same window carries more
#   traffic, and measures detection accuracy as a function of it.
#
#   --udpInterval feeds RecomputeWindowTiming(), so the per-window datagram
#   count follows automatically: (windowSeconds - 4) / udpInterval.
#     2.0 s -> 18 datagrams   (the reported campaign; already on disk)
#     1.0 s -> 36   (2x)
#     0.5 s -> 72   (4x)
#     0.25 s -> 144 (8x)
#   The default 2.0 is NOT run here.
#
# ⚠️ DENSITY ONLY, NOT BURSTINESS. UdpClient emits at a fixed interval; a bursty
#   source would need an OnOff application. The answer must say only density was
#   measured.
#
# EVERY bundle call passes --arm listener. Nothing else is ever built.
# Stops at the first failing phase. Idempotent: re-running skips finished work.
#
# USAGE
#   ./run_traffic_all.sh                   # both modes, 2000 seeds, 1.0 + 0.5
#   SMOKE_ONLY=1 ./run_traffic_all.sh      # 50-seed smoke, then stop
#   INTERVALS="0.5" MODES="static" ./run_traffic_all.sh
# ============================================================================
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-$HOME/miniconda3/envs/manet/bin/python}"
[ -x "$PY" ] || PY=python3
BIN="$ROOT/build/scratch/ns3.47-iolsr-tests-corrected-default"
SWEEP="$ROOT/simulations_v347_trafficsweep"
ARMSBASE="$ROOT/analysis/arms_trafficsweep"
# The reported campaign is the SINGLE-LISTENER, 17-OBSERVABLE arm
# (arms_r34_17feat). Stage 1 selects that space through DCFM_PIPELINE
# (run_arm.py:39). Without it run_arm.py falls back to analysis/pipeline, which
# carries 33 metrics, and this sweep would sit in a different feature space from
# the campaign it is compared against. Verified 26/8: pipeline_17 -> 17 METRICS,
# analysis/pipeline -> 33.
PIPE17="$ROOT/analysis/arms_r34_17feat/pipeline_17"
LOG="$ROOT/traffic_all.log"

INTERVALS="${INTERVALS:-1.0 0.5}"
MODES="${MODES:-static mobile}"
N_SEEDS="${N_SEEDS:-2000}"
NUM_WORKERS="${NUM_WORKERS:-22}"
RUN_TIMEOUT="${RUN_TIMEOUT:-900}"
SMOKE_ONLY="${SMOKE_ONLY:-0}"
NICE="${NICE:-}"
SCEN=(baseline attack_only defense_only defense_vs_attack)

say () { echo; echo "############################################################"
         echo "# $* :: $(date '+%F %H:%M:%S')"
         echo "############################################################"; }
die () { echo "!! ABORT: $*"; exit 1; }

# ---------------------------------------------------------------- campaign --
run_campaign () {   # $1=mode
    local M="$1" MOB FEATDIR SRC I
    case "$M" in
      static) MOB=0; FEATDIR=features_static
              SRC="$ROOT/simulations_v347_hopablation_10k/manifests/C_all.csv" ;;
      mobile) MOB=1; FEATDIR=features_mobile
              SRC="$ROOT/simulations_v347_hopablation_mobile_10k/manifests/C_all.csv" ;;
      *) die "bad mode $M" ;;
    esac
    [ -f "$SRC" ] || die "manifest not found: $SRC"

    local STATUS="$SWEEP/traffic_status.csv"
    mkdir -p "$SWEEP"
    [ -f "$STATUS" ] || echo "interval,mode,run_id,seed,result,timestamp" > "$STATUS"
    local LOCK="$SWEEP/.lock"; : > "$LOCK"

    for I in $INTERVALS; do
        for s in "${SCEN[@]}"; do mkdir -p "$SWEEP/i$I/$FEATDIR/$s"; done
        mkdir -p "$SWEEP/i$I/logs"
    done

    have_all () {   # $1=I  $2=id
        local I="$1" id="$2" s
        for s in baseline attack_only defense_only defense_vs_attack; do
            [ -s "$SWEEP/i$I/$FEATDIR/$s/metrics_output-$id.csv" ]   || return 1
            [ -s "$SWEEP/i$I/$FEATDIR/$s/observer_metrics-$id.csv" ] || return 1
            [ -s "$SWEEP/i$I/$FEATDIR/$s/observer_detail-$id.csv" ]  || return 1
        done
        return 0
    }
    one () {   # $1=I  $2=id  $3=seed
        local I="$1" id="$2" seed="$3"
        if have_all "$I" "$id"; then return 0; fi
        local log="$SWEEP/i$I/logs/run_${id}.log"
        timeout "$RUN_TIMEOUT" "$BIN" \
            --run="$id" --RngRun="$seed" --bMobility="$MOB" \
            --udpInterval="$I" --enforceHopFilter=0 \
            --outputDir="$SWEEP/i$I/$FEATDIR/" \
            --topologyProbeFile="$SWEEP/i$I/probe.csv" \
            > "$log" 2>&1
        local rc=$? res=OK
        { [ $rc -eq 0 ] && have_all "$I" "$id"; } || res="FAILED rc=$rc"
        ( flock -x 9; echo "$I,$M,$id,$seed,$res,$(date '+%F %T')" >> "$STATUS" ) 9>"$LOCK"
        [ "$res" = OK ] || echo "[$I/$id] $res (log: $log)"
    }
    export -f one have_all
    export BIN SWEEP RUN_TIMEOUT STATUS LOCK MOB FEATDIR M INTERVALS

    echo "mode=$M  intervals=[$INTERVALS]  N=$N_SEEDS  workers=$NUM_WORKERS"
    echo "seeds: $SRC (first $N_SEEDS)   hop filter DISABLED"
    # manifest columns: seed,run_id,source,hop_category
    # ⛔ Do NOT use `tail -n +2 "$SRC" | head -"$N_SEEDS"` here. The manifest holds
    # 10,000 rows; head closes the pipe at N_SEEDS, tail keeps writing and takes
    # SIGPIPE, and `set -o pipefail` surfaces that as rc=141. It is invisible while
    # xargs is busy simulating, and becomes the pipeline's status the moment every
    # run is already on disk -- so a COMPLETED campaign aborts on re-entry.
    # Measured 26/8: producer alone rc=141, 4000 lines, 0 ids missing.
    # One awk pass reads the whole file and closes nothing early.
    for I in $INTERVALS; do
        awk -F, -v n="$N_SEEDS" -v i="$I" \
            'NR>1 && NR<=n+1 { print i, $2, $1 }' "$SRC"
    done | xargs -P "$NUM_WORKERS" -n 3 bash -c 'one "$0" "$1" "$2"'
}

{
say "PHASE 0 — verify the binary carries --udpInterval"
cd "$ROOT" || die "cd"
[ -x "$BIN" ] || die "binary not built: $BIN"
"$BIN" --PrintHelp 2>&1 | grep -q "udpInterval" \
    || die "binary has no --udpInterval. Build first: ./ns3 build iolsr-tests-corrected"
echo "OK: --udpInterval present"

say "PHASE 1 — smoke: 50 seeds, interval 0.5 s (72 datagrams), static"
( INTERVALS="0.5" N_SEEDS=50 NUM_WORKERS=8; run_campaign static ) 2>&1 | tail -6
grep -q "FAILED" "$SWEEP/traffic_status.csv" 2>/dev/null && die "smoke produced FAILED runs"
SF=$(ls "$SWEEP/i0.5/features_static/defense_only/"metrics_output-*.csv 2>/dev/null | head -1)
[ -n "$SF" ] || die "smoke wrote no defense_only output"
echo "smoke output: $SF"
echo "--- datagrams per window actually produced (expect 72) ---"
grep -m1 ",UdpPacketsExpected," "$SF" | cut -d, -f1-3
echo "--- the simulator's own line ---"
grep -m1 -h "packetsPerWindow" "$SWEEP/i0.5/logs/"*.log | head -1
[ "$SMOKE_ONLY" = "1" ] && { echo; echo "SMOKE_ONLY=1 — stopping here."; exit 0; }

say "PHASE 2 — campaigns"
for M in $MODES; do
    say "  campaign: mode=$M"
    # ⛔ The exit status of a 22-worker xargs pipeline is not a sound success
    # test (see the SIGPIPE note above, and xargs' own 123-on-any-failure rule).
    # The authoritative record is the FAILED column the workers write themselves.
    before=$(grep -c ",FAILED" "$SWEEP/traffic_status.csv" 2>/dev/null) || before=0
    before=${before:-0}
    run_campaign "$M" 2>&1 | tail -8
    after=$(grep -c ",FAILED" "$SWEEP/traffic_status.csv" 2>/dev/null) || after=0
    after=${after:-0}
    [ "$after" -eq "$before" ] \
        || die "campaign mode=$M produced $((after-before)) FAILED runs (see $SWEEP/traffic_status.csv)"
    echo "  mode=$M OK: no new FAILED rows"
done
echo
echo "campaign totals:"
awk -F, 'NR>1{c[$1","$2","$5]++} END{for(k in c) printf "  %-30s %s\n", k, c[k]}' \
    "$SWEEP/traffic_status.csv" | sort

say "PHASE 3 — bundles (--arm listener ONLY) + stage 1"
for M in $MODES; do
  for I in $INTERVALS; do
    A="$ARMSBASE/i${I}_${M}"
    mkdir -p "$A"
    if [ -s "$A/listener/results/$M/results.csv" ]; then
        echo "[skip] i$I/$M already has results.csv"; continue
    fi
    echo "--- bundle i$I $M ---"
    DCFM_SIM_ROOT="$SWEEP/i$I" DCFM_ARMS_ROOT="$A" \
      $NICE "$PY" analysis/make_arm_bundles.py --arm listener --mode "$M" \
      > "$A/bundle.log" 2>&1 || die "bundle failed i$I/$M (see $A/bundle.log)"
    ls -d "$A"/*/ 2>/dev/null | grep -qvE "listener/?$" \
      && echo "  !! WARNING: an arm other than listener exists under $A"
    echo "--- stage 1 i$I $M ---"
    DCFM_ARMS_ROOT="$A" DCFM_PIPELINE="$PIPE17" \
      $NICE "$PY" -u analysis/run_arm.py --arm listener \
      --script defense_detection_v2.py --mode "$M" \
      -- --no-augmentation --split-validation --add-linearsvc --calibrate-stacking \
      > "$A/stage1.log" 2>&1 || echo "  !! stage 1 failed i$I/$M (see $A/stage1.log)"
  done
done

say "RESULTS — accuracy vs traffic density"
printf "%-10s %-10s %-8s %-22s %s\n" "interval" "datagrams" "mode" "best model" "accuracy"
for M in $MODES; do
  for I in $INTERVALS; do
    R="$ARMSBASE/i${I}_${M}/listener/results/$M/results.csv"
    NPK=$(awk -v i="$I" 'BEGIN{printf "%d", (40-4)/i}')
    if [ -s "$R" ]; then
        best=$(tail -n +2 "$R" | cut -d, -f1,4 | sort -t, -k2 -rn | head -1)
        printf "%-10s %-10s %-8s %-22s %s\n" "$I" "$NPK" "$M" "${best%%,*}" "${best##*,}"
    else
        printf "%-10s %-10s %-8s %-22s %s\n" "$I" "$NPK" "$M" "-" "MISSING"
    fi
  done
done
echo
echo "anchor already on disk: interval 2.0 s = 18 datagrams = the reported campaign"
echo "# done $(date '+%F %H:%M:%S')"
} 2>&1 | tee -a "$LOG"
