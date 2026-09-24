#!/usr/bin/env bash
# ============================================================================
# run_campaign_v347.sh  —  the heavy ns-3.47 isolation-attack campaign.
#
# Runs candidate seeds until TARGET_ACCEPTED are accepted, writing the paper's
# directory layout:
#
#   simulations_v347/
#     features_static/  |  features_mobile/
#         baseline/ attack_only/ defense_only/ defense_vs_attack/
#             metrics_output-<id>.csv
#             observer_metrics-<id>.csv   (10 vantage rows + 1 global row idx -1)
#             observer_detail-<id>.csv
#     logs_parallel_<mode>/
#         worker_<w>.log
#     run_status_<mode>.csv       one row per attempt: id,seed,result,reason,ts
#     accepted_seeds_<mode>.csv   id,seed  (contiguous ids 1..N, for reproducibility)
#     topology_probes_<mode>.csv
#
# Acceptance (paper criterion): network fully connected at t=60 AND the UDP sender
# is >=3 hops from the victim (1-2 hop packets do not use the TC table and are
# unaffected by the isolation attack). Every REJECT is classified and counted.
#
# Accepted runs get CONTIGUOUS ids (1..N) so the learning pipeline and the bundle
# builder see a dense id space; the id->seed map is recorded for replay.
#
# Writes ONLY under simulations_v347/. Idempotent: a completed run (all 12 output
# files present for its id) is not recomputed.
#
# Usage (env-overridable):
#   MODE=static TARGET_ACCEPTED=10000 NUM_WORKERS=22 ./run_campaign_v347.sh
#   MODE=mobile TARGET_ACCEPTED=10000                ./run_campaign_v347.sh
# ============================================================================
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$ROOT/build/scratch/ns3.47-iolsr-tests-corrected-default"

MODE="${MODE:-static}"
if [ "$MODE" = mobile ]; then MOB=1; else MOB=0; MODE=static; fi
TARGET_ACCEPTED="${TARGET_ACCEPTED:-10000}"
NUM_WORKERS="${NUM_WORKERS:-22}"
RUN_TIMEOUT="${RUN_TIMEOUT:-900}"
MAX_SEEDS="${MAX_SEEDS:-2000000}"          # hard safety cap on the search
TMP_ID_BASE=10000000                       # temp-id offset; must exceed MAX_SEEDS
                                           # and TARGET_ACCEPTED (see worker())
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-30}"

OUT="$ROOT/simulations_v347"
FEAT="$OUT/features_${MODE}"
LOGDIR="$OUT/logs_parallel_${MODE}"
STATUS="$OUT/run_status_${MODE}.csv"
ACCEPTED="$OUT/accepted_seeds_${MODE}.csv"
PROBE="$OUT/topology_probes_${MODE}.csv"
SCEN=(baseline attack_only defense_only defense_vs_attack)

if [ ! -x "$BIN" ]; then
    echo "ERROR: scenario binary not built: $BIN"
    echo "  build: cd $ROOT && ./ns3 build iolsr-tests-corrected"
    exit 1
fi

# Guard: never write anywhere but simulations_v347/
case "$OUT" in */simulations_v347) : ;; *) echo "REFUSING: bad OUT $OUT"; exit 1 ;; esac

export LD_LIBRARY_PATH="$ROOT/build/lib:${LD_LIBRARY_PATH:-}"
cd "$ROOT"

mkdir -p "$LOGDIR"
for s in "${SCEN[@]}"; do mkdir -p "$FEAT/$s"; done
[ -f "$STATUS" ]   || echo "run_id,seed,result,reason,timestamp" > "$STATUS"
[ -f "$ACCEPTED" ] || echo "run_id,seed" > "$ACCEPTED"

# ---- shared, flock-guarded state -------------------------------------------
STATE="$OUT/.state_${MODE}"; mkdir -p "$STATE"
LOCK="$STATE/lock"; : > "$LOCK"
# resume-aware: next id continues past whatever is already accepted
[ -f "$STATE/next_seed" ] || echo 0 > "$STATE/next_seed"
if [ ! -f "$STATE/accepted" ]; then
    a=$(($(wc -l < "$ACCEPTED") - 1)); [ "$a" -lt 0 ] && a=0; echo "$a" > "$STATE/accepted"
fi

outputs_present() {  # $1 = run_id
    local id="$1" s
    for s in "${SCEN[@]}"; do
        [ -s "$FEAT/$s/metrics_output-$id.csv" ]  || return 1
        [ -s "$FEAT/$s/observer_metrics-$id.csv" ] || return 1
        [ -s "$FEAT/$s/observer_detail-$id.csv" ]  || return 1
    done
    return 0
}

cleanup_id() {  # remove partial outputs for a rejected/failed id
    local id="$1" s
    for s in "${SCEN[@]}"; do
        rm -f "$FEAT/$s/metrics_output-$id.csv" \
              "$FEAT/$s/observer_metrics-$id.csv" \
              "$FEAT/$s/observer_detail-$id.csv"
    done
}

record() {  # id seed result reason
    ( flock -x 9
      echo "$1,$2,$3,$4,$(date '+%F %T')" >> "$STATUS" ) 9>"$LOCK"
}

# Atomically claim the next (seed, id). Accepted ids are contiguous: an id is only
# consumed on ACCEPT, so a rejected seed does not burn an id.
claim() {
    ( flock -x 9
      local acc ns
      acc=$(cat "$STATE/accepted"); ns=$(cat "$STATE/next_seed")
      if (( acc >= TARGET_ACCEPTED || ns >= MAX_SEEDS )); then echo "STOP STOP"
      else echo $((ns+1)) > "$STATE/next_seed"; echo "$((ns+1)) $((acc+1))"; fi
    ) 9>"$LOCK"
}

commit_accept() {  # id seed -> assign the contiguous id, record; or FULL
    ( flock -x 9
      local acc; acc=$(cat "$STATE/accepted")
      # Cap at exactly TARGET_ACCEPTED. A run that finishes AFTER the target was
      # reached (up to NUM_WORKERS were in flight when it crossed) is surplus:
      # not committed, so the accepted count is exact. The caller logs it.
      if (( acc >= TARGET_ACCEPTED )); then echo "FULL"; else
        local newid=$((acc+1))
        echo "$newid" > "$STATE/accepted"
        echo "$newid,$2" >> "$ACCEPTED"
        echo "$newid"
      fi ) 9>"$LOCK"
}

# ---- worker ----------------------------------------------------------------
worker() {
    local wid=$1
    local wlog="$LOGDIR/worker_${wid}.log"
    : > "$wlog"
    while :; do
        local claimed seed provid
        claimed=$(claim); seed=${claimed% *}; provid=${claimed#* }
        [ "$seed" = STOP ] && break

        # Run into a TEMP id namespace, then rename to a contiguous id only on
        # accept, so ids stay dense without cross-worker id races.
        #
        # The temp namespace MUST be disjoint from the accepted-id namespace.
        # It used to be the bare seed, which shares the filename space with the
        # accepted ids: while the seed counter and the accepted counter are still
        # close (only in a campaign's first seconds, before the ~37% accept rate
        # pulls them apart), one worker's cleanup_id/mv on temp id N could hit
        # another worker's already-committed file for accepted id N. That cost
        # the 11/8 static campaign ids 2 and 13 outright, and crossed one window
        # of id 16 with a different run. Offsetting past MAX_SEEDS makes the two
        # spaces provably disjoint. g_currentRun only names output files (it is
        # not RngRun), so the offset cannot change any simulation result.
        local tmp=$((seed + TMP_ID_BASE))
        cleanup_id "$tmp"
        echo "[w$wid] try seed=$seed" >> "$wlog"
        # This run's output goes to a PER-RUN file first. The reject classifier
        # below greps it, and grepping the accumulated worker log instead is a
        # bug: the log holds every earlier run of this worker, so the first
        # branch that ever matched wins for the rest of the campaign. That made
        # the 12/8 campaign report 99.96% "connectivity" when the true share was
        # 78.5% static / 84.4% mobile, and hid ~8,700 distance_lt3hops rejects.
        local rlog="$LOGDIR/.run_${wid}.tmp"
        : > "$rlog"
        timeout "$RUN_TIMEOUT" "$BIN" \
            --run="$tmp" --RngRun="$seed" --bMobility="$MOB" \
            --outputDir="$FEAT/" --topologyProbeFile="$PROBE" \
            >> "$rlog" 2>&1
        local rc=$?
        cat "$rlog" >> "$wlog"

        if [ $rc -eq 0 ] && outputs_present "$tmp"; then
            local id; id=$(commit_accept "$tmp" "$seed")
            if [ "$id" = FULL ]; then
                # Target already reached while this run was in flight. Log it (not a
                # silent drop) and stop this worker so the count stays exact.
                record "-" "$seed" SURPLUS target_reached
                cleanup_id "$tmp"
                echo "[w$wid] SURPLUS seed=$seed (target reached)" >> "$wlog"
                break
            fi
            local s
            for s in "${SCEN[@]}"; do
                mv -f "$FEAT/$s/metrics_output-$tmp.csv"  "$FEAT/$s/metrics_output-$id.csv"
                mv -f "$FEAT/$s/observer_metrics-$tmp.csv" "$FEAT/$s/observer_metrics-$id.csv"
                mv -f "$FEAT/$s/observer_detail-$tmp.csv"  "$FEAT/$s/observer_detail-$id.csv"
            done
            record "$id" "$seed" ACCEPT outputs_ok
            echo "[w$wid] ACCEPT seed=$seed -> id=$id" >> "$wlog"
        else
            local reason
            if   grep -q "Assert connectivity failed" "$rlog"; then reason=connectivity
            elif grep -q "within two hops of victim"   "$rlog"; then reason=distance_lt3hops
            elif grep -q "Attacker not available"       "$rlog"; then reason=attacker_unavailable
            elif (( rc == 124 ));                             then reason=timeout
            else                                                   reason=other
            fi
            record "-" "$seed" REJECT "$reason"
            cleanup_id "$tmp"
        fi
    done
}

# ---- live progress monitor -------------------------------------------------
monitor() {
    local start; start=$(date +%s)
    while :; do
        sleep "$PROGRESS_INTERVAL"
        local acc tried el spa eta arate
        acc=$(cat "$STATE/accepted" 2>/dev/null || echo 0)
        tried=$(( $(wc -l < "$STATUS") - 1 ))
        el=$(( $(date +%s) - start ))
        eta="?"
        if (( acc > 0 )); then
            spa=$(( el / acc ))                              # seconds per accept
            eta="$(( spa * (TARGET_ACCEPTED - acc) / 60 ))m"
        fi
        # accept-rate %, pure bash integer arithmetic (no awk, no ternary quirks)
        arate=0
        (( tried > 0 )) && arate=$(( 100 * acc / tried ))
        printf "[%s] %s  accepted %d/%d  tried %d  accept-rate %d%%  elapsed %dm  eta %s\n" \
            "$(date '+%T')" "$MODE" "$acc" "$TARGET_ACCEPTED" "$tried" \
            "$arate" $(( el / 60 )) "$eta"
        (( acc >= TARGET_ACCEPTED )) && break
    done
}

echo "============================================================"
echo "ns-3.47 campaign   mode=$MODE   target=$TARGET_ACCEPTED   workers=$NUM_WORKERS"
echo "output: $FEAT"
echo "accept = fully connected AND sender >=3 hops from victim"
echo "started $(date '+%F %T')"
echo "============================================================"

monitor & MON=$!
PIDS=()
for (( w=0; w<NUM_WORKERS; w++ )); do worker "$w" & PIDS+=("$!"); done
for pid in "${PIDS[@]}"; do wait "$pid"; done
kill "$MON" 2>/dev/null

ACC=$(cat "$STATE/accepted"); TRIED=$(( $(wc -l < "$STATUS") - 1 ))
echo
echo "==================== DONE ($MODE) ===================="
echo "accepted        : $ACC"
echo "seeds tried     : $TRIED"
echo "rejections by reason:"
awk -F, 'NR>1 && $3=="REJECT"{c[$4]++} END{for(r in c) printf "  %-22s %d\n", r, c[r]}' "$STATUS"
echo "outputs  : $FEAT/<scenario>/"
echo "id->seed : $ACCEPTED"
echo "status   : $STATUS"
echo "finished $(date '+%F %T')"
