#!/usr/bin/env bash
# ============================================================================
# run_fraction_all.sh — R#3.11 end to end, one command.
#
#   [0] build  ->  [1] smoke  ->  [2] campaigns  ->  [3] bundles + stage 1
#
# Produces the accuracy-vs-activation-fraction curve. 0% is baseline and 100%
# is the reported campaign; both already exist, so only 0.25/0.50/0.75 are run.
#
# EVERY bundle call passes --arm listener. Nothing else is ever built.
# Stops at the first failing phase. Idempotent: re-running skips finished work.
#
# USAGE
#   ./run_fraction_all.sh                 # full: both modes, 2000 seeds
#   SMOKE_ONLY=1 ./run_fraction_all.sh    # build + 50-seed smoke, then stop
#   MODES="static" ./run_fraction_all.sh  # static only
# ============================================================================
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-$HOME/miniconda3/envs/manet/bin/python}"
[ -x "$PY" ] || PY=python3
BIN="$ROOT/build/scratch/ns3.47-iolsr-tests-corrected-default"
SWEEP="$ROOT/simulations_v347_fractionsweep"
ARMSBASE="$ROOT/analysis/arms_fractionsweep"
LOG="$ROOT/fraction_all.log"

FRACTIONS="${FRACTIONS:-0.25 0.5 0.75}"
MODES="${MODES:-static mobile}"
N_SEEDS="${N_SEEDS:-2000}"
NUM_WORKERS="${NUM_WORKERS:-16}"
SMOKE_ONLY="${SMOKE_ONLY:-0}"

say () { echo; echo "############################################################"
         echo "# $* :: $(date '+%F %H:%M:%S')"
         echo "############################################################"; }
die () { echo "!! ABORT: $*"; exit 1; }

{
say "PHASE 0 — build (one scratch program; src/ is not touched)"
cd "$ROOT" || die "cd"
 ./ns3 build iolsr-tests-corrected 2>&1 | tail -15
"$BIN" --PrintHelp 2>&1 | grep -q "defenseActiveFraction" \
    || die "binary still has no --defenseActiveFraction"
echo "OK: --defenseActiveFraction present in the binary"

say "PHASE 1 — smoke: 50 seeds, fraction 0.5, static"
MODE=static FRACTIONS="0.5" N_SEEDS=50 NUM_WORKERS=8 \
    ./run_fractionsweep.sh 2>&1 | tail -12
grep -q "FAILED" "$SWEEP/fraction_status.csv" && die "smoke produced FAILED runs"
SF=$(ls "$SWEEP/f0.5/features_static/defense_only/"metrics_output-*.csv 2>/dev/null | head -1)
[ -n "$SF" ] || die "smoke wrote no defense_only output"
echo "smoke output present: $SF"
echo "--- the defense really was switched off mid-window? ---"
grep -m1 "Defense active fraction" "$SWEEP/f0.5/logs/"$(basename "$SF" | sed 's/metrics_output-/run_/;s/\.csv/.log/') 2>/dev/null \
  || grep -m1 -h "Defense active fraction" "$SWEEP/f0.5/logs/"*.log | head -1
[ "$SMOKE_ONLY" = "1" ] && { echo; echo "SMOKE_ONLY=1 — stopping here."; exit 0; }

say "PHASE 2 — campaigns"
for M in $MODES; do
    say "  campaign: mode=$M fractions=[$FRACTIONS] N=$N_SEEDS"
    MODE="$M" FRACTIONS="$FRACTIONS" N_SEEDS="$N_SEEDS" NUM_WORKERS="$NUM_WORKERS" \
         ./run_fractionsweep.sh 2>&1 | tail -12 \
        || die "campaign failed for mode=$M"
done
echo
echo "campaign totals:"
awk -F, 'NR>1{c[$1","$2","$5]++} END{for(k in c) printf "  %-30s %s\n", k, c[k]}' \
    "$SWEEP/fraction_status.csv" | sort

say "PHASE 3 — bundles (--arm listener ONLY) + stage 1"
for M in $MODES; do
  for F in $FRACTIONS; do
    A="$ARMSBASE/f${F}_${M}"
    mkdir -p "$A"
    if [ -s "$A/listener/results/$M/results.csv" ]; then
        echo "[skip] f$F/$M already has results.csv"; continue
    fi
    echo "--- bundle f$F $M ---"
    DCFM_SIM_ROOT="$SWEEP/f$F" DCFM_ARMS_ROOT="$A" \
       "$PY" analysis/make_arm_bundles.py --arm listener --mode "$M" \
      > "$A/bundle.log" 2>&1 || die "bundle failed f$F/$M (see $A/bundle.log)"
    ls "$A"/*/ -d 2>/dev/null | grep -qvE "listener/?$" \
      && echo "  !! WARNING: an arm other than listener exists under $A"
    echo "--- stage 1 f$F $M ---"
    DCFM_ARMS_ROOT="$A" \
       "$PY" -u analysis/run_arm.py --arm listener \
      --script defense_detection_v2.py --mode "$M" \
      -- --no-augmentation --split-validation --add-linearsvc --calibrate-stacking \
      > "$A/stage1.log" 2>&1 || echo "  !! stage 1 failed f$F/$M (see $A/stage1.log)"
  done
done

say "RESULTS — accuracy vs activation fraction"
printf "%-10s %-8s %-22s %s\n" "fraction" "mode" "best model" "accuracy"
for M in $MODES; do
  for F in $FRACTIONS; do
    R="$ARMSBASE/f${F}_${M}/listener/results/$M/results.csv"
    if [ -s "$R" ]; then
        best=$(tail -n +2 "$R" | cut -d, -f1,4 | sort -t, -k2 -rn | head -1)
        printf "%-10s %-8s %-22s %s\n" "$F" "$M" "${best%%,*}" "${best##*,}"
    else
        printf "%-10s %-8s %-22s %s\n" "$F" "$M" "-" "MISSING"
    fi
  done
done
echo
echo "anchors already on disk: 0% = baseline, 100% = the reported campaign"
echo "# done $(date '+%F %H:%M:%S')"
} 2>&1 | tee -a "$LOG"
