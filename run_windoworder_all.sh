#!/usr/bin/env bash
# ============================================================================
# run_windoworder_all.sh — the whole window-order experiment, both modes.
#
# Six serial stages (each needs the CPU to itself, and each depends on the one
# before it). Launch ONCE, detached:
#
#   cd ~/ns3/nv347/ns-3.47
#   nohup bash run_windoworder_all.sh >> windoworder_all.log 2>&1 &
#
#   tail -f windoworder_all.log        # follow
#
#   1. static   canonical  simulations   (--scenarioOrder=0,1,2,3)
#   2. static   shuffled   simulations   (--scenarioOrder=random, same seeds)
#   3. static   bundles + 2 learnings
#   4. mobile   canonical  simulations
#   5. mobile   shuffled   simulations
#   6. mobile   bundles + 2 learnings
#
# Both arms run with --formationLeadIn=60 so all four slots are structurally
# identical 100 s blocks (STATE §25.20). Canonical runs first in each mode
# because it fixes the seed set the shuffled arm must match.
#
# RESUMABLE: every stage skips itself if its output is already complete, so
# re-launching after an interruption continues where it stopped. Nothing is
# ever overwritten and no --force is used anywhere.
#
#   TARGET=500 bash run_windoworder_all.sh    # smaller/faster rehearsal
#   STAGES=1,2,3 bash run_windoworder_all.sh  # static only
# ============================================================================
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WO="${WO:-$ROOT/simulations_v347_windoworder}"
TARGET="${TARGET:-2000}"
NUM_WORKERS="${NUM_WORKERS:-22}"
MAX_JOBS="${MAX_JOBS:-22}"
STAGES="${STAGES:-1,2,3,4,5,6}"

want() { case ",$STAGES," in *",$1,"*) return 0;; *) return 1;; esac; }
banner() {
  echo
  echo "=========================================================="
  echo " STAGE $* "
  echo " $(date '+%F %T')"
  echo "=========================================================="
}
t0=$(date +%s)
elapsed() { local s=$(( $(date +%s) - t0 )); printf "%dh%02dm" $((s/3600)) $(((s%3600)/60)); }

if pgrep -f "defense_detection_v2.py" >/dev/null 2>&1; then
  echo "ABORT: a learning job is already running:"; pgrep -af "defense_detection_v2.py"; exit 1
fi

echo "window-order experiment — full sequence"
echo "  target per arm : $TARGET      workers: $NUM_WORKERS"
echo "  output         : $WO"
echo "  stages         : $STAGES"

sim_arm() { # $1=mode  $2=arm
  local mode="$1" arm="$2" mf="$WO/$2/manifest_$1.csv"
  if [ -f "$mf" ] && [ "$(( $(wc -l < "$mf") - 1 ))" -ge "$TARGET" ]; then
    echo "  [skip] $arm/$mode already has $(( $(wc -l < "$mf") - 1 )) accepted runs"
    return 0
  fi
  local extra=()
  [ "$arm" = shuffled ] && extra=(SEEDS_FROM="$WO/canonical/manifest_${mode}.csv")
  env ARM="$arm" MODE="$mode" OUT="$WO" TARGET="$TARGET" \
      NUM_WORKERS="$NUM_WORKERS" "${extra[@]}" \
      bash "$ROOT/run_windoworder.sh"
}

learn_mode() { # $1=mode
  MODE="$1" WO="$WO" MAX_JOBS="$MAX_JOBS" bash "$ROOT/run_windoworder_learn.sh"
}

for mode in static mobile; do
  case "$mode" in static) s1=1; s2=2; s3=3;; mobile) s1=4; s2=5; s3=6;; esac

  if want "$s1"; then
    banner "$s1/6  $mode — CANONICAL simulations   [elapsed $(elapsed)]"
    sim_arm "$mode" canonical || { echo "STAGE $s1 FAILED"; exit 1; }
  fi
  if want "$s2"; then
    banner "$s2/6  $mode — SHUFFLED simulations    [elapsed $(elapsed)]"
    sim_arm "$mode" shuffled  || { echo "STAGE $s2 FAILED"; exit 1; }
  fi
  if want "$s3"; then
    banner "$s3/6  $mode — bundles + learning x2   [elapsed $(elapsed)]"
    learn_mode "$mode" || { echo "STAGE $s3 FAILED"; exit 1; }
  fi
done

banner "DONE   [total $(elapsed)]"
echo
echo "  RESULTS — window order shuffled vs fixed, paired, listener arm"
printf "  %-8s %-11s %10s %10s\n" MODE ARM ACCURACY AUC
for mode in static mobile; do
  for arm in canonical shuffled; do
    f="$ROOT/analysis/arms_windoworder/$arm/listener/results/$mode/results.csv"
    a=$(awk -F, '$1=="Stacking_Ensemble"{print $4}' "$f" 2>/dev/null)
    u=$(awk -F, '$1=="Stacking_Ensemble"{print $3}' "$f" 2>/dev/null)
    printf "  %-8s %-11s %10.6s %10.6s\n" "$mode" "$arm" "${a:--}" "${u:--}"
  done
done
echo
echo "  A shuffled result close to canonical answers R#4 #1: the classifier is"
echo "  not keying on window position. Interpret WITHIN each mode only."
echo "  Follow-up available with no extra runs: the manifests record each run's"
echo "  slot assignment and every bundle row carries _StartTime, so accuracy can"
echo "  be broken down by slot offline (STATE §25.20)."
