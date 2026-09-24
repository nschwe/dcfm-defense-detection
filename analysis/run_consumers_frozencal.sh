#!/usr/bin/env bash
# STATE 25.188.9 step (1): re-run the pickle-consuming analyses against the
# corrected (frozencal) models. Each consumer writes to its own frozencal out-dir
# so the old outputs stay on disk for comparison.
#
# Usage: bash run_consumers_frozencal.sh [name ...]   (default: all)
set -u
AN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-$HOME/miniconda3/envs/manet/bin/python}"
[ -x "$PY" ] || PY=python3
FZ=$AN/frozencal/pipeline_17
TS=$(date +%Y%m%d_%H%M%S)
LOGDIR=$AN/frozencal_consumer_logs; mkdir -p "$LOGDIR"
GITADD=$AN/frozencal_gitadd_consumers_${TS}.txt
MANIFEST=$AN/frozencal_marathon_manifest.md
cd "$AN" || exit 1

log_row() {  # name logfile status note
    echo "| \`frozencal_consumer_logs/$(basename "$2")\` | consumer: $1 (frozencal models) | $3 $4 |" >> "$MANIFEST"
    echo "analysis/frozencal_consumer_logs/$(basename "$2")" >> "$GITADD"
}

run() {      # name  command...
    local name=$1; shift
    local lg=$LOGDIR/${name}_${TS}.log
    echo "--- $name  $(date '+%T')"
    ( "$@" ) > "$lg" 2>&1
    local st=$?
    if [ $st -eq 0 ]; then
        echo "OK   $name"; log_row "$name" "$lg" "✅" "exit 0"
    else
        echo "FAIL $name (exit $st) -> $lg"; log_row "$name" "$lg" "⛔" "exit $st"
    fi
    tail -3 "$lg" | sed 's/^/     /'
    return 0     # never abort the batch: consumers are independent
}

want() { [ $# -eq 0 ] && return 0; return 1; }
SEL=("$@")
sel() { [ ${#SEL[@]} -eq 0 ] && return 0; for s in "${SEL[@]}"; do [ "$s" = "$1" ] && return 0; done; return 1; }

# --- R#3.11: fraction sweep scored against the CORRECTED campaign model -------
# FRAC_PIPELINE/FRAC_REPORTED let it read the frozencal arm instead of the
# reported one (both are 77-column, so the frozen scaler still matches).
if sel frac; then
  run fraction_frozen_score env \
      FRAC_PIPELINE="$FZ" \
      FRAC_REPORTED="$AN/arms_r34_17feat_frozencal" \
      "$PY" -u "$AN/arms_fractionsweep/fraction_frozen_score.py" \
      --mode both --out-dir "$AN/arms_fractionsweep/frozen17_frozencal"
fi

# --- R#2.3 / R#5.3: realistic-channel paired bootstrap, each arm vs its own baseline
if sel r23; then
  for m in static mobile; do
    # R23_PIPELINE must be the 77-column generator the frozencal arms were
    # relearned with; its default (cleanfeat, 67 columns) would not match the
    # scaler stored in the new pickles.
    run "r23_prop_$m" env R23_PIPELINE="$FZ" "$PY" -u "$AN/r23_paired_bootstrap.py" \
        --realistic "$AN/arms_propmodel_17" \
        --baseline  "$AN/arms_propmodel_baseline_17" \
        --mode "$m" --model Stacking_Ensemble \
        --out-dir "$AN/r23_paired_bootstrap_frozencal"
    run "r23_rl42_$m" env R23_PIPELINE="$FZ" "$PY" -u "$AN/r23_paired_bootstrap.py" \
        --realistic "$AN/arms_propmodel_rl42_17" \
        --baseline  "$AN/arms_propmodel_rl42_baseline_17" \
        --mode "$m" --model Stacking_Ensemble \
        --out-dir "$AN/r23_paired_bootstrap_frozencal"
  done
fi

# --- R#2.4 / R#2.8: cost-sensitive thresholds + the flat-curve ceiling --------
if sel cost; then
  for m in static mobile; do
    run "cost_sensitive_$m" "$PY" -u "$AN/cost_sensitive_threshold.py" \
        --arm-root "$AN/arms_r34_17feat_frozencal/listener" --mode "$m"
    run "posthoc_$m" "$PY" -u "$AN/threshold_posthoc_check.py" \
        --arm-root "$AN/arms_r34_17feat_frozencal/listener" --mode "$m"
  done
fi

# --- R#4.1: window-order accuracy by slot ------------------------------------
if sel wo; then
  for m in static mobile; do
    run "windoworder_$m" "$PY" -u "$AN/windoworder_by_slot.py" \
        --arms-root "$AN/arms_r34_17feat_frozencal/listener" --mode "$m" \
        --arm canonical
  done
fi

echo
echo "=== consumers done $(date '+%F %T') ==="
echo "manifest rows appended to $MANIFEST"
echo "git list: $GITADD"
