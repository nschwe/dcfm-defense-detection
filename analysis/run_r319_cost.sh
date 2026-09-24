#!/usr/bin/env bash
# run_r319_cost.sh -- R#3.19: the whole cost measurement, in one log.
#
# WHAT IT RUNS
#   Four invocations of inference_cost_report.py on the reported campaign:
#
#     1. static, full width      3. static, one pinned core
#     2. mobile, full width      4. mobile, one pinned core
#
#   Each one walks every model stored in the arm's pickle (--all-models), so the
#   output is a measured size / latency / accuracy frontier rather than a single
#   number for the ensemble, and each one re-runs the real feature engineering
#   (--with-feature-eng), which is what makes the space check exact.
#
# WHY THE ONE-CORE PAIR EXISTS
#   Runs 3 and 4 limit BLAS/OMP to one thread and pin the process to one CPU,
#   which gives a less parallel reference point on the same machine.
#   ⛔ It is not a hardware emulation and not a bound on anything: nothing
#   measured here says what a slower or embedded processor would do.
#
# WHAT IT DOES NOT DO
#   ⛔ It never fits anything. Every estimator, scaler, feature list and
#   threshold comes from the arm's stored pickle.
#   ⛔ It writes only under inference_cost_out/<arm>/frontier/, so the 3/9 21:32
#   full-width CSVs and the 20/8 arms_c10k_stackcal CSVs are untouched. A
#   --threads run carries a _t<N> suffix, so it cannot overwrite a full-width
#   one either.
#
# GATES -- the tool aborts, and this script stops with it (pipefail + -e):
#   METRICS = 17; FrozenEstimator present; exactly one arm and it is listener;
#   bundle 27 columns; the model has a row in results.csv; every stored feature
#   name is reproduced by the gated generator; and the engineered and selected
#   counts equal the ones the arm's own stage-1 log recorded. That log also
#   supplies Total Runtime -- the training time the comment asks for.
#
# USAGE
#   cd ~/ns3/nv347/ns-3.47/analysis && bash run_r319_cost.sh
#   Idle machine: the tool refuses to time anything above load average 4.

set -euo pipefail

AN="$HOME/ns3/nv347/ns-3.47/analysis"
PY="$HOME/miniconda3/envs/manet/bin/python"
ARM_NAME="${ARM_NAME:-arms_r34_17feat_frozencal}"
ARM="$AN/$ARM_NAME/listener"
OUT="$AN/inference_cost_out/$ARM_NAME/frontier"
LOG="$AN/r319_cost_$(date +%Y%m%d_%H%M%S).log"

export R23_PIPELINE="${R23_PIPELINE:-$AN/frozencal/pipeline_17}"

for p in "$PY" "$AN/inference_cost_report.py" "$ARM" "$R23_PIPELINE"; do
    [ -e "$p" ] || { echo "ABORT: missing $p"; exit 1; }
done

{
    echo "########################################################################"
    echo "# R#3.19 cost measurement -- $(date '+%Y-%m-%d %H:%M:%S')"
    echo "# arm      : $ARM"
    echo "# pipeline : $R23_PIPELINE"
    echo "# out      : $OUT"
    echo "# host     : $(uname -srm), $(nproc) cpus, load $(cut -d' ' -f1-3 /proc/loadavg)"
    echo "########################################################################"

    for mode in static mobile; do
        for threads in 0 1; do
            label="$mode, full width"
            extra=()
            if [ "$threads" -ne 0 ]; then
                label="$mode, one pinned core"
                extra=(--threads 1)
            fi
            echo
            echo "########################################################################"
            echo "# RUN: $label   ($(date '+%H:%M:%S'))"
            echo "########################################################################"
            "$PY" -u "$AN/inference_cost_report.py" \
                --arm-root "$ARM" --mode "$mode" \
                --all-models --with-feature-eng \
                --out-dir "$OUT" "${extra[@]}"
        done
    done

    echo
    echo "########################################################################"
    echo "# DONE $(date '+%Y-%m-%d %H:%M:%S'). Files written:"
    ls -la "$OUT"
    echo "# Untouched, as intended:"
    ls -la "$AN/inference_cost_out/arms_r34_17feat_frozencal/"*.csv \
           "$AN/inference_cost_out/"*.csv
    echo "########################################################################"
} 2>&1 | tee "$LOG"

echo "log: $LOG"
