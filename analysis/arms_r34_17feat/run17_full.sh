#!/bin/bash
# Full pipeline on the 17-observable space, isolated tree only.
#
# Stage 1 (defense_detection_v2) already ran on 24/8 -- 0.8995 static /
# 0.92825 mobile -- and is NOT repeated. This runs the rest:
#   2 cohens_d, 3 separability, 4 feature importance (selects the set),
#   5 K sweep + select_optimal_k, 8 threshold decomposition, 9 variance
#   decomposition.
# Stages 6-7 (the DJ ablations) are deliberately absent: their subject, the six
# delay/jitter metrics, does not exist in this space.
set -u
W="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-$HOME/miniconda3/envs/manet/bin/python}"
[ -x "$PY" ] || PY=python3
LOG=$W/run17_full.log
cd "$W" || exit 1

run_stage () {  # run_stage <script> [mode]
    local script=$1 mode=${2:-}
    echo "============================================================"
    echo "=== $script ${mode:+($mode)}   $(date '+%H:%M:%S')"
    echo "============================================================"
    if [ -n "$mode" ]; then
        DCFM_ARMS_ROOT="$W" DCFM_PIPELINE="$W/pipeline_17" \
            nice -n 19 "$PY" -u run_arm.py --arm listener \
            --script "$script" --mode "$mode"
    else
        DCFM_ARMS_ROOT="$W" DCFM_PIPELINE="$W/pipeline_17" \
            nice -n 19 "$PY" -u run_arm.py --arm listener --script "$script"
    fi
    local st=$?
    echo "=== $script finished, status $st   $(date '+%H:%M:%S')"
    return $st
}

{
echo "############################################################"
echo "# full 17-space pipeline, started $(date '+%F %H:%M:%S')"
echo "############################################################"
echo "preflight:"
echo -n "  ARMS: "; grep -m1 "^ARMS = " run_arm.py
echo -n "  stage-4 METRICS: "
"$PY" -c "import sys; sys.path.insert(0,'pipeline_17'); import feature_importance_sensitivity_v2 as f; print(len(f.METRICS))"
echo

# Stages 2, 3 and 9 are independent of stage 4 and of each other -- run them
# concurrently while stage 4 (the long one) works. The real dependency chain is
# only 4 -> 5 -> 8.
run_stage compute_cohens_d_v2.py     > "$W/.s2.log" 2>&1 || echo "!! stage 2 failed" &
P2=$!
run_stage compute_separability_v2.py > "$W/.s3.log" 2>&1 || echo "!! stage 3 failed" &
P3=$!
run_stage variance_decomposition_v2.py > "$W/.s9.log" 2>&1 || echo "!! stage 9 failed" &
P9=$!

# Stage 4 dominates the wall clock. runtime_guard's defaults (MAX_JOBS=2,
# OMP_NUM_THREADS=1) starve it -- STATE 5162: "the starvation is the thread cap,
# not a mis-wired flag". Both are setdefault, so exporting here wins. 10 workers
# is deliberate: the stage-1 OOM was at 16 with the nested stacking ensemble,
# which stage 4 does not build.
MAX_JOBS=10 OMP_NUM_THREADS=2 \
run_stage feature_importance_sensitivity_v2.py || { echo "!! stage 4 failed -- stages 5+ depend on it, ABORT"; kill $P2 $P3 $P9 2>/dev/null; exit 1; }

echo "=== waiting for stages 2, 3, 9 ==="
wait $P2 $P3 $P9
for s in 2 3 9; do echo "--- stage $s log tail ---"; tail -5 "$W/.s$s.log"; done

run_stage k_sweep_universal4_v2.py          || { echo "!! stage 5 failed -- 8 depends on it, ABORT"; exit 1; }

ks="$W/listener/results/k_sweep_universal4_v2"
if [ -f "$ks/k_sweep_results.csv" ]; then
    echo "=== select_optimal_k ==="
    nice -n 19 "$PY" select_optimal_k.py \
        --k-sweep-csv "$ks/k_sweep_results.csv" --out "$ks/optimal_k.json" \
        || echo "!! select_optimal_k failed"
fi

run_stage threshold_decomposition_v2.py     || echo "!! stage 8 failed, continuing"
run_stage variance_decomposition_v2.py      || echo "!! stage 9 failed, continuing"

echo
echo "=== outputs ==="
find "$W/listener/results" -newer "$W/run17_full.sh" -name "*.csv" -o \
     -newer "$W/run17_full.sh" -name "*.json" 2>/dev/null | head -20
echo "# done $(date '+%F %H:%M:%S')"
} 2>&1 | tee -a "$LOG"
