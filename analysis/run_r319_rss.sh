#!/usr/bin/env bash
# run_r319_rss.sh -- R#3.19: runtime memory of the deployed detector, both modes.
#
# The gap this closes: 712.90 MB is a STORAGE footprint and 1,761 MB is a
# process that loaded all seventeen models. Neither is the RAM a deployment
# needs. measure_deployed_rss.py extracts the deployed artefact and measures
# resident memory in a FRESH process that loads nothing else.
#
# ⛔ Nothing is fitted; the artefact comes from the arm's stored pickle. The
# temporary copy is written under a mktemp directory and removed afterwards.
set -euo pipefail

AN="$HOME/ns3/nv347/ns-3.47/analysis"
PY="$HOME/miniconda3/envs/manet/bin/python"
ARM_NAME="${ARM_NAME:-arms_r34_17feat_frozencal}"
ARM="$AN/$ARM_NAME/listener"
LOG="$AN/r319_rss_$(date +%Y%m%d_%H%M%S).log"
export R23_PIPELINE="${R23_PIPELINE:-$AN/frozencal/pipeline_17}"

{
    echo "# R#3.19 deployed-process memory -- $(date '+%Y-%m-%d %H:%M:%S')"
    echo "# arm: $ARM"
    for mode in static mobile; do
        echo
        echo "########################################################################"
        echo "# $mode   ($(date '+%H:%M:%S'))"
        echo "########################################################################"
        "$PY" -u "$AN/measure_deployed_rss.py" --arm-root "$ARM" --mode "$mode"
    done
    echo
    echo "# DONE $(date '+%H:%M:%S')"
} 2>&1 | tee "$LOG"
echo "log: $LOG"
