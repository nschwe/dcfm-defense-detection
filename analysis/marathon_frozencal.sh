#!/usr/bin/env bash
# marathon_frozencal.sh -- STATE 25.188.5 re-run queue under the corrected
# pipeline (frozencal/pipeline_17: FrozenEstimator calibration + stacking from
# raw models). Runs stage 1 for every quoted arm, IN PAIRS (two concurrent
# runs, the measured-safe parallelism of 25.188.8).
#
# Per run: backs up the existing results.csv, runs with --force, checks the
# [frozen-calibration] N/N gate and FINAL RESULTS, and appends a manifest row
# to frozencal_marathon_manifest.md (merged into STATE 25.188.7 afterwards).
# A failed gate STOPS the whole queue -- no ploughing on past a broken run.
#
# Usage:  bash marathon_frozencal.sh            # run the queue
#         bash marathon_frozencal.sh --dry-run  # print what would run
set -u
AN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER=$AN/arms_r34_17feat/run_arm.py
PIPE=${PIPE:-$AN/frozencal/pipeline_17}
PY="${PY:-$HOME/miniconda3/envs/manet/bin/python}"
[ -x "$PY" ] || PY=python3
TS=$(date +%Y%m%d_%H%M%S)
LOGDIR=$AN/frozencal_marathon_logs
MANIFEST=$AN/frozencal_marathon_manifest.md
GITADD=$AN/frozencal_gitadd_${TS}.txt   # exactly what this round produced, for git add
CAMPAIGN_FLAGS="--no-augmentation --split-validation --add-linearsvc --calibrate-stacking"
NOSPLIT_FLAGS="--no-augmentation --add-linearsvc --calibrate-stacking"
DRY=${1:-}

# ---------------------------------------------------------------- the queue
# "arm_root|flagset"  -- modes are auto-detected from the bundle files.
# flagset: campaign (default) | nosplit (the R#3.13 submitted-protocol arm).
QUEUE=(
  "arms_fractionsweep/f0.25_static|campaign"
  "arms_fractionsweep/f0.25_mobile|campaign"
  "arms_fractionsweep/f0.5_static|campaign"
  "arms_fractionsweep/f0.5_mobile|campaign"
  "arms_fractionsweep/f0.75_static|campaign"
  "arms_fractionsweep/f0.75_mobile|campaign"
  "arms_windowsweep/w10_17|campaign"
  "arms_windowsweep/w20_17|campaign"
  "arms_trafficsweep/i0.5_static|campaign"
  "arms_trafficsweep/i0.5_mobile|campaign"
  "arms_trafficsweep/i1.0_static|campaign"
  "arms_trafficsweep/i1.0_mobile|campaign"
  "arms_gm_control_17|campaign"
  "arms_gm10k_17|campaign"
  "arms_propmodel_17|campaign"
  "arms_propmodel_baseline_17|campaign"
  "arms_propmodel_rl42_17|campaign"
  "arms_propmodel_rl42_baseline_17|campaign"
  "arms_r34_17feat_2k_w_17|campaign"
  "arms_r34_17feat_nosplit|nosplit"
)
# Single-arm re-run (3/9): the queue above is the 2/9 round. To re-learn one
# arm that the round missed -- arms_r34_17feat_2k, R#3.6's 1,200-run training
# volume arm -- pass it in without editing the list:
#     QUEUE_OVERRIDE="arms_r34_17feat_2k|campaign" bash marathon_frozencal.sh
# Everything else (backup, pickle archiving, --force, both gates, the manifest
# row and the git-add list) is unchanged, so a single arm goes through exactly
# the path the 29 did.
if [ -n "${QUEUE_OVERRIDE:-}" ]; then
    QUEUE=( ${QUEUE_OVERRIDE} )
fi

# NOT in this queue, on purpose:
#  - arms_r34_17feat_frozencal      -- already run (the 25.188.8 gate pair)
#  - thrbias, FPNT                  -- their own pipeline copies need the two
#                                      edits first (25.188.8 decision 1)
#  - windoworder / hopablation-window arms -- provenance (which arm fed
#                                      r4_1/3.7/5.5) to be pinned down first

# ---------------------------------------------------------------- helpers
expand_jobs() {           # arm_root|flagset -> lines "arm_root|mode|flagset"
    local root=$1 flags=$2 cd_=$AN/$1/listener/colab_data m
    for m in static mobile; do
        if ls "$cd_/wide_${m}."* >/dev/null 2>&1; then
            echo "${root}|${m}|${flags}"
        fi
    done
}

run_one() {               # root mode flagset  (foreground; caller backgrounds)
    local root=$1 mode=$2 flagset=$3
    local name=$(echo "$root" | tr '/' '_')_${mode}
    local log=$LOGDIR/${name}_${TS}.log
    local res=$AN/$root/listener/results/$mode/results.csv
    local flags=$CAMPAIGN_FLAGS
    [ "$flagset" = nosplit ] && flags=$NOSPLIT_FLAGS
    [ -f "$res" ] && cp "$res" "${res}.bak_pre_frozencal_${TS}"
    # A run writes best_model_<winner>.pkl. When the winner changes, the old
    # pickle survives beside the new one and every consumer aborts with
    # "expected 1". Archive (never delete) whatever is there before the run.
    local rd=$(dirname "$res")
    if ls "$rd"/best_model_*.pkl >/dev/null 2>&1; then
        mkdir -p "$rd/superseded_prefrozencal"
        mv "$rd"/best_model_*.pkl "$rd/superseded_prefrozencal/" 2>/dev/null
    fi
    MAX_JOBS=8 OMP_NUM_THREADS=2 \
    DCFM_ARMS_ROOT=$AN/$root DCFM_PIPELINE=$PIPE \
        nice -n 19 "$PY" -u "$RUNNER" --arm listener \
        --script defense_detection_v2.py --mode "$mode" --force \
        -- $flags > "$log" 2>&1
    local st=$?
    # gate 1: frozen-calibration N/N with N==N
    local gate=$(grep -o 'frozen-calibration] [0-9]*/[0-9]*' "$log" | tail -1)
    local a=${gate##*] }; a=${a%%/*}; local b=${gate##*/}
    # gate 2: the run reached its final table
    local fin=$(grep -c 'FINAL RESULTS' "$log")
    # winner row: after 'FINAL RESULTS' come a rule line, the header, then row 1
    # -- so -A3. VERIFIED against the 2/9 logs (smoke_parse_test.sh); -A4/-A5
    # land on the runners-up. bestsum is the pipeline's own 'Best Model:' line,
    # kept as a cross-check on the parse.
    local best=$(grep -A3 'FINAL RESULTS' "$log" | tail -1 | awk '{print $1, "acc", $4, "auc", $3}')
    local bestsum=$(grep -m1 'Best Model:' "$log" | sed 's/^ *//')
    if [ $st -ne 0 ] || [ -z "$gate" ] || [ "$a" != "$b" ] || [ "$fin" -eq 0 ]; then
        echo "GATE-FAIL|$root|$mode|$log|status=$st gate='$gate' final=$fin" >> "$MANIFEST"
        echo "FAIL $root/$mode  (see $log)"
        return 1
    fi
    echo "| \`$LOGDIR/${name}_${TS}.log\` | $root $mode, $flagset flags, frozencal pipeline | ✅ gate $gate; top: $best; $bestsum |" >> "$MANIFEST"
    # everything this run produced, relative to the tree root, for the paper repo
    echo "analysis/frozencal_marathon_logs/${name}_${TS}.log" >> "$GITADD"
    echo "analysis/$root/listener/results/$mode/results.csv" >> "$GITADD"
    echo "OK   $root/$mode  gate=$gate  top: $best"
    return 0
}

# ---------------------------------------------------------------- main
mkdir -p "$LOGDIR"
JOBS=()
for item in "${QUEUE[@]}"; do
    root=${item%%|*}; flags=${item##*|}
    if [ ! -d "$AN/$root/listener/colab_data" ]; then
        echo "SKIP $root -- no listener/colab_data (recorded)" | tee -a "$MANIFEST"
        continue
    fi
    while IFS= read -r j; do JOBS+=("$j"); done < <(expand_jobs "$root" "$flags")
done

echo "=== frozencal marathon: ${#JOBS[@]} runs, in pairs, started $(date '+%F %T') ==="
if [ "$DRY" = "--dry-run" ]; then printf '%s\n' "${JOBS[@]}"; exit 0; fi
if [ "$DRY" = "--smoke" ]; then
    echo "=== SMOKE: first job only, full path (backup/run/gates/manifest/gitadd) ==="
    JOBS=("${JOBS[0]}")
fi

echo "## frozencal marathon $TS -- manifest rows for STATE 25.188.7" >> "$MANIFEST"
{   # the round's fixed assets -- code before results, so the list stands alone
    echo "analysis/marathon_frozencal.sh"
    echo "analysis/frozencal/pipeline_17/defense_detection_v2.py"
    echo "analysis/arms_r34_17feat/linearsvc_csweep.py"
    echo "analysis/frozencal_marathon_manifest.md"
} >> "$GITADD"
FAILED=0
i=0
while [ $i -lt ${#JOBS[@]} ]; do
    j1=${JOBS[$i]}; j2=""
    [ $((i+1)) -lt ${#JOBS[@]} ] && j2=${JOBS[$((i+1))]}
    IFS='|' read -r r1 m1 f1 <<< "$j1"
    echo "--- pair: $j1  +  ${j2:-<none>}   $(date '+%T')"
    run_one "$r1" "$m1" "$f1" & P1=$!
    P2=""
    if [ -n "$j2" ]; then
        IFS='|' read -r r2 m2 f2 <<< "$j2"
        run_one "$r2" "$m2" "$f2" & P2=$!
    fi
    wait $P1 || FAILED=1
    [ -n "$P2" ] && { wait $P2 || FAILED=1; }
    if [ $FAILED -ne 0 ]; then
        echo "=== GATE FAILURE -- queue STOPPED after job $((i+1))/${#JOBS[@]}. Fix, then rerun; completed runs keep their results."
        exit 1
    fi
    i=$((i+2))
done
echo "=== marathon complete $(date '+%F %T') -- manifest rows in $MANIFEST ==="
