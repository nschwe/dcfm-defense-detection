#!/usr/bin/env bash
# ============================================================================
# run_prop17.sh — re-learn the four propagation arms in the 17-observable space
#
# R#5.3. The propagation experiment (STATE §25.36-25.44) ran 18-19/8, BEFORE the
# 17-observable listener arm and before --split-validation. All four arms score
# 16 models, i.e. the pre-split pipeline in the 33-metric / 141-column space.
# Their figures therefore CANNOT be set beside the reported campaign's
# 0.8995 / 0.9283, which is what R#5.3's answer needs to do.
#
# This script re-runs STAGE 1 ONLY, on the bundles that already exist, in the
# configuration of the reported campaign:
#     DCFM_PIPELINE = analysis/cleanfeat/pipeline_17      (17 metrics -> 67 cols)
#     flags         = --no-augmentation --split-validation
#                     --add-linearsvc --calibrate-stacking
#
# ⛔ NO SIMULATION. NO BUNDLE REBUILD. The eight bundles are read-only inputs;
#    they are copied into new *_17 arm roots and the originals are never touched.
#
# ⛔ THE 23/8-19/8 RESULTS ARE PRESERVED. Everything lands under <arm>_17.
#
#   bash run_prop17.sh                 # all four arms, both modes
#   MODES=static bash run_prop17.sh    # one mode
#   ARMS_LIST="arms_propmodel arms_propmodel_baseline" bash run_prop17.sh
#   DRYRUN=1 bash run_prop17.sh        # print the plan and stop
# ============================================================================
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-$HOME/miniconda3/envs/manet/bin/python}"
[ -x "$PY" ] || PY=python3
ANALYSIS="$ROOT/analysis"
PIPELINE="${PIPELINE:-$ANALYSIS/cleanfeat/pipeline_17}"
LEARN_FLAGS="${LEARN_FLAGS:---no-augmentation --split-validation --add-linearsvc --calibrate-stacking}"
MAX_JOBS_LEARN="${MAX_JOBS_LEARN:-22}"
SUFFIX="${SUFFIX:-_17}"
DRYRUN="${DRYRUN:-0}"

read -r -a ARMS <<< "${ARMS_LIST:-arms_propmodel arms_propmodel_baseline arms_propmodel_rl42 arms_propmodel_rl42_baseline}"
read -r -a MODES_A <<< "${MODES:-static mobile}"

# ---------------------------------------------------------------- guards ----
[ -x "$PY" ] || { echo "ABORT: manet python missing: $PY"; exit 1; }
[ -d "$PIPELINE" ] || { echo "ABORT: PIPELINE not a directory: $PIPELINE"; exit 1; }
[ -f "$PIPELINE/defense_detection_v2.py" ] || {
  echo "ABORT: no defense_detection_v2.py under $PIPELINE"; exit 1; }

# ⛔ The 17-metric check is a GATE, not a printout. A silent fallback to
#    analysis/pipeline (33 metrics) is exactly the defect of STATE §25.112.1.
# ⚠️ The pipeline module prints a "[GPU] PyTorch not available" banner on import.
#    A first version of this gate captured that banner into NMET and refused a
#    perfectly good pipeline. Keep the last numeric line only.
NMET=$("$PY" - "$PIPELINE" <<'PYEOF' 2>/dev/null | grep -E '^[0-9]+$' | tail -1
import sys, importlib.util, os
spec = importlib.util.spec_from_file_location("dd", os.path.join(sys.argv[1], "defense_detection_v2.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print(len(m.DefenseDetector.METRICS))
PYEOF
)
if [ "$NMET" != "17" ]; then
  echo "ABORT: $PIPELINE exposes $NMET metrics, expected 17."
  echo "       Learning in the wrong feature space is the whole reason this script exists."
  exit 1
fi

if pgrep -f "defense_detection_v2.py" >/dev/null 2>&1; then
  echo "ABORT: a stage-1 learner is already running. Stage 1 must not share the machine."
  exit 1
fi

echo "================================================================"
echo " run_prop17.sh"
echo "   pipeline    : $PIPELINE   (METRICS=$NMET ✅)"
echo "   flags       : $LEARN_FLAGS"
echo "   arms        : ${ARMS[*]}"
echo "   modes       : ${MODES_A[*]}"
echo "   suffix      : $SUFFIX"
echo "   MAX_JOBS    : $MAX_JOBS_LEARN"
echo "================================================================"
date '+%F %T'

# ------------------------------------------------------------- the cells ----
PLANNED=0; SKIPPED=0; DONE=0; FAILED=0

for A in "${ARMS[@]}"; do
  SRC="$ROOT/analysis/$A"
  DST="$ROOT/analysis/${A}${SUFFIX}"
  [ -d "$SRC" ] || { echo "  [warn] source arm missing, skipping: $SRC"; continue; }

  for M in "${MODES_A[@]}"; do
    sb="$SRC/listener/colab_data/wide_${M}.csv.gz"
    db="$DST/listener/colab_data/wide_${M}.csv.gz"
    res="$DST/listener/results/$M/results.csv"

    if [ ! -f "$sb" ]; then
      echo "  [warn] no source bundle: $sb"
      continue
    fi
    PLANNED=$((PLANNED+1))

    if [ -f "$res" ] && [ "${FORCE:-0}" != 1 ]; then
      echo "  [skip] already learned: ${A}${SUFFIX}/$M   (FORCE=1 to redo)"
      SKIPPED=$((SKIPPED+1))
      continue
    fi

    echo
    echo "--------- ${A}${SUFFIX} / $M ---------"
    date '+%F %T'

    if [ "$DRYRUN" = 1 ]; then
      echo "  DRYRUN: would copy $sb -> $db and learn"
      continue
    fi

    mkdir -p "$DST/listener/colab_data"
    if [ ! -f "$db" ]; then
      cp -p "$sb" "$db"
    fi
    # ⛔ Verify the copy is identical on the DECOMPRESSED stream: gzip embeds an
    #    mtime, so comparing .gz bytes is not a content check (STATE §25.135.5).
    ha=$(zcat "$sb" | sha256sum | cut -c1-16)
    hb=$(zcat "$db" | sha256sum | cut -c1-16)
    if [ "$ha" != "$hb" ]; then
      echo "  *** ABORT: copied bundle differs from source ($ha vs $hb)"
      exit 1
    fi
    echo "  bundle: $(zcat "$db" | wc -l) rows, $(zcat "$db" | head -1 | tr ',' '\n' | wc -l) cols, sha=$hb ✅"

    env DCFM_ARMS_ROOT="$DST" DCFM_PIPELINE="$PIPELINE" MAX_JOBS="$MAX_JOBS_LEARN" \
      "$PY" -u "$ANALYSIS/run_arm.py" \
      --arm listener --script defense_detection_v2.py --mode "$M" \
      -- $LEARN_FLAGS
    rc=$?
    if [ $rc -ne 0 ]; then
      echo "  [warn] ${A}${SUFFIX}/$M exited $rc"
      FAILED=$((FAILED+1))
    else
      DONE=$((DONE+1))
    fi

    # ⛔ Single-listener post-check, hard. Past runs silently built original /
    #    corrected / combined and threw away hours.
    stray=$(ls -d "$DST"/*/ 2>/dev/null | sed 's#.*/\([^/]*\)/#\1#' | grep -vxF -e listener || true)
    if [ -n "$stray" ]; then
      echo "  *** ERROR: non-listener arm(s) under $DST:"
      echo "$stray" | sed 's/^/        /'
      exit 1
    fi
  done
done

# ---------------------------------------------------------------- summary ---
echo
echo "================ SUMMARY ================"
date '+%F %T'
echo "  cells planned=$PLANNED  learned=$DONE  skipped=$SKIPPED  failed=$FAILED"
echo

acc () {  # arm mode -> best-model accuracy, or empty
  local f="$ROOT/analysis/$1/listener/results/$2/results.csv"
  [ -f "$f" ] && sed -n 2p "$f" | cut -d, -f4
}
mdl () {
  local f="$ROOT/analysis/$1/listener/results/$2/results.csv"
  [ -f "$f" ] && sed -n 2p "$f" | cut -d, -f1
}
nmod () {
  local f="$ROOT/analysis/$1/listener/results/$2/results.csv"
  [ -f "$f" ] && echo $(( $(wc -l < "$f") - 1 ))
}

printf "  %-34s %-7s %-11s %-22s %s\n" arm mode accuracy "best model" models
for A in "${ARMS[@]}"; do
  for M in "${MODES_A[@]}"; do
    a=$(acc "${A}${SUFFIX}" "$M")
    [ -n "$a" ] && printf "  %-34s %-7s %-11s %-22s %s\n" \
      "${A}${SUFFIX}" "$M" "$a" "$(mdl "${A}${SUFFIX}" "$M")" "$(nmod "${A}${SUFFIX}" "$M")"
  done
done

echo
echo "  ---- paired deltas (realistic minus its OWN staged baseline) ----"
echo "  ⛔ Each realistic arm has its own paired baseline. Do not cross them:"
echo "     arms_propmodel      <-> arms_propmodel_baseline        (RL 46.75 / 46)"
echo "     arms_propmodel_rl42 <-> arms_propmodel_rl42_baseline   (RL 42)"
echo
for pair in "arms_propmodel:arms_propmodel_baseline:RL 46.75/46 (degree-matched)" \
            "arms_propmodel_rl42:arms_propmodel_rl42_baseline:RL 42 (route-viability)"; do
  R="${pair%%:*}"; rest="${pair#*:}"; B="${rest%%:*}"; LABEL="${rest#*:}"
  for M in "${MODES_A[@]}"; do
    ra=$(acc "${R}${SUFFIX}" "$M"); ba=$(acc "${B}${SUFFIX}" "$M")
    if [ -n "$ra" ] && [ -n "$ba" ]; then
      awk -v r="$ra" -v b="$ba" -v m="$M" -v l="$LABEL" 'BEGIN{
        d = r - b
        printf "  %-34s %-7s realistic=%.4f  disk=%.4f  delta=%+.4f  %s\n",
               l, m, r, b, d, (d<0.014 && d>-0.014) ? "<- INSIDE the +/-0.014 paired noise floor" : ""
      }'
    fi
  done
done

echo
echo "  ⚠️  The paired noise floor for this comparison is +/-0.014 (STATE §25.33)."
echo "      A delta inside it means INDISTINGUISHABLE, not 'restored' or 'improved'."
echo "  ⚠️  At RL=42 the realistic graph is 40-52 % denser than the disk (§25.44),"
echo "      so fading tolerance and the compensating density CANNOT be separated."
echo "  ⛔  Report BOTH calibration endpoints. §25.40-b makes this mandatory."
