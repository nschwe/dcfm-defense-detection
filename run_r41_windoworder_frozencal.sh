#!/usr/bin/env bash
# ============================================================================
# run_r41_windoworder_frozencal.sh -- R#4.1, STATE 25.199.3 steps 2-4, ONE pass.
#
# WHY THIS EXISTS.
#   The whole first half of r4_1_answer.tex rests on
#   analysis/arms_windoworder/canonical_17 and shuffled_17, written 1/9
#   04:08-04:26 by
#       MAX_JOBS_LEARN=8 \
#       ARMS_LIST="arms_windoworder/canonical arms_windoworder/shuffled" \
#       bash run_prop17.sh
#   whose default PIPELINE is analysis/cleanfeat/pipeline_17 -- a copy in which
#   "frozen-calibration" appears 0 times (mtime 26/8, never patched). Those two
#   arms therefore carry the CalibratedClassifierCV refit defect of 25.188.
#
#   The 2/9 replacement attempt landed in canonical_17_17 / shuffled_17_17
#   because run_prop17.sh:35 appends its own _17 to arm names that already end
#   in _17, and it ran at 09:48 -- BEFORE the gate was ported into
#   frozencal67_pipeline_17 at 22:15 the same day. So the _17_17 cells are
#   ungated as well, which is why 25.199.3 refuses them and 25.189.1's
#   description of them as "the corrected cells from 2/9" is wrong.
#
#   25.200.1 closed step 1: frozencal67_pipeline_17 now prints the gate.
#   This script is steps 2, 3 and 4.
#
# WHAT IT DOES.
#   1. Guards, all hard: manet python, pipeline is 17-METRIC, pipeline carries
#      the frozen-calibration gate, no stage-1 learner already running.
#   2. Prints the exact list of files it will overwrite, and copies the small
#      text artefacts of the four target cells to a timestamped backup dir.
#   3. Stage 1 relearn of the four *_17_17 cells by calling run_arm.py --force
#      directly, with DCFM_PIPELINE pinned and MAX_JOBS=8. ⛔ NOT through
#      run_prop17.sh: its FORCE only defeats its own skip, not run_arm.py's.
#   4. Verifies the two standing gates of 25.189.2 on the new log.
#   5. Both consumers, into NEW output directories so the 1/9 artefacts stay:
#         r23_paired_bootstrap.py    -> r41_windoworder_bootstrap_frozencal/
#         windoworder_by_slot_test.py-> r41_byslot_17_frozencal/
#   6. Summary on the Stacking_Ensemble row, never the best-model row (20.7).
#
# ⛔ NO SIMULATION, NO BUNDLE REBUILD. Stage 1 on the bundles that exist.
# ⛔ ONE PASS (25.199.3 step 3): four accuracies, the paired difference and both
#    intervals, four slot spreads and four chi-squares all move together. Do not
#    quote any of them until every cell in this run has printed its gate.
# ⛔ It is the paired bootstrap r23_paired_bootstrap.py that produced the
#    intervals now in the answer, NOT windoworder_paired_ci.py. 25.199.3 step 2
#    names the wrong script; r4_1_answer.tex's own OURS block records that
#    windoworder_paired_ci.py is no longer used here.
#
#   DRYRUN=1 bash run_r41_windoworder_frozencal.sh    # default: plan + guards
#   DRYRUN=0 bash run_r41_windoworder_frozencal.sh    # actually run
# ============================================================================
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-$HOME/miniconda3/envs/manet/bin/python}"
[ -x "$PY" ] || PY=python3
ANALYSIS="$ROOT/analysis"
PIPELINE="${PIPELINE:-$ANALYSIS/frozencal67_pipeline_17}"
MAX_JOBS_LEARN="${MAX_JOBS_LEARN:-8}"
DRYRUN="${DRYRUN:-1}"
# Identical to run_prop17.sh:33 and to line 4 of frozencal_windoworder_20260902_094855.log.
LEARN_FLAGS="${LEARN_FLAGS:---no-augmentation --split-validation --add-linearsvc --calibrate-stacking}"

TS=$(date +%Y%m%d_%H%M%S)
LOG="$ANALYSIS/r41_windoworder_frozencal_${TS}.log"
BOOT_OUT="$ANALYSIS/r41_windoworder_bootstrap_frozencal"
SLOT_OUT="$ANALYSIS/r41_byslot_17_frozencal"
BACKUP="$ANALYSIS/arms_windoworder/pregate_backup_${TS}"

SRC_ARMS="arms_windoworder/canonical_17 arms_windoworder/shuffled_17"
DST_CANON="arms_windoworder/canonical_17_17"
DST_SHUF="arms_windoworder/shuffled_17_17"
MODES="static mobile"

# ---------------------------------------------------------------- guards ----
[ -x "$PY" ] || { echo "ABORT: manet python missing: $PY"; exit 1; }
[ -d "$PIPELINE" ] || { echo "ABORT: PIPELINE not a directory: $PIPELINE"; exit 1; }
[ -f "$PIPELINE/defense_detection_v2.py" ] || {
  echo "ABORT: no defense_detection_v2.py under $PIPELINE"; exit 1; }

# GATE 1 -- 17 metrics. A silent fallback to analysis/pipeline (33 metrics) or to
# cleanfeat/pipeline_17 (17 metrics but NO frozen calibration) is exactly how the
# 1/9 arms were produced. Keep the last numeric line only: the module prints a
# "[GPU] PyTorch not available" banner on import.
NMET=$("$PY" - "$PIPELINE" <<'PYEOF' 2>/dev/null | grep -E '^[0-9]+$' | tail -1
import sys, importlib.util, os
spec = importlib.util.spec_from_file_location("dd", os.path.join(sys.argv[1], "defense_detection_v2.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print(len(m.DefenseDetector.METRICS))
PYEOF
)
if [ "$NMET" != "17" ]; then
  echo "ABORT: $PIPELINE exposes ${NMET:-?} metrics, expected 17."
  exit 1
fi

# GATE 2 -- the calibration fix is present in the pipeline that will be used.
NGATE=$(grep -c 'frozen-calibration' "$PIPELINE/defense_detection_v2.py")
if [ "$NGATE" -lt 1 ]; then
  echo "ABORT: $PIPELINE/defense_detection_v2.py has no frozen-calibration gate."
  echo "       This is the 1/9 defect. 25.190.1 / 25.200.1."
  exit 1
fi

if pgrep -f "defense_detection_v2.py" >/dev/null 2>&1; then
  echo "ABORT: a stage-1 learner is already running. Stage 1 must not share the machine."
  exit 1
fi

for A in $SRC_ARMS; do
  for M in $MODES; do
    b="$ANALYSIS/$A/listener/colab_data/wide_${M}.csv.gz"
    [ -f "$b" ] || { echo "ABORT: source bundle missing: $b"; exit 1; }
  done
done

echo "================================================================"
echo " run_r41_windoworder_frozencal.sh"
echo "   pipeline    : $PIPELINE   (METRICS=$NMET, gate lines=$NGATE)"
echo "   source arms : $SRC_ARMS"
echo "   target arms : $DST_CANON  $DST_SHUF"
echo "   MAX_JOBS    : $MAX_JOBS_LEARN"
echo "   DRYRUN      : $DRYRUN"
echo "   log         : $LOG"
echo "================================================================"
date '+%F %T'

# ------------------------------------------- what will be overwritten -------
# Never a destructive step without the match list printed first.
echo
echo "---- files this run will OVERWRITE ----"
NOVER=0
for D in "$DST_CANON" "$DST_SHUF"; do
  for M in $MODES; do
    d="$ANALYSIS/$D/listener/results/$M"
    [ -d "$d" ] || continue
    while IFS= read -r f; do
      printf "  %10s  %s\n" "$(du -h "$f" | cut -f1)" "$f"
      NOVER=$((NOVER+1))
    done < <(find "$d" -maxdepth 1 -type f 2>/dev/null | sort)
  done
done
echo "  ($NOVER files; they are the UNGATED 2/9 09:48-09:54 outputs, quotable nowhere)"

echo
echo "---- text artefacts backed up to $BACKUP ----"
if [ "$DRYRUN" != 1 ]; then
  for D in "$DST_CANON" "$DST_SHUF"; do
    for M in $MODES; do
      s="$ANALYSIS/$D/listener/results/$M"
      [ -d "$s" ] || continue
      t="$BACKUP/$(basename "$D")/$M"
      mkdir -p "$t"
      for f in results.csv confusion_matrix.csv; do
        [ -f "$s/$f" ] && cp -p "$s/$f" "$t/$f"
      done
    done
  done
  find "$BACKUP" -type f | sort | sed 's/^/  /'
else
  echo "  DRYRUN: nothing copied"
fi

if [ "$DRYRUN" = 1 ]; then
  echo
  echo "DRYRUN complete. Re-run with DRYRUN=0 to learn."
  exit 0
fi

# ------------------------------------------------------ stage 1 relearn -----
echo
echo "======== STAGE 1 ========"
# ⛔ WHY run_arm.py IS CALLED DIRECTLY AND NOT THROUGH run_prop17.sh.
#    The 4/9 09:20 attempt went through run_prop17.sh with FORCE=1. That variable
#    only defeats run_prop17.sh's OWN skip (:100). The inner call at :129-132 has
#    no --force, so run_arm.py's own guard fired on all four cells --
#    "[skip] listener/defense_detection_v2.py/static: results already present" --
#    and run_prop17.sh still reported "learned=4". Nothing was relearned, and the
#    gate check below is what caught it. This block reproduces run_prop17.sh's
#    inner invocation exactly, plus --force.
# ⚠️ The bundles are NOT copied here: they already sit in the target arms and
#    run_prop17.sh verified them on the decompressed stream at 09:20
#    (8001 rows, 27 cols = 21 metrics + 6, sha printed per cell).
: > "$LOG"
for PAIR in "$DST_CANON" "$DST_SHUF"; do
  DST="$ANALYSIS/$PAIR"
  for M in $MODES; do
    db="$DST/listener/colab_data/wide_${M}.csv.gz"
    [ -f "$db" ] || { echo "ABORT: target bundle missing: $db"; exit 1; }
    echo | tee -a "$LOG"
    echo "--------- $PAIR / $M ---------" | tee -a "$LOG"
    date '+%F %T' | tee -a "$LOG"
    echo "  bundle: $(zcat "$db" | wc -l) rows, $(zcat "$db" | head -1 | tr ',' '\n' | wc -l) cols" | tee -a "$LOG"
    env DCFM_ARMS_ROOT="$DST" DCFM_PIPELINE="$PIPELINE" MAX_JOBS="$MAX_JOBS_LEARN" \
      "$PY" -u "$ANALYSIS/run_arm.py" \
      --arm listener --script defense_detection_v2.py --mode "$M" --force \
      -- $LEARN_FLAGS 2>&1 | tee -a "$LOG"
    rc=${PIPESTATUS[0]}
    [ "$rc" -ne 0 ] && { echo "ABORT: $PAIR/$M exited $rc"; exit 1; }
    # Single-listener post-check, hard -- run_prop17.sh:141-148. Past runs
    # silently built original / corrected / combined and threw away hours.
    stray=$(ls -d "$DST"/*/ 2>/dev/null | sed 's#.*/\([^/]*\)/#\1#' | grep -vxF -e listener || true)
    [ -n "$stray" ] && { echo "ABORT: non-listener arm(s) under $DST: $stray"; exit 1; }
  done
done

# ------------------------------------------------------------- gate check ---
echo
echo "======== GATES (25.189.2) ========"
NFROZ=$(grep -c 'frozen-calibration' "$LOG")
NFEAT=$(grep -c 'Created 67 features (from 18)' "$LOG")
NSHAPE=$(grep -c 'Shape: (8000, 18)' "$LOG")
echo "  [frozen-calibration] lines : $NFROZ   (need 4)"
echo "  Created 67 features (18)   : $NFEAT   (need 4)"
echo "  Shape: (8000, 18)          : $NSHAPE  (need 4)"
if [ "$NFROZ" -ne 4 ] || [ "$NFEAT" -ne 4 ] || [ "$NSHAPE" -ne 4 ]; then
  echo "  *** GATES NOT MET. The consumers are NOT run and nothing here is quotable."
  exit 1
fi
echo "  ✅ all four cells gated"

# -------------------------------------------------------------- consumers ---
echo
echo "======== CONSUMER 1: paired cluster bootstrap ========"
for M in $MODES; do
  echo "-- $M --"
  env R23_PIPELINE="$PIPELINE" "$PY" -u "$ANALYSIS/r23_paired_bootstrap.py" \
    --realistic "$DST_SHUF" --baseline "$DST_CANON" --mode "$M" \
    --model Stacking_Ensemble --n-boot 2000 --seed 42 \
    --out-dir "$BOOT_OUT" 2>&1 | tee -a "$LOG"
done

echo
echo "======== CONSUMER 2: by-slot on the held-out test split ========"
for D in "$DST_CANON" "$DST_SHUF"; do
  for M in $MODES; do
    echo "-- $(basename "$D") / $M --"
    env WO_PIPELINE="$PIPELINE" "$PY" -u "$ANALYSIS/windoworder_by_slot_test.py" \
      --arms-root "$ANALYSIS/arms_windoworder" --mode "$M" \
      --arm "$(basename "$D")" --model Stacking_Ensemble \
      --out-dir "$SLOT_OUT" 2>&1 | tee -a "$LOG"
  done
done

# ---------------------------------------------------------------- summary ---
echo
echo "================ SUMMARY -- Stacking_Ensemble rows only ================"
date '+%F %T'
printf "  %-34s %-7s %-10s %-10s\n" arm mode accuracy AUC
for D in "$DST_CANON" "$DST_SHUF"; do
  for M in $MODES; do
    f="$ANALYSIS/$D/listener/results/$M/results.csv"
    [ -f "$f" ] || continue
    row=$(grep -m1 '^Stacking_Ensemble,' "$f")
    printf "  %-34s %-7s %-10s %-10s\n" "$(basename "$D")" "$M" \
      "$(echo "$row" | cut -d, -f4)" "$(echo "$row" | cut -d, -f3)"
  done
done

echo
echo "  ---- paired difference (shuffled minus canonical) ----"
for M in $MODES; do
  j=$(find "$BOOT_OUT" -name "*${M}.json" | head -1)
  [ -n "$j" ] && { echo "  $M: $j"; grep -E '"delta_pp"|"ci95_pp"|"p_two_sided"|"acc_' "$j" | sed 's/^/      /'; }
done

echo
echo "  ---- by-slot spreads ----"
for f in "$SLOT_OUT"/*.csv; do
  [ -f "$f" ] || continue
  printf "  %-52s spread=%s\n" "$(basename "$f")" "$(sed -n 2p "$f" | cut -d, -f8)"
done

echo
echo "  ⛔ NOTHING HERE IS QUOTABLE PIECEMEAL. r4_1_answer.tex is rewritten in one"
echo "     pass: two accuracies per mode, the paired difference and both intervals,"
echo "     four slot spreads and four chi-squares (25.199.3 step 3)."
echo "  ⛔ The chi-square values are NOT computed here. They come from a 2x4"
echo "     contingency test on the per-slot correct/incorrect counts in $SLOT_OUT."
echo "  ⚠️ The 1/9 artefacts are untouched: r41_byslot_17/ and"
echo "     r41_windoworder_bootstrap/. Do not mix the two sets."
echo "  ⚠️ canonical_17 / shuffled_17 are untouched as well; they remain the"
echo "     pre-fix record, exactly as arms_r34_17feat/pipeline_17 does (25.189.1)."
