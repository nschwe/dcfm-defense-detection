#!/usr/bin/env bash
# ============================================================================
# run_fpnt_arm_all.sh — the whole FPNT second-defence arm, pools to learning.
#
# R#3.1 / R#5.2: "only one defence, and it is the authors' own". This runs the
# second defence (Tan et al. 2015, FPNT-OLSR) through the same detection
# pipeline as the GCOP arm, static and mobile, and stops at the end of stage 1
# learning.
#
# ⚠️ 27/8 — the line above used to read "over the full four-arm panel". That was
# never what it did: ARMS defaults to `listener` alone (see ARMS_LIST below) and
# the 23/8 campaign built only the listener arm. The stale comment misled a
# reader into thinking DCFM_PIPELINE would affect four arms. Single vantage.
#
# SCENARIO CONFIGURATION — pinned deliberately, all four confirmed 23/8:
#   --admission=full          every node's routing table complete at t=60, i.e.
#                             every node can reach every other, not necessarily
#                             directly. Scenario default, passed implicitly.
#   --enforceHopFilter=0      NO minimum distance between UDP source and victim.
#                             Must be explicit: the default is 1, which rejects
#                             runs closer than 3 hops. A 1-hop run is NOT thrown
#                             out by distance; it simply has no forwarder to make
#                             a black-hole out of and ends "Attacker not
#                             available", which the admission log records.
#   (no propagation flags)    NO realistic channel. The Nakagami/RefLoss=42 work
#                             of §25.36-43 is a different experiment.
#   (no --scenarioOrder)      NO window mixing. Fixed order
#                             baseline -> attack_only -> defense_only ->
#                             defense_vs_attack.
#   --defence=fpnt --bBlackholeAttack=1 --bBlackholeOnPath=1
#                             The two flags are INDEPENDENT (STATE 25.82).
#                             Mode 1 and mode 2 are measured equivalent on this
#                             population, static and mobile (25.84.3, 25.85.9).
#
# FEATURE COMPATIBILITY — verified, not assumed (23/8). Same binary, same output
# code; --defence never touches the writer. Checked file by file against a real
# GCOP campaign output, all four windows:
#     metrics_output-N.csv    63 metrics, same names, same order
#     observer_metrics-N.csv  44 columns, 11 rows
#     observer_detail-N.csv   40 rows
# identical throughout. The single-listener panel is built from the same
# observer files and the same X_* metrics.
#
# ⛔ nv347 IS NEVER WRITTEN TO. The analysis scripts live there and are read
#    from there, but DCFM_ARMS_ROOT is redirected under fpntbh347 and phase
#    `guard` snapshots nv347's mtimes before and after so a stray write is
#    caught rather than assumed away.
#
# Every phase is resumable and idempotent: re-running continues where it left
# off. Nothing here creates .fpnt_campaign_ready — that stays the author's call.
#
#   bash run_fpnt_arm_all.sh                  # everything, in order
#   PHASE=pools bash run_fpnt_arm_all.sh      # one phase
#   REHEARSE=1 bash run_fpnt_arm_all.sh       # 40-seed dry run of the whole chain
#
# Phases: guard pools campaign bundles learn summary
#
# ----------------------------------------------------------------------------
# 27/8 — ADDITIVE, GATED overrides for the 17-observable re-run (R#5.11/R#3.1).
# EVERY DEFAULT BELOW REPRODUCES THE 23/8 BEHAVIOUR EXACTLY. Set none of them
# and this script does what it did before, byte for byte.
#
#   ARMS_ROOT=<dir>     where bundles and results are written.
#                       DEFAULT: $FPNT/analysis/arms_fpnt — the 23/8 arm.
#                       ⛔ Re-running PHASE=learn against the default OVERWRITES
#                       the 23/8 results (0.9481 / 0.9469), which have no backup.
#                       Point this at a NEW directory for any re-run.
#
#   PIPELINE=<dir>      value of DCFM_PIPELINE for stage 1. DEFAULT: unset, so
#                       run_arm.py:39 falls back to analysis/pipeline (33
#                       metrics) — which is what the 23/8 arm used. Set it to
#                       $NV/analysis/cleanfeat/pipeline_17 for the 17-observable
#                       / 67-engineered space. Same defect and same fix as
#                       run_traffic_all.sh (STATE 25.112.1).
#
#   LEARN_FLAGS=<str>   flags passed after `--` to defense_detection_v2.py.
#                       DEFAULT: "--no-augmentation" — the 23/8 set.
#                       The reported campaign arms_r34_17feat additionally used
#                       --split-validation --add-linearsvc --calibrate-stacking;
#                       measured 27/8 from run17.log (two split-validation and
#                       two calibrate-stacking banners) and from its model
#                       roster, which carries SVM_linear where arms_fpnt does
#                       not (17 models vs 16).
#
#   ALLOW_MULTIARM=1    permit an ARMS list other than exactly `listener`.
#                       Without it the script REFUSES to start. Past runs
#                       silently built original/corrected/combined and wasted
#                       hours; a warning was not enough.
# ============================================================================
set -uo pipefail

# ---- configuration ---------------------------------------------------------
FPNT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$FPNT/ns-3.47"
BIN="$ROOT/build/scratch/ns3.47-iolsr-tests-corrected-default"
NV="${NV:-$HOME/ns3/nv347/ns-3.47}"
ANALYSIS="$NV/analysis"
PY="${PY:-$HOME/miniconda3/envs/manet/bin/python}"
[ -x "$PY" ] || PY=python3

SIM="$FPNT/simulations_fpnt"                      # features_static + features_mobile
# Overridable, default identical to 23/8. See the ARMS_ROOT note in the header:
# re-running learn against the default overwrites the 23/8 results in place.
# ARMS_ROOT_SET records whether the caller chose one, so REHEARSE below does not
# silently drag an explicit root back to the rehearsal directory.
ARMS_ROOT_SET="${ARMS_ROOT+yes}"
ARMS_ROOT="${ARMS_ROOT:-$FPNT/analysis/arms_fpnt}"
POOL_STATIC="$FPNT/simulations_fpnt_static_10k"
POOL_MOBILE="$FPNT/simulations_fpnt_mobile_10k"
STAMP="$FPNT/.nv347_mtimes"

TARGET="${TARGET:-2000}"          # accepted seeds per mode
MAXSCAN_S="${MAXSCAN_S:-4000}"
MAXSCAN_M="${MAXSCAN_M:-6000}"
# 22 of 24, the campaign's own figure: run_hopablation_learning.sh uses
# MAX_JOBS=22 and §25.80.1 measured ~38 windows/min at 22 workers. Two cores are
# left for the system. The phases here run SEQUENTIALLY, so unlike
# run_nohopfilter_mobile.sh — which drops to 8 because learning shares the
# machine with it — nothing else is competing and the full 22 are available.
JOBS="${JOBS:-22}"
RUN_TIMEOUT="${RUN_TIMEOUT:-900}"
MAX_JOBS_LEARN="${MAX_JOBS_LEARN:-22}"
PHASE="${PHASE:-all}"
# ⛔ SINGLE LISTENER ONLY. The GCOP arm's comparability claim is about one
# vantage point, so `listener` is the arm this experiment is about; original,
# corrected and combined would quadruple the learning time to answer a question
# nobody asked. Override with ARMS_LIST="listener corrected" if that changes.
read -r -a ARMS <<< "${ARMS_LIST:-listener}"

# ---- 17-observable / learning-flag overrides (see header) -------------------
PIPELINE="${PIPELINE:-}"
LEARN_FLAGS="${LEARN_FLAGS:---no-augmentation}"

# ⛔ SINGLE-LISTENER GATE. Past runs built original/corrected/combined and threw
# away hours. run_traffic_all.sh only WARNS about this; a warning that prints
# after the fact is not a gate, so this refuses to start instead.
if [ "${#ARMS[@]}" -ne 1 ] || [ "${ARMS[0]}" != "listener" ]; then
  if [ "${ALLOW_MULTIARM:-0}" != 1 ]; then
    echo "REFUSING: arm list is '${ARMS[*]}', not exactly 'listener'."
    echo "  This campaign is a single-vantage experiment. Building original,"
    echo "  corrected or combined quadruples the learning time to answer a"
    echo "  question nobody asked."
    echo "  Set ALLOW_MULTIARM=1 only if that is genuinely intended."
    exit 1
  fi
  echo "*** ALLOW_MULTIARM=1: proceeding with arms '${ARMS[*]}' ***"
fi

if [ "${REHEARSE:-0}" = 1 ]; then
  TARGET=20; MAXSCAN_S=60; MAXSCAN_M=80; JOBS=8
  SIM="$FPNT/simulations_fpnt_rehearse"
  [ -z "$ARMS_ROOT_SET" ] && ARMS_ROOT="$FPNT/analysis/arms_fpnt_rehearse"
  POOL_STATIC="$FPNT/simulations_fpnt_static_rehearse"
  POOL_MOBILE="$FPNT/simulations_fpnt_mobile_rehearse"
  echo "*** REHEARSAL: target=$TARGET, separate output roots ***"
fi

# ---- guards ----------------------------------------------------------------
[ -x "$BIN" ]  || { echo "ERROR: binary not built: $BIN"; exit 1; }
[ -x "$PY" ]   || { echo "ERROR: manet python not found: $PY"; exit 1; }
[ -d "$ANALYSIS" ] || { echo "ERROR: analysis dir not found: $ANALYSIS"; exit 1; }
case "$SIM" in "$FPNT"/*) : ;; *) echo "REFUSING: SIM outside fpntbh347: $SIM"; exit 1 ;; esac
case "$ARMS_ROOT" in "$FPNT"/*) : ;; *) echo "REFUSING: ARMS_ROOT outside fpntbh347"; exit 1 ;; esac

export LD_LIBRARY_PATH="$ROOT/build/lib:${LD_LIBRARY_PATH:-}"

want () { [ "$PHASE" = all ] || [ "$PHASE" = "$1" ]; }
hdr  () { echo; echo "================ $* ================"; date '+%F %T'; }

# ============================================================================
# PHASE guard — snapshot nv347 so a stray write is caught, not assumed away
# ============================================================================
snapshot_nv () {
  # catboost_info is excluded, and the exclusion is measured rather than assumed:
  # the 23/8 rehearsal tripped the guard on exactly four files —
  # catboost_training.json, learn/events.out.tfevents, learn_error.tsv,
  # time_left.tsv — all under analysis/pipeline/catboost_info. run_arm.py runs
  # with cwd=analysis/pipeline and CatBoost writes its training log to
  # ./catboost_info by default, so any learning run rewrites them, nv347's own
  # included. Same file set, mtimes only. Source, scenario, manifests and
  # campaign data were untouched. A guard that fires on a training library's
  # scratch directory teaches the reader to ignore it, so it is excluded here
  # and the reason is recorded instead.
  find "$NV/src" "$NV/scratch" "$NV/analysis" "$NV/simulations_v347_hopablation_10k/manifests" \
       "$NV/simulations_v347_hopablation_mobile_10k/manifests" \
       -type f -not -path '*/catboost_info/*' \
       -printf '%T@ %p\n' 2>/dev/null | sort > "$1"
}

if want guard || [ "$PHASE" = all ]; then
  hdr "PHASE guard — nv347 mtime snapshot"
  snapshot_nv "$STAMP.before"
  echo "recorded $(wc -l < "$STAMP.before") files under nv347"
fi

# ============================================================================
# PHASE pools — re-admit the campaign population under black-hole + this mode
# ============================================================================
if want pools; then
  hdr "PHASE pools — static"
  MOB=0 TARGET="$TARGET" MAXSCAN="$MAXSCAN_S" JOBS="$JOBS" OUT="$POOL_STATIC" \
    bash "$FPNT/build_fpnt_mobile_pool.sh"

  hdr "PHASE pools — mobile"
  MOB=1 TARGET="$TARGET" MAXSCAN="$MAXSCAN_M" JOBS="$JOBS" OUT="$POOL_MOBILE" \
    bash "$FPNT/build_fpnt_mobile_pool.sh"
fi

# ============================================================================
# PHASE campaign — one run per seed, contributing all four of its windows
# ============================================================================
run_campaign () {
  mode="$1"; pool="$2"; feat="$3"; ridbase="$4"
  man="$pool/manifests/C_all.csv"
  [ -s "$man" ] || { echo "ERROR: pool manifest missing: $man  (run PHASE=pools)"; return 1; }

  mkdir -p "$SIM/logs_$mode"
  for s in baseline attack_only defense_only defense_vs_attack; do
    mkdir -p "$SIM/$feat/$s"
  done
  META="$SIM/hop_metadata_$mode.csv"
  LOCK="$SIM/.lock_$mode"; : > "$LOCK"
  [ -f "$META" ] || echo "seed,run_id,hop_category,result,timestamp" > "$META"

  awk -F, 'NR>1 && $1!="" {print $1}' "$man" > "$SIM/seeds_$mode.txt"
  echo "$(wc -l < "$SIM/seeds_$mode.txt") seeds, $JOBS workers -> $SIM/$feat"

  export SIM BIN RUN_TIMEOUT META LOCK feat ridbase mode

  one_run () {
    seed="$1"
    rid=$((seed + ridbase))
    complete=1
    for s in baseline attack_only defense_only defense_vs_attack; do
      for f in metrics_output observer_metrics observer_detail; do
        [ -s "$SIM/$feat/$s/$f-$rid.csv" ] || complete=0
      done
    done
    [ "$complete" = 1 ] && { echo "[skip] seed=$seed"; return 0; }

    rlog="$SIM/logs_$mode/run_${seed}.log"
    timeout "$RUN_TIMEOUT" "$BIN" \
      --run="$rid" --RngRun="$seed" --bMobility="$([ "$mode" = mobile ] && echo 1 || echo 0)" \
      --enforceHopFilter=0 \
      --defence=fpnt --bBlackholeAttack=1 --bBlackholeOnPath=1 \
      --outputDir="$SIM/$feat/" \
      --topologyProbeFile="$SIM/topology_probes_$mode.csv" > "$rlog" 2>&1
    rc=$?

    complete=1
    for s in baseline attack_only defense_only defense_vs_attack; do
      [ -s "$SIM/$feat/$s/metrics_output-$rid.csv" ] || complete=0
    done
    # LAST HOP_CATEGORY, not the first: the scenario prints one per window and
    # under mobility the distance genuinely varies between them.
    cat=$(grep -oP 'HOP_CATEGORY=\K[0-9]' "$rlog" 2>/dev/null | tail -1); cat="${cat:-?}"
    res=OK; { [ $rc -ne 0 ] || [ "$complete" != 1 ]; } && res="FAIL rc=$rc"
    ( flock -x 9; echo "$seed,$rid,$cat,$res,$(date '+%F %T')" >> "$META" ) 9>"$LOCK"
    echo "[$res] seed=$seed rid=$rid hop=$cat"
  }
  export -f one_run

  xargs -a "$SIM/seeds_$mode.txt" -P "$JOBS" -n 1 -I{} bash -c 'one_run {}'

  echo "--- $mode outcomes ---"
  tail -n +2 "$META" | awk -F, '{c[$4" cat"$3]++} END {for (k in c) printf "  %-14s %d\n", k, c[k]}' | sort
}

if want campaign; then
  hdr "PHASE campaign — static"
  run_campaign static "$POOL_STATIC" features_static 0
  hdr "PHASE campaign — mobile"
  run_campaign mobile "$POOL_MOBILE" features_mobile 1000000
fi

# ============================================================================
# PHASE bundles — one wide bundle per arm, from THIS campaign's features
# ============================================================================
if want bundles; then
  hdr "PHASE bundles"
  for m in static mobile; do
    n=$(ls "$SIM/features_$m/baseline"/metrics_output-*.csv 2>/dev/null | wc -l)
    echo "  features_$m: $n runs"
    [ "$n" -gt 0 ] || { echo "ERROR: no features_$m — run PHASE=campaign"; exit 1; }
  done
  mkdir -p "$ARMS_ROOT"
  # One --arm per pass: without it make_arm_bundles builds all four, and only
  # the single-listener arm is wanted here.
  for a in "${ARMS[@]}"; do
    echo "  building bundle: $a"
    DCFM_SIM_ROOT="$SIM" DCFM_ARMS_ROOT="$ARMS_ROOT" \
      "$PY" -u "$ANALYSIS/make_arm_bundles.py" --arm "$a"
  done
  echo "--- bundles built ---"
  for a in "${ARMS[@]}"; do
    for m in static mobile; do
      f="$ARMS_ROOT/$a/colab_data/wide_$m.csv.gz"
      printf "  %-10s %-7s %s\n" "$a" "$m" "$([ -f "$f" ] && du -h "$f" | cut -f1 || echo MISSING)"
    done
  done
fi

# ============================================================================
# PHASE learn — stage 1 only, full four-arm panel, both modes
# ============================================================================
if want learn; then
  hdr "PHASE learn — defense_detection_v2.py, ${#ARMS[@]} arm(s) x 2 modes"

  # Resolved configuration, printed so the log records what actually ran rather
  # than what the header says it should.
  echo "  arms        : ${ARMS[*]}"
  echo "  ARMS_ROOT   : $ARMS_ROOT"
  echo "  DCFM_PIPELINE: ${PIPELINE:-<unset -> run_arm.py default, 33 metrics>}"
  echo "  learn flags : $LEARN_FLAGS"

  LEARN_ENV=(DCFM_ARMS_ROOT="$ARMS_ROOT" MAX_JOBS="$MAX_JOBS_LEARN")
  if [ -n "$PIPELINE" ]; then
    [ -d "$PIPELINE" ] || { echo "REFUSING: PIPELINE not a directory: $PIPELINE"; exit 1; }
    [ -f "$PIPELINE/defense_detection_v2.py" ] || {
      echo "REFUSING: no defense_detection_v2.py under $PIPELINE"; exit 1; }
    LEARN_ENV+=(DCFM_PIPELINE="$PIPELINE")
  fi

  for arm in "${ARMS[@]}"; do
    for m in static mobile; do
      echo; echo "--------- $arm / $m ---------"; date '+%F %T'
      # LEARN_FLAGS is deliberately unquoted: it is a flag list, not one word.
      env "${LEARN_ENV[@]}" \
        "$PY" -u "$ANALYSIS/run_arm.py" \
        --arm "$arm" --script defense_detection_v2.py --mode "$m" \
        -- $LEARN_FLAGS
      rc=$?
      [ $rc -ne 0 ] && echo "[warn] $arm/$m exited $rc"
    done
  done

  # ⛔ POST-CHECK, hard. Catches an arm built by an earlier or stray invocation
  # into the same root — the failure the single-listener gate exists to prevent.
  stray=$(ls -d "$ARMS_ROOT"/*/ 2>/dev/null | sed 's#.*/\([^/]*\)/#\1#' \
          | grep -vxF -e listener || true)
  if [ -n "$stray" ]; then
    echo "*** ERROR: non-listener arm(s) present under $ARMS_ROOT:"
    echo "$stray" | sed 's/^/      /'
    echo "*** This campaign is single-vantage. Investigate before using results."
    exit 1
  fi
  echo; echo "  ✅ single-listener check: only 'listener' exists under $ARMS_ROOT"
fi

# ============================================================================
# PHASE summary
# ============================================================================
if want summary || [ "$PHASE" = all ]; then
  hdr "SUMMARY"

  echo "--- nv347 write guard ---"
  snapshot_nv "$STAMP.after"
  if diff -q "$STAMP.before" "$STAMP.after" >/dev/null 2>&1; then
    echo "  nv347 UNTOUCHED (mtimes identical, $(wc -l < "$STAMP.after") files)"
  else
    echo "  *** nv347 CHANGED — investigate before trusting anything below ***"
    diff "$STAMP.before" "$STAMP.after" | head -20
  fi

  echo
  echo "--- admission ---"
  for p in "$POOL_STATIC" "$POOL_MOBILE"; do
    l="$p/admission_log.csv"
    [ -s "$l" ] || continue
    a=$(grep -c ',accept,' "$l"); r=$(grep -c ',reject,' "$l"); t=$((a+r))
    printf "  %-40s %d/%d accepted (%.1f%%)\n" "$(basename "$p")" "$a" "$t" \
           "$(awk -v a="$a" -v t="$t" 'BEGIN{print t?100*a/t:0}')"
    awk -F, 'NR>1 && $2=="reject"{c[$3]++} END{for(k in c) printf "     %-32s %d\n", k, c[k]}' "$l"
  done

  echo
  echo "--- campaign ---"
  for m in static mobile; do
    f="$SIM/features_$m/baseline"
    [ -d "$f" ] && printf "  %-8s %s runs\n" "$m" "$(ls "$f"/metrics_output-*.csv 2>/dev/null | wc -l)"
  done

  echo
  echo "--- stage 1 accuracy ---"
  printf "  %-10s %-8s %s\n" arm mode accuracy
  for arm in "${ARMS[@]}"; do
    for m in static mobile; do
      f="$ARMS_ROOT/$arm/results/$m/results.csv"
      [ -f "$f" ] && printf "  %-10s %-8s %s\n" "$arm" "$m" \
        "$(head -2 "$f" | tail -1 | cut -d, -f4)"
    done
  done

  echo
  echo "note: the Section 5.2 piggyback adds 4 bytes per advertised neighbour to"
  echo "      every TC, attack or no attack, so byte-volume features carry a"
  echo "      defence-presence signal on their own (STATE 25.80.10 / 25.85.10)."
  echo "      If RoutingOverheadBytesRatio or X_TransmissionByteRate dominate the"
  echo "      importances, ablate them and re-read the accuracy."
  date '+%F %T'
fi
