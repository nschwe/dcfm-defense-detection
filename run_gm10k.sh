#!/usr/bin/env bash
# ============================================================================
# run_gm10k.sh -- the Gauss-Markov arm at FULL SCALE (Reviewer 3, Comment 3.2).
#
# WHY A SECOND SCRIPT RATHER THAN TARGET=10000 ON run_gm_arm.sh.
# The 2,000-run arm showed the feature selection is not stable at that size:
# even the RandomWalk2d CONTROL -- the same data and the same procedure as the
# published campaign, only smaller -- reproduced the published K=3 set in just
# 6 of 12 (method, variant) combinations, against 5 of 12 for Gauss-Markov. A
# difference of one cell cannot be attributed to the mobility model when the
# control itself does not reproduce the reference. Presenting that to a referee
# would open a stability question on R#3.3 while trying to answer R#3.2.
#
# At 10,000 the comparison is like-for-like against the published selection, so
# no control arm is needed: the main campaign IS the control.
#
# WHAT IS REUSED, WHAT IS NEW.
#   reused : the 2,000 Gauss-Markov runs already simulated -- identical flags
#            (--mobilityModel=gaussmarkov --enforceHopFilter=0, admission=full),
#            so they are a valid part of the 10,000. The campaign is
#            resume-aware and adds the missing 8,000 beside them.
#   reused : the static and RandomWalk2d halves of the main campaign. Neither
#            is re-simulated. Only the static half is COPIED into this arm's
#            tree, because stages 4 and 5 need a features_static next to
#            features_mobile. RandomWalk2d is not copied anywhere -- it is the
#            comparison, and its selection already exists in arms_c10k_stackcal.
#   new    : 8,000 Gauss-Markov mobile runs, and this arm's analysis tree.
#
# NOTHING IS OVERWRITTEN.
#   analysis/arms_gm10k/           <- NEW. The 2,000-run analysis in
#                                     analysis/arms_gaussmarkov/ is left intact.
#   simulations_v347_gaussmarkov/  <- grows. accepted_seeds and run_status are
#                                     appended to; no existing run is deleted or
#                                     rewritten.
#   simulations_v347_gm_control/   <- not touched at all; the control is not
#                                     needed at this scale.
#   everything else                <- read-only, and verified so by `summary`.
#
# THE DIFFERENCE FROM THE 2,000 RUN, besides scale: stage 4 runs WITHOUT
# --skip-evaluation and stage 5 is added, so this arm selects its own K instead
# of importing K=3 from the other domain. Importing it was the same defect we
# charge in R#3.3 -- a decision carried over from a domain that should have been
# held out.
#
# Usage:
#   bash run_gm10k.sh                 # all phases
#   PHASE=campaign bash run_gm10k.sh  # one phase
#   Do NOT pipe stdout into gm10k.log -- the script already tees there.
# ============================================================================
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

TARGET="${TARGET:-10000}"
WORKERS="${WORKERS:-22}"
NICE="${NICE:-10}"
PHASE="${PHASE:-all}"

# --- read-only sources ------------------------------------------------------
SRC_STATIC="$ROOT/simulations_v347_hopablation_10k"
SRC_MOBILE="$ROOT/simulations_v347_hopablation_mobile_10k"

# --- this arm ---------------------------------------------------------------
GM_SIM="$ROOT/simulations_v347_gaussmarkov"      # resumed, never truncated
GM_ARMS="$ROOT/analysis/arms_gm10k"              # NEW tree

SNAP="$ROOT/.gm10k_src_snapshot"
LOG="$ROOT/gm10k.log"
PY="${PY:-$HOME/miniconda3/envs/manet/bin/python}"
[ -x "$PY" ] || PY=python3

say () { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
die () { say "ABORT: $*"; exit 1; }
want () { [ "$PHASE" = all ] || [ "$PHASE" = "$1" ]; }

# --- refuse to write anywhere except this arm's own two trees ---------------
for p in "$GM_SIM" "$GM_ARMS"; do
    case "$p" in
        "$ROOT"/simulations_v347_gaussmarkov|"$ROOT"/analysis/arms_gm10k) : ;;
        *) die "output path is not one of this arm's trees: $p" ;;
    esac
done
[ -d "$SRC_STATIC" ] || die "missing source tree: $SRC_STATIC"
[ -d "$SRC_MOBILE" ] || die "missing source tree: $SRC_MOBILE"
[ -e "$GM_ARMS" ] && say "NOTE: $GM_ARMS already exists; phases are idempotent and will resume"

# ------------------------------------------------------------------ guard ---
if want guard; then
    say "PHASE guard -- snapshotting the read-only source trees"
    { find "$SRC_STATIC" -maxdepth 3 -printf '%p %T@\n'
      find "$SRC_MOBILE" -maxdepth 3 -printf '%p %T@\n'; } 2>/dev/null | sort > "$SNAP"
    say "  $(wc -l < "$SNAP") entries recorded"
    say "  2,000-run analysis preserved at: $ROOT/analysis/arms_gaussmarkov"
fi

# --------------------------------------------------------------- campaign ---
if want campaign; then
    have=$(( $(wc -l < "$GM_SIM/accepted_seeds_mobile.csv" 2>/dev/null || echo 1) - 1 ))
    say "PHASE campaign -- target $TARGET, already have $have, need $((TARGET - have))"
    [ -f "$ROOT/run_campaign_gaussmarkov.sh" ] || die "run prep on run_gm_arm.sh first"
    grep -q -- "--enforceHopFilter=0" "$ROOT/run_campaign_gaussmarkov.sh" \
        || die "campaign script does not disable the distance gate"
    MODE=mobile TARGET_ACCEPTED="$TARGET" NUM_WORKERS="$WORKERS" \
        nice -n "$NICE" bash "$ROOT/run_campaign_gaussmarkov.sh" 2>&1 | tee -a "$LOG"
    acc=$(( $(wc -l < "$GM_SIM/accepted_seeds_mobile.csv" 2>/dev/null || echo 1) - 1 ))
    say "  accepted: $acc"
    [ "$acc" -ge "$TARGET" ] || die "campaign short of target ($acc < $TARGET)"
    grep -q distance_lt3hops "$GM_SIM/run_status_mobile.csv" 2>/dev/null \
        && die "distance rejections present -- wrong population"
    say "  verified: zero distance rejections"
fi

# ------------------------------------------------------------------- copy ---
if want copy; then
    say "PHASE copy -- static half, $TARGET runs, selected through the manifest"
    "$PY" copy_subset.py --src "$SRC_STATIC" --dst "$GM_SIM" \
        --mode static --limit "$TARGET" 2>&1 | tee -a "$LOG" || die "copy failed"
fi

# ---------------------------------------------------------------- bundles ---
if want bundles; then
    say "PHASE bundles -- listener arm, both modes, into the NEW analysis tree"
    for m in static mobile; do
        say "  $m"
        # No --limit: it slices by numeric id and would keep only the >=3-hop
        # runs of a C_all tree. copy_subset.py already cut the tree to size.
        DCFM_SIM_ROOT="$GM_SIM" DCFM_ARMS_ROOT="$GM_ARMS" \
            nice -n "$NICE" "$PY" analysis/make_arm_bundles.py --arm listener --mode "$m" 2>&1 | tee -a "$LOG"
        f="$GM_ARMS/listener/colab_data/wide_$m.csv.gz"
        [ -f "$f" ] || die "no bundle produced: $f"
        say "    $(du -h "$f" | cut -f1)  $f"
    done
fi

# ------------------------------------------------------------------ learn ---
if want learn; then
    say "PHASE learn -- stage 1, mobile"
    DCFM_ARMS_ROOT="$GM_ARMS" nice -n "$NICE" "$PY" -u analysis/run_arm.py \
        --arm listener --script defense_detection_v2.py --mode mobile \
        -- --no-augmentation --split-validation --add-linearsvc --calibrate-stacking \
        2>&1 | tee -a "$LOG"
fi

# ------------------------------------------------------------------- rank ---
if want rank; then
    say "PHASE rank -- stage 4, WITH the evaluation loop (this arm picks its own K)"
    DCFM_ARMS_ROOT="$GM_ARMS" nice -n "$NICE" "$PY" -u analysis/run_arm.py \
        --arm listener --script feature_importance_sensitivity_v2.py \
        2>&1 | tee -a "$LOG"
fi

# ----------------------------------------------------------------- ksweep ---
if want ksweep; then
    say "PHASE ksweep -- stage 5, then select_optimal_k"
    DCFM_ARMS_ROOT="$GM_ARMS" nice -n "$NICE" "$PY" -u analysis/run_arm.py \
        --arm listener --script k_sweep_universal4_v2.py --mode mobile \
        2>&1 | tee -a "$LOG"
    ks="$GM_ARMS/listener/results/k_sweep_universal4_v2"
    if [ -f "$ks/k_sweep_results.csv" ]; then
        nice -n "$NICE" "$PY" analysis/select_optimal_k.py \
            --k-sweep-csv "$ks/k_sweep_results.csv" --out "$ks/optimal_k.json" 2>&1 | tee -a "$LOG" \
            || say "  select_optimal_k failed"
    else
        say "  no k_sweep_results.csv; stage 5 did not produce a sweep"
    fi
fi

# ---------------------------------------------------------------- summary ---
if want summary; then
    say "PHASE summary"

    echo "--- stage 1, mobile (Gauss-Markov, 10k) ---" | tee -a "$LOG"
    f="$GM_ARMS/listener/results/mobile/results.csv"
    [ -f "$f" ] && awk -F, 'NR==2 {n=$10+$11+$12+$13;
        printf "  %-20s AUC=%-8.4f acc=%-8.4f test_windows=%d test_runs=%d\n",
        $1,$3,$4,n,n/4}' "$f" | tee -a "$LOG"

    echo "--- selected set at each K (what answers 3.2) ---" | tee -a "$LOG"
    m="$GM_ARMS/listener/results/feature_importance_sensitivity_v2/universal_set_members.csv"
    if [ -f "$m" ]; then
        echo "  published (main campaign, K=3):" | tee -a "$LOG"
        sed 's/^/    /' "$ROOT/analysis/arms_c10k_stackcal/listener/results/feature_importance_sensitivity_v2/universal_set_k3/universal_set.txt" | tee -a "$LOG"
        echo "  this arm, K=3, all 12 (method,variant):" | tee -a "$LOG"
        awk -F, '$3==3 {printf "    %-16s %-7s %s\n", $1, $2, $7}' "$m" | tee -a "$LOG"
    else
        echo "  (no universal_set_members.csv yet)" | tee -a "$LOG"
    fi

    echo "--- K chosen by this arm ---" | tee -a "$LOG"
    cat "$GM_ARMS/listener/results/k_sweep_universal4_v2/optimal_k.json" 2>/dev/null | tee -a "$LOG" \
        || echo "  (not produced yet)" | tee -a "$LOG"

    echo "--- source-tree integrity ---" | tee -a "$LOG"
    if [ -f "$SNAP" ]; then
        { find "$SRC_STATIC" -maxdepth 3 -printf '%p %T@\n'
          find "$SRC_MOBILE" -maxdepth 3 -printf '%p %T@\n'; } 2>/dev/null | sort > "$SNAP.now"
        if diff -q "$SNAP" "$SNAP.now" >/dev/null; then
            say "  source trees UNTOUCHED"
        else
            say "  *** SOURCE TREES CHANGED -- stop and investigate:"
            diff "$SNAP" "$SNAP.now" | head -20 | tee -a "$LOG"
        fi
        rm -f "$SNAP.now"
    fi
    echo "--- 2,000-run arm still present? ---" | tee -a "$LOG"
    ls -d "$ROOT/analysis/arms_gaussmarkov" "$ROOT/analysis/arms_gm_control" 2>/dev/null | sed 's/^/  /' | tee -a "$LOG"
fi
