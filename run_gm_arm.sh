#!/usr/bin/env bash
# ============================================================================
# run_gm_arm.sh -- the Gauss-Markov mobility arm (Reviewer 3, Comment 3.2).
#
# THE QUESTION. The submitted cross-domain setup compares static against one
# mobility model (RandomWalk2d). R#3.2 asks whether the selected feature set
# survives a qualitatively different kind of motion. This arm adds Gauss-Markov
# -- temporally correlated velocity and direction -- as a second mobile domain.
#
# THE DESIGN. Static is the fixed anchor; only the mobile half is swapped, so
# the comparison is (static, RandomWalk2d) against (static, GaussMarkov) and
# differs in exactly one thing. Both mobile halves are 2,000 runs: Gauss-Markov
# is simulated here, RandomWalk2d is copied out of the reported 10,000-run
# campaign. The static half is the SAME 2,000 runs in both trees.
#
# ⛔ WHERE THE CONTROL DATA COMES FROM. run_item8_full.sh:100-101 is the only
# authority on which tree backs the paper's numbers:
#     static : simulations_v347_hopablation_10k/C_all
#     mobile : simulations_v347_hopablation_mobile_10k/C_all
# NOT simulations_v347 -- that is an earlier tree of the same size and is easy
# to mistake for the campaign. C_all is the UNRESTRICTED hop population (its
# manifests/ also holds A_ge3 and B_ge2), which is why the campaign below runs
# --enforceHopFilter=0. A gate left on would build a different population from
# the control and break the comparison silently.
#
# ⛔ DATA IS COPIED, NEVER READ IN PLACE. The source trees belong to another
# campaign. Reading from them leaves every later step one typo away from writing
# into them. copy_subset.py refuses to write inside its own source.
#
# WHAT IS AND IS NOT MEASURED. Stage 4 runs with --skip-evaluation: we take the
# importance RANKING in each domain pair and ask whether the same three features
# come out on top. We do not re-derive the optimal K, and we do not repeat the
# twelve-criteria agreement sweep -- that exists to answer the circularity
# charge (R#3.3 / R#5.10) on the main campaign and is a different question.
#
# Usage:
#   bash run_gm_arm.sh                    # all phases in order
#   PHASE=campaign bash run_gm_arm.sh     # one phase
#   TARGET=200 bash run_gm_arm.sh         # smaller, for a rehearsal
# ============================================================================
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

TARGET="${TARGET:-2000}"
WORKERS="${WORKERS:-22}"
NICE="${NICE:-10}"
PHASE="${PHASE:-all}"

# --- sources: read-only, copied out of, never written to --------------------
# Staging ROOTS: each holds manifests/C_all.csv beside C_all/features_<mode>/.
SRC_STATIC="$ROOT/simulations_v347_hopablation_10k"
SRC_MOBILE="$ROOT/simulations_v347_hopablation_mobile_10k"

# --- everything this arm creates --------------------------------------------
GM_SIM="$ROOT/simulations_v347_gaussmarkov"
CT_SIM="$ROOT/simulations_v347_gm_control"
GM_ARMS="$ROOT/analysis/arms_gaussmarkov"
CT_ARMS="$ROOT/analysis/arms_gm_control"

SNAP="$ROOT/.gm_arm_src_snapshot"
LOG="$ROOT/gm_arm.log"

PY="${PY:-$HOME/miniconda3/envs/manet/bin/python}"
[ -x "$PY" ] || PY=python3

say () { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
die () { say "ABORT: $*"; exit 1; }
want () { [ "$PHASE" = all ] || [ "$PHASE" = "$1" ]; }

# --- guard: nothing this script creates may sit inside a source tree ---------
for p in "$GM_SIM" "$CT_SIM" "$GM_ARMS" "$CT_ARMS"; do
    case "$p" in
        "$SRC_STATIC"*|"$SRC_MOBILE"*|"$ROOT"/simulations_v347/*)
            die "output path sits inside a source tree: $p" ;;
    esac
done
[ -d "$SRC_STATIC" ] || die "missing source tree: $SRC_STATIC"
[ -d "$SRC_MOBILE" ] || die "missing source tree: $SRC_MOBILE"

# ------------------------------------------------------------------ guard ---
if want guard; then
    say "PHASE guard -- snapshotting both source trees"
    { find "$SRC_STATIC" -maxdepth 2 -printf '%p %T@\n'
      find "$SRC_MOBILE" -maxdepth 2 -printf '%p %T@\n'; } 2>/dev/null | sort > "$SNAP"
    say "  $(wc -l < "$SNAP") entries recorded"
fi

# ------------------------------------------------------------------- prep ---
if want prep; then
    say "PHASE prep -- generating the Gauss-Markov campaign script"
    "$PY" "$ROOT/patch_gm_campaign.py" 2>&1 | tee -a "$LOG" || die "patch failed"
    ./build/scratch/ns3.47-iolsr-tests-corrected-default --PrintHelp 2>&1 \
        | grep -q mobilityModel || die "built binary has no --mobilityModel; run ./ns3 build"
    grep -q -- "--enforceHopFilter=0" run_campaign_gaussmarkov.sh \
        || die "generated campaign script does not disable the distance gate"
    say "  binary carries --mobilityModel; campaign script disables the distance gate"
fi

# --------------------------------------------------------------- campaign ---
if want campaign; then
    say "PHASE campaign -- $TARGET accepted runs, mobile, Gauss-Markov, no distance gate"
    MODE=mobile TARGET_ACCEPTED="$TARGET" NUM_WORKERS="$WORKERS" \
        nice -n "$NICE" bash "$ROOT/run_campaign_gaussmarkov.sh" 2>&1 | tee -a "$LOG"
    acc=$(( $(wc -l < "$GM_SIM/accepted_seeds_mobile.csv" 2>/dev/null || echo 1) - 1 ))
    say "  accepted: $acc"
    [ "$acc" -ge "$TARGET" ] || die "campaign short of target ($acc < $TARGET)"
    if grep -q distance_lt3hops "$GM_SIM/run_status_mobile.csv" 2>/dev/null; then
        die "distance rejections present -- the gate was ON. Discard $GM_SIM and re-run prep."
    fi
    say "  verified: zero distance rejections"
fi

# ------------------------------------------------------------------- copy ---
if want copy; then
    say "PHASE copy -- building two self-contained trees at $TARGET runs each"
    say "  static half (identical in both trees)"
    "$PY" copy_subset.py --src "$SRC_STATIC" --dst "$GM_SIM" --mode static --limit "$TARGET" 2>&1 | tee -a "$LOG" || die "copy failed"
    "$PY" copy_subset.py --src "$SRC_STATIC" --dst "$CT_SIM" --mode static --limit "$TARGET" 2>&1 | tee -a "$LOG" || die "copy failed"
    say "  RandomWalk2d mobile half (control only)"
    "$PY" copy_subset.py --src "$SRC_MOBILE" --dst "$CT_SIM" --mode mobile --limit "$TARGET" 2>&1 | tee -a "$LOG" || die "copy failed"
    say "  (Gauss-Markov mobile half was simulated, not copied)"
fi

# ---------------------------------------------------------------- bundles ---
if want bundles; then
    say "PHASE bundles -- listener arm, both modes, both trees"
    for pair in "$GM_SIM:$GM_ARMS:gaussmarkov" "$CT_SIM:$CT_ARMS:control"; do
        sim="${pair%%:*}"; rest="${pair#*:}"; arms="${rest%%:*}"; name="${rest##*:}"
        for m in static mobile; do
            say "  $name / $m"
            # ⛔ No --limit here. make_arm_bundles sorts by numeric id and slices,
            # which on a C_all tree would keep only the >=3-hop runs. The trees
            # bundled here were already cut to size by copy_subset.py, which
            # selects through the manifest instead.
            DCFM_SIM_ROOT="$sim" DCFM_ARMS_ROOT="$arms" \
                nice -n "$NICE" "$PY" analysis/make_arm_bundles.py --arm listener --mode "$m" 2>&1 | tee -a "$LOG"
            f="$arms/listener/colab_data/wide_$m.csv.gz"
            [ -f "$f" ] || die "no bundle produced: $f"
        done
    done
fi

# ------------------------------------------------------------------ learn ---
if want learn; then
    say "PHASE learn -- stage 1, mobile, on both trees"
    for pair in "$GM_ARMS:gaussmarkov" "$CT_ARMS:control"; do
        arms="${pair%%:*}"; name="${pair##*:}"
        say "  $name"
        DCFM_ARMS_ROOT="$arms" nice -n "$NICE" "$PY" -u analysis/run_arm.py \
            --arm listener --script defense_detection_v2.py --mode mobile \
            -- --no-augmentation --split-validation --add-linearsvc --calibrate-stacking \
            2>&1 | tee -a "$LOG"
    done
fi

# ------------------------------------------------------------------- rank ---
if want rank; then
    say "PHASE rank -- stage 4, ranking only (--skip-evaluation)"
    for pair in "$GM_ARMS:gaussmarkov" "$CT_ARMS:control"; do
        arms="${pair%%:*}"; name="${pair##*:}"
        say "  $name"
        DCFM_ARMS_ROOT="$arms" nice -n "$NICE" "$PY" -u analysis/run_arm.py \
            --arm listener --script feature_importance_sensitivity_v2.py \
            -- --skip-evaluation \
            2>&1 | tee -a "$LOG"
    done
fi

# ---------------------------------------------------------------- summary ---
if want summary; then
    say "PHASE summary"

    echo "--- stage 1, mobile ---" | tee -a "$LOG"
    for pair in "$GM_ARMS:gaussmarkov" "$CT_ARMS:control"; do
        arms="${pair%%:*}"; name="${pair##*:}"
        f="$arms/listener/results/mobile/results.csv"
        if [ -f "$f" ]; then
            awk -F, -v L="$name" 'NR==2 {n=$10+$11+$12+$13;
                printf "%-12s model=%-20s AUC=%-8.4f acc=%-8.4f test_windows=%d test_runs=%d\n",
                L, $1, $3, $4, n, n/4}' "$f" | tee -a "$LOG"
        else
            printf "%-12s (no results yet)\n" "$name" | tee -a "$LOG"
        fi
    done

    echo "--- top features (this is what answers 3.2) ---" | tee -a "$LOG"
    for pair in "$GM_ARMS:gaussmarkov" "$CT_ARMS:control"; do
        arms="${pair%%:*}"; name="${pair##*:}"
        d="$arms/listener/results/feature_importance_sensitivity_v2"
        f=$(find "$d" -name 'universal_set*.txt' -print -quit 2>/dev/null)
        if [ -n "$f" ]; then
            echo "  $name: $(tr '\n' ' ' < "$f")" | tee -a "$LOG"
        else
            echo "  $name: (no ranking yet; looked under $d)" | tee -a "$LOG"
        fi
    done

    echo "--- source-tree integrity ---" | tee -a "$LOG"
    if [ -f "$SNAP" ]; then
        { find "$SRC_STATIC" -maxdepth 2 -printf '%p %T@\n'
          find "$SRC_MOBILE" -maxdepth 2 -printf '%p %T@\n'; } 2>/dev/null | sort > "$SNAP.now"
        if diff -q "$SNAP" "$SNAP.now" >/dev/null; then
            say "  source trees UNTOUCHED"
        else
            say "  *** SOURCE TREES CHANGED -- stop and investigate:"
            diff "$SNAP" "$SNAP.now" | head -20 | tee -a "$LOG"
        fi
        rm -f "$SNAP.now"
    fi
fi
