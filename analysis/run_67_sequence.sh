#!/usr/bin/env bash
# run_67_sequence.sh -- move every paper figure still learned with the 77-column
# generator onto the 67-column one (FILES_FOR_SUBMISSION 9.1, STATE 25.393/25.397).
#
#   bash run_67_sequence.sh              # = dry: prints the plan, runs nothing
#   bash run_67_sequence.sh learn        # stage 1 at 67 for the 8 arms (~35-40 min)
#   bash run_67_sequence.sh consumers    # window bootstrap, reuse control, fraction (~15 min)
#   bash run_67_sequence.sh summary      # new figures beside the manuscript's; writes nothing
#
# Nothing is deleted. marathon_frozencal.sh backs up every results.csv and
# archives the old pickles before it re-learns; the consumers write into NEW
# directories (suffix _67), so the 77-column outputs stay where they are.
set -u
AN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-$HOME/miniconda3/envs/manet/bin/python}"
[ -x "$PY" ] || PY=python3
P67=$AN/frozencal67_pipeline_17
STAGE=${1:-dry}
TS=$(date +%Y%m%d_%H%M%S)
LOG=$AN/run_67_sequence_${STAGE}_${TS}.log

QUEUE="arms_r34_17feat_2k|campaign arms_r34_17feat_2k_w_17|campaign \
arms_gm_control_17|campaign arms_windowsweep/w10_17|campaign arms_windowsweep/w20_17|campaign \
arms_trafficsweep/i1.0_static|campaign arms_trafficsweep/i1.0_mobile|campaign \
arms_trafficsweep/i0.5_static|campaign arms_trafficsweep/i0.5_mobile|campaign \
arms_perwindowseed_17|campaign arms_r34_17feat_nosplit|nosplit"

say() { echo "$*" | tee -a "$LOG"; }
die() { say "ABORT: $*"; exit 1; }

guards() {
    [ -f "$P67/defense_detection_v2.py" ] || die "no $P67/defense_detection_v2.py"
    grep -q FrozenEstimator "$P67/defense_detection_v2.py" || die "$P67 has no FrozenEstimator"
    n=$(pgrep -fc 'defense_detection_v2.py|marathon_frozencal.sh' 2>/dev/null || true)
    [ "${n:-0}" -eq 0 ] || die "a learning job is already running ($n); wait for it"
    say "guards OK: pipeline $P67, no learning running"
}

case "$STAGE" in
dry)
    say "=== PLAN (nothing runs) ==="
    say "learn     : marathon_frozencal.sh, PIPE=$P67, queue:"
    for q in $QUEUE; do say "              $q"; done
    say "            then every new log must say 'Created 67 features (from 18)' (checked here)"
    say "consumers : run_r36_bootstrap.sh  R36_PIPELINE=$P67 R36_SUFFIX=_67"
    say "            reuse_control.py --arm arms_r34_17feat_frozencal67 --mode both  (REUSE_PIPELINE=$P67)"
    say "            fraction_frozen_score.py --mode both --out-dir frozen17_frozencal67"
    say "              (FRAC_REPORTED=arms_r34_17feat_frozencal67 FRAC_PIPELINE=$P67)"
    say "summary   : prints the new figures beside the manuscript's"
    say "--- marathon's own dry run:"
    PIPE=$P67 QUEUE_OVERRIDE="$QUEUE" bash "$AN/marathon_frozencal.sh" --dry-run 2>&1 | tee -a "$LOG"
    ;;
learn)
    guards
    say "=== BACKUP of every results directory the round touches ==="
    # marathon backs up results.csv only; confusion/figure files are overwritten in
    # place and an archived best_model_*.pkl of the same name would be replaced by
    # its mv. So: a full snapshot per arm. Models (*.pkl) are hard-linked -- never
    # rewritten in place, so a link preserves them at ~0 disk cost -- and every
    # other file is a real copy, because those ARE overwritten in place.
    for q in $QUEUE; do
        root=${q%%|*}; src=$AN/$root/listener/results; dst=$AN/$root/listener/results.bak_pre67_$TS
        [ -d "$src" ] || die "no results dir: $src"
        [ -e "$dst" ] && die "backup target exists: $dst"
        cp -al "$src" "$dst" || die "hard-link copy failed: $src"
        find "$dst" -type f ! -name '*.pkl' -print0 | while IFS= read -r -d '' f; do
            cp -p "$f" "$f.tmp67" && mv -f "$f.tmp67" "$f" || { echo "copy failed: $f"; exit 1; }
        done || die "breaking links failed under $dst"
        a=$(find "$src" -type f | wc -l); b=$(find "$dst" -type f | wc -l)
        [ "$a" -eq "$b" ] || die "backup incomplete for $root ($a files vs $b)"
        say "  backup OK  $root  ($b files) -> $(basename "$dst")"
    done
    say "=== LEARN at 67, started $(date '+%F %T') ==="
    START=$(date +%s)
    PIPE=$P67 QUEUE_OVERRIDE="$QUEUE" bash "$AN/marathon_frozencal.sh" 2>&1 | tee -a "$LOG"
    [ "${PIPESTATUS[0]}" -eq 0 ] || die "marathon failed -- see above; completed runs keep their results"
    say "--- gate: 'Created 67 features (from 18)' in every log written by this round"
    bad=0
    for q in $QUEUE; do
        root=${q%%|*}; name=$(echo "$root" | tr '/' '_'); found=0
        for f in "$AN"/frozencal_marathon_logs/${name}_*.log; do
            [ -f "$f" ] || continue
            [ "$(stat -c %Y "$f")" -ge "$START" ] || continue      # this round only
            found=1
            if grep -q 'Created 67 features (from 18)' "$f"; then say "  67 OK  $(basename "$f")"
            else say "  NOT 67 $(basename "$f")"; bad=1; fi
        done
        [ $found -eq 1 ] || { say "  NO NEW LOG for $root"; bad=1; }
    done
    [ $bad -eq 0 ] || die "a log did not report 67 columns"
    say "=== LEARN done $(date '+%F %T') ==="
    ;;
consumers)
    guards
    # every consumer must write only where nothing exists yet
    for p in "$AN/r36_window_bootstrap_67" "$AN/r36_window_bootstrap_mobile_67" \
             "$AN/arms_fractionsweep/frozen17_frozencal67" \
             "$AN/arms_r34_17feat_frozencal67/listener/results/static/reuse_control.csv" \
             "$AN/arms_r34_17feat_frozencal67/listener/results/mobile/reuse_control.csv"; do
        [ -e "$p" ] && die "would overwrite an existing output: $p"
    done
    say "guards OK: no consumer output exists yet"
    say "=== CONSUMERS at 67, started $(date '+%F %T') ==="
    R36_PIPELINE=$P67 R36_SUFFIX=_67 bash "$AN/run_r36_bootstrap.sh" 2>&1 | tee -a "$LOG"
    [ "${PIPESTATUS[0]}" -eq 0 ] || die "run_r36_bootstrap.sh failed"
    ( cd "$AN" && REUSE_PIPELINE=$P67 "$PY" -u reuse_control.py \
        --arm arms_r34_17feat_frozencal67 --mode both ) 2>&1 | tee -a "$LOG"
    [ "${PIPESTATUS[0]}" -eq 0 ] || die "reuse_control.py failed"
    ( cd "$AN/arms_fractionsweep" && FRAC_REPORTED=$AN/arms_r34_17feat_frozencal67 \
        FRAC_PIPELINE=$P67 "$PY" -u fraction_frozen_score.py --mode both \
        --out-dir frozen17_frozencal67 ) 2>&1 | tee -a "$LOG"
    [ "${PIPESTATUS[0]}" -eq 0 ] || die "fraction_frozen_score.py failed"
    say "=== CONSUMERS done $(date '+%F %T') ==="
    ;;
summary)
    say "=== NEW FIGURES vs MANUSCRIPT (Stacking_Ensemble; accuracy / AUC) ==="
    row() {  # arm mode published
        local f=$AN/$1/listener/results/$2/results.csv
        local r=$(grep '^Stacking_Ensemble,' "$f" 2>/dev/null | awk -F, '{printf "%.4f / %.4f", $4, $3}')
        printf '%-34s %-6s now %-17s manuscript %s\n' "$1" "$2" "${r:-MISSING}" "$3" | tee -a "$LOG"
    }
    row arms_r34_17feat_2k            static "0.8856 / 0.9326"
    row arms_r34_17feat_2k            mobile "0.9063 / 0.9525"
    row arms_trafficsweep/i1.0_static static "0.8913"
    row arms_trafficsweep/i1.0_mobile mobile "0.9000"
    row arms_trafficsweep/i0.5_static static "0.8894"
    row arms_trafficsweep/i0.5_mobile mobile "0.9063"
    row arms_perwindowseed_17         static "0.8888 / 0.9277"
    row arms_perwindowseed_17         mobile "0.9137 / 0.9548"
    row arms_r34_17feat_nosplit       static "0.8988"
    row arms_r34_17feat_nosplit       mobile "0.9269"
    say "--- training volume (MS 4.1: 1.42 / 2.35 pp) = headline minus 2k, from the rows above"
    say "--- window sweep (MS 4.1: static +0.38 [-0.75,+1.56] @20s, +0.94 [-0.13,+1.94] @10s;"
    say "    mobile -0.75 [-2.50,+0.88] @20s, -2.56 [-4.50,-0.63] @10s)"
    for f in "$AN"/r36_window_bootstrap_67/*.json "$AN"/r36_window_bootstrap_mobile_67/*.json; do
        [ -f "$f" ] || continue
        say "  $(basename "$f"): $(grep -o '"delta_pp": *[-0-9.]*' "$f" | head -1)  $(grep -o '"ci95_pp": *\[[^]]*\]' "$f" | head -1)"
    done
    say "--- reuse control (MS 5.3: mean -0.01 / -0.06 pp, median 0.00 / -0.02)"
    ls "$AN"/arms_r34_17feat_frozencal67/listener/results/*/reuse_control.csv 2>/dev/null | tee -a "$LOG"
    say "--- fraction (MS 4.1: mobile f=0.25 0.8832, -0.0461 [-0.0587,-0.0354]; static f=0.50 +0.0219 [+0.0128,+0.0316])"
    ls "$AN"/arms_fractionsweep/frozen17_frozencal67/*.json 2>/dev/null | tee -a "$LOG"
    say "=== read the reuse and fraction files named above; nothing here was written ==="
    ;;
*) die "unknown stage '$STAGE' (dry | learn | consumers | summary)";;
esac
