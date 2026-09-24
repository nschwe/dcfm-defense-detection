#!/usr/bin/env bash
# ============================================================================
# run_gm10k17_all.sh — migrate the Gauss-Markov arm (R#3.2) into the
# 17-observable reported space, unattended, end to end.
#
#   stage 1        run_prop17.sh   -> analysis/arms_gm10k_17
#   stage 4 + 4b   run_item8_full.sh (feature importance, universal set, wiring)
#   stage 5        k sweep, anchored on THIS arm's stage 4, --k-list 1..17
#
# ⛔ WHY THIS EXISTS: arms_gm10k stage 4 ranked a pool of 33 columns of which
#    12 are zero-filled phantoms (AverageJitter, PacketDeliveryRatio, ...).
#    The anchor and the feature-stability claim in r3_2_answer.tex rest on it.
#    Stage 5 was already clean (it drops phantoms; it loaded 22 columns), so
#    the transfer table moves little -- but the selection does not.
#
#   nohup bash run_gm10k17_all.sh > /dev/null 2>&1 &
#
# Every step is gated. Any gate failure stops the run and says why.
# Nothing existing is overwritten: all output lands under arms_gm10k_17.
# ============================================================================
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-$HOME/miniconda3/envs/manet/bin/python}"
[ -x "$PY" ] || PY=python3
SRC="$ROOT/analysis/arms_gm10k"
DST="$ROOT/analysis/arms_gm10k_17"
PIPE="$ROOT/analysis/cleanfeat/pipeline_17"
HP="$HOME/ns3/Final_Project_NS3-master/strict_observable_v2/colab_data/results_cache/hp_search_extended"
WANT_ROWS=40001
K_LIST=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17

STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$ROOT/gm10k17_all_${STAMP}.log"
exec > >(tee -a "$LOG") 2>&1

t_start=$(date +%s)
say()  { echo; echo "############ $* ############"; echo "  $(date '+%F %T')"; }
ok()   { echo "  [ok] $*"; }
die()  { echo; echo "⛔⛔ ABORT: $*"; echo "  $(date '+%F %T')"; echo "  log: $LOG"; exit 1; }
rows() { zcat "$1" 2>/dev/null | wc -l; }
md5()  { zcat "$1" 2>/dev/null | md5sum | cut -d' ' -f1; }

# the 12 columns the detector zero-fills and stage 5 drops as phantoms
PHANTOMS="AverageEndToEndDelay AverageJitter AvgFlowDelay AvgFlowJitter \
AvgFlowLossRate FlowCount FlowDelayStd FlowJitterStd FlowLossRateStd \
PacketDeliveryRatio PacketLossRatio RxTxPacketRatio"

echo "=========================================================="
echo " gm10k -> 17-observable space (R#3.2)"
echo "   source : $SRC"
echo "   dest   : $DST"
echo "   log    : $LOG"
echo "=========================================================="

# ---------------------------------------------------------------- PRE ------
say "GATE 0 — preconditions"

[ -x "$PY" ] || die "manet python missing: $PY"
ok "python"

[ -d "$PIPE" ] || die "pipeline missing: $PIPE"

[ -d "$HP" ] || die "hp results cache missing: $HP  (stage 5 needs --hp-results-dir)"
ok "hp results cache"

# ⛔⛔ PROVE, BEFORE SPENDING 10 HOURS, THAT THE 141-COLUMN SPACE IS UNREACHABLE.
# run_arm.py:39 reads DCFM_PIPELINE and falls back to analysis/pipeline, which is
# the 33-metric / 141-column one. runstage() does not set it. So:
#   (a) the variable name must still be the one run_arm.py reads,
#   (b) the fallback must really be the 33-space one (else this gate proves nothing),
#   (c) the pipeline we export must expose 17.
grep -q 'os.environ.get("DCFM_PIPELINE"' "$ROOT/analysis/run_arm.py" \
  || die "run_arm.py no longer reads DCFM_PIPELINE — the export below would be ignored"
ok "run_arm.py still reads DCFM_PIPELINE"

count_metrics() {
  "$PY" - "$1" <<'PYEOF' 2>/dev/null | grep -E '^[0-9]+$' | tail -1
import sys, importlib.util, os
spec = importlib.util.spec_from_file_location("dd", os.path.join(sys.argv[1], "defense_detection_v2.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print(len(m.DefenseDetector.METRICS))
PYEOF
}
FALLBACK="$ROOT/analysis/pipeline"
FMET=$(count_metrics "$FALLBACK")
[ "$FMET" = "17" ] && die "the fallback pipeline also reports 17 — this gate cannot discriminate, refusing to trust it"
ok "fallback pipeline ($FALLBACK) exposes ${FMET:-?} metrics — the gate discriminates"

NMET=$(count_metrics "$PIPE")
[ "$NMET" = "17" ] || die "pipeline exposes '${NMET:-?}' metrics, expected 17 — this is the 33-space guard"
ok "pipeline we will export exposes 17 metrics — the 141-column space is unreachable"

for m in static mobile; do
  f="$SRC/listener/colab_data/wide_${m}.csv.gz"
  [ -f "$f" ] || die "source bundle missing: $f"
  r=$(rows "$f")
  [ "$r" = "$WANT_ROWS" ] || die "source bundle $m has $r rows, expected $WANT_ROWS"
done
ok "source bundles present, both $WANT_ROWS rows"

SRC_MOBILE_MD5=$(md5 "$SRC/listener/colab_data/wide_mobile.csv.gz")
SRC_STATIC_MD5=$(md5 "$SRC/listener/colab_data/wide_static.csv.gz")
echo "  source mobile md5 : $SRC_MOBILE_MD5   (Gauss-Markov)"
echo "  source static md5 : $SRC_STATIC_MD5"

if [ "${SKIP_STAGE1:-0}" = "1" ]; then
  # Resume path: stage 1 already produced this arm. Verify it instead of
  # rebuilding it -- same checks GATE 1 would have run.
  [ -d "$DST" ] || die "SKIP_STAGE1=1 but $DST does not exist"
  for m in static mobile; do
    f="$DST/listener/colab_data/wide_${m}.csv.gz"
    [ -f "$f" ] || die "SKIP_STAGE1=1 but migrated bundle missing: $f"
    r=$(rows "$f"); [ "$r" = "$WANT_ROWS" ] || die "migrated bundle $m has $r rows, expected $WANT_ROWS"
    [ -d "$DST/listener/results/$m" ] || die "SKIP_STAGE1=1 but stage 1 left no results/$m — stage 1 did not finish"
  done
  [ "$(md5 "$DST/listener/colab_data/wide_mobile.csv.gz")" = "$SRC_MOBILE_MD5" ] \
    || die "existing mobile bundle is not the Gauss-Markov data"
  [ "$(md5 "$DST/listener/colab_data/wide_static.csv.gz")" = "$SRC_STATIC_MD5" ] \
    || die "existing static bundle does not match the source"
  ok "SKIP_STAGE1: existing arm verified — bundles, row counts, md5, stage-1 results for both modes"

  # ⛔ The importance cache is keyed by FILENAME ONLY -- no fingerprint of the
  # data it was computed on. `cache hit` will happily reload a cache built in
  # the wrong space. So inspect its contents, not its existence.
  CACHE="$DST/listener/results/feature_importance_sensitivity_v2/importance_cache"
  if [ -d "$CACHE" ] && [ -n "$(ls -A "$CACHE" 2>/dev/null)" ]; then
    "$PY" - "$CACHE" <<'PYEOF'
import glob, os, sys
import joblib as jl
PH = {"AverageEndToEndDelay","AverageJitter","AvgFlowDelay","AvgFlowJitter",
      "AvgFlowLossRate","FlowCount","FlowDelayStd","FlowJitterStd",
      "FlowLossRateStd","PacketDeliveryRatio","PacketLossRatio","RxTxPacketRatio"}
bad = []
files = sorted(glob.glob(os.path.join(sys.argv[1], "*.pkl")))
if not files:
    sys.exit(0)
checked = 0
for p in files:
    c = jl.load(p)
    for mode in ("static", "mobile"):
        s = c.get(mode) or []
        if len(s) != 20:
            bad.append(f"{os.path.basename(p)}/{mode}: {len(s)} seeds, expected 20"); continue
        # EVERY seed, not just the first -- one divergent seed is a wrong-space seed
        for i, ser in enumerate(s):
            idx = set(ser.index)
            if idx & PH:
                bad.append(f"{os.path.basename(p)}/{mode} seed {i}: phantoms {sorted(idx & PH)}")
            elif len(idx) != 17:
                bad.append(f"{os.path.basename(p)}/{mode} seed {i}: {len(idx)} features, expected 17")
            checked += 1
if bad:
    print("CACHE REJECTED:"); [print("   " + b) for b in bad[:10]]; sys.exit(1)
print(f"   cache verified: {len(files)} methods x 2 modes x 20 seeds = {checked} series, all 17 features, phantom-free")
PYEOF
    [ $? -eq 0 ] || die "the importance cache is not the 17-observable space — delete $CACHE and let it recompute"
    ok "importance cache content verified (not merely present)"
  fi
else
  [ -e "$DST" ] && die "destination already exists: $DST  (refusing to overwrite; move it aside first, or set SKIP_STAGE1=1 to reuse its stage 1)"
  ok "destination is free"
fi

n=$(pgrep -fa 'defense_detection_v2.py|run_item8_full.sh|run_prop17.sh' 2>/dev/null | grep -v "$$" | grep -vc 'run_gm10k17_all')
[ "${n:-0}" -eq 0 ] || { pgrep -fa 'defense_detection_v2.py|run_item8_full.sh|run_prop17.sh'; die "learning already running ($n) — refusing to compete for cores"; }
ok "no learning running"

# --------------------------------------------------------------- STAGE 1 ---
if [ "${SKIP_STAGE1:-0}" = "1" ]; then
  say "STAGE 1 — SKIPPED, already complete and verified in GATE 0"
else
say "STAGE 1 — re-learn in the 17-observable space (run_prop17.sh)"
# JOBS=16 is the pipeline's own DEFAULT_MAX_JOBS and what the 10-11 h reference
# run used. runtime_guard.py reads MAX_JOBS from the environment and pins every
# BLAS library to one thread, so a low value throttles the inner fits.
# ⚠️ Do not raise it above 16: runtime_guard exists because nested n_jobs
# reached 144 threads on 24 cores (§25.166.1).
ARMS_LIST="arms_gm10k" MAX_JOBS_LEARN="${JOBS:-16}" bash "$ROOT/run_prop17.sh"
rc=$?
[ $rc -eq 0 ] || die "run_prop17.sh exited $rc"

say "GATE 1 — stage 1 output"
for m in static mobile; do
  f="$DST/listener/colab_data/wide_${m}.csv.gz"
  [ -f "$f" ] || die "migrated bundle missing: $f"
  r=$(rows "$f")
  [ "$r" = "$WANT_ROWS" ] || die "migrated bundle $m has $r rows, expected $WANT_ROWS"
done
ok "migrated bundles, both $WANT_ROWS rows"

# ⛔ the decisive one: the Gauss-Markov data must have survived the copy.
NEW_MOBILE_MD5=$(md5 "$DST/listener/colab_data/wide_mobile.csv.gz")
NEW_STATIC_MD5=$(md5 "$DST/listener/colab_data/wide_static.csv.gz")
[ "$NEW_MOBILE_MD5" = "$SRC_MOBILE_MD5" ] || die "mobile bundle changed in the copy: $NEW_MOBILE_MD5 != $SRC_MOBILE_MD5 — this would mean the Gauss-Markov data was replaced"
[ "$NEW_STATIC_MD5" = "$SRC_STATIC_MD5" ] || die "static bundle changed in the copy"
ok "Gauss-Markov mobile data preserved byte-for-byte"

# the log is written through a tee subprocess; let it flush before reading it
sync; sleep 3
if grep -qE 'Created 141 features|\(from 34\)|Shape: \([0-9]+, 34\)' "$LOG"; then
  grep -nE 'Created 141 features|\(from 34\)|Shape: \([0-9]+, 34\)' "$LOG" | head
  die "a 33-space banner appeared in stage 1 — the wrong pipeline ran"
fi
grep -q '(from 18)' "$LOG" || die "no '(from 18)' banner in stage 1 — cannot confirm the 17-observable space"
ok "space banners: '(from 18)' present, no 34/141 anywhere"
fi   # end SKIP_STAGE1

# ------------------------------------------------------------ STAGE 4 + 5 --
say "GATE 2 — pre-flight for stages 4/5"

# ⛔⛔ run_item8_full.sh rebuilds a bundle unless it already has EXACTLY
#     N*4+1 rows, and it would rebuild the mobile side from
#     simulations_v347_hopablation_mobile_10k -- RandomWalk2d, NOT
#     Gauss-Markov. The row check above is what makes it skip. Re-assert it.
for m in static mobile; do
  r=$(rows "$DST/listener/colab_data/wide_${m}.csv.gz")
  [ "$r" = "$WANT_ROWS" ] || die "bundle $m is $r rows — run_item8_full would REBUILD it from the RandomWalk2d tree"
done
ok "bundle row counts will make run_item8_full skip its rebuild"

export DCFM_PIPELINE="$PIPE"   # runstage() does NOT set this; without it -> silent 33-space
ok "DCFM_PIPELINE exported: $DCFM_PIPELINE"

say "STAGE 4 + 4b + 5 — expect ~10-11 h for stage 4"
ARMS_ROOT="$DST" STAGES=4,5 K_LIST="$K_LIST" K_FULL=17 MAX_JOBS="${JOBS:-16}" \
  bash "$ROOT/run_item8_full.sh"
rc=$?
[ $rc -eq 0 ] || echo "  [warn] run_item8_full.sh exited $rc — checking outputs anyway"

# ---------------------------------------------------------------- GATES ----
say "GATE 3 — stage 4 output"
FI="$DST/listener/results/feature_importance_sensitivity_v2"
MEM="$FI/universal_set_members.csv"
[ -f "$MEM" ] || die "stage 4 produced no universal_set_members.csv"

MAXK=$(awk -F',' 'NR>1 && $3+0>m {m=$3+0} END{print m+0}' "$MEM")
[ "$MAXK" = "17" ] || die "stage 4 pool max K is $MAXK, expected 17 — the pool is not the 17-observable one"
ok "stage 4 pool max K = 17"

MEMROW=$(awk -F',' 'NR>1 && $3+0==17 {print; exit}' "$MEM")
[ -n "$MEMROW" ] || die "no K=17 row in $MEM"
BAD=""
for p in $PHANTOMS; do
  case "$MEMROW" in *"$p"*) BAD="$BAD $p";; esac
done
[ -z "$BAD" ] || die "PHANTOM columns present in stage 4 pool:$BAD"
ok "no phantom columns in the stage 4 pool"

US="$FI/universal_set.txt"
[ -f "$US" ] || die "stage 4b did not write universal_set.txt — stage 5 would fall back to the paper's Universal-4"
ok "anchor written: $(tr '\n' ' ' < "$US")"

say "GATE 4 — stage 5 output"
sync; sleep 3
grep -q 'Anchor set OVERRIDDEN' "$LOG" || die "stage 5 did not report an overridden anchor — it may have used the built-in Universal-4"
ok "anchor override confirmed in the log"

KS="$DST/listener/results/k_sweep_universal4_v2/k_sweep_summary.txt"
[ -f "$KS" ] || die "stage 5 produced no k_sweep_summary.txt"
ok "k sweep summary written"

# --------------------------------------------------------------- REPORT ----
say "RESULT — Gauss-Markov in the 17-observable space"
cat "$KS"

echo
echo "  ---- the superseded 33-column run, for comparison (do NOT mix them) ----"
echo "     K | S->S    | GM->GM  | S->GM   | GM->S   | asym"
echo "     3 | 0.9030  | 0.9339  | 0.8330  | 0.5610  | 0.2720"
echo "     1 | 0.7670  | 0.8484  | 0.8079  | 0.7321  | 0.0757"
echo "  reported campaign, 17-space, K=3: S->S 0.9032  M->M 0.9291  asym 0.0098"
echo
echo "  anchor (this arm's own stage 4): $(tr '\n' ' ' < "$US")"

t_end=$(date +%s); mins=$(( (t_end - t_start) / 60 ))
say "DONE — ${mins} min total"
echo "  outputs : $DST/listener/results/"
echo "  log     : $LOG"
echo
echo "  ⛔ Next, by hand: r3_2_answer.tex still quotes the 33-column table."
echo "     Nothing here edits any answer file."
