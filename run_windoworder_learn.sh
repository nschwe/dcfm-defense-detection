#!/usr/bin/env bash
# ============================================================================
# run_windoworder_learn.sh — pairing + bundles + learning for the window-order
# experiment (STATE §23.10 / §25.18). Run AFTER run_windoworder.sh.
#
# Builds the PAIRED comparison the design requires: for every (id, seed) in the
# shuffled manifest, the SAME seed's canonical run is staged under the SAME new
# id, so the two datasets differ in window order and in nothing else.
#
#   simulations_v347_windoworder_pair/features_<mode>/...   <- symlinks into the
#                                                              canonical campaign
#   analysis/arms_windoworder/shuffled/listener/            <- bundle + results
#   analysis/arms_windoworder/canonical/listener/           <- bundle + results
#
# ADDITIVE ONLY: canonical campaign read-only (symlinks); no --force; existing
# results are never overwritten. Learning is ~2h per (variant, mode) at N=2,000.
#
#   nohup bash run_windoworder_learn.sh >> windoworder_learn_static.log 2>&1 &
#   MODE=mobile nohup bash run_windoworder_learn.sh >> windoworder_learn_mobile.log 2>&1 &
#   STAGE_ONLY=1 bash run_windoworder_learn.sh     # pairing + bundles, no learning
# ============================================================================
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${DCFM_PYTHON:-$HOME/miniconda3/envs/manet/bin/python}"
[ -x "$PY" ] || PY=python3
SRC="$ROOT/simulations_v347"                       # canonical campaign (read-only)
WO="${WO:-$ROOT/simulations_v347_windoworder}"     # shuffled tree (from run_windoworder.sh)
PAIR="${PAIR:-$ROOT/simulations_v347_windoworder_pair}"  # canonical subset, re-id'd
ARMS="${ARMS_OUT:-$ROOT/analysis/arms_windoworder}"
MODE="${MODE:-static}"
MAX_JOBS="${MAX_JOBS:-22}"
STAGE_ONLY="${STAGE_ONLY:-0}"
SCENARIOS="baseline attack_only defense_only defense_vs_attack"
KINDS="metrics_output observer_metrics observer_detail"

MANIFEST="$WO/canonical/manifest_${MODE}.csv"

say() { echo; echo "############ $* ############"; echo "  $(date '+%F %T')"; }

for m in canonical shuffled; do
  [ -f "$WO/$m/manifest_${MODE}.csv" ] || {
    echo "ERROR: $m manifest missing: $WO/$m/manifest_${MODE}.csv"
    echo "  run: ARM=$m MODE=$MODE bash run_windoworder.sh"; exit 1; }
done
if pgrep -f "defense_detection_v2.py" >/dev/null 2>&1; then
  echo "ABORT: a learning job is already running:"; pgrep -af "defense_detection_v2.py"; exit 1
fi

N=$(( $(wc -l < "$MANIFEST") - 1 ))
echo "=========================================================="
echo " window-order pairing + learning"
echo "   mode     : $MODE      N (from manifest): $N"
echo "   shuffled : $WO"
echo "   paired   : $PAIR"
echo "=========================================================="

# ---------------------------------------------------- 1. paired staging -----
# Both arms are now REAL simulation trees produced by run_windoworder.sh under
# the same --formationLeadIn, so there is nothing to symlink (STATE §25.20).
# What must be checked is that they cover exactly the same seeds under the same
# ids — that is what makes the comparison paired.
say "1/4  verifying the two arms are paired (same seed at every id)"
"$PY" - "$WO/canonical/manifest_${MODE}.csv" "$WO/shuffled/manifest_${MODE}.csv" <<'PYEOF'
import csv, sys
def load(p):
    with open(p) as fh:
        return {r["id"]: r for r in csv.DictReader(fh)}
try:
    c, s = load(sys.argv[1]), load(sys.argv[2])
except FileNotFoundError as e:
    print(f"  ERROR: {e}"); sys.exit(1)
print(f"  canonical ids: {len(c)}   shuffled ids: {len(s)}")
common = sorted(set(c) & set(s), key=int)
mismatch = [i for i in common if c[i]["seed"] != s[i]["seed"]]
if mismatch:
    print(f"  ERROR: {len(mismatch)} ids carry different seeds, first 5: "
          f"{[(i, c[i]['seed'], s[i]['seed']) for i in mismatch[:5]]}")
    sys.exit(1)
only_c, only_s = sorted(set(c) - set(s), key=int), sorted(set(s) - set(c), key=int)
if only_c or only_s:
    print(f"  ERROR: {len(only_c)} ids only in canonical, {len(only_s)} only in shuffled.")
    print("  Rerun the shuffled arm with SEEDS_FROM=<canonical manifest> so both")
    print("  arms settle on the same seed set.")
    sys.exit(1)
diff_order = sum(1 for i in common
                 if [c[i][f"slot{k}"] for k in range(4)] != [s[i][f"slot{k}"] for k in range(4)])
print(f"  PAIRED: {len(common)} ids, identical seeds")
print(f"  runs whose window order actually differs: {diff_order} / {len(common)}")
PYEOF
rc=$?; [ $rc -ne 0 ] && { echo "ABORT: the two arms are not paired"; exit $rc; }

# ---------------------------------------------------- 2. bundles ------------
say "2/4  bundles (listener arm, both variants)"
want_rows=$(( N * 4 + 1 ))
build_bundle() { # $1=variant  $2=sim_root
  local b="$ARMS/$1/listener/colab_data/wide_${MODE}.csv.gz"
  if [ -f "$b" ] && [ "$(zcat "$b" 2>/dev/null | wc -l)" -eq "$want_rows" ]; then
    echo "  [skip] $1 bundle exists with $((want_rows-1)) rows"; return 0
  fi
  DCFM_SIM_ROOT="$2" DCFM_ARMS_ROOT="$ARMS/$1" \
    "$PY" "$ROOT/analysis/make_arm_bundles.py" --arm listener --mode "$MODE" --limit "$N"
}
build_bundle shuffled  "$WO/shuffled"  || exit 1
build_bundle canonical "$WO/canonical" || exit 1

[ "$STAGE_ONLY" = 1 ] && { echo "STAGE_ONLY=1 — stopping before learning."; exit 0; }

# ---------------------------------------------------- 3. learning -----------
for variant in shuffled canonical; do
  say "3/4  learning — $variant / $MODE"
  DCFM_ARMS_ROOT="$ARMS/$variant" MAX_JOBS="$MAX_JOBS" \
    "$PY" -u "$ROOT/analysis/run_arm.py" \
    --arm listener --script defense_detection_v2.py --mode "$MODE" \
    -- --no-augmentation || echo "  [warn] $variant learning exited $?"
done

# ---------------------------------------------------- 4. summary ------------
say "4/4  SUMMARY — shuffled vs canonical (paired, same seeds, N=$N)"
printf "  %-11s %10s %10s\n" VARIANT ACCURACY AUC
for variant in shuffled canonical; do
  f="$ARMS/$variant/listener/results/$MODE/results.csv"
  a=$(awk -F, '$1=="Stacking_Ensemble"{print $4}' "$f" 2>/dev/null)
  u=$(awk -F, '$1=="Stacking_Ensemble"{print $3}' "$f" 2>/dev/null)
  printf "  %-11s %10.6s %10.6s\n" "$variant" "${a:--}" "${u:--}"
done
echo
echo "  Both arms are the SAME seeds under --formationLeadIn=60; the only"
echo "  difference is window order. Compare within this table only (this is not"
echo "  the 400 s main-campaign number)."
