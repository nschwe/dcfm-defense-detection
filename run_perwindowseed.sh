#!/usr/bin/env bash
# ============================================================================
# run_perwindowseed.sh — the independent-topology-per-window experiment
#                        (R#5.5 / item 30; plan in STATE §25.78).
#
# WHAT IT ANSWERS
#   Reviewers 3 and 5 argue that the four measurement windows of a run share a
#   topology, so the classifier may be reading run identity rather than the
#   defence's footprint. This builds a campaign in which EVERY window comes
#   from its own simulation with its own seed, so no information — neither
#   topology nor protocol state — can cross from one window to the next.
#
# WHY ONLY WINDOW 0 IS KEPT
#   Within one simulation the later windows inherit state from the earlier
#   ones (the scenario sets a flag when the attack first executes and never
#   clears it). Slot 0 is the ONLY window with no predecessor, so it is the
#   only clean one. Recomposing existing runs was considered and REJECTED for
#   exactly this reason: it would remove the shared topology but keep the
#   inherited state, i.e. it would not test the mechanism under dispute.
#
#   Cost of that decision: one simulation yields one usable window, so a
#   TARGET of N costs 4*N simulations per mode.
#
# NO SIMULATOR CHANGE IS NEEDED
#   --scenarioOrder puts any phase in slot 0 and --formationLeadIn gives that
#   slot a proper formation period with its config OFF:
#       0-60 formation | 60 config | 60-120 stabilise | 120-160 measure
#   Both flags already exist and are validated (STATE §25.20). This script
#   therefore touches NO .cc file and NO existing script.
#
# ⛔ WHAT THIS SCRIPT WILL NOT DO
#   * write anywhere except $OUT (a new tree)
#   * write into any existing simulations_* tree, or into the previous paper's
#     tree (~/ns3/Final_Project_NS3-master, ns-3.19 — frozen)
#   * regenerate manifests/*.csv; it READS the population manifest only.
#     build_hopablation*.py is never invoked (STATE §25.23: running it would
#     redefine the seed pool every later experiment is drawn from)
#   * run while a learning job is running
#   All four are enforced as hard guards below, not left to discipline.
#
# MODES
#   TIMING=1 (default)  rehearsal: N_TIMING sims per phase into a scratch dir,
#                       measures real wall-clock, extrapolates the full cost,
#                       verifies slot 0 is written for all four phases, then
#                       stops. Writes NOTHING into $OUT.
#   TIMING=0            the real campaign.
#
# USAGE
#   bash run_perwindowseed.sh                       # timing rehearsal, static
#   MODE=mobile bash run_perwindowseed.sh           # timing rehearsal, mobile
#   TIMING=0 nohup bash run_perwindowseed.sh   > pws_static.log 2>&1 &
#   TIMING=0 MODE=mobile nohup bash run_perwindowseed.sh > pws_mobile.log 2>&1 &
# ============================================================================
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------- guards ----
# G1. Right tree. This experiment belongs to the revision campaign in
#     nv347/ns-3.47. The previous paper lives in Final_Project_NS3-master on
#     ns-3.19/waf and is frozen; fpnt347 is an unrelated protocol tree.
case "$ROOT" in
  */nv347/ns-3.47) ;;
  *) echo "ABORT: wrong tree: $ROOT"
     echo "  expected .../nv347/ns-3.47 (the revision campaign)."
     echo "  Final_Project_NS3-master is the PREVIOUS paper and is frozen."
     exit 1 ;;
esac

BIN="$ROOT/build/scratch/ns3.47-iolsr-tests-corrected-default"
OUT="${OUT:-$ROOT/simulations_v347_perwindowseed}"

MODE="${MODE:-static}"
if [ "$MODE" = mobile ]; then MOB=1; else MOB=0; MODE=static; fi

TARGET="${TARGET:-2000}"
TIMING="${TIMING:-1}"
N_TIMING="${N_TIMING:-5}"          # sims per phase in the rehearsal
NUM_WORKERS="${NUM_WORKERS:-22}"
RUN_TIMEOUT="${RUN_TIMEOUT:-900}"
LEADIN="${LEADIN:-60}"
HOPFILTER="${HOPFILTER:-0}"        # C_all includes cat1/cat2; the filter must be off
POP="${POP:-C_all}"
SAMPLE_SEED="${SAMPLE_SEED:-20260821}"   # fixed and recorded: the draw is reproducible

# G2. Hard cap. 2,000 per mode was the decision (21/8); refuse silently larger.
if [ "$TARGET" -gt 2000 ]; then
  echo "ABORT: TARGET=$TARGET exceeds the agreed cap of 2000 per mode."; exit 1
fi

# ⛔ The _10k staging, not the 2,000 one. The revision's campaign (arms_c10k)
# is C_all at N=10,000, so this experiment must draw from the same population
# to be comparable — and 4*TARGET=8,000 draws do not fit in a 2,000 pool.
# run_windoworder.sh points at the 2,000 staging because it predates the 10k
# population; copying its path here cost a failed launch on 21/8.
if [ "$MODE" = mobile ]; then STAGE_DIR="$ROOT/simulations_v347_hopablation_mobile_10k"
else                          STAGE_DIR="$ROOT/simulations_v347_hopablation_10k"; fi
POP_MANIFEST="$STAGE_DIR/manifests/${POP}.csv"

PHASES="baseline attack_only defense_only defense_vs_attack"
# phase name -> its index in the scenario's own numbering
declare -A PIDX=( [baseline]=0 [attack_only]=1 [defense_only]=2 [defense_vs_attack]=3 )

FEAT="$OUT/features_${MODE}"
TMPROOT="$OUT/.tmp_${MODE}"
MANIFEST="$OUT/manifest_${MODE}.csv"
ATTEMPTS="$OUT/attempts_${MODE}.csv"
# timing rehearsals must leave the campaign tree completely untouched, the
# attempts log included — otherwise a rehearsal leaves a stale attempts file
# at the campaign root that looks like a real run's.
[ "$TIMING" = 1 ] && { FEAT="$OUT/.timing_${MODE}/features"
                       TMPROOT="$OUT/.timing_${MODE}/tmp"
                       ATTEMPTS="$OUT/.timing_${MODE}/attempts_${MODE}.csv"; }

say() { echo; echo "############ $* ############"; echo "  $(date '+%F %T')"; }

# G3. Binary and population must exist; we never build either.
[ -x "$BIN" ] || { echo "ERROR: binary missing: $BIN"
                   echo "  build: cd $ROOT && ./ns3 build iolsr-tests-corrected"; exit 1; }
[ -f "$POP_MANIFEST" ] || { echo "ERROR: population manifest not found: $POP_MANIFEST"
                            echo "  ⛔ do NOT run build_hopablation*.py to create it —"
                            echo "     that rewrites the seed pool every experiment uses."; exit 1; }

# ⛔ Check the pool against the REAL target, even in a rehearsal. A 20-seed
# rehearsal cannot notice that the pool is too small for 8,000 draws — that is
# exactly how the 21/8 launch failed three minutes in. The rehearsal's job is
# to clear the real run, so it validates the real run's requirement.
POOL_N=$(( $(wc -l < "$POP_MANIFEST") - 1 ))
if [ "$POOL_N" -lt "$(( TARGET * 4 ))" ]; then
  echo "ABORT: population pool too small for the campaign."
  echo "  pool: $POOL_N seeds in $POP_MANIFEST"
  echo "  need: $(( TARGET * 4 )) (TARGET=$TARGET pseudo-runs x 4 windows, drawn"
  echo "        WITHOUT replacement so no topology is ever reused)"
  echo "  Either point POP/STAGE_DIR at a larger staging, or lower TARGET."
  exit 1
fi

# G4. $OUT must be a NEW tree, never inside an existing campaign.
for existing in "$ROOT"/simulations_v347 "$ROOT"/simulations_v347_hopablation* \
                "$ROOT"/simulations_v347_propmodel* "$ROOT"/simulations_v347_windoworder* \
                "$ROOT"/simulations_v347_nohopfilter "$ROOT"/simulations_v347_propscan; do
  [ -e "$existing" ] || continue
  case "$OUT" in "$existing"|"$existing"/*)
    echo "ABORT: OUT would land inside the existing tree $existing"; exit 1;; esac
done
case "$OUT" in *Final_Project_NS3-master*|*fpnt347*)
  echo "ABORT: OUT points at another project's tree: $OUT"; exit 1;; esac

# G5. Never contend with a learning job.
if pgrep -f "defense_detection_v2.py" >/dev/null 2>&1; then
  echo "ABORT: a learning job is running; it would contend for the CPU:"
  pgrep -af "defense_detection_v2.py"; exit 1
fi

mkdir -p "$FEAT" "$TMPROOT"
for p in $PHASES; do mkdir -p "$FEAT/$p"; done

N_PER_PHASE=$([ "$TIMING" = 1 ] && echo "$N_TIMING" || echo "$TARGET")
NEED=$(( N_PER_PHASE * 4 ))

echo "=========================================================="
echo " per-window-seed experiment   (R#5.5 / item 30)"
echo "   tree            : $ROOT"
echo "   mode            : $MODE (bMobility=$MOB)"
echo "   population      : $POP  (read-only: $POP_MANIFEST)"
echo "   per phase       : $N_PER_PHASE      total sims: $NEED"
echo "   lead-in         : ${LEADIN}s -> slot 0 measures at 120-160s"
echo "   sampling        : WITHOUT replacement, SAMPLE_SEED=$SAMPLE_SEED"
echo "   output          : $OUT"
echo "   workers         : $NUM_WORKERS   timeout: ${RUN_TIMEOUT}s"
[ "$TIMING" = 1 ] && echo "   *** TIMING REHEARSAL — writes only under .timing_${MODE}/ ***"
echo "=========================================================="

# ------------------------------------------------- seed draw (reproducible) --
# Every window in the campaign gets a DIFFERENT accepted seed: 4*N distinct
# seeds drawn without replacement, so no topology is reused anywhere. The draw
# is a deterministic shuffle of the population manifest under SAMPLE_SEED.
mapfile -t DRAW < <(python3 - "$POP_MANIFEST" "$SAMPLE_SEED" "$NEED" <<'PY'
import csv, random, sys
manifest, sample_seed, need = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
seeds = []
with open(manifest) as fh:
    for r in csv.DictReader(fh):
        s = (r.get("seed") or "").strip()
        if s:
            seeds.append(s)
seeds = sorted(set(seeds), key=int)          # order-independent of file order
if len(seeds) < need:
    sys.exit(f"POOL_TOO_SMALL {len(seeds)} < {need}")
random.Random(sample_seed).shuffle(seeds)
print("\n".join(seeds[:need]))
PY
)
if [ "${#DRAW[@]}" -ne "$NEED" ]; then
  echo "ERROR: seed draw returned ${#DRAW[@]} of $NEED — pool too small?"; exit 1
fi
echo "  drew ${#DRAW[@]} distinct seeds (no seed is used twice, in any phase)"

# ------------------------------------------------------------- one run ------
# Runs ONE simulation with $phase in slot 0 and keeps ONLY that slot's files.
run_one() {
  local phase="$1" seed="$2" id="$3"
  local pi="${PIDX[$phase]}"
  # slot 0 = the wanted phase; the rest keep their natural relative order.
  local rest="" i
  for i in 0 1 2 3; do [ "$i" -ne "$pi" ] && rest="${rest},$i"; done
  local order="${pi}${rest}"

  local d="$TMPROOT/${phase}_${seed}"
  rm -rf "$d"; mkdir -p "$d/feat"
  ( cd "$d" && timeout "$RUN_TIMEOUT" "$BIN" \
      --run=1 --RngRun="$seed" --bMobility="$MOB" \
      --outputDir="$d/feat/" \
      --formationLeadIn="$LEADIN" --scenarioOrder="$order" \
      --enforceHopFilter="$HOPFILTER" > run.log 2>&1 )
  local rc=$?

  # Keep slot 0 only. Its files live under the PHASE's own directory, because
  # phases keep their meaning regardless of the slot they occupy.
  local ok=1 f
  for f in metrics_output observer_metrics observer_detail; do
    if [ -f "$d/feat/$phase/${f}-1.csv" ]; then
      cp "$d/feat/$phase/${f}-1.csv" "$FEAT/$phase/${f}-${id}.csv"
    else
      ok=0
    fi
  done
  # ⛔ order contains commas ("1,0,2,3"). Written raw it splits the CSV into
  # nine fields and every positional read downstream — the kept count, the
  # manifest builder — silently reads the wrong column. Recorded pipe-separated.
  echo "$phase,$seed,$id,${order//,/|},$rc,$ok" >> "$ATTEMPTS"
  [ "$ok" = 1 ] && rm -rf "$d"
  return $((1 - ok))
}
export -f run_one

# ------------------------------------------------------------- dispatch -----
say "phase A — $NEED simulations, $NUM_WORKERS workers"
echo "phase,seed,id,order,rc,kept" > "$ATTEMPTS"
t0=$(date +%s)

k=0
for p in $PHASES; do
  for ((i = 1; i <= N_PER_PHASE; i++)); do
    seed="${DRAW[$k]}"; k=$((k + 1))
    while [ "$(jobs -rp | wc -l)" -ge "$NUM_WORKERS" ]; do wait -n; done
    run_one "$p" "$seed" "$i" &
  done
done
wait
t1=$(date +%s); ELAPSED=$((t1 - t0))

# -------------------------------------------------------------- report ------
KEPT=$(awk -F, 'NR>1 && $6==1' "$ATTEMPTS" | wc -l)
say "phase B — result"
echo "  simulations run : $NEED"
echo "  slot-0 kept     : $KEPT"
echo "  wall clock      : $((ELAPSED / 60)) min $((ELAPSED % 60)) s"
for p in $PHASES; do
  echo "    $p: $(ls "$FEAT/$p" 2>/dev/null | grep -c '^metrics_output-') slot-0 windows"
done

if [ "$KEPT" -ne "$NEED" ]; then
  echo
  echo "  ⚠️  $((NEED - KEPT)) simulations did not yield slot 0. Inspect:"
  awk -F, 'NR>1 && $6!=1 {print "    " $0}' "$ATTEMPTS" | head
fi

if [ "$TIMING" = 1 ]; then
  PER=$(python3 -c "print(f'{$ELAPSED/$NEED:.2f}')")
  FULL=$(python3 -c "print(f'{$ELAPSED/$NEED*$TARGET*4/3600:.1f}')")
  say "TIMING REHEARSAL — extrapolation"
  echo "  seconds per simulation : $PER  (at $NUM_WORKERS workers)"
  echo "  projected for TARGET=$TARGET ($((TARGET * 4)) sims, $MODE): $FULL h"
  echo
  echo "  Nothing was written into the campaign tree. Scratch: $OUT/.timing_${MODE}/"
  echo "  If the projection is acceptable and all four phases yielded slot 0:"
  echo "    TIMING=0 MODE=$MODE nohup bash $0 > pws_${MODE}.log 2>&1 &"
  exit 0
fi

# manifest: one row per pseudo-run, naming the four seeds it is composed of
say "phase C — manifest"
python3 - "$ATTEMPTS" "$MANIFEST" "$N_PER_PHASE" <<'PY'
import csv, sys
attempts, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
by = {}
with open(attempts) as fh:
    for r in csv.DictReader(fh):
        if r["kept"] == "1":
            by.setdefault(int(r["id"]), {})[r["phase"]] = r["seed"]
phases = ["baseline", "attack_only", "defense_only", "defense_vs_attack"]
with open(out, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["id"] + [f"seed_{p}" for p in phases])
    complete = 0
    for i in range(1, n + 1):
        row = by.get(i, {})
        if all(p in row for p in phases):
            w.writerow([i] + [row[p] for p in phases]); complete += 1
print(f"  complete pseudo-runs (all four windows present): {complete} / {n}")
PY
echo "  wrote $MANIFEST"
echo
echo "  NEXT: bundle this tree into its own arm, then learn. Use a NEW arm name"
echo "  (e.g. arms_perwindowseed) so no existing arm's results are overwritten."
