#!/usr/bin/env bash
# ============================================================================
# run_windoworder.sh — the window-order experiment (STATE §23.10, §25.20, R#4 #1).
#
# Runs seeds from the accepted-seed list under a chosen window order, into a
# separate tree. Run it TWICE per mode — once per arm:
#
#   ARM=shuffled   --scenarioOrder=random    (permuted order)
#   ARM=canonical  --scenarioOrder=0,1,2,3   (fixed order, same timeline)
#
# BOTH arms use --formationLeadIn=60, so the timeline is
#
#   0-60 formation | 60 config | 60-120 stabilise | 120-160 window | ...
#
# i.e. every phase is a self-contained 100 s block (60 s settled with its
# config + 40 s measurement) and every slot is structurally identical. The
# canonical arm is REGENERATED rather than symlinked from the main campaign
# because that campaign has no lead-in (400 s vs 460 s); pairing across
# different timelines would let mobility drift differ between the arms.
# Verified 17/8: with the lead-in, 12/12 random-order seeds complete and the
# "Attacker not available" abort — which used to kill ~50% of permutations —
# no longer occurs, so all 24 orderings are now reachable.
#
#   simulations_v347_windoworder/<arm>/
#     features_<mode>/{baseline,attack_only,defense_only,defense_vs_attack}/
#         metrics_output-<id>.csv  observer_metrics-<id>.csv  observer_detail-<id>.csv
#     manifest_<mode>.csv  id,seed,hop_category,slot0..slot3  (accepted only)
#     attempts_<mode>.csv  seed,result,reason,slots,hop_category  (every attempt)
#
# The seed pool is the POPULATION manifest (default C_all), NOT accepted_seeds
# — see the POP block below; hop_category is carried through so B_ge2/A_ge3 can
# be re-derived offline without re-simulating.
#
# ids are contiguous 1..N in ACCEPTED-SEED ORDER, so the same seed gets the
# same id in both arms and the pairing is exact.
#
# ⚠️ Seeds can still be rejected for the ordinary reasons (connectivity, the
# hop criterion). Run the canonical arm FIRST, then pass its manifest as
# SEEDS_FROM= to the shuffled arm so both arms end on the same seed set.
#
# Usage (launch detached; each arm is ~1 h):
#   ARM=canonical nohup bash run_windoworder.sh >> wo_canon_static.log 2>&1 &
#   ARM=shuffled SEEDS_FROM=.../canonical/manifest_static.csv \
#       nohup bash run_windoworder.sh >> wo_shuf_static.log 2>&1 &
#   ARM=canonical MODE=mobile nohup bash run_windoworder.sh >> wo_canon_mobile.log 2>&1 &
#
#   DRY=1 TARGET=5 bash run_windoworder.sh     # tiny rehearsal, prints commands
# ============================================================================
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$ROOT/build/scratch/ns3.47-iolsr-tests-corrected-default"
SRC="$ROOT/simulations_v347"
OUT="${OUT:-$ROOT/simulations_v347_windoworder}"

MODE="${MODE:-static}"
if [ "$MODE" = mobile ]; then MOB=1; else MOB=0; MODE=static; fi
TARGET="${TARGET:-2000}"
NUM_WORKERS="${NUM_WORKERS:-22}"
RUN_TIMEOUT="${RUN_TIMEOUT:-900}"
DRY="${DRY:-0}"

# Which arm, and therefore which window order. Both arms share the lead-in.
ARM="${ARM:-canonical}"
LEADIN="${LEADIN:-60}"
case "$ARM" in
  canonical) ORDER="0,1,2,3" ;;
  shuffled)  ORDER="random"  ;;
  *) echo "ERROR: ARM must be 'canonical' or 'shuffled', got: $ARM"; exit 1 ;;
esac
# Restrict to a seed set produced by a previous arm, so both arms end up on
# exactly the same seeds (accepts that arm's manifest or any csv with a seed column).
SEEDS_FROM="${SEEDS_FROM:-}"

# ---------------------------------------------------------------------------
# POPULATION (STATE §23.1 decision 2, §25.23).
#
# The seed pool MUST come from the hop-ablation manifests, not from
# accepted_seeds_<mode>.csv. accepted_seeds holds only runs that already passed
# the >=3-hop distance criterion, so drawing from it silently reproduces the
# OLD B_ge2-style population no matter what --enforceHopFilter says — the
# filter never fires because the input is pre-filtered. That defect cost the
# 17/8 20:36 run (2,000 canonical runs on the wrong population, discarded).
#
# manifests/C_all.csv is exactly what build_hopablation.py assembled for the
# paper: main-campaign runs plus the distance-rejected reruns, with each seed's
# hop_category recorded. Static C_all = 247 cat1 + 306 cat2 + 1447 cat3.
POP="${POP:-C_all}"
if [ "$MODE" = mobile ]; then STAGE_DIR="$ROOT/simulations_v347_hopablation_mobile"
else                          STAGE_DIR="$ROOT/simulations_v347_hopablation"; fi
POP_MANIFEST="$STAGE_DIR/manifests/${POP}.csv"

# Category 1/2 seeds are exactly the ones the binary's built-in filter rejects,
# so it MUST be off or the C_all tail is silently dropped.
HOPFILTER="${HOPFILTER:-0}"

ARMDIR="$OUT/$ARM"
SEEDS_CSV="$POP_MANIFEST"
FEAT="$ARMDIR/features_${MODE}"
TMPROOT="$ARMDIR/.tmp_${MODE}"
MANIFEST="$ARMDIR/manifest_${MODE}.csv"
ATTEMPTS="$ARMDIR/attempts_${MODE}.csv"
SCENARIOS="baseline attack_only defense_only defense_vs_attack"

say() { echo; echo "############ $* ############"; echo "  $(date '+%F %T')"; }

# ---------------------------------------------------------------- guards ----
[ -x "$BIN" ] || { echo "ERROR: binary missing: $BIN"; echo "  build: cd $ROOT && ./ns3 build iolsr-tests-corrected"; exit 1; }
[ -f "$SEEDS_CSV" ] || { echo "ERROR: population manifest not found: $SEEDS_CSV"; echo "  (POP=$POP; build it with analysis/build_hopablation*.py)"; exit 1; }
if pgrep -f "defense_detection_v2.py" >/dev/null 2>&1; then
  echo "ABORT: a learning job is running; it would contend for the CPU:"
  pgrep -af "defense_detection_v2.py"; exit 1
fi
# The canonical tree is READ-ONLY here. Refuse if OUT would land inside it.
case "$OUT" in "$SRC"|"$SRC"/*) echo "ABORT: OUT is inside the canonical tree"; exit 1;; esac

mkdir -p "$FEAT" "$TMPROOT"
for s in $SCENARIOS; do mkdir -p "$FEAT/$s"; done

echo "=========================================================="
echo " window-order experiment"
echo "   arm             : $ARM   (--scenarioOrder=$ORDER)"
echo "   formation lead-in: ${LEADIN}s   -> windows at 120/220/320/420"
echo "   mode            : $MODE (bMobility=$MOB)"
echo "   population      : $POP   (--enforceHopFilter=$HOPFILTER)"
echo "   target accepted : $TARGET"
echo "   candidate seeds : ${SEEDS_FROM:-$SEEDS_CSV}"
echo "   output          : $ARMDIR"
echo "   workers         : $NUM_WORKERS   timeout: ${RUN_TIMEOUT}s"
[ "$DRY" = 1 ] && echo "   *** DRY RUN ***"
echo "=========================================================="

# Candidate seeds. Either the full accepted list, or the seed set an earlier
# arm settled on (SEEDS_FROM), which makes the two arms share seeds exactly.
if [ -n "$SEEDS_FROM" ]; then
  [ -f "$SEEDS_FROM" ] || { echo "ERROR: SEEDS_FROM not found: $SEEDS_FROM"; exit 1; }
  # column named 'seed' in that file's header
  mapfile -t CAND < <("${PY:-python3}" - "$SEEDS_FROM" <<'PY'
import csv,sys
with open(sys.argv[1]) as fh:
    for r in csv.DictReader(fh):
        if r.get("seed"): print(r["seed"])
PY
)
else
  # population manifest: seed,run_id,source,hop_category  -> seed is column 1
  mapfile -t CAND < <(awk -F, 'NR>1 && $1 != "" {print $1}' "$SEEDS_CSV")
fi

# seed -> hop_category, so the manifest can carry it and B_ge2/A_ge3 can be
# re-derived offline from a C_all run without simulating again. Hop category
# never enters metrics_output (it correlates with difficulty and would leak).
declare -A HOPCAT
while IFS=, read -r hs _hr _hsrc hc; do
  [ "$hs" = seed ] && continue
  HOPCAT["$hs"]="$hc"
done < "$POP_MANIFEST"
echo "  population $POP: ${#HOPCAT[@]} seeds, hop mix: $(awk -F, 'NR>1{c[$4]++} END{for(k in c) printf "cat%s=%d ", k, c[k]}' "$POP_MANIFEST")"
NCAND=${#CAND[@]}
# With the lead-in every permutation is runnable (verified 17/8), so the only
# losses are the ordinary connectivity/hop rejections. 1.4x plus slack is ample;
# the early stop trims the surplus.
WANT_CAND=$(( TARGET * 14 / 10 + 200 ))
[ "$WANT_CAND" -gt "$NCAND" ] && WANT_CAND=$NCAND
echo "  candidates available: $NCAND   will attempt up to: $WANT_CAND"
[ "$WANT_CAND" -lt "$TARGET" ] && echo "  ⚠️  fewer candidates than target; yield will be short"

# ---------------------------------------------------- phase A: run them -----
# Each seed runs isolated in its own temp dir with --run=1, so no id arbitration
# is needed while parallel. Status goes to one line per seed; phase B assigns
# the contiguous ids deterministically.
say "PHASE A/2  running $WANT_CAND candidates ($POP) with --scenarioOrder=$ORDER"

run_one() {
  local seed="$1"
  local d="$TMPROOT/$seed"
  rm -rf "$d"; mkdir -p "$d"
  if [ "$DRY" = 1 ]; then
    echo "  DRY: $BIN --run=1 --RngRun=$seed --bMobility=$MOB --outputDir=$d/feat/ --formationLeadIn=$LEADIN --scenarioOrder=$ORDER --enforceHopFilter=$HOPFILTER"
    echo "$seed,DRY,dry_run,-,-" >> "$d/.status"; return
  fi
  ( cd "$d" && timeout "$RUN_TIMEOUT" "$BIN" \
      --run=1 --RngRun="$seed" --bMobility="$MOB" \
      --outputDir="$d/feat/" \
      --formationLeadIn="$LEADIN" --scenarioOrder="$ORDER" \
      --enforceHopFilter="$HOPFILTER" > run.log 2>&1 )
  local rc=$?
  # HOP_CATEGORY is stdout-only by design; the last line is the run's final
  # (minimum-over-windows) category. Fall back to the population manifest.
  local hop; hop=$(grep '^HOP_CATEGORY=' "$d/run.log" 2>/dev/null | tail -1 | cut -d= -f2)
  [ -z "$hop" ] && hop="${HOPCAT[$seed]:--}"
  local slots; slots=$(sed -n 's/.*slots=\[\(.*\)\].*/\1/p' "$d/run.log" | head -1)
  [ -z "$slots" ] && slots="-"

  # accepted only if the run exited clean AND all four windows were written
  local n=0 s
  for s in $SCENARIOS; do [ -f "$d/feat/$s/metrics_output-1.csv" ] && n=$((n+1)); done

  # FIELD ORDER MATTERS: $slots is itself comma-separated, so it must be the
  # LAST field — anything after it would be swallowed by `IFS=, read` in phase B.
  if [ "$rc" -eq 0 ] && [ "$n" -eq 4 ]; then
    echo "$seed,OK,-,$hop,$slots" > "$d/.status"
  else
    local reason
    if   [ "$rc" -eq 124 ];                                     then reason=timeout
    # With LEADIN>0 this should never fire; if it does, the lead-in is too
    # short for the topology to form and the run must not be silently dropped.
    elif grep -q "Attacker not available" "$d/run.log" 2>/dev/null; then reason=attacker_unavailable
    # With HOPFILTER=0 this must never fire either: the C_all tail is cat 1/2
    # by definition, and dropping it would rebuild the old B_ge2 population.
    elif grep -q "within two hops of victim" "$d/run.log" 2>/dev/null; then reason=hopfilter_REJECT_BUG
    elif grep -q "Terminated"            "$d/run.log" 2>/dev/null; then reason=terminated_admission
    else reason="incomplete_rc${rc}_files${n}"
    fi
    echo "$seed,REJECT,$reason,$hop,$slots" > "$d/.status"
    rm -rf "$d/feat"          # keep run.log for forensics, drop partial output
  fi
}

count_ok() { grep -l ',OK,' "$TMPROOT"/*/.status 2>/dev/null | wc -l; }

running=0; launched=0
STOP_AT=$(( TARGET + TARGET / 20 + 3 ))    # small surplus over the target
for i in $(seq 0 $((WANT_CAND - 1))); do
  run_one "${CAND[$i]}" &
  running=$((running + 1)); launched=$((launched + 1))
  if [ "$running" -ge "$NUM_WORKERS" ]; then
    wait -n 2>/dev/null || wait; running=$((running - 1))
    # early stop: enough accepted already — don't burn the whole candidate list
    if [ $((launched % 20)) -eq 0 ] && [ "$(count_ok)" -ge "$STOP_AT" ]; then
      echo "  early stop: $(count_ok) accepted >= $STOP_AT (launched $launched/$WANT_CAND)"
      break
    fi
  fi
done
wait
echo "  phase A done: $(date '+%F %T')   attempted: $launched   accepted so far: $(count_ok)"

# ------------------------------------------- phase B: commit in order -------
# Serial and fast. Walks candidates in accepted-seed order, assigns contiguous
# ids to successes, stops at TARGET. Deterministic: same inputs -> same mapping.
say "PHASE B/2  assigning ids and committing"

echo "seed,result,reason,hop_category,slot0,slot1,slot2,slot3" > "$ATTEMPTS"
echo "id,seed,hop_category,slot0,slot1,slot2,slot3" > "$MANIFEST"

id=0; nrej=0
declare -A REJ
for i in $(seq 0 $((WANT_CAND - 1))); do
  seed="${CAND[$i]}"
  st="$TMPROOT/$seed/.status"
  [ -f "$st" ] || { echo "$seed,MISSING,no_status,-,-" >> "$ATTEMPTS"; continue; }
  cat "$st" >> "$ATTEMPTS"

  # .status is  seed,result,reason,hop,slot0,slot1,slot2,slot3  — slots last,
  # so `read` collects all four of them into $slots.
  IFS=, read -r _s res reason hop slots < "$st"
  if [ "$res" != OK ]; then
    nrej=$((nrej + 1)); REJ[$reason]=$(( ${REJ[$reason]:-0} + 1 ))
    continue
  fi
  [ "$id" -ge "$TARGET" ] && continue          # target met; leave the rest

  id=$((id + 1))
  for s in $SCENARIOS; do
    for kind in metrics_output observer_metrics observer_detail; do
      src="$TMPROOT/$seed/feat/$s/${kind}-1.csv"
      [ -f "$src" ] && mv "$src" "$FEAT/$s/${kind}-${id}.csv"
    done
  done
  echo "$id,$seed,$hop,${slots//;/,}" >> "$MANIFEST"
done

say "SUMMARY"
printf "  accepted : %d / %d target\n" "$id" "$TARGET"
printf "  rejected : %d\n" "$nrej"
for r in "${!REJ[@]}"; do printf "    %-32s %d\n" "$r" "${REJ[$r]}"; done
echo
echo "  manifest : $MANIFEST"
echo "  attempts : $ATTEMPTS"
if [ "$id" -lt "$TARGET" ]; then
  echo
  echo "  ⚠️  SHORT of target. Raise the candidate pool (more accepted seeds) or"
  echo "      lower TARGET. Do NOT pair a short shuffled set against the full"
  echo "      canonical set — the comparison must use the SAME seeds in both."
fi
if [ -n "${REJ[attacker_unavailable]:-}" ]; then
  echo
  echo "  ⚠️  attacker_unavailable fired ${REJ[attacker_unavailable]} times with"
  echo "      LEADIN=$LEADIN. That should not happen — the lead-in exists to give"
  echo "      slot 0 a formed topology. Investigate before using these results."
fi
echo
echo "  temp dirs kept for forensics: $TMPROOT   (run.log per seed)"
if [ "$ARM" = canonical ]; then
  echo "  next: run the SHUFFLED arm restricted to these seeds —"
  echo "        ARM=shuffled MODE=$MODE SEEDS_FROM='$MANIFEST' bash run_windoworder.sh"
else
  echo "  next: bundles + learning for both arms —"
  echo "        MODE=$MODE bash run_windoworder_learn.sh"
fi
echo "finished $(date '+%F %T')"
