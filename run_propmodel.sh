#!/usr/bin/env bash
# ============================================================================
# run_propmodel.sh — the realistic-propagation sensitivity experiment
# (STATE §23.11, §21.4, §25.26; answers R#2 and R#5 on channel realism).
#
# Re-runs the SAME C_all seeds the paper's listener result uses, changing ONLY
# the channel: range guard + LogDistance + Nakagami instead of the 190 m disk.
# Everything else is the paper's configuration — 400 s timeline, fixed window
# order, no formation lead-in. The comparison is therefore directly against the
# EXISTING C_all/listener numbers (static 0.8944, mobile 0.9081): same seeds,
# same timeline, same pipeline code, one physical difference.
#
# ⚠️ CALIBRATION IS MANDATORY. --propRefLoss must be the value run_propcal.sh
# calibrated so the mean OLSR neighbour degree matches the 190 m disk (6.014).
# Without it realism and density change together and nothing is attributable
# (§23.11). This script REFUSES to run without an explicit REFLOSS.
#
# ADDITIVE ONLY: writes simulations_v347_propmodel/ and
# analysis/arms_propmodel/. The canonical campaign is read-only; no --force.
#
#   REFLOSS=45 nohup bash run_propmodel.sh >> propmodel_static.log 2>&1 &
#   REFLOSS=45 MODE=mobile nohup bash run_propmodel.sh >> propmodel_mobile.log 2>&1 &
#
#   REFLOSS=45 STAGES=sim bash run_propmodel.sh     # simulations only
#   REFLOSS=45 TARGET=20 NUM_WORKERS=8 bash run_propmodel.sh   # rehearsal
# ============================================================================
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${DCFM_PYTHON:-$HOME/miniconda3/envs/manet/bin/python}"
BIN="$ROOT/build/scratch/ns3.47-iolsr-tests-corrected-default"
SRC="$ROOT/simulations_v347"
OUT="${OUT:-$ROOT/simulations_v347_propmodel}"
ARMS="${ARMS_OUT:-$ROOT/analysis/arms_propmodel}"

MODE="${MODE:-static}"
if [ "$MODE" = mobile ]; then MOB=1; else MOB=0; MODE=static; fi
TARGET="${TARGET:-2000}"
NUM_WORKERS="${NUM_WORKERS:-22}"
MAX_JOBS="${MAX_JOBS:-22}"
RUN_TIMEOUT="${RUN_TIMEOUT:-900}"
STAGES="${STAGES:-sim,bundle,learn}"

# ---- population: the SAME rule as every other derived experiment (§25.23) ---
# The pool is the N=10,000 C_all manifest (item 7's staging), not the N=2,000
# one: under the faded channel only ~22% of seeds keep a victim->source route
# at t=60 (measured 18/8, 9/40), so reaching TARGET=2000 admitted runs needs
# a ~9,000-seed pool. Same population definition, deeper draw. The admission
# rate itself is a headline finding and is reported by this script.
POP="${POP:-C_all}"
if [ "$MODE" = mobile ]; then STAGE_DIR="${STAGE_DIR:-$ROOT/simulations_v347_hopablation_mobile_10k}"
else                          STAGE_DIR="${STAGE_DIR:-$ROOT/simulations_v347_hopablation_10k}"; fi
POP_MANIFEST="${POP_MANIFEST:-$STAGE_DIR/manifests/${POP}.csv}"
export POP_MANIFEST          # the baseline-staging heredoc reads it
# cat1/cat2 seeds ARE the C_all tail; the binary's filter must be off or the
# population silently collapses back to B_ge2.
HOPFILTER="${HOPFILTER:-0}"

# ---- the channel ----
PROPMODEL="${PROPMODEL:-realistic}"
REFLOSS="${REFLOSS:-}"
EXPONENT="${EXPONENT:-3.0}"
GUARD="${GUARD:-250}"
# scenario-viability admission (STATE §25.36); 'full' would reject ~all runs
# under the faded channel. Overridable but scenario is the intended default here.
ADMISSION="${ADMISSION:-scenario}"

FEAT="$OUT/features_${MODE}"
TMPROOT="$OUT/.tmp_${MODE}"
MANIFEST="$OUT/manifest_${MODE}.csv"
ATTEMPTS="$OUT/attempts_${MODE}.csv"
SCENARIOS="baseline attack_only defense_only defense_vs_attack"
want() { case ",$STAGES," in *",$1,"*) return 0;; *) return 1;; esac; }
say() { echo; echo "############ $* ############"; echo "  $(date '+%F %T')"; }

# ---------------------------------------------------------------- guards ----
[ -x "$BIN" ] || { echo "ERROR: binary missing: $BIN"; exit 1; }
[ -f "$POP_MANIFEST" ] || { echo "ERROR: population manifest missing: $POP_MANIFEST"; exit 1; }
if [ -z "$REFLOSS" ]; then
  echo "ABORT: REFLOSS is not set."
  echo "  The calibrated ReferenceLoss is mandatory (STATE §23.11): without it the"
  echo "  experiment changes realism AND density together and is uninterpretable."
  echo "  Run:  bash run_propcal.sh     then re-launch with REFLOSS=<value>"
  exit 1
fi
if pgrep -f "defense_detection_v2.py" >/dev/null 2>&1; then
  echo "ABORT: a learning job is running; it would contend for the CPU:"
  pgrep -af "defense_detection_v2.py"; exit 1
fi
case "$OUT" in "$SRC"|"$SRC"/*) echo "ABORT: OUT is inside the canonical tree"; exit 1;; esac

echo "=========================================================="
echo " realistic-propagation sensitivity experiment"
echo "   mode        : $MODE (bMobility=$MOB)"
echo "   population  : $POP   (--enforceHopFilter=$HOPFILTER)"
echo "   seeds from  : $POP_MANIFEST"
echo "   channel     : $PROPMODEL  RefLoss=$REFLOSS  exponent=$EXPONENT  guard=${GUARD}m"
echo "   timeline    : the paper's 400 s, fixed window order, NO lead-in"
echo "   target      : $TARGET      workers: $NUM_WORKERS"
echo "   output      : $OUT"
echo "   stages      : $STAGES"
echo "=========================================================="

# ---------------------------------------------------------------- sim -------
if want sim; then
  say "1/3  simulations — $MODE, realistic channel"
  mkdir -p "$FEAT" "$TMPROOT"
  for s in $SCENARIOS; do mkdir -p "$FEAT/$s"; done

  mapfile -t CAND < <(awk -F, 'NR>1 && $1 != "" {print $1}' "$POP_MANIFEST")
  declare -A HOPCAT
  while IFS=, read -r hs _hr _hsrc hc; do
    [ "$hs" = seed ] && continue
    HOPCAT["$hs"]="$hc"
  done < "$POP_MANIFEST"
  echo "  candidates: ${#CAND[@]}   hop mix: $(awk -F, 'NR>1{c[$4]++} END{for(k in c) printf "cat%s=%d ", k, c[k]}' "$POP_MANIFEST")"

  run_one() {
    local seed="$1" d="$TMPROOT/$1"
    rm -rf "$d"; mkdir -p "$d"
    # --admission=scenario: the paper's all-nodes-converged check is unmeetable
    # under Nakagami fading (STATE §25.36); admit on scenario viability instead
    # (victim has a route to the source AND >=1 neighbour).
    ( cd "$d" && timeout "$RUN_TIMEOUT" "$BIN" \
        --run=1 --RngRun="$seed" --bMobility="$MOB" \
        --outputDir="$d/feat/" \
        --enforceHopFilter="$HOPFILTER" \
        --admission="$ADMISSION" \
        --propagationModel="$PROPMODEL" --propRefLoss="$REFLOSS" \
        --propExponent="$EXPONENT" --propMaxRange="$GUARD" \
        > run.log 2>&1 )
    local rc=$? n=0 s
    local hop; hop=$(grep '^HOP_CATEGORY=' "$d/run.log" 2>/dev/null | tail -1 | cut -d= -f2)
    [ -z "$hop" ] && hop="${HOPCAT[$seed]:--}"
    for s in $SCENARIOS; do [ -f "$d/feat/$s/metrics_output-1.csv" ] && n=$((n+1)); done
    if [ "$rc" -eq 0 ] && [ "$n" -eq 4 ]; then
      echo "$seed,OK,-,$hop" > "$d/.status"
    else
      local reason
      if   [ "$rc" -eq 124 ]; then reason=timeout
      elif grep -q "within two hops of victim" "$d/run.log" 2>/dev/null; then reason=hopfilter_REJECT_BUG
      elif grep -q "Attacker not available"    "$d/run.log" 2>/dev/null; then reason=attacker_unavailable
      elif grep -q "Terminated"                "$d/run.log" 2>/dev/null; then reason=terminated_admission
      else reason="incomplete_rc${rc}_files${n}"
      fi
      echo "$seed,REJECT,$reason,$hop" > "$d/.status"
      rm -rf "$d/feat"
    fi
  }
  count_ok() { grep -l ',OK,' "$TMPROOT"/*/.status 2>/dev/null | wc -l; }

  running=0; launched=0
  STOP_AT=$(( TARGET + TARGET / 20 + 3 ))
  for seed in "${CAND[@]}"; do
    run_one "$seed" &
    running=$((running+1)); launched=$((launched+1))
    if [ "$running" -ge "$NUM_WORKERS" ]; then wait -n; running=$((running-1)); fi
    if [ $((launched % 100)) -eq 0 ]; then
      echo "  progress: launched=$launched accepted=$(count_ok)  $(date '+%H:%M:%S')"
      if [ "$(count_ok)" -ge "$STOP_AT" ]; then
        echo "  early stop: $(count_ok) >= $STOP_AT"; break
      fi
    fi
  done
  wait
  echo "  simulations done: launched=$launched accepted=$(count_ok)"

  say "2/3  committing ids in population order"
  echo "seed,result,reason,hop_category" > "$ATTEMPTS"
  echo "id,seed,hop_category" > "$MANIFEST"
  id=0; nrej=0; declare -A REJ
  for seed in "${CAND[@]}"; do
    st="$TMPROOT/$seed/.status"
    [ -f "$st" ] || continue
    cat "$st" >> "$ATTEMPTS"
    IFS=, read -r _s res reason hop < "$st"
    if [ "$res" != OK ]; then
      nrej=$((nrej+1)); REJ[$reason]=$(( ${REJ[$reason]:-0} + 1 )); continue
    fi
    [ "$id" -ge "$TARGET" ] && continue
    id=$((id+1))
    for s in $SCENARIOS; do
      for kind in metrics_output observer_metrics observer_detail; do
        src="$TMPROOT/$seed/feat/$s/${kind}-1.csv"
        [ -f "$src" ] && mv "$src" "$FEAT/$s/${kind}-${id}.csv"
      done
    done
    echo "$id,$seed,$hop" >> "$MANIFEST"
  done
  echo "  accepted : $id / $TARGET"
  echo "  rejected : $nrej"
  for k in "${!REJ[@]}"; do printf "    %-32s %s\n" "$k" "${REJ[$k]}"; done
  tried=$(( id + nrej ))
  if [ "$tried" -gt 0 ]; then
    echo "  ADMISSION RATE under the realistic channel: $id/$tried = $(( id * 100 / tried ))%"
    echo "  (a headline finding: the share of disk-viable deployments that keep a"
    echo "   victim->source route under fading — report it, STATE §25.36)"
  fi
  if [ -n "${REJ[hopfilter_REJECT_BUG]:-}" ]; then
    echo "  ⚠️  the hop filter fired — the C_all tail was dropped. DO NOT USE."
  fi
fi

# ---------------------------------------------------------------- bundle ----
N=$(( $(wc -l < "$MANIFEST") - 1 ))
if want bundle; then
  say "bundle — listener arm, N=$N"
  b="$ARMS/listener/colab_data/wide_${MODE}.csv.gz"
  want_rows=$(( N * 4 + 1 ))
  if [ -f "$b" ] && [ "$(zcat "$b" 2>/dev/null | wc -l)" -eq "$want_rows" ]; then
    echo "  [skip] bundle exists with $N runs"
  else
    DCFM_SIM_ROOT="$OUT" DCFM_ARMS_ROOT="$ARMS" \
      "$PY" "$ROOT/analysis/make_arm_bundles.py" --arm listener --mode "$MODE" --limit "$N" || exit 1
  fi
fi

# ------------------------------------------------- paired baseline staging --
# The admitted seeds are known only AFTER the realistic run. Stage the SAME
# seeds' range-channel (paper) data — which already exists for every C_all seed
# in the main + rerun trees — into a paired baseline arm, so the comparison is
# seed-for-seed, channel the only difference (user's correction, STATE §25.36).
# No new simulation: the range-channel windows are reused via symlinks.
BASE_ARMS="${BASE_ARMS:-$ROOT/analysis/arms_propmodel_baseline}"
BASE_SIM="${BASE_SIM:-$ROOT/simulations_v347_propmodel_baseline}"
if want learn; then
  say "3a  staging the paired range-channel baseline for the admitted seeds"
  bfeat="$BASE_SIM/features_${MODE}"
  for s in $SCENARIOS; do mkdir -p "$bfeat/$s"; done
  "$PY" - "$MANIFEST" "$SRC" "$ROOT/simulations_v347_nohopfilter" "$bfeat" "$MODE" <<'PYEOF'
import csv, os, sys
manifest, main_root, rerun_root, dst, mode = sys.argv[1:6]
SCEN = ["baseline","attack_only","defense_only","defense_vs_attack"]
KINDS = ["metrics_output","observer_metrics","observer_detail"]
# manifest of the realistic run: id,seed,hop_category. Recover source+run_id
# from the population manifest by seed.
popm = os.environ["POP_MANIFEST"]
seed2 = {}
for r in csv.DictReader(open(popm)):
    seed2[r["seed"]] = (r["run_id"], r["source"])
made, miss = 0, []
for r in csv.DictReader(open(manifest)):
    newid, seed = r["id"], r["seed"]
    if seed not in seed2:
        miss.append((newid, seed, "seed_not_in_pop")); continue
    rid, source = seed2[seed]
    root = main_root if source == "main" else rerun_root
    for scen in SCEN:
        for k in KINDS:
            src = os.path.join(root, f"features_{mode}", scen, f"{k}-{rid}.csv")
            d = os.path.join(dst, scen, f"{k}-{newid}.csv")
            if not os.path.isfile(src):
                miss.append((newid, seed, f"{scen}/{k}-{rid}")); continue
            if os.path.islink(d) or os.path.exists(d): os.remove(d)
            os.symlink(src, d); made += 1
print(f"  staged {made} range-channel links for the admitted seeds")
if miss:
    print(f"  MISSING {len(miss)}, first 5: {miss[:5]}"); sys.exit(1)
PYEOF
  rc=$?; [ $rc -ne 0 ] && { echo "  ABORT: baseline staging incomplete"; exit $rc; }

  bN=$(( $(wc -l < "$MANIFEST") - 1 ))
  say "3b  bundle — paired baseline, N=$bN"
  DCFM_SIM_ROOT="$BASE_SIM" DCFM_ARMS_ROOT="$BASE_ARMS" \
    "$PY" "$ROOT/analysis/make_arm_bundles.py" --arm listener --mode "$MODE" --limit "$bN" || exit 1

  say "3c  learning — REALISTIC channel, $MODE"
  DCFM_ARMS_ROOT="$ARMS" MAX_JOBS="$MAX_JOBS" \
    "$PY" -u "$ROOT/analysis/run_arm.py" \
    --arm listener --script defense_detection_v2.py --mode "$MODE" \
    -- --no-augmentation || echo "  [warn] realistic learning exited $?"

  say "3d  learning — paired BASELINE (range channel, same seeds), $MODE"
  DCFM_ARMS_ROOT="$BASE_ARMS" MAX_JOBS="$MAX_JOBS" \
    "$PY" -u "$ROOT/analysis/run_arm.py" \
    --arm listener --script defense_detection_v2.py --mode "$MODE" \
    -- --no-augmentation || echo "  [warn] baseline learning exited $?"
fi

# ---------------------------------------------------------------- summary ---
# PAIRED: both arms are the same admitted seeds, the range channel vs the faded
# channel the only difference. The baseline reuses existing range-channel data
# (no new simulation). The ~22% admission rate is itself a reported result.
say "SUMMARY — realistic vs range channel, PAIRED on the admitted seeds ($MODE)"
BASE="$BASE_ARMS/listener/results/$MODE/results.csv"
NEW="$ARMS/listener/results/$MODE/results.csv"
a_new=$(awk -F, '$1=="Stacking_Ensemble"{print $4}' "$NEW"  2>/dev/null)
a_base=$(awk -F, '$1=="Stacking_Ensemble"{print $4}' "$BASE" 2>/dev/null)
bN=$(( $(wc -l < "$MANIFEST") - 1 ))
printf "  %-30s %10s   (N=%s, same seeds)\n" "range channel (paper, RL n/a)" "${a_base:--}" "$bN"
printf "  %-30s %10s\n" "realistic channel (RL=$REFLOSS)" "${a_new:--}"
if [ -n "$a_new" ] && [ -n "$a_base" ]; then
  d=$(awk -v n="$a_new" -v b="$a_base" 'BEGIN{printf "%+.4f", n-b}')
  echo "  paired delta (realistic - range): $d"
  echo "  measured noise floor for a comparison of this kind is ~±0.014 (STATE §25.33);"
  echo "  interpret the delta against the paired CI, not a fixed cut-off."
fi
echo "finished $(date '+%F %T')"
