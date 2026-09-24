#!/usr/bin/env bash
# ============================================================================
# run_propcal.sh — calibration harness for the realistic propagation model
# (STATE §21.4, §23.11, §25.26; answers R#2/R#5 realism).
#
# GOAL: find the --propRefLoss under which the realistic channel (range guard +
# LogDistance + Nakagami) yields the SAME mean OLSR neighbour degree as the
# paper's 190 m disk. Without this the sensitivity experiment changes realism
# and density together and nothing is attributable (§23.11).
#
# HOW:
#   target  — computed from the EXISTING campaign's topology probes (t=59
#             OLSR neighbour sets, accepted static seeds only). No simulation.
#   sweep   — for each candidate RefLoss: K short runs (probe fires at t=59,
#             sim ends right after), mean degree, delta vs target.
#   output  — a calibration table + the argmin recommendation.
#
# Cheap by design: a 65 s sim is ~6 s wall; the default sweep is
# 7 values x 10 seeds = 70 runs ≈ 2 CPU-minutes at NUM_WORKERS=4, polite
# enough to run alongside a campaign.
#
#   bash run_propcal.sh                        # default sweep
#   SWEEP="40 45 50" K=20 bash run_propcal.sh  # custom
# ============================================================================
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$ROOT/build/scratch/ns3.47-iolsr-tests-corrected-default"
SRC="$ROOT/simulations_v347"
OUT="${OUT:-$ROOT/propcal}"
SWEEP="${SWEEP:-35 40 45 50 55 60 65}"
K="${K:-10}"                       # seeds per sweep point
NUM_WORKERS="${NUM_WORKERS:-4}"    # polite default: leave the campaign its CPU
EXPONENT="${EXPONENT:-3.0}"
GUARD="${GUARD:-250}"

# ⚠️ Calibrate PER MODE. Measured 18/8 from the existing probes: the mean OLSR
# neighbour degree is 6.014 static but 6.639 mobile — a 10% gap, because
# RandomWaypoint's stationary distribution concentrates nodes toward the centre.
# Reusing the static RefLoss for mobile would leave mobile mis-calibrated by
# exactly the amount the calibration exists to remove (STATE §23.11).
MODE="${MODE:-static}"
if [ "$MODE" = mobile ]; then MOB=1; PSUF="_mobile"; else MOB=0; MODE=static; PSUF=""; fi

# The target is computed over the population the EXPERIMENT will use (C_all),
# not over accepted_seeds. The gap is small (static +0.005, mobile +0.021) but
# the data is on hand, so there is no reason to approximate.
POP="${POP:-C_all}"
if [ "$MODE" = mobile ]; then STAGE_DIR="$ROOT/simulations_v347_hopablation_mobile"
else                          STAGE_DIR="$ROOT/simulations_v347_hopablation"; fi
POP_MANIFEST="$STAGE_DIR/manifests/${POP}.csv"

say() { echo; echo "############ $* ############"; echo "  $(date '+%F %T')"; }
[ -x "$BIN" ] || { echo "ERROR: binary missing"; exit 1; }
mkdir -p "$OUT"

# ---------------------------------------------------------------- target ----
say "TARGET  mean neighbour degree of the 190 m disk — $MODE / $POP (existing probes, no sims)"
[ -f "$POP_MANIFEST" ] || { echo "ERROR: population manifest missing: $POP_MANIFEST"; exit 1; }

# Probes live in two trees: the main campaign and the distance-rejected rerun.
# C_all draws from both, so both must be read or the cat1/cat2 tail is dropped.
read -r TARGET NTARGET < <(awk -F, -v want="$MOB" '
  FILENAME==ARGV[1] { if (FNR>1) pop[$1]=1; next }                 # manifest: seed,...
  FNR>1 && ($1 in pop) && $3==want && !seen[$1]++ { s+=$4; n++ }   # probes: seed,run_id,mobility,avg_neighbor_count
  END { if (n>0) printf "%.4f %d", s/n, n; else printf "NA 0" }
' "$POP_MANIFEST" "$SRC/topology_probes_${MODE}.csv" \
  "$ROOT/simulations_v347_nohopfilter/topology_probes${PSUF}.csv")

echo "  target = $TARGET   (mean avg_neighbor_count over $NTARGET $POP runs, $MODE)"
[ "$TARGET" != NA ] || { echo "ERROR: could not compute target"; exit 1; }

# calibration seeds: the first K of the SAME population
mapfile -t SEEDS < <(awk -F, 'NR>1{print $1}' "$POP_MANIFEST" | head -"$K")
echo "  calibration seeds: ${SEEDS[*]}"

# ---------------------------------------------------------------- sweep -----
one_point() { # $1=refloss -> writes $OUT/probe_<rl>.csv, prints "rl mean delta"
  local rl="$1"
  local probe="$OUT/probe_rl${rl}.csv"
  : > "$probe"
  local running=0
  for seed in "${SEEDS[@]}"; do
    local d="$OUT/.tmp_rl${rl}_s${seed}"
    rm -rf "$d"; mkdir -p "$d"
    # each seed writes its OWN probe file: concurrent appends to one file are
    # not safe, and a per-seed file also survives a partial sweep
    ( cd "$d" && timeout 120 "$BIN" \
        --run=1 --RngRun="$seed" --bMobility="$MOB" \
        --outputDir="$d/feat/" --nSimulationSeconds=65 \
        --enforceHopFilter=0 \
        --propagationModel=realistic --propRefLoss="$rl" \
        --propExponent="$EXPONENT" --propMaxRange="$GUARD" \
        --topologyProbeFile="$d/probe.csv" > run.log 2>&1 ) &
    running=$((running+1))
    if [ "$running" -ge "$NUM_WORKERS" ]; then wait -n; running=$((running-1)); fi
  done
  wait
  # gather the per-seed probe files into one, then average
  awk -F, -v m="$MOB" 'FNR>1 && $3==m {print}' "$OUT"/.tmp_rl${rl}_s*/probe.csv >> "$probe" 2>/dev/null
  local mean
  mean=$(awk -F, '{s+=$4; n++} END {if (n>0) printf "%.4f", s/n; else print "nan"}' "$probe")
  echo "$rl $mean"
}

say "SWEEP  RefLoss in: $SWEEP   (K=$K seeds each, exponent=$EXPONENT, guard=${GUARD}m)"
# mode in the filename: static and mobile calibrate to different targets and
# must not overwrite one another
TABLE="$OUT/calibration_table_${MODE}.csv"
echo "ref_loss_db,mean_neighbor_degree,target,delta" > "$TABLE"
best_rl=""; best_abs=""
for rl in $SWEEP; do
  read -r _rl mean <<< "$(one_point "$rl")"
  delta=$(awk -v m="$mean" -v t="$TARGET" 'BEGIN{printf "%+.4f", m-t}')
  absd=$(awk -v m="$mean" -v t="$TARGET" 'BEGIN{d=m-t; if(d<0)d=-d; printf "%.4f", d}')
  echo "$rl,$mean,$TARGET,$delta" >> "$TABLE"
  printf "  RefLoss %-6s -> degree %-9s (target %s, delta %s)\n" "$rl" "$mean" "$TARGET" "$delta"
  if [ -z "$best_abs" ] || awk -v a="$absd" -v b="$best_abs" 'BEGIN{exit !(a<b)}'; then
    best_abs="$absd"; best_rl="$rl"
  fi
done

say "RESULT"
echo "  table: $TABLE"
echo "  closest: RefLoss=$best_rl (|delta|=$best_abs)"
echo
echo "  Refine with a narrower sweep around it, e.g.:"
echo "    SWEEP=\"$(awk -v r="$best_rl" 'BEGIN{for(i=-2;i<=2;i++) printf "%s ", r+i}')\" K=30 bash run_propcal.sh"
echo
echo "  Then the paired 2,000-run campaign (STATE §23.11) uses:"
echo "    --propagationModel=realistic --propRefLoss=<calibrated> --propExponent=$EXPONENT --propMaxRange=$GUARD"
echo "finished $(date '+%F %T')"
