#!/usr/bin/env bash
# build_fpnt_mobile_pool.sh — admitted seed pool for the FPNT mobile arm.
#
# WHY THIS EXISTS (STATE 25.85.9). A seed pool is specific to its ATTACK CLASS,
# not just to its mobility mode. simulations_v347_hopablation_mobile_10k was
# admitted under the ISOLATION attack. The black-hole configuration allocates
# attacker positions through a separate allocator, so it consumes a different
# sequence of draws from the same RngRun and lands on a different topology.
# Measured 23/8: mobile seeds 4 and 5 pass the t=60 connectivity assert under
# isolation and fail it under the black-hole — identically for gcop and fpnt, so
# it is a property of the attack, not of a defence. Reusing the old pool cost
# 3 rejections out of 5 and two degenerate survivors.
#
# WHAT IT DOES. Draws candidates from the campaign's own mobile population, in
# manifest order, runs each one exactly as the FPNT arm will run it, and keeps
# the seeds whose four windows all produced a metrics CSV.
#
# ⛔ Population continuity is the point: candidates come from the campaign's
#    C_all (N=10,000). The FPNT mobile arm is therefore a declared SUBSET of the
#    campaign population, not a differently-drawn sample. Say so in the paper.
# ⛔ Writes only under fpntbh347. nv347 is never touched.
# ⛔ No hop_category prefilter. It would look like free compute — hop_category=1
#    seeds self-exclude anyway with "Attacker not available" — but the value in
#    the manifest was measured under the ISOLATION attack and is not the same
#    quantity as this run's hop distance, so filtering on it would be filtering
#    on the wrong number. Rejections are measured, not predicted.
#
# Resumable: already-decided seeds are skipped, so re-running continues.
#
#   JOBS=22 TARGET=2000 MAXSCAN=6000 bash build_fpnt_mobile_pool.sh
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"      # fpntbh347/
ROOT="$HERE/ns-3.47"
# the main tree that holds the C_all manifests: in the repository it is the parent of
# fpntbh347/; in another layout set NV347 to the ns-3.47 tree of the main campaign
NV347="${NV347:-$(cd "$HERE/.." && pwd)}"
BIN="$ROOT/build/scratch/ns3.47-iolsr-tests-corrected-default"

# The same argument applies to BOTH modes: the static pool was admitted under
# the isolation attack too. MOB picks which population is being re-admitted.
MOB="${MOB:-1}"
if [ "$MOB" = 1 ]; then
  SRC="$NV347/simulations_v347_hopablation_mobile_10k/manifests/C_all.csv"
  DEF_OUT="$HERE/simulations_fpnt_mobile_10k"
else
  SRC="$NV347/simulations_v347_hopablation_10k/manifests/C_all.csv"
  DEF_OUT="$HERE/simulations_fpnt_static_10k"
fi

OUT="${OUT:-$DEF_OUT}"
TARGET="${TARGET:-2000}"      # accepted seeds wanted
MAXSCAN="${MAXSCAN:-6000}"    # candidates to try before giving up
JOBS="${JOBS:-22}"   # 22 of 24 — the campaign's own figure (§25.80.1)
RUN_TIMEOUT="${RUN_TIMEOUT:-900}"
SCRATCH="${SCRATCH:-/tmp/fpnt_pool_work}"

[ -x "$BIN" ] || { echo "binary missing: $BIN"; exit 1; }
[ -f "$SRC" ] || { echo "source manifest missing: $SRC"; exit 1; }

mkdir -p "$OUT/manifests" "$SCRATCH"
LOG="$OUT/admission_log.csv"
LOCK="$OUT/.lock"
touch "$LOCK"
[ -s "$LOG" ] || echo "seed,verdict,reason,seconds" > "$LOG"

export BIN OUT LOG LOCK SCRATCH RUN_TIMEOUT TARGET MOB

# ---- one candidate --------------------------------------------------------
probe_seed () {
  seed="$1"

  # Stop early once enough have been accepted.
  # grep -c already prints 0 and exits 1 when there is no match; an "|| echo 0"
  # here appends a SECOND line and every [ -ge ] then dies on "0\n0".
  n=$(flock "$LOCK" -c "grep -c ',accept,' '$LOG'" 2>/dev/null | head -1)
  [ "${n:-0}" -ge "$TARGET" ] && return 0

  d="$SCRATCH/$seed"
  rm -rf "$d"; mkdir -p "$d/feat"
  t0=$(date +%s)
  ( cd "$d" && timeout "$RUN_TIMEOUT" "$BIN" \
      --run=1 --RngRun="$seed" --bMobility="$MOB" \
      --outputDir="$d/feat/" --enforceHopFilter=0 \
      --defence=fpnt --bBlackholeAttack=1 --bBlackholeOnPath=1 \
      > run.log 2>&1 )
  t1=$(date +%s)

  complete=1
  for p in baseline attack_only defense_only defense_vs_attack; do
    [ -s "$d/feat/$p/metrics_output-1.csv" ] || complete=0
  done

  if [ "$complete" = 1 ]; then
    verdict=accept; reason=ok
  else
    verdict=reject
    reason=$(grep -m1 -oE 'Assert connectivity failed|Attacker not available|Received 0 packets|Terminated' "$d/run.log" 2>/dev/null)
    reason="${reason:-unknown}"
    reason="${reason// /_}"
  fi

  flock "$LOCK" -c "echo '$seed,$verdict,$reason,$((t1-t0))' >> '$LOG'"
  rm -rf "$d"
}
export -f probe_seed

# ---- candidate list, minus what is already decided -------------------------
awk -F, 'NR>1 && $1!="" {print $1}' "$SRC" | head -"$MAXSCAN" > "$SCRATCH/all_candidates.txt"
awk -F, 'NR>1 {print $1}' "$LOG" | sort -u > "$SCRATCH/decided.txt"
grep -vxF -f "$SCRATCH/decided.txt" "$SCRATCH/all_candidates.txt" > "$SCRATCH/todo.txt" || true

echo "candidates: $(wc -l < "$SCRATCH/all_candidates.txt")"
echo "already decided: $(wc -l < "$SCRATCH/decided.txt")"
echo "to run: $(wc -l < "$SCRATCH/todo.txt")   jobs=$JOBS   target=$TARGET"
echo "log: $LOG"
echo

xargs -a "$SCRATCH/todo.txt" -P "$JOBS" -I{} bash -c 'probe_seed "$@"' _ {}

# ---- manifest --------------------------------------------------------------
# Same four columns as the campaign manifest, rows copied VERBATIM from the
# source so nothing downstream sees a changed schema. hop_category stays the
# value the campaign measured; this run's own hop distances live in the
# diagnostic log, never in the manifest.
awk -F, -v n="$TARGET" '
  NR==FNR { if ($2=="accept") acc[$1]=1; next }
  FNR==1  { print; next }
  ($1 in acc) && kept<n { print; kept++ }
' "$LOG" "$SRC" > "$OUT/manifests/C_all.csv"

acc=$(grep -c ',accept,' "$LOG")
rej=$(grep -c ',reject,' "$LOG")
tot=$((acc+rej))
echo
echo "===== FPNT mobile pool ====="
echo "tried    : $tot"
echo "accepted : $acc   ($(awk -v a="$acc" -v t="$tot" 'BEGIN{printf "%.1f", t?100*a/t:0}')%)"
echo "rejected : $rej"
awk -F, 'NR>1 && $2=="reject" {c[$3]++} END {for (k in c) printf "   %-32s %d\n", k, c[k]}' "$LOG"
echo
echo "manifest : $OUT/manifests/C_all.csv  ($(( $(wc -l < "$OUT/manifests/C_all.csv") - 1 )) seeds)"
echo "log      : $LOG"
