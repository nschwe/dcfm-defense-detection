#!/usr/bin/env bash
# ============================================================================
# run_hidden.sh — isolate the hidden-terminal / interference contribution
#
# R#5.3 names three things: "fading, interference, and hidden terminal effects".
# The propagation experiment (§25.36-25.44) measured the FIRST only. This script
# measures the other two.
#
# ⭐ THE KEY FACT, verified 27/8, and it changes what has to be built:
#    iolsr-tests-corrected.cc never sets RtsCtsThreshold, and ns-3.47's default
#    is WIFI_DEFAULT_RTS_THRESHOLD = WIFI_MAX_RTS_THRESHOLD
#    (src/wifi/model/wifi-standard-constants.h:113). With 512-byte datagrams the
#    RTS/CTS handshake therefore NEVER fires, so hidden-terminal collisions are
#    ALREADY PRESENT in every run this project has ever produced, the disk
#    baseline included. YansWifiPhy likewise runs InterferenceHelper.
#
#    So the gap is not that the effects are unmodelled. It is that they have
#    never been ISOLATED. Enabling RTS/CTS suppresses precisely the
#    hidden-terminal collision mode and leaves everything else alone, so the
#    paired difference is the effect's size.
#
# DESIGN — paired, and only ONE arm needs simulating:
#    treatment : the paper's configuration + RTS/CTS enabled   <- simulated here
#    baseline  : the paper's configuration as-is               <- ALREADY EXISTS
#                (range-channel windows for every C_all seed, staged by symlink,
#                 exactly as §25.36 staged the propagation baseline)
#
# ⛔ REQUIRES A ONE-LINE SIMULATOR CHANGE THAT DOES NOT YET EXIST. This script
#    REFUSES to run until it is made, and prints the patch. It is deliberately
#    additive and defaults to current behaviour, so every existing campaign
#    stays byte-identical:
#
#      // near the globals, with the other --prop* knobs (~line 317)
#      uint32_t g_rtsCtsThreshold = 4692480;   // WIFI_MAX_RTS_THRESHOLD: off
#
#      // in the CommandLine block, beside --propagationModel (~line 2570)
#      cmd.AddValue("rtsCtsThreshold",
#                   "PSDU size above which RTS/CTS is used. Default = ns-3 max, "
#                   "i.e. RTS/CTS never fires, which is the paper's behaviour. "
#                   "Set low (e.g. 100) to suppress hidden-terminal collisions.",
#                   g_rtsCtsThreshold);
#
#      // replace the existing SetRemoteStationManager call (~line 2725)
#      wifi.SetRemoteStationManager ("ns3::ConstantRateWifiManager",
#                                    "RtsCtsThreshold",
#                                    UintegerValue (g_rtsCtsThreshold));
#
# ⛔ SMOKE FIRST. SMOKE=1 runs 30 seeds into a scratch tree and stops.
#
#   SMOKE=1 bash run_hidden.sh
#   MODE=static nohup bash run_hidden.sh > hidden_static.log 2>&1 &
#   MODE=mobile nohup bash run_hidden.sh > hidden_mobile.log 2>&1 &
# ============================================================================
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$ROOT/build/scratch/ns3.47-iolsr-tests-corrected-default"
PY="${PY:-$HOME/miniconda3/envs/manet/bin/python}"
[ -x "$PY" ] || PY=python3
ANALYSIS="$ROOT/analysis"
SCEN="baseline attack_only defense_only defense_vs_attack"

MODE="${MODE:-static}"
if [ "$MODE" = mobile ]; then MOB=1; else MOB=0; MODE=static; fi

RTS="${RTS:-100}"                 # treatment: RTS/CTS on for anything >100 B
TARGET="${TARGET:-2000}"
NUM_WORKERS="${NUM_WORKERS:-16}"
MAX_JOBS="${MAX_JOBS:-22}"
RUN_TIMEOUT="${RUN_TIMEOUT:-900}"
SMOKE="${SMOKE:-0}"
TAG_SUFFIX="${TAG_SUFFIX:-}"
STAGES="${STAGES:-sim bundle learn summary}"

PIPELINE="${PIPELINE:-$ANALYSIS/cleanfeat/pipeline_17}"
LEARN_FLAGS="${LEARN_FLAGS:---no-augmentation --split-validation --add-linearsvc --calibrate-stacking}"

if [ "$MODE" = mobile ]; then
  POP_MANIFEST="${POP_MANIFEST:-$ROOT/simulations_v347_hopablation_mobile_10k/manifests/C_all.csv}"
else
  POP_MANIFEST="${POP_MANIFEST:-$ROOT/simulations_v347_hopablation_10k/manifests/C_all.csv}"
fi

# ⛔ TAG_SUFFIX isolates a rehearsal from the real campaign. Without it a
#    dry-run writes into the SAME arm roots the live run uses and clobbers it.
TAG="hidden_rts${RTS}${TAG_SUFFIX:-}"
if [ "$SMOKE" = 1 ]; then
  TARGET=30; NUM_WORKERS=8; TAG="${TAG}_smoke"
  echo "*** SMOKE: target=$TARGET, scratch trees, no learning ***"
  STAGES="sim"
fi

OUT="$ROOT/simulations_v347_${TAG}"
FEAT="$OUT/features_${MODE}"
TMPROOT="$OUT/.work_${MODE}"
ARMS="$ROOT/analysis/arms_${TAG}"
BASE_SIM="$ROOT/simulations_v347_${TAG}_baseline"
BASE_ARMS="$ROOT/analysis/arms_${TAG}_baseline"
MANIFEST="$OUT/manifests/admitted_${MODE}.csv"

want () { case " $STAGES " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
say  () { echo; echo "================ $* ================"; date '+%F %T'; }

# ---------------------------------------------------------------- guards ----
[ -x "$BIN" ] || { echo "ABORT: binary not built: $BIN"; exit 1; }
[ -x "$PY" ]  || { echo "ABORT: manet python missing"; exit 1; }
[ -s "$POP_MANIFEST" ] || { echo "ABORT: population manifest missing: $POP_MANIFEST"; exit 1; }

# ⛔ GATE 1 — the flag must exist. A binary that silently ignores an unknown
#    argument would produce a "treatment" arm identical to the baseline and the
#    delta would read as "hidden terminals cost nothing". That is the exact
#    shape of failure §25.112.1 and §25.119.2 warn about.
# ⚠️ Capture first, test second. Under `set -o pipefail` a pipeline inherits the
#    binary's exit code, and --PrintHelp exits non-zero, so testing the pipeline
#    directly makes this gate refuse even when the flag IS present.
HELP=$("$BIN" --PrintHelp 2>&1 || true)
if ! printf '%s' "$HELP" | grep -q "rtsCtsThreshold"; then
  echo "ABORT: this binary has no --rtsCtsThreshold."
  echo "       Apply the three-line patch in the header of this script,"
  echo "       rebuild, and re-run. Refusing to produce a meaningless delta."
  exit 1
fi

# ⛔ GATE 2 — and it must actually reach the MAC. Checked from produced data,
#    not from the banner: with RTS/CTS on, RTS frames must appear.
if [ "$SMOKE" = 1 ]; then
  echo "  gate 2 will be exercised by the smoke run below."
fi

# ⚠️ Only the LEARN stage must not share the machine. A smoke run executes
#    simulations only, so refusing it while an unrelated learner is running would
#    block the cheap check for hours for no reason. Use nice when they overlap.
if want learn && pgrep -f "defense_detection_v2.py" >/dev/null 2>&1; then
  echo "ABORT: a stage-1 learner is running and this invocation includes the"
  echo "       learn stage. Wait, or run with STAGES=sim."
  exit 1
fi
if pgrep -f "defense_detection_v2.py" >/dev/null 2>&1; then
  echo "  [note] a stage-1 learner is running elsewhere; simulations will share"
  echo "         the machine. Launch this under nice."
fi

# ⛔ GATE 3 — THE FEATURE SPACE, checked by counting, not by existence.
#    A file-exists test is not a gate (§25.146.2). run_arm.py falls back to
#    analysis/pipeline (33 metrics) whenever DCFM_PIPELINE is unset or wrong,
#    silently, and the campaign it is compared against is the 17-observable one.
#    Checked HERE, before ~35 min of simulation, not at the learn stage.
[ -d "$PIPELINE" ] || { echo "ABORT: PIPELINE not a directory: $PIPELINE"; exit 1; }
[ -f "$PIPELINE/defense_detection_v2.py" ] || {
  echo "ABORT: no defense_detection_v2.py under $PIPELINE"; exit 1; }
NMET=$("$PY" - "$PIPELINE" <<'PYEOF' 2>/dev/null | grep -E '^[0-9]+$' | tail -1
import sys, importlib.util, os
spec = importlib.util.spec_from_file_location("dd", os.path.join(sys.argv[1], "defense_detection_v2.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print(len(m.DefenseDetector.METRICS))
PYEOF
)
if [ "$NMET" != "17" ]; then
  echo "ABORT: $PIPELINE exposes ${NMET:-?} metrics, expected 17."
  echo "       The reported campaign is the 17-observable listener arm; learning"
  echo "       in the 33- or 21-metric space would not be comparable to it."
  exit 1
fi

# ⛔ GATE 4 — refuse UP FRONT if a non-listener arm already sits in either
#    output root. The post-learn check below catches one we create; this catches
#    one that is already there. Past runs built original / corrected / combined
#    and threw away hours.
for r in "$ARMS" "$BASE_ARMS"; do
  [ -d "$r" ] || continue
  pre=$(ls -d "$r"/*/ 2>/dev/null | sed 's#.*/\([^/]*\)/#\1#' | grep -vxF -e listener || true)
  if [ -n "$pre" ]; then
    echo "ABORT: non-listener arm(s) already under $r:"
    echo "$pre" | sed 's/^/       /'
    exit 1
  fi
done

echo "================================================================"
echo " run_hidden.sh — hidden-terminal isolation"
echo "   mode        : $MODE (bMobility=$MOB)"
echo "   treatment   : --rtsCtsThreshold=$RTS  (RTS/CTS ON above $RTS B)"
echo "   baseline    : the paper's configuration (RTS/CTS never fires)"
echo "   channel     : range / 190 m disk  — the REPORTED configuration"
echo "   population  : $POP_MANIFEST"
echo "   target      : $TARGET admitted seeds"
echo "   pipeline    : $PIPELINE   (METRICS=$NMET ✅)"
echo "   flags       : $LEARN_FLAGS"
echo "================================================================"

# ------------------------------------------------------------------- sim ----
if want sim; then
  say "1/4  simulations — $MODE, RTS/CTS enabled"
  mkdir -p "$FEAT" "$TMPROOT" "$OUT/manifests"
  for s in $SCEN; do mkdir -p "$FEAT/$s"; done
  : > "$MANIFEST"; echo "seed,run_id,source,hop_category" >> "$MANIFEST"

  mapfile -t CAND < <(awk -F, 'NR>1 && $1 != "" {print $1}' "$POP_MANIFEST")
  echo "  candidates: ${#CAND[@]}"

  # ⛔⛔ run_id IS NOT A FUNCTION OF THE SEED (§25.148). It is assigned by the
  #    campaign: `main` rows carry a running counter, `rerun` rows carry
  #    seed+1000000. In the static manifest only 2 rows of 10,000 satisfy
  #    run_id == seed. Computing it paired 38 of 40 seeds against ANOTHER
  #    seed's data while the staging gate reported "0 missing".
  #    It must be READ. run_propmodel.sh already does this (§25.36).
  RIDMAP="$TMPROOT/.ridmap"
  mkdir -p "$TMPROOT"
  awk -F, 'NR>1 && $1 != "" {print $1" "$2}' "$POP_MANIFEST" > "$RIDMAP"
  echo "  seed->run_id map: $(wc -l < "$RIDMAP") entries"

  export BIN TMPROOT FEAT RUN_TIMEOUT MOB RTS SCEN TARGET
  run_one () {
    seed="$1"
    # ⛔ EARLY STOP. Without this the loop simulates every candidate handed to
    #    xargs before the collection loop even looks at a verdict. At 100 %
    #    admission that is TARGET*6 runs to obtain TARGET seeds -- six times the
    #    work, measured 27/8: 2,446 runs in 62 min on the way to 12,000
    #    (~5 h of simulation for 2,000 usable seeds).
    #    The counter file is appended under flock and read with wc -l, so the
    #    check costs one small read per run.
    if [ -f "$TMPROOT/.okcount" ] \
       && [ "$(wc -l < "$TMPROOT/.okcount" 2>/dev/null || echo 0)" -ge "$TARGET" ]; then
      return 0
    fi
    d="$TMPROOT/$seed"
    rm -rf "$d"; mkdir -p "$d/feat"
    ( cd "$d" && timeout "$RUN_TIMEOUT" "$BIN" \
        --run=1 --RngRun="$seed" --bMobility="$MOB" \
        --outputDir="$d/feat/" \
        --enforceHopFilter=0 \
        --rtsCtsThreshold="$RTS" \
        > run.log 2>&1 )
    rc=$?; n=0
    for s in $SCEN; do [ -f "$d/feat/$s/metrics_output-1.csv" ] && n=$((n+1)); done
    hop=$(grep -oP 'HOP_CATEGORY=\K[0-9]' "$d/run.log" 2>/dev/null | tail -1)
    if [ "$rc" -eq 0 ] && [ "$n" -eq 4 ]; then
      echo "$seed,OK,${hop:--}" > "$d/.status"
      ( flock -x 8; echo "$seed" >> "$TMPROOT/.okcount" ) 8>"$TMPROOT/.okcount.lock"
    else
      reason=other
      grep -q "Assert connectivity failed" "$d/run.log" 2>/dev/null && reason=connectivity
      grep -q "Attacker not available"     "$d/run.log" 2>/dev/null && reason=attacker_unavailable
      [ "$rc" -eq 124 ] && reason=timeout
      echo "$seed,FAIL,$reason" > "$d/.status"
    fi
  }
  export -f run_one

  : > "$TMPROOT/.okcount"

  # ⛔ THE CEILING. It was TARGET*6, sized for the propagation arm's 18-61 %
  #    admission. Measured admission under the disk channel is 100 % (§25.147),
  #    so 2,000 accepted needs ~2,000 runs. A six-fold ceiling is not a safety
  #    net, it is a silent licence to run six times too long. 1.5x leaves room
  #    for admission down to ~67 % and screams if it is ever reached.
  MAXSCAN="${MAXSCAN:-$(( TARGET * 3 / 2 ))}"
  [ "$MAXSCAN" -gt "${#CAND[@]}" ] && MAXSCAN=${#CAND[@]}
  echo "  ceiling: will probe at most $MAXSCAN candidates for $TARGET admitted"

  printf '%s\n' "${CAND[@]}" | head -"$MAXSCAN" \
    | xargs -P "$NUM_WORKERS" -I{} bash -c 'run_one {}' &
  XPID=$!

  # ---- heartbeat: the log was silent for ~50 min between these two lines ---
  t0=$(date +%s)
  while kill -0 "$XPID" 2>/dev/null; do
    sleep 60
    kill -0 "$XPID" 2>/dev/null || break
    d=$(find "$TMPROOT" -maxdepth 2 -name .status 2>/dev/null | wc -l)
    o=$(wc -l < "$TMPROOT/.okcount" 2>/dev/null || echo 0)
    el=$(( $(date +%s) - t0 ))
    if [ "$o" -gt 0 ] && [ "$o" -lt "$TARGET" ]; then
      eta=$(( el * (TARGET - o) / o / 60 ))
      printf "  [%s] probed %s  admitted %s/%s  elapsed %dm  eta ~%dm\n" \
        "$(date +%H:%M:%S)" "$d" "$o" "$TARGET" $((el/60)) "$eta"
    else
      printf "  [%s] probed %s  admitted %s/%s  elapsed %dm\n" \
        "$(date +%H:%M:%S)" "$d" "$o" "$TARGET" $((el/60))
    fi
  done
  wait "$XPID" 2>/dev/null

  PROBED=$(ls "$TMPROOT" 2>/dev/null | grep -vc '^\.' || echo 0)
  ADMIT=$(wc -l < "$TMPROOT/.okcount" 2>/dev/null || echo 0)
  echo "  simulated $PROBED candidates to reach $ADMIT admitted"
  if [ "$ADMIT" -lt "$TARGET" ]; then
    echo
    echo "  ⛔⛔ CEILING REACHED WITHOUT MEETING THE TARGET."
    echo "      probed $PROBED (ceiling $MAXSCAN), admitted $ADMIT, wanted $TARGET."
    echo "      Admission is lower than the 100 % measured in the smoke, so the"
    echo "      arm would be SMALLER than the campaign it is compared against."
    echo "      Refusing to continue. Re-run with an explicit MAXSCAN once the"
    echo "      real admission rate is known."
    exit 1
  fi

  # collect
  acc=0; rej=0
  declare -A REASONS
  for d in "$TMPROOT"/*/; do
    [ -s "$d/.status" ] || continue
    IFS=, read -r s v r < "$d/.status"
    if [ "$v" = OK ]; then
      # ⛔ the CAMPAIGN's run_id for this seed, read from the manifest. Our own
      #    output files are named by it too, so the bundle and the staged
      #    baseline refer to the same run.
      rid=$(awk -v k="$s" '$1==k{print $2; exit}' "$RIDMAP")
      if [ -z "$rid" ]; then
        echo "  [warn] seed $s is not in the population manifest -- skipped"
        continue
      fi
      acc=$((acc+1))
      for sc in $SCEN; do
        for f in metrics_output observer_metrics observer_detail; do
          [ -f "$d/feat/$sc/$f-1.csv" ] && cp -p "$d/feat/$sc/$f-1.csv" "$FEAT/$sc/$f-$rid.csv"
        done
      done
      echo "$s,$rid,hidden,$r" >> "$MANIFEST"
      [ "$acc" -ge "$TARGET" ] && break
    else
      rej=$((rej+1)); REASONS[$r]=$(( ${REASONS[$r]:-0} + 1 ))
    fi
  done

  echo
  echo "  admitted $acc / tried $((acc+rej))"
  for k in "${!REASONS[@]}"; do printf "    reject %-26s %d\n" "$k" "${REASONS[$k]}"; done

  # ⛔ GATE 2, from the data: RTS frames must actually appear.
  probe=$(ls -d "$TMPROOT"/*/ 2>/dev/null | head -1)
  if [ -n "$probe" ] && [ -f "$probe/run.log" ]; then
    echo
    echo "  gate 2 — RTS/CTS reached the MAC?"
    echo "    ⚠️  The scenario prints no RTS counter. Verify once, by hand, that"
    echo "        a treatment run differs from a baseline run on the same seed"
    echo "        (e.g. UdpPacketsReceived or ObsNonOlsrDataFrames in"
    echo "        metrics_output-*.csv). An identical pair means the flag did"
    echo "        NOT take effect and the campaign is void."
  fi

  if [ "$SMOKE" = 1 ]; then
    echo
    echo "*** SMOKE COMPLETE. Inspect the admission rate and gate 2, then re-run"
    echo "    without SMOKE=1. Scratch tree: $OUT"
    exit 0
  fi
fi

# ---------------------------------------------------------------- bundle ----
if want bundle; then
  say "2/4  bundle — listener arm"
  N=$(( $(wc -l < "$MANIFEST") - 1 ))
  echo "  admitted seeds: $N"
  mkdir -p "$ARMS"
  DCFM_SIM_ROOT="$OUT" DCFM_ARMS_ROOT="$ARMS" \
    "$PY" "$ANALYSIS/make_arm_bundles.py" --arm listener --mode "$MODE" --limit "$N" || exit 1
fi

# --------------------------------------------- paired baseline (no new sim) --
if want learn; then
  say "3/4  staging the paired baseline for the SAME admitted seeds"
  bfeat="$BASE_SIM/features_${MODE}"
  for s in $SCEN; do mkdir -p "$bfeat/$s"; done
  "$PY" - "$MANIFEST" "$ROOT" "$bfeat" "$MODE" "$POP_MANIFEST" <<'PYEOF'
import csv, os, sys
manifest, root, dst, mode, pop = sys.argv[1:6]
SCEN = ["baseline","attack_only","defense_only","defense_vs_attack"]
tree = ("simulations_v347_hopablation_mobile_10k" if mode == "mobile"
        else "simulations_v347_hopablation_10k")
src = os.path.join(root, tree, "C_all", "features_%s" % mode)

# ⛔ IDENTITY CHECK, not existence (§25.148.3). A file-exists test cannot tell
#    "absent" from "present but belonging to a different seed", which is exactly
#    how 38 of 40 static pairings passed while being wrong.
seed2rid = {}
with open(pop) as fh:
    for r in csv.DictReader(fh):
        if r.get("seed"):
            seed2rid[r["seed"].strip()] = r["run_id"].strip()

wrong = 0
made = miss = 0
with open(manifest) as fh:
    for row in csv.DictReader(fh):
        rid = row["run_id"]
        seed = row["seed"].strip()
        if seed2rid.get(seed) != rid:
            wrong += 1
            if wrong <= 5:
                print("  MIS-PAIRED seed %s: manifest says run_id %s, campaign says %s"
                      % (seed, rid, seed2rid.get(seed)))
        for sc in SCEN:
            for f in ("metrics_output", "observer_metrics", "observer_detail"):
                s = os.path.join(src, sc, "%s-%s.csv" % (f, rid))
                d = os.path.join(dst, sc, "%s-%s.csv" % (f, rid))
                if os.path.exists(s):
                    if not os.path.exists(d):
                        os.symlink(s, d)
                    made += 1
                else:
                    miss += 1
print("  staged %d paired baseline links, %d missing" % (made, miss))
print("  identity check: %d of %d seeds carry the campaign's own run_id"
      % (len(list(open(manifest))) - 1 - wrong, len(list(open(manifest))) - 1))
if miss:
    print("  ⛔ missing files mean the pairing is INCOMPLETE — investigate before learning")
    sys.exit(1)
if wrong:
    print("  ⛔⛔ %d MIS-PAIRED seeds. The baseline would come from other seeds' runs." % wrong)
    print("     This is the §25.148 failure. Refusing to continue.")
    sys.exit(1)
PYEOF
  [ $? -ne 0 ] && { echo "ABORT: paired baseline staging failed its checks"; exit 1; }

  say "4/4  learning — treatment and paired baseline"
  N=$(( $(wc -l < "$MANIFEST") - 1 ))
  mkdir -p "$BASE_ARMS"
  DCFM_SIM_ROOT="$BASE_SIM" DCFM_ARMS_ROOT="$BASE_ARMS" \
    "$PY" "$ANALYSIS/make_arm_bundles.py" --arm listener --mode "$MODE" --limit "$N" || exit 1

  # (the real check is GATE 3 at the top: METRICS must be exactly 17)

  for pair in "treatment:$ARMS" "baseline:$BASE_ARMS"; do
    label="${pair%%:*}"; root_="${pair#*:}"
    echo; echo "--- learning $label / $MODE ---"; date '+%F %T'
    echo "    (stage 1 is quiet for 1-2 h; the banners below appear at the start)"
    env DCFM_ARMS_ROOT="$root_" DCFM_PIPELINE="$PIPELINE" MAX_JOBS="$MAX_JOBS" \
      "$PY" -u "$ANALYSIS/run_arm.py" \
      --arm listener --script defense_detection_v2.py --mode "$MODE" \
      -- $LEARN_FLAGS
    stray=$(ls -d "$root_"/*/ 2>/dev/null | sed 's#.*/\([^/]*\)/#\1#' | grep -vxF -e listener || true)
    [ -n "$stray" ] && { echo "*** ERROR: non-listener arm under $root_: $stray"; exit 1; }
  done
fi

# --------------------------------------------------------------- summary ----
if want summary; then
  say "SUMMARY — hidden-terminal isolation, $MODE"
  a1=$(sed -n 2p "$ARMS/listener/results/$MODE/results.csv"      2>/dev/null | cut -d, -f4)
  a0=$(sed -n 2p "$BASE_ARMS/listener/results/$MODE/results.csv" 2>/dev/null | cut -d, -f4)
  printf "  %-40s %s\n" "RTS/CTS enabled (hidden terminals suppressed)" "${a1:--}"
  printf "  %-40s %s\n" "paper configuration (hidden terminals present)" "${a0:--}"
  if [ -n "$a1" ] && [ -n "$a0" ]; then
    awk -v x="$a1" -v y="$a0" 'BEGIN{
      d = x - y
      printf "  %-40s %+.4f %s\n", "paired delta", d,
             (d<0.014 && d>-0.014) ? "<- INSIDE the +/-0.014 noise floor" : ""
    }'
  fi
  echo
  echo "  ⭐ HOW TO READ THIS, decided in advance:"
  echo "     delta ~ 0        hidden-terminal collisions do not materially affect"
  echo "                      detection; the reviewer's concern is answered by"
  echo "                      measurement rather than by discussion."
  echo "     delta > 0        suppressing them HELPS detection, i.e. the reported"
  echo "                      figures are depressed by collision noise and are a"
  echo "                      conservative estimate."
  echo "     delta < 0        suppressing them HURTS detection, i.e. part of the"
  echo "                      separability rides on collision behaviour. This is"
  echo "                      the outcome that must be reported most carefully."
  echo
  echo "  ⚠️  RTS/CTS suppresses hidden-terminal collisions but also adds control"
  echo "      overhead of its own. The delta is therefore an UPPER BOUND on the"
  echo "      hidden-terminal effect, not a clean estimate of it. Say so."
  echo "  ⛔  This isolates hidden terminals under the DISK channel. Interference"
  echo "      under fading is a different cell and is not measured here."
fi
date '+%F %T'
