#!/usr/bin/env bash
# ============================================================================
# fp_ablate.sh -- does FPNT detection survive without message-size information?
#
# The arm scored listener 0.9481 static / 0.9469 mobile, the highest this system
# has produced, and recall jumped from GCOP's 0.80-0.89 to 0.94. Per-feature
# separation (fp_importance.py) says why: mean frame size, SniffedBytes /
# SniffedFrames, separates defended from undefended windows at AUC 0.876 static
# and 0.857 mobile, standardised mean difference 1.59 and 1.41 — and 4 of the
# top 5 features are byte-volume. The Section 5.2 piggyback adds 4 bytes per
# advertised neighbour to every TC, attack or no attack, which is exactly that
# quantity.
#
# So: rebuild the listener bundle with every byte-volume column removed and run
# the SAME pipeline on it. Accuracy that holds up is a claim about control-plane
# behaviour. Accuracy that collapses means the detector was reading message size,
# and the result has to be stated that way.
#
#   SET=volume  (default) all seven byte-volume columns
#   SET=size    only the mean-frame-size pair, the dominant one
#
#   bash fp_ablate.sh                 # build ablated bundles + run both modes
#   PHASE=build bash fp_ablate.sh     # bundles only
# ============================================================================
set -uo pipefail

FPNT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-$HOME/miniconda3/envs/manet/bin/python}"
[ -x "$PY" ] || PY=python3
ANALYSIS="${ANALYSIS:-$HOME/ns3/nv347/ns-3.47/analysis}"
SET="${SET:-volume}"
# ---------------------------------------------------------------------------
# 27/8 — ADDITIVE, GATED overrides, mirroring run_fpnt_arm_all.sh.
# EVERY DEFAULT BELOW REPRODUCES THE 23/8 BEHAVIOUR EXACTLY.
#   SRC          source bundle dir  (default: the 23/8 arms_fpnt bundle)
#   ARMS_ROOT    output root        (default: arms_fpnt_ablate_$SET)
#   FULL_REF     the "full" arm the closing table compares against
#   PIPELINE     DCFM_PIPELINE for stage 1; unset => run_arm.py default (33)
#   LEARN_FLAGS  flags after `--`   (default: --no-augmentation)
# ⛔ Re-running learn against the default ARMS_ROOT OVERWRITES the 23/8 ablation
#    (0.9394 static / 0.8200 mobile). Point it at a new directory.
# ---------------------------------------------------------------------------
SRC="${SRC:-$FPNT/analysis/arms_fpnt/listener/colab_data}"
ARMS_ROOT="${ARMS_ROOT:-$FPNT/analysis/arms_fpnt_ablate_$SET}"
DST="$ARMS_ROOT/listener/colab_data"
FULL_REF="${FULL_REF:-$FPNT/analysis/arms_fpnt}"
PIPELINE="${PIPELINE:-}"
LEARN_FLAGS="${LEARN_FLAGS:---no-augmentation}"
PHASE="${PHASE:-all}"
MAX_JOBS_LEARN="${MAX_JOBS_LEARN:-22}"

case "$SET" in
  volume) DROP="RoutingOverheadRatio,RoutingOverheadBytesRatio,AvgTxPacketSize,AvgRxPacketSize,AvgTxBytesPerFlow,AvgRxBytesPerFlow,AvgFlowThroughput" ;;
  size)   DROP="AvgTxPacketSize,AvgRxPacketSize" ;;
  *) echo "unknown SET=$SET (volume|size)"; exit 1 ;;
esac

command -v "$PY" >/dev/null 2>&1 || { echo "ERROR: no python interpreter: $PY"; exit 1; }
[ -f "$SRC/wide_static.csv.gz" ] || { echo "ERROR: source bundle missing: $SRC"; exit 1; }

if [ "$PHASE" = all ] || [ "$PHASE" = build ]; then
  echo "=== building ablated bundles: SET=$SET"
  echo "    dropping: $DROP"
  mkdir -p "$DST"
  DROP="$DROP" SRC="$SRC" DST="$DST" "$PY" - <<'PYEOF'
import gzip, os
import pandas as pd
drop = os.environ["DROP"].split(",")
src, dst = os.environ["SRC"], os.environ["DST"]
for mode in ("static", "mobile"):
    f = f"{src}/wide_{mode}.csv.gz"
    if not os.path.isfile(f):
        print(f"  {mode}: source missing, skipped"); continue
    df = pd.read_csv(f)
    present = [c for c in drop if c in df.columns]
    missing = [c for c in drop if c not in df.columns]
    out = df.drop(columns=present)
    o = f"{dst}/wide_{mode}.csv.gz"
    out.to_csv(o, index=False, compression="gzip")
    print(f"  {mode}: {df.shape[1]} -> {out.shape[1]} columns, "
          f"{len(out)} rows, dropped {len(present)}")
    if missing:
        print(f"    NOT PRESENT (nothing to drop): {missing}")
    assert not [c for c in drop if c in out.columns], "a drop column survived"
PYEOF
fi

if [ "$PHASE" = all ] || [ "$PHASE" = learn ]; then
  echo "  ARMS_ROOT    : $ARMS_ROOT"
  echo "  DCFM_PIPELINE: ${PIPELINE:-<unset -> run_arm.py default, 33 metrics>}"
  echo "  learn flags  : $LEARN_FLAGS"
  LEARN_ENV=(DCFM_ARMS_ROOT="$ARMS_ROOT" MAX_JOBS="$MAX_JOBS_LEARN")
  if [ -n "$PIPELINE" ]; then
    [ -f "$PIPELINE/defense_detection_v2.py" ] || {
      echo "REFUSING: no defense_detection_v2.py under $PIPELINE"; exit 1; }
    LEARN_ENV+=(DCFM_PIPELINE="$PIPELINE")
  fi
  for m in static mobile; do
    echo; echo "=== learning without $SET features: $m ==="; date '+%F %T'
    # LEARN_FLAGS is deliberately unquoted: it is a flag list, not one word.
    env "${LEARN_ENV[@]}" \
      "$PY" -u "$ANALYSIS/run_arm.py" \
      --arm listener --script defense_detection_v2.py --mode "$m" \
      -- $LEARN_FLAGS
  done
fi

echo
echo "================ ABLATION RESULT ================"
printf "  %-30s %-7s %-8s %-9s %-9s %s\n" bundle mode AUC Accuracy Recall model
for pair in "full:$FULL_REF" "ablated($SET):$ARMS_ROOT"; do
  label="${pair%%:*}"; base="${pair#*:}"
  for m in static mobile; do
    f="$base/listener/results/$m/results.csv"
    [ -f "$f" ] || continue
    l=$(head -2 "$f" | tail -1)
    printf "  %-30s %-7s %-8.4f %-9.4f %-9.4f %s\n" "$label" "$m" \
      "$(echo "$l" | cut -d, -f3)" "$(echo "$l" | cut -d, -f4)" \
      "$(echo "$l" | cut -d, -f6)" "$(echo "$l" | cut -d, -f1)"
  done
done
date '+%F %T'
