#!/usr/bin/env bash
# run_r36_bootstrap.sh -- R#3.6's four window comparisons, re-run 3/9.
#
# WHY IT MUST BE RE-RUN
#   The four intervals in r3_6_answer.tex were produced 1/9 (00:08-02:53) from
#   the stage-1 pickles of arms_windowsweep/w{10,20}_17, arms_gm_control_17 and
#   arms_r34_17feat_2k_w_17. The frozencal marathon re-learned all four of those
#   arms on 2/9 between 01:00 and 01:38, so every one of those pickles has since
#   been replaced. r23_paired_bootstrap.py reconstructs from the pickle; the
#   1/9 outputs therefore describe models that no longer exist.
#
# WHY --model Stacking_Ensemble
#   Without it each arm is scored with its own winner. On 1/9 that paired
#   w10_17's Stacking_Ensemble against arms_gm_control_17's SVM_linear -- a
#   delta across estimators, which measures the estimator and not the window.
#   The script now aborts on that mismatch (r23_paired_bootstrap.py:249), and
#   the house rule is the Stacking_Ensemble row anyway (STATE 20.7, 25.200.3).
#   The pickle carries every fitted estimator and its tuned threshold, so this
#   is still a reconstruction, never a refit.
#
# ANCHORS -- deliberately different per mode, see r3_6's OURS:
#   static  40 s : arms_gm_control_17          (2,000 ids)
#   mobile  40 s : arms_r34_17feat_2k_w_17     (1,999 ids, window-matched;
#                  gm_control_17 is NOT paired with the mobile sweep arms)
#
# WHY R23_PIPELINE MUST BE SET -- checked 3/9, do not drop it
#   ⛔ NOT a calibration fix, and it must not be described as one. This script
#   never fits and never calibrates: grep '.fit(' / CalibratedClassifierCV /
#   FrozenEstimator in r23_paired_bootstrap.py returns nothing. It uses the
#   pipeline module only for dd.Config() and dd.DefenseDetector(cfg) -- load,
#   preprocess, engineer, split -- and then applies the model, scaler and
#   threshold stored in each arm's best_model_*.pkl. The corrected calibration
#   is INSIDE those pickles, put there by the 2/9 frozencal marathon
#   (gate [frozen-calibration] 16/16 in every cell). No choice of pipeline here
#   can change it.
#   The one thing the choice does change is the FEATURE GENERATOR WIDTH.
#   r23_paired_bootstrap.py:51 defaults to analysis/cleanfeat/pipeline_17, which
#   skips ten formulas whose base metrics are absent (67 columns); frozencal
#   emits all 77. The marathon logs record these arms as "Created 77 features
#   (from 18)", so a 67-column reconstruction is refused by the stored scaler --
#   exactly what aborted the first mobile attempt on 1/9 (STATE 25.179).
#   Pinning frozencal therefore means: reconstruct with the same module that
#   trained the arm.
#   ⚠️ Separately true and NOT this script's business: cleanfeat/pipeline_17 has
#   no FrozenEstimator. That matters only for arms LEARNED through it -- today
#   just arms_fpnt_17. Its re-run landed 2/9 as arms_fpnt_17_frozencal,
#   learned through frozencal67_pipeline_17, which does carry one. The
#   reported FPNT figures come from that arm and not from this one
#   (STATE 25.200.2, 25.336.9).
#   Both pipelines carry METRICS = 17 and every bundle here is 27 columns
#   (21 canonical + 6 meta), so the SPACE is right either way.
#
# Usage:  bash run_r36_bootstrap.sh
set -u
AN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-$HOME/miniconda3/envs/manet/bin/python}"
[ -x "$PY" ] || PY=python3
# R36_PIPELINE: the generator the four arms were LEARNED with. Default is the
# 77-column copy of the 2/9 marathon; after the 67-column re-learn (FFS 9.1)
# pass R36_PIPELINE=$AN/frozencal67_pipeline_17 and a fresh R36_SUFFIX so the
# 2/9 intervals are not overwritten.
R36_PIPELINE="${R36_PIPELINE:-$AN/frozencal/pipeline_17}"
R36_SUFFIX="${R36_SUFFIX:-}"
export R23_PIPELINE=$R36_PIPELINE
TS=$(date +%Y%m%d_%H%M%S)
LOG=$AN/r36_bootstrap_${TS}.log

run_one() {   # realistic baseline mode out-dir
    echo "--- $1 vs $2 [$3]" | tee -a "$LOG"
    PYTHONIOENCODING=utf-8 R23_PIPELINE=$R36_PIPELINE     nice -n 19 "$PY" -u "$AN/r23_paired_bootstrap.py" \
        --realistic "$1" --baseline "$2" --mode "$3" \
        --model Stacking_Ensemble --n-boot 2000 --seed 42 \
        --out-dir "$4" >> "$LOG" 2>&1
    local st=$?
    [ $st -ne 0 ] && { echo "FAIL ($st) -- see $LOG"; return 1; }
    grep -E '"delta_pp"|"ci95_pp"|"p_two_sided"|"model_' -A 3 \
        "$4/r23_paired_$(basename "$1")_$3.json" 2>/dev/null | tail -12
    return 0
}

echo "R23_PIPELINE=$R36_PIPELINE  out suffix='${R36_SUFFIX}'" | tee -a "$LOG"
run_one arms_windowsweep/w10_17 arms_gm_control_17        static \
        "$AN/r36_window_bootstrap${R36_SUFFIX}"        || exit 1
run_one arms_windowsweep/w20_17 arms_gm_control_17        static \
        "$AN/r36_window_bootstrap${R36_SUFFIX}"        || exit 1
run_one arms_windowsweep/w10_17 arms_r34_17feat_2k_w_17   mobile \
        "$AN/r36_window_bootstrap_mobile${R36_SUFFIX}" || exit 1
run_one arms_windowsweep/w20_17 arms_r34_17feat_2k_w_17   mobile \
        "$AN/r36_window_bootstrap_mobile${R36_SUFFIX}" || exit 1

echo "=== all four written; log $LOG ==="
