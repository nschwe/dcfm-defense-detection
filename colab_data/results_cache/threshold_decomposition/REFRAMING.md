# Threshold-vs-Ranking Decomposition — Results and Proposed Re-framing

**Analysis:** `threshold_decomposition_v2.py` (20 bootstrap seeds 42–61, identical
CatBoost pipeline, group-aware 40% target-test fold, paired comparisons).
**Verdict:** **Route B (qualified)** — the feature-invariance effect is real and
statistically dominant, **but** ~45% of the S→M headline gap is a decision-threshold
(calibration) artifact, not feature discriminability. The "0.67 vs 0.86" sentence
over-attributes the gap to features by ~1.8× and should be re-framed.

---

## 1. Headline decomposition (S→M, the "collapse" direction)

All three regimes evaluated on the **same** held-out 40% target-test fold per seed,
varying **only** the decision threshold. The per-seed identity
`Δ_total = Δ_ranking + Δ_threshold` holds exactly.

| Component | Meaning | Mean | 95% paired-t CI | bootstrap CI | share |
|---|---|---|---|---|---|
| **Δ_total** | K4 − K33, source threshold (the headline) | **+0.1921** | [0.1837, 0.2006] | [0.1907, 0.1936] | 100% |
| **Δ_ranking** | K4 − K33, **oracle** threshold (genuine feature part) | **+0.1060** | [0.0969, 0.1151] | [0.1046, 0.1075] | **55%** |
| **Δ_threshold** | residual = calibration artifact | **+0.0862** | [0.0784, 0.0939] | [0.0846, 0.0877] | **45%** |

All p < 1e-14. Paired-t and run-level cluster-bootstrap CIs agree (robust).

**Reproduction check:** under the source threshold the script measures K33 S→M
accuracy = **0.6716** [0.663, 0.680] and K4 = **0.8638** — i.e. it reproduces the
paper's 0.6718 / 0.8639 headline exactly on the held-out fold, so the decomposition
is anchored to the published numbers.

### The 2×2×3 operating-point table (accuracy ± 95% CI)

| Dir | K | source-thr (operational) | oracle (target-tuned) | Δ recovered by threshold alone |
|---|---|---|---|---|
| **S→M** | 33 | 0.6716 [0.663, 0.680] | **0.7588** [0.749, 0.769] | **+0.087** |
| S→M | 4 | 0.8638 [0.862, 0.866] | 0.8648 [0.863, 0.867] | +0.001 |
| M→S | 33 | 0.8419 [0.840, 0.844] | 0.8453 [0.844, 0.847] | +0.003 |
| M→S | 4 | 0.8584 [0.857, 0.860] | 0.8590 [0.858, 0.860] | +0.001 |

Pure threshold re-tuning lifts K33 S→M from 0.672 to **0.759** (+0.087) — recovering
**45% of the entire 0.192 gap** without changing a single feature. The achievable
oracle (tuned on a disjoint target-calibration pool) equals the in-test ceiling to 4
decimals, so the calibration pool is large enough that this is a true operating-point
limit, not an estimation artifact.

> **M→S asymmetry:** in the non-collapsing direction the threshold artifact is
> negligible (Δ_threshold = 0.0028, 17% of an already-tiny 0.016 gap). The calibration
> failure is specific to S→M — consistent with mobility-induced score drift appearing
> when a static-trained model meets mobile traffic.

---

## 2. Few-shot recalibration — the operational payoff

Threshold re-tuned on `n` labeled **target runs** (group-level), evaluated on the
target-test fold (`fig_fewshot_recalibration.{png,pdf}`):

| n (runs) | K33 S→M accuracy | vs. source-thr 0.672 → oracle 0.759 |
|---|---|---|
| 25 | **0.746** | recovers ~85% of the threshold gap |
| 100 | 0.755 | ~91% |
| 500 | 0.758 | ≈ oracle |
| 250–500 + isotonic | **0.779–0.780** | **exceeds** the single-threshold oracle |

**25 labeled target scenarios** recover most of the K=33 "collapse" by moving the
threshold alone; K=4 barely benefits (0.857→0.864) because it is already calibrated.
Isotonic recalibration (n≥250) pushes past the single-threshold ceiling, indicating a
mild monotone score-warp on top of the location shift.

---

## 3. Mechanism — score-distribution drift, not feature failure

`fig_score_drift.{png,pdf}` + `mechanism_stats.csv`. Class priors are balanced by
construction (each run = 2 positive + 2 negative windows) and identical across
domains, so any optimal-threshold shift reflects **score-distribution drift, not prior
shift**.

| Dir | K | \|t*_src − t*_tgt\| | W₁(logit) | mean P(target-test) |
|---|---|---|---|---|
| S→M | 33 | **0.364** | **1.283** | 0.686 (drifts up from 0.495 source) |
| S→M | 4 | 0.057 | 0.613 | 0.453 (stable, source 0.496) |
| M→S | 33 | 0.130 | 1.630 | — |
| M→S | 4 | 0.053 | 0.923 | — |

At K=33 the mobile score mass migrates toward 1.0, so the static-tuned threshold
(≈0.51) lands far from the mobile-optimal (≈0.90) and accuracy craters even though the
ranking (AUC = 0.835) is largely intact. UNIVERSAL-4's score distribution is
domain-stable (|Δt*| ≈ 0.06), so its threshold transfers. **This is "score-distribution
invariance," a stronger and more precise claim than "feature discriminative invariance."**

---

## 4. Why the feature story still survives (Route B)

The oracle-threshold gap Δ_ranking = **0.106** (p<1e-15) is not only significant, it is
**larger** than the AUC gap (0.897 − 0.835 = 0.062). So at the best achievable operating
point UNIVERSAL-4 still beats Standard-33 by ~0.11 accuracy — the feature reduction
improves separability at the decision boundary beyond what the AUC summary alone
reveals. Feature invariance remains the dominant single contributor (55%); it is simply
no longer the *whole* story.

---

## 5. Proposed text changes

### 5a. Abstract — soft addition (no numbers; keep the abstract's flow)

The abstract does **not** explicitly over-attribute the gap to features, so the
least-defensive move is to leave it nearly intact and append one qualitative
sentence after "…near-symmetric cross-domain transfer (≈0.86 in both directions)":

> "Analysis indicates that the cross-domain degradation arises from both reduced
> class separability and decision-threshold transfer, the latter being largely
> recoverable through lightweight target-domain recalibration."

(Avoid putting the 55/45 split or raw numbers in the abstract — too technical;
they belong in the new §VI subsection and §VII-C. The numeric decomposition lives
in `paper_insert.tex` Insert 1.)

### 5b. §VII-C — "Feature Invariance as the Determinant of Generalization"

Re-title to **"Feature Invariance and Score Calibration as Co-Determinants of
Generalization"** and insert:

> The static→mobile accuracy collapse of the full 33-feature model (0.6718) is
> frequently read as a loss of discriminative power. The threshold-free AUC contradicts
> this: it falls only to 0.8348 (vs. 0.8974 for UNIVERSAL-4), a 0.06 gap against a 0.19
> accuracy gap. We therefore decompose the accuracy gap on identical predicted scores
> by varying only the decision threshold across three regimes — the operational
> source-tuned threshold, an oracle threshold tuned on a disjoint target-calibration
> fold, and few-shot recalibration on `n` labeled target runs — over the same 20
> bootstrap seeds with paired confidence intervals (Table&nbsp;[2×2×3]).
>
> Of the 0.192±0.008 static→mobile gap, **0.106±0.009 is a ranking (feature) effect**
> that survives oracle thresholding, and **0.086±0.008 is a threshold (calibration)
> artifact** (both p<10⁻¹⁴; cluster-bootstrap CIs agree). Mechanistically, under
> mobility the 33-feature score distribution drifts (logit-Wasserstein 1.28; optimal
> threshold shifts by 0.36), so a single source-tuned threshold cannot transfer;
> UNIVERSAL-4's score distribution is stable (shift 0.06), so its threshold transfers
> unchanged. Re-tuning only the threshold recovers the 33-feature detector to 0.759, and
> 25 labeled target scenarios already reach 0.746 — a practical recalibration recipe.
>
> We therefore refine the contribution from "feature invariance prevents collapse" to
> the more precise and better-supported claim that **UNIVERSAL-4 confers
> score-distribution invariance**: it is the dominant driver of cross-domain
> generalization (55% of the gap, and a larger effect than AUC alone indicates), while
> the remaining 45% is an avoidable calibration failure of the high-dimensional model
> rather than an intrinsic loss of signal.

### 5c. New table to add (from `decomposition_summary.csv` / `decomposition_components.csv`)

The 2×2×3 accuracy table (§1) plus the component table; the few-shot curve
(`fig_fewshot_recalibration`) and score-drift figure (`fig_score_drift`) as new
figures.

---

## 6. Artifacts
- `decomposition_per_seed.csv`, `decomposition_summary.csv`, `decomposition_components.csv`
- `fewshot_curve.csv`, `mechanism_stats.csv`, `hist_data.npz`
- `fig_fewshot_recalibration.{png,pdf}`, `fig_score_drift.{png,pdf}`
- Scripts: `threshold_decomposition_v2.py`, `plot_threshold_decomposition.py`
- Hygiene: group-aware split at run level; threshold tuned on validation/calibration
  only; target-test fold held out; same RobustScaler-free StandardScaler pipeline and
  tuned hyperparameters as the existing tables (so numbers are directly comparable).
