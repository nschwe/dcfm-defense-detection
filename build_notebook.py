#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_notebook.py
=================
Generate DCFM_defense_detection.ipynb — a full-reproduction Colab notebook for
the ML "learning" part of the paper "Passive Reconnaissance of Routing-Layer
Defenses in OLSR-Based MANETs using ML" (DCFM / Fictive-Mitigation detection).

Design:
  * Loads the compact git-shipped bundle (colab_data/) instead of the 80k raw
    CSVs and 18.7 GB of fitted models (see make_colab_dataset.py / make_*.py).
  * DEFAULT mode (RETRAIN=False): cheap analyses run live from the bundle;
    expensive analyses are displayed from the shipped result cache. Runs in
    minutes on free Colab.
  * RETRAIN=True: recompute the heavy stages (hp-search, K-sweep, ablations,
    threshold decomposition) from scratch.

Run:  python build_notebook.py   ->  writes DCFM_defense_detection.ipynb
"""
import json

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": text.rstrip("\n").splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "outputs": [],
                  "execution_count": None,
                  "source": text.strip("\n").splitlines(keepends=True)})


# ======================================================================
# Title
# ======================================================================
md(r"""
# Passive Detection of the DCFM Defense in OLSR MANETs — ML Reproduction

This notebook reproduces the **machine-learning ("learning") part** of the paper
*Passive Reconnaissance of Routing-Layer Defenses in OLSR-Based MANETs using ML*.

**Task.** Binary classification: from passively observable OLSR control-/data-plane
statistics, decide whether the **DCFM / Fictive-Mitigation** defense is active
(`defense_only`, `attack_defense` → positive) or not (`baseline`, `attack_only` → negative),
under **static** and **mobile** regimes.

### What each paper section maps to

| Paper section | Script | Notebook part |
|---|---|---|
| §4.1 Variance decomposition (R = σ²w/σ²b) | `variance_decomposition_v2.py` | Part 1 |
| §4.2 / Table 2 topology Cohen's d | `table_ii_cohens_d.py` | Part 1 |
| §4.3 Feature filtering (25/37 counts) | `verify_iv_c_numbers.py`, `inspect_removed_features.py` | Part 1 |
| §5–6.1 In-domain classification (Table 4) | `unified_hp_search_v2.py`, `rebuild_stacking_v2.py` | Part 2 |
| §5.5 Cluster-bootstrap CIs + ROC (Fig 1) | `cluster_bootstrap_ci.py`, `generate_roc_figure.py` | Part 2 |
| §6.2 Per-feature Cohen's d (Table 5) | `compute_cohens_d_v2.py` | Part 3 |
| §6.3 Joint separability (Table 6) | `compute_separability_v2.py` | Part 3 |
| §6.5 Universal-4 selection (Table 7) | `feature_importance_sensitivity_v2.py` | Part 4 |
| §6.6 K-sweep + Fig 2/3 | `k_sweep_universal4_v2.py`, `plot_figure1_optimal_k.py`, `confusion_matrices_universal4_v2.py` | Part 4 |
| §6.7 DJ ablation / augmentation (Table 9) | `feature_eng_ablation_v2.py`, `dj_ablation_breakdown*.py`, `flowthroughputstd_analysis_v2.py` | Part 5 |
| §6.8 Threshold/ranking decomposition (Table 10) | `threshold_decomposition_v2.py`, `plot_threshold_decomposition.py` | Part 6 |

> **Data note.** The 80,080 raw per-window CSVs (~314 MB) are pre-aggregated into two
> compact tables `colab_data/wide_{static,mobile}.csv.gz` (~5 MB each). The 18.7 GB of
> fitted ensembles are **not** shipped: cross-domain scripts only need the tuned
> *hyper-parameters*, which travel as ~6 KB stub pickles. Final tables/figures are also
> shipped pre-computed so the default run is fast.
""")

# ======================================================================
# 0. Setup
# ======================================================================
md(r"""
## 0 · Setup — code, dependencies, and the data bundle

**To run this notebook, make the code + data available (pick one):**
1. **Upload the zip** — run the *Upload* cell below and choose `reproduction_bundle.zip`.
2. **Google Drive** — put the `strict_observable_v2/` folder at `MyDrive/strict_observable_v2`; the setup cell mounts Drive and finds it.
3. **GitHub** — set `DCFM_REPO_URL` to the repository; the setup cell clones it.

Then run: *Upload* (if used) → *Smart setup* → *Install* → *Stage bundle* → *Switches*.
""")

code(r"""
# --- (optional) UPLOAD: run to upload reproduction_bundle.zip, then run the Smart setup cell ---
import os, zipfile
if os.path.exists("defense_detection_v2.py"):
    print("Code already present; skip this cell.")
else:
    try:
        from google.colab import files
        up = files.upload()                      # choose reproduction_bundle.zip
        for name in up:
            if name.endswith(".zip"):
                with zipfile.ZipFile(name) as z:
                    z.extractall("_bundle")
                print("Extracted", name, "-> _bundle/. Now run the Smart setup cell.")
    except ImportError:
        print("Not on Colab — just open this notebook from inside the unzipped folder.")
""")

code(r"""
# --- SMART SETUP: find the code via GitHub / Drive / uploaded-zip, then cd into it ---
import os, sys, subprocess, glob

REPO_URL = os.environ.get("DCFM_REPO_URL", "")   # e.g. https://github.com/<you>/<repo>.git

def have_code():
    return os.path.exists("defense_detection_v2.py")

def _enter(root):
    for r, _d, files in os.walk(root):
        if "defense_detection_v2.py" in files:
            os.chdir(r); return True
    return False

if not have_code():
    if REPO_URL:                                          # 1) GitHub
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, "_repo"], check=True)
        _enter("_repo")
    elif os.path.isdir("_bundle"):                        # 2) uploaded zip
        _enter("_bundle")
    else:                                                 # 3) Google Drive
        try:
            from google.colab import drive
            drive.mount("/content/drive")
            cand = "/content/drive/MyDrive/strict_observable_v2"
            if os.path.exists(os.path.join(cand, "defense_detection_v2.py")):
                os.chdir(cand)
            else:
                _enter("/content/drive/MyDrive")
        except Exception as e:
            print("Drive not available:", e)

assert have_code(), (
    "Could not find defense_detection_v2.py. Use ONE of:\n"
    "  (1) the Upload cell above with reproduction_bundle.zip,\n"
    "  (2) Google Drive with the folder at MyDrive/strict_observable_v2, or\n"
    "  (3) set DCFM_REPO_URL to the GitHub repo and re-run."
)
print("Working dir:", os.getcwd())
""")

code(r"""
# --- Install dependencies PINNED to the exact versions used to produce the paper ---
# This makes RETRAIN reproduce the original numbers and lets the param-stubs unpickle cleanly.
PINS = ["scikit-learn==1.7.2", "xgboost==3.2.0", "lightgbm==4.6.0", "catboost==1.2.10",
        "imbalanced-learn==0.14.1", "joblib==1.5.3", "pandas==2.3.3", "scipy==1.15.3", "seaborn"]
subprocess.run([sys.executable, "-m", "pip", "-q", "install", *PINS], check=False)
subprocess.run([sys.executable, "-m", "pip", "-q", "install", "numpy==2.2.6"], check=False)
print("deps ready (pinned to the paper's environment: sklearn 1.7.2, xgboost 3.2.0, "
      "lightgbm 4.6.0, catboost 1.2.10).")
print("If Colab asks to RESTART the runtime after a numpy change, do it and re-run from here.")
""")

code(r"""
# --- Point every script at the compact bundle, and stage the shipped result cache ---
import shutil

BUNDLE = os.path.abspath("colab_data")
assert os.path.isdir(BUNDLE), "colab_data/ not found next to the code — include it in the repo."
os.environ["DCFM_DATA_BUNDLE"] = BUNDLE     # <- read by the data loaders (see colab_bundle.py)
os.environ.setdefault("MAX_JOBS", "2")      # required by a couple of scripts at import time

# Stage pre-computed results so display cells and downstream scripts find their inputs.
os.makedirs("results", exist_ok=True)
for src in glob.glob(os.path.join(BUNDLE, "results_cache", "*")):
    dst = os.path.join("results", os.path.basename(src))
    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)

# table_ii_cohens_d.py reads simulations/{topology_probes,run_status}_*.csv (cwd-relative).
os.makedirs("simulations", exist_ok=True)
for f in glob.glob(os.path.join(BUNDLE, "support", "*")):
    shutil.copy2(f, os.path.join("simulations", os.path.basename(f)))

print("Bundle staged. DCFM_DATA_BUNDLE =", BUNDLE)
print("wide tables:", [os.path.basename(p) for p in glob.glob(BUNDLE + "/wide_*")])
""")

code(r"""
# ====================== MASTER SWITCHES ======================
RETRAIN = False   # True  -> recompute heavy stages from scratch (hours; Colab Pro / high-RAM advised)
FAST    = True    # in RETRAIN, use fewer seeds / smaller grids to finish sooner
N_SEEDS = 3 if FAST else 20
# =============================================================

import pandas as pd, numpy as np
from IPython.display import Image, display, Markdown

STATIC_ROOT = "../simulations/features_static"   # only the 'static'/'mobile' substring matters under the bundle
MOBILE_ROOT = "../simulations/features_mobile"

def run(cmd, title=None, tail=4000):
    if title:
        print(">>", title)
    print("$", " ".join(str(c) for c in cmd))
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    print(out[-tail:])
    if r.returncode != 0:
        print(f"[exit {r.returncode}]")
    return r.returncode, out

def show_csv(path, n=None, **kw):
    if not os.path.exists(path):
        print("(missing)", path); return None
    df = pd.read_csv(path, **kw)
    display(df.head(n) if n else df)
    return df

def show_img(path, width=760):
    if os.path.exists(path):
        display(Image(filename=path, width=width))
    else:
        print("(missing)", path)

def show_text(path, tail=60):
    if os.path.exists(path):
        print("".join(open(path, encoding="utf-8", errors="replace").readlines()[-tail:]))
    else:
        print("(missing)", path)

print("RETRAIN =", RETRAIN, "| FAST =", FAST, "| N_SEEDS =", N_SEEDS)
""")

# ======================================================================
# Part 1 — Data & acceptance
# ======================================================================
md(r"""
## Part 1 · Dataset, leakage controls & feature filtering (§4)

Each accepted simulation run contributes four 40 s windows (one per scenario) sharing one
RNG seed. Grouped (run-level) splitting prevents leakage. Two controls support that the
signal is scenario-driven, not topology-driven:

* **§4.1 Variance decomposition** — every retained feature has within/between ratio R ≥ 0.5.
* **Table 2** — accepted vs rejected runs differ only negligibly on topology metrics (|d| small).
* **§4.3** — variance + |r|>0.95 correlation filtering leaves **25** features (static) / **37** (mobile).
""")

code(r"""
# §4.1 Variance decomposition  (default: show shipped result; RETRAIN: recompute)
if RETRAIN:
    run(["python", "variance_decomposition_v2.py"], "variance decomposition")
print("STATIC:"); show_csv("results/static/variance_decomposition.csv", n=12)
print("MOBILE:"); show_csv("results/mobile/variance_decomposition.csv", n=12)
""")

code(r"""
# Table 2 — topology Cohen's d (accepted vs rejected). Runs live (seconds; support files shipped).
run(["python", "table_ii_cohens_d.py"], "Table 2: topology metrics Cohen's d")
""")

code(r"""
# §4.3 Final feature counts after variance + correlation filtering (expected 25 static / 37 mobile).
# This rebuilds 141 engineered features over 40k samples -> a few minutes; gated behind RETRAIN.
if RETRAIN:
    run(["python", "verify_iv_c_numbers.py"], "verify §IV-C feature counts")
else:
    print("Skipped live (slow). Paper result: 25 retained features (static), 37 (mobile).")
    print("Set RETRAIN=True to recompute via verify_iv_c_numbers.py / inspect_removed_features.py.")
""")

# ======================================================================
# Part 2 — In-domain classification
# ======================================================================
md(r"""
## Part 2 · In-domain classification (§5–6.1, Table 4)

Twelve tuned classifiers + a stacking ensemble, evaluated within each configuration
(60/20/20 run-level split, robust scaling, validation-tuned thresholds, isotonic calibration).

**Headline (paper):** Stacking accuracy **0.887** (static) / **0.910** (mobile), AUC **0.95–0.96**.
Ensembles improve under mobility; Logistic Regression collapses (0.55) while Ridge/LDA hold.

Default mode shows the shipped per-model results and the cached ROC/CI artifacts.
`RETRAIN=True` reruns the full hyper-parameter search (hours) then rebuilds the stacking model.
""")

code(r"""
# Table 4 — per-model in-domain accuracy / AUC (shipped from the tuned 'extended' search).
for cfg in ("static", "mobile"):
    print("="*30, cfg.upper(), "="*30)
    show_csv(f"results/hp_search_extended/{cfg}/combined_results.csv")
print("\n--- Stacking ensemble summary ---")
show_text("results/hp_search_final/static/stacking_summary.txt")
show_text("results/hp_search_final/mobile/stacking_summary.txt")
""")

code(r"""
# Figure 1 — ROC curves (static + mobile). Runs live from the shipped prediction cache.
run(["python", "generate_roc_figure.py"], "ROC figure")
show_img("results/cluster_bootstrap/roc_curves.png")
# §5.5 cluster-aware bootstrap confidence intervals for Table 4:
show_csv("results/cluster_bootstrap/bootstrap_results.csv")
""")

code(r"""
# RETRAIN: full hyper-parameter search + stacking rebuild (HOURS; high-RAM).
if RETRAIN:
    run(["python", "unified_hp_search_v2.py",
         "--config", "both", "--grid-mode", "extended",
         "--static-root", STATIC_ROOT, "--mobile-root", MOBILE_ROOT,
         "--out-dir", "./results/hp_search_extended"], "FULL hp-search (extended)")
    run(["python", "unified_hp_search_v2.py", "--config", "both", "--grid-mode", "focused",
         "--static-root", STATIC_ROOT, "--mobile-root", MOBILE_ROOT,
         "--out-dir", "./results/hp_search_focused"], "SVM focused search")
    run(["python", "rebuild_stacking_v2.py",
         "--static-root", STATIC_ROOT, "--mobile-root", MOBILE_ROOT], "rebuild stacking")
    run(["python", "generate_scaler_v2.py"], "scaler/split artifacts")
    run(["python", "cluster_bootstrap_ci.py"], "cluster bootstrap CIs")
else:
    print("Showing cached Table 4. Set RETRAIN=True to retrain all 12 models from scratch.")
""")

# ======================================================================
# Part 3 — Separability
# ======================================================================
md(r"""
## Part 3 · Feature separability (§6.2–6.3, Tables 5–6)

* **Per-feature Cohen's d (Table 5).** Control-plane features (`AverageMprCount`,
  `AverageAdvertisedLinksPerTCMessage`) stay discriminative across regimes; delay/jitter
  features collapse under mobility.
* **Joint separability (Table 6).** Mahalanobis distance and the LDA ratio both *increase*
  under mobility (disjoint 95% CIs, permutation p < 1e-5) — discriminative content is
  redistributed across features, which tree ensembles exploit.

Both run live from the bundle (Cohen's d: seconds; separability: a few minutes).
""")

code(r"""
# §6.2 Per-feature Cohen's d across configs (Table 5 source).
run(["python", "compute_cohens_d_v2.py",
     "--static", STATIC_ROOT, "--mobile", MOBILE_ROOT,
     "--out", "results/cohens_d"], "Cohen's d per feature")
show_csv("results/cohens_d/cohens_d_all_features.csv")
""")

code(r"""
# §6.3 Multivariate separability (Mahalanobis + LDA). FAST shrinks the permutation count.
nperm = 2000 if FAST else 100000
run(["python", "compute_separability_v2.py",
     "--static", STATIC_ROOT, "--mobile", MOBILE_ROOT,
     "--n_bootstrap", "100", "--n_permutations", str(nperm),
     "--out", "results/separability"], f"separability ({nperm} perms)")
show_csv("results/separability/separability_summary.csv")
show_img("results/separability/separability_distributions.png")
""")

# ======================================================================
# Part 4 — Universal-4 and cross-domain
# ======================================================================
md(r"""
## Part 4 · Universal-4 subset & cross-domain transfer (§6.5–6.6, Tables 7–8, Figs 2–3)

The cross-domain optimum is a **4-feature invariant subset (Universal-4)**:
`AverageAdvertisedLinksPerTCMessage`, `AverageMprCount`, `DataPacketRate`, `FlowThroughputStd`.

* **Table 7** — CatBoost PVC top-9 intersection (static ∩ mobile) → these 4 features.
* **Table 8 / Fig 2** — cross-domain accuracy vs K peaks near-symmetrically at **K=4**
  (S→M 0.864, M→S 0.859); S→M degrades sharply beyond K=5 while M→S stays flat.
* **Fig 3** — Universal-4 keeps in-domain accuracy (88/90%) and symmetric transfer (86/86%).
""")

code(r"""
# §6.5 Universal-4 selection (Table 7) — shipped (full sweep is hours).
show_csv("results/feature_importance/recommended_K_per_method.csv")
show_text("results/feature_importance/summary.txt", tail=40)
show_img("results/feature_importance/optimal_k_curves.png")
if RETRAIN:
    run(["python", "feature_importance_sensitivity_v2.py",
         "--static-root", STATIC_ROOT, "--mobile-root", MOBILE_ROOT,
         "--hp-results-dir", "./results/hp_search_extended",
         "--out-dir", "./results/feature_importance",
         "--n-seeds", str(N_SEEDS)], "FULL importance/Universal-4 sweep")
""")

code(r"""
# §6.6 K-sweep (Table 8) + Figure 2. Default shows shipped CSV; live plot from it.
if RETRAIN:
    run(["python", "k_sweep_universal4_v2.py",
         "--static-root", STATIC_ROOT, "--mobile-root", MOBILE_ROOT,
         "--hp-results-dir", "./results/hp_search_extended",
         "--out-dir", "./results/k_sweep_universal4",
         "--n-seeds", str(N_SEEDS)], "K-sweep (live)")
show_csv("results/k_sweep_universal4/k_sweep_results.csv")
run(["python", "plot_figure1_optimal_k.py",
     "--in-csv", "results/k_sweep_universal4/k_sweep_results.csv",
     "--out-dir", "results/k_sweep_universal4", "--optimal-k", "4"], "Figure 2")
show_img("results/k_sweep_universal4/figure1_optimal_k.png")
""")

code(r"""
# Figure 3 — Universal-4 confusion matrices (in-domain + cross-domain).
if RETRAIN:
    run(["python", "confusion_matrices_universal4_v2.py",
         "--static-root", STATIC_ROOT, "--mobile-root", MOBILE_ROOT,
         "--hp-results-dir", "./results/hp_search_extended",
         "--out-dir", "./results/confusion_matrices_universal4",
         "--n-seeds", str(N_SEEDS)], "confusion matrices (live, ~tens of min)")
show_img("results/confusion_matrices_universal4/confusion_matrices.png")
show_text("results/confusion_matrices_universal4/confusion_matrices_summary.txt", tail=40)
""")

# ======================================================================
# Part 5 — Ablations
# ======================================================================
md(r"""
## Part 5 · Feature ablation & augmentation (§6.7, Table 9)

* Removing the **six delay/jitter (DJ) features** from the 33-feature set raises S→M accuracy
  from 0.664 → 0.795 (Δ=+0.131), but leaves M→S unchanged — a *family-level*, direction-specific
  effect (random / instability-based removals do **not** reproduce it).
* Feature **augmentation** (ratios, logs, poly) does **not** close the gap.
* **FlowThroughputStd** has near-zero univariate d under mobility yet contributes through
  interactions (dropping it lowers in-domain accuracy in both regimes).

Heavy sweeps are shipped pre-computed; `flowthroughputstd_analysis` can run live.
""")

code(r"""
# Table 9 — DJ ablation & augmentation summaries (shipped; RETRAIN recomputes).
print("--- feature-engineering ablation (S->M, Table 9) ---")
show_csv("results/feature_eng_ablation_v2/feature_eng_summary.csv")
print("--- DJ-ablation breakdown  S->M ---");  show_csv("results/dj_ablation_breakdown/summary.csv")
print("--- DJ-ablation breakdown  M->S ---");  show_csv("results/dj_ablation_breakdown_MS/summary.csv")
if RETRAIN:
    run(["python", "feature_eng_ablation_v2.py",
         "--static-root", STATIC_ROOT, "--mobile-root", MOBILE_ROOT,
         "--n-seeds", str(N_SEEDS)], "feature-eng ablation (live)")
    run(["python", "dj_ablation_breakdown.py"], "DJ breakdown S->M")
    run(["python", "dj_ablation_breakdown_MS.py"], "DJ breakdown M->S")
""")

code(r"""
# FlowThroughputStd interaction analysis (minutes; uses the param stubs). Live when RETRAIN.
if RETRAIN:
    run(["python", "flowthroughputstd_analysis_v2.py",
         "--static-root", STATIC_ROOT, "--mobile-root", MOBILE_ROOT,
         "--hp-results-dir", "./results/hp_search_extended",
         "--out-dir", "./results/flowthroughputstd_analysis",
         "--n-seeds", str(N_SEEDS)], "FlowThroughputStd analysis")
show_text("results/flowthroughputstd_analysis/summary.txt", tail=40)
show_csv("results/flowthroughputstd_analysis/analysis_A_summary.csv")
""")

# ======================================================================
# Part 6 — Threshold decomposition
# ======================================================================
md(r"""
## Part 6 · Sources of cross-domain error (§6.8, Table 10)

The S→M gap of the full feature set decomposes into a **ranking** (feature-separability)
component and a **threshold-transfer** (score-distribution shift) component:

`Δtotal = Δranking + Δthreshold`.

For S→M, Δtotal ≈ 0.192 = Δranking 0.106 + Δthreshold 0.086 — i.e. a genuine separability
advantage *plus* a recoverable threshold/calibration effect. The few-shot recalibration curve
shows the threshold part is largely recovered with a handful of labeled target runs.
""")

code(r"""
# Table 10 — decomposition + figures (shipped; RETRAIN recomputes the full sweep).
if RETRAIN:
    run(["python", "threshold_decomposition_v2.py",
         "--static-root", STATIC_ROOT, "--mobile-root", MOBILE_ROOT,
         "--hp-results-dir", "./results/hp_search_extended",
         "--out-dir", "./results/threshold_decomposition",
         "--n-seeds", str(N_SEEDS)], "threshold decomposition (live)")
show_csv("results/threshold_decomposition/decomposition_summary.csv")
run(["python", "plot_threshold_decomposition.py",
     "--in-dir", "results/threshold_decomposition"], "decomposition figures")
show_img("results/threshold_decomposition/fig_fewshot_recalibration.png")
show_img("results/threshold_decomposition/fig_score_drift.png")
""")

md(r"""
## Appendix · Diagnostics & robustness checks (§5.2, §6.7, §7.1)

Supporting checks behind the appendix tables: the SVM C=10⁵ boundary probe, the
RobustScaler-vs-StandardScaler control for the DJ-ablation effect, and the edge-of-grid
hyper-parameter extension. Outputs are shipped; recompute via `RETRAIN`.
""")

code(r"""
import json
def show_json(path, n=1200):
    if os.path.exists(path):
        print(path, "\n", json.dumps(json.load(open(path)), indent=2)[:n])
    else:
        print("(missing)", path)

# SVM C=1e6 probe — is the C=1e5 search boundary a true plateau? (§5.2 / Tables A.11–A.14)
show_json("results/hp_search_svm_c1e6/static/svm_c1e6_results.json")
# RobustScaler vs StandardScaler control for the +0.13 DJ-ablation effect (§6.7)
show_csv("results/robust_scaler_check/scaler_comparison.csv")
# Edge-of-grid hyper-parameter extension (§5.2 follow-up)
for cfg in ("static", "mobile"):
    show_json(f"results/hp_search_edge_extension/{cfg}/extension_results.json", n=500)

if RETRAIN:
    run(["python", "check_robust_scaler_vs_standard.py"], "scaler control (live)")
    run(["python", "test_svm_high_c.py"], "SVM C=1e6 probe (live, slow)")
print("\nNote: count_attacker_unavailable.py (§4.2 rejection profile) needs the raw "
      "simulation logs, which are not part of the compact bundle.")
""")

md(r"""
## Done

**Default run** reproduces every paper section from the shipped bundle in minutes:
Tables 2/4/5/6/7/8/9/10 and Figures 1/2/3.

To recompute the heavy stages from the aggregated data, set `RETRAIN = True` (top of Part 0)
and re-run — start with `FAST = True` for a reduced-seed sanity pass, then `FAST = False` for the
full 20-seed numbers. The full hyper-parameter search is the only multi-hour stage and benefits
from a high-RAM / Pro runtime.
""")

# ======================================================================
# Serialize
# ======================================================================
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
        "colab": {"provenance": [], "toc_visible": True},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open("DCFM_defense_detection.ipynb", "w", encoding="utf-8") as fh:
    json.dump(nb, fh, ensure_ascii=False, indent=1)

print(f"wrote DCFM_defense_detection.ipynb with {len(cells)} cells")
