# DCFM Defense-Detection — Colab reproduction

`DCFM_defense_detection.ipynb` reproduces the full machine-learning part of the paper
*Passive Reconnaissance of Routing-Layer Defenses in OLSR-Based MANETs using ML*
(DCFM = the Fictive-Mitigation defense) — Tables 2, 4, 5, 6, 7, 8, 9, 10 and Figures 1, 2, 3.

## Code and Data Availability

This repository provides the complete machine-learning pipeline and the processed feature
dataset underlying the reported results, together with a self-contained Colab notebook
(`DCFM_defense_detection.ipynb`) that regenerates every reported table and figure. The
processed dataset is the per-measurement-window feature representation derived from the
ns-3 simulations; it is shipped in compact form (`colab_data/`) and has been verified to be
bit-for-bit identical to the full raw extraction. The raw ns-3 packet-level traces
(~314 MB) and the trained model ensembles (~18.7 GB) are not included here for size reasons
but are available from the corresponding author upon reasonable request. The notebook runs
end-to-end on a free Colab instance and pins the exact library versions used to produce the
results, so the published numbers are reproducible without access to the raw simulation
data.

## How the data is shipped

The raw simulation output (80,080 per-window CSVs, ~314 MB) and the trained ensembles
(~18.7 GB) are **not** committed. Instead, a compact bundle (`colab_data/`, ~39 MB) is:

| Path | What | Size |
|---|---|---|
| `colab_data/wide_{static,mobile}.csv.gz` | one row per measurement window, all 46 metrics | ~5 MB each |
| `colab_data/results_cache/` | pre-computed result tables/figures (so the default run is instant) | ~21 MB |
| `colab_data/results_cache/hp_search_extended/*/best_models.pkl` | **params-only stubs** (tuned hyper-params, not fitted weights) | ~6 KB |
| `colab_data/support/` | `topology_probes_*`, `run_status_*` for Table 2 | ~9 MB |

A single environment variable, `DCFM_DATA_BUNDLE`, makes every script read this bundle
instead of the raw CSVs (loader patched in `defense_detection_v2.py`; the few scripts that
glob directly use `colab_bundle.py`). Verified: the bundle reproduces the exact 25/37
retained-feature counts and matches the originally-trained model features bit-for-bit.

## Run it

Open `DCFM_defense_detection.ipynb` in Colab, then make the code + data available to it
in any one of these ways (Part 0 auto-detects all three):

- **Upload** — run the *Upload* cell and pick `reproduction_bundle.zip`.
- **Google Drive** — put `strict_observable_v2/` at `MyDrive/strict_observable_v2`.
- **GitHub** — push this folder to a repo and set `DCFM_REPO_URL` in Part 0.

Then `Runtime → Run all`.

**Default (`RETRAIN=False`)** — finishes in minutes on free Colab: cheap analyses
(Cohen's d, separability, ROC, K-plot, Table 2) run live from the bundle; the heavy
sweeps are shown from the cache.

**`RETRAIN=True`** — recompute the heavy stages from the aggregated data. Start with
`FAST=True` (3 seeds) for a sanity pass, then `FAST=False` (20 seeds) for the paper numbers.
The full hyper-parameter search (`unified_hp_search_v2.py`) is the only multi-hour stage and
wants a high-RAM / Pro runtime.

## Regenerating the bundle (after new simulations)

From inside `strict_observable_v2/` (with `../simulations/` present):

```bash
python make_colab_dataset.py     # raw CSVs  -> colab_data/wide_*.csv.gz + support/
python make_results_cache.py     # results/  -> colab_data/results_cache/ (+ param stubs)
python build_notebook.py         # regenerate the .ipynb
```
