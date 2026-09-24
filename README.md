# Passive detection of routing-layer defenses in OLSR MANETs

Reproduction guide for the code and processed data behind the paper.

---

## 1. What this project does

An OLSR mobile ad hoc network is simulated in ns-3. In some simulation windows a
routing-layer defense is active and in others it is not. A passive listener,
which is one ordinary node that only overhears traffic reaching its own radio,
records statistics of what it hears. A classifier is then trained to decide,
from those statistics alone, whether the defense was active.

The question the paper answers is whether the defense leaves a trace that such a
listener can detect. It does.

Two defenses are evaluated, each against its own attack. The first injects
fictitious nodes into the topology and is evaluated against a node-isolation
attack. The second attaches trust metadata to topology-control messages and is
evaluated against a black-hole attack.

### The numbers you should get

| Result | Static | Mobile |
|---|---|---|
| Accuracy, first defense | 0.8998 | 0.9298 |
| AUC, first defense | 0.9403 | 0.9729 |
| Accuracy, second defense | 0.9488 | 0.9394 |
| Cross-domain accuracy, three features | 0.8730 (static to mobile) | 0.8632 (mobile to static) |
| Cross-domain accuracy, all 17 observables | 0.8753 (static to mobile) | 0.8522 (mobile to static) |

Published values are rounded to four decimals, half away from zero. On disk the
same numbers are stored at full precision, for example `0.89975` for the static
headline.

---

## 2. Getting started

### Requirements

* Linux, or Windows with WSL.
* Python 3.11 with scikit-learn, XGBoost, CatBoost, LightGBM, pandas, numpy,
  matplotlib and joblib. The project was run under a conda environment named
  `manet`.
* About 16 GB of RAM and 8 cores for a comfortable run. Less works but is slow.
* ns-3 version 3.47, only if you intend to regenerate the simulation data. If
  you use the processed data shipped here, you do not need ns-3 at all.

Throughout this document:

```
PY=~/miniconda3/envs/manet/bin/python
REPO=$(pwd)                       # the repository root, where this file is
```

Every `cd` below is relative to the repository root, and the launchers that
need an absolute path take it from `$REPO`.

Substitute your own interpreter if it lives elsewhere.

### A two minute check that it works

Run this before anything else. It only prints, and it finishes in seconds.

```
cd analysis
$PY table4_cluster_ci.py --help
$PY build_roc_cache.py --help
```

If both print a usage block, you are ready. A message about PyTorch not being
available is normal and harmless; the project runs on CPU.

---

## 3. Layout and flow

### Directory map

| Location | What lives there |
|---|---|
| `analysis/` | analysis scripts, the campaign arms, and the pipelines |
| `analysis/frozencal67_pipeline_17/` | the pipeline that produced every reported number |
| `analysis/arms_*/` | one campaign each, holding its input bundle and its results |
| tree root | the launchers that drive the ns-3 simulator |
| `analysis/hp_selected/` | the selected hyperparameters, one parameter file per configuration |
| `fpntbh347/` | the second defense: its own simulator and analysis |

Inside one arm:

```
arms_r34_17feat_frozencal67/
  listener/
    colab_data/                  the measurement bundle, one CSV per configuration
    results/
      static/   results.csv, best_model_Stacking_Ensemble.pkl, confusion_matrix.csv
      mobile/   the same for the mobile configuration
      roc_cache/                 prediction caches for the ROC figure
```

### How the parts fit together

```
  a campaign launcher (Section 6)            writes a simulation tree
            |
            v   features_{static,mobile}/
            |
            +---> make_arm_bundles.py        the tree becomes an arm bundle,
            |                                ARM/listener/colab_data/
            |
            v
  hp_selected/{static,mobile}/best_params.json   the selected hyperparameters (Step 1)
            |
            v
  defense_detection_v2.py   (via run_arm.py) trains and scores the campaign
            |
            v   results.csv  +  best_model_*.pkl
            |
            +---> table4_cluster_ci.py       confidence intervals for Table 4
            +---> build_roc_cache.py -> plot_roc_17.py      Figure 1
            +---> k_sweep_universal4_v2.py -> plot_figure1_optimal_k_17.py   Figure 2
            +---> r45_confusion_universal3_t16.py           Figure 3
            +---> cost_sensitive_report.py   the cost-sensitive appendix table
            +---> inference_cost_report.py   deployment cost
```

The order matters. Four analyses read the extended grids and cannot run before
they exist: `k_sweep_universal4_v2`, `feature_importance_sensitivity_v2`,
`feature_eng_ablation_v2` and `instability_ablation_v2`.

### What a copy must contain

A copy of this project is complete for the purposes of this guide when it carries:

| Item | Why it is needed |
|---|---|
| `analysis/frozencal67_pipeline_17/` | the pipeline that produced every reported number; Step 0 points at it |
| `analysis/thrbias/make_supplementary_table.py` | builds the appendix feature table and runs the `--verify` check of Step 10 |
| `analysis/thrbias/rename_67.py` | emits the appendix block |
| `analysis/thrbias/audit_cleanup.py` | the exhaustive recomputation of Step 10 |
| `analysis/thrbias/enumerate_engineered.py` | enumerates the generated columns |
| `analysis/arms_r34_17feat_frozencal67/` | the arm behind Section 1's numbers, with its bundle and results |

### Files that appear more than once, and why

Several names occur in more than one place in this package. Each copy exists
because a different step needs it, and using the wrong copy is the commonest way
to get a different number without an error. Every experiment starts from the same
seventeen single-listener observables; what differs is which later steps it goes
through.

**One input, several learnings.** The reported campaign's bundle,
`arms_r34_17feat/listener/colab_data/`, is the input of three arms. The other two
reach it through relative links, so they read the same measurement windows:

| Arm | What is done to that input |
|---|---|
| `arms_r34_17feat` | the feature-level analyses: Tables 5 to 9 and Figures 2 and 3. They work on the seventeen observables directly and train no calibrated detector |
| `arms_r34_17feat_frozencal67` | the reported detector: engineered to 67 columns, calibrated, thresholded (Table 4, Figure 1, Steps 3 to 5 and 8 to 9). The reuse control and the activation-fraction scoring read it too |
| `arms_r34_17feat_nosplit` | the same learning with the ensemble threshold chosen on the whole validation partition, for the threshold-protocol comparison |

Not every experiment needs every step. The propagation calibration of Section 6
applies only to the propagation arms. The frozen-detector scoring applies only to
the activation-fraction arms. The feature-importance and feature-size analyses
calibrate nothing. That these arms read files of the same shape is intended.

**Three copies of the pipeline.** All three compute the same seventeen observables.
They differ in the columns they generate, and each is needed by a named consumer:

| Directory | Used by |
|---|---|
| `analysis/frozencal67_pipeline_17/` | the reported campaign and everything scored from it. Generates the 67 columns of Appendix C; point `DCFM_PIPELINE` here unless a step below says otherwise. Every detector the paper reports was learned with it, and a stored scaler accepts only the column count it was fitted on, so score those arms with this copy |
| `analysis/arms_r34_17feat/pipeline_17/` | the feature-level stages of `arms_r34_17feat`, which use the seventeen observables without engineering |
| `analysis/cleanfeat/pipeline_17/` | the default of four tree-root launchers (`run_gm10k17_all.sh`, `run_hidden.sh`, `run_prop17.sh`, `run_r41_windoworder_frozencal.sh`). The reported cells of the last three were re-scored under `frozencal67_pipeline_17`, as Step 3 describes |

The generator announces itself: every learning log prints `Created 67 features
(from 18)`. A log that prints `Created 77 features` came from an earlier copy of
the generator that does not ship; no reported number was learned from one. The
eighteenth input is the window's duration, which turns counts into rates and is
not an observable.

**Other duplicated names:**

| Name | Why there are two |
|---|---|
| `run_arm.py` in `analysis/` and in `analysis/arms_r34_17feat/` | Step 3 and `marathon_frozencal.sh` use the second, which runs the listener arm only. `run_prop17.sh` and `run_r41_windoworder_frozencal.sh` call the first, both with `--arm listener` |
| `runtime_guard.py`, the same two directories | a plain import resolves beside the running script, so each `run_arm.py` needs its own copy |
| `paper_names.py`, `select_optimal_k.py`, the same two directories | byte-identical copies, kept so that scripts in either directory find them |
| `universal_set.txt`, two files | the four-feature ranking anchor that the size sweep starts from, and the three features that the sweep selects (Step 6) |
| `scratch/iolsr-tests-corrected.cc`, one per ns-3 tree | the fictitious-node campaign and the second defense each have their own scenario file; build each in its own tree |
| `supplementary_features.csv` | the one under `thrbias/out67/` is the 67-column table; no other copy ships |

**Column names.** The bundles label the seventeen observables with the canonical
names used internally, and several of those names are borrowed from network-wide
metrics (`AverageMprCount`, `FlowThroughputStd`). Every value in a bundle is
computed from frames received by the single listener; `paper_names.py` maps each
column to the name the paper uses (Step 10).

The quickest way to tell a complete copy from an incomplete one is Step 10's `--verify`,
which rebuilds the 67 engineered columns from the 17 observables and checks each closed-form
definition against the column the pipeline produced.

---

## 4. Step by step

### Step 0. Point at the right pipeline

`run_arm.py` silently falls back to a different, earlier feature space when the
environment variable `DCFM_PIPELINE` is not set. Always set it.

```
cd analysis
export DCFM_PIPELINE=$PWD/frozencal67_pipeline_17
```

### Step 1. The hyperparameters

The configurations the paper's analyses use were selected in an earlier round
of the study by a unified search (Section 5.2 and the appendix tables). The
search is not part of this package; its result is, as a parameter file:

```
analysis/hp_selected/static/best_params.json
analysis/hp_selected/mobile/best_params.json
```

Each file maps a model name to the parameters the search selected for that
configuration, as `get_params()` returned them. Every analysis that takes
`--hp-results-dir` reads `<dir>/{static,mobile}/best_params.json`; `run_arm.py`
passes `analysis/hp_selected` unless `DCFM_HP_RESULTS` says otherwise. Each
analysis keeps only the parameters of the model it builds (`k_sweep`,
`feature_eng_ablation` and the confusion matrices: CatBoost;
`feature_importance_sensitivity`: CatBoost and XGBoost).
`analysis/hp_search_spaces.py` holds the grids the search drew from, as data;
`catboost_retune.py` searches them again on the reported campaign for the
re-tuning control of Section 5.2. `analysis/dump_hp_params.py` is the tool
that wrote the parameter files from the search's own output.

### Step 2. (merged into Step 1)

There is no separate grid-extension step: the parameter file already holds
the configurations selected on the extended grids.

### Step 3. The scoring run

Every accuracy in the paper comes from this one command. What changes between
them is which campaign `DCFM_ARMS_ROOT` points at; the script, the pipeline and
the four learn flags never change.

```
cd analysis
MAX_JOBS=8 OMP_NUM_THREADS=2 \
DCFM_ARMS_ROOT=$PWD/ARM \
DCFM_PIPELINE=$PWD/frozencal67_pipeline_17 \
$PY -u arms_r34_17feat/run_arm.py \
    --arm listener --script defense_detection_v2.py --mode static --force \
    -- --no-augmentation --split-validation --add-linearsvc --calibrate-stacking
```

Repeat with `--mode mobile`. This is the expensive step: several hours per
configuration, so run at most two at a time.

`runtime_guard.py` must stay beside the script that imports it. Both this step
and Step 4 import it as their first statement, before numpy is loaded, because it
is what sets the thread and `MAX_JOBS` caps for the run; there is no fallback, so
a missing copy stops the command rather than degrading it. It is a module and not
an entry point, so it is never invoked directly. Two copies travel with the
package, `analysis/runtime_guard.py` and `analysis/arms_r34_17feat/runtime_guard.py`,
because a plain `import` resolves against the directory of the script being run:
this step runs `arms_r34_17feat/run_arm.py` and Step 4 runs
`analysis/table4_cluster_ci.py`.

#### What to put in ARM, and what you get

| ARM | Yields |
|---|---|
| `arms_r34_17feat_frozencal67` | the headline result and every row of Table 4 |
| `arms_propmodel_rl42_17` and `arms_propmodel_rl42_baseline_17` | the propagation comparison at the route-viability calibration, Table 10 |
| `arms_propmodel_17` and `arms_propmodel_baseline_17` | the same comparison at the degree-matched calibration |
| `arms_hidden_rts100` and `arms_hidden_rts100_baseline` | the RTS/CTS comparison, Table 10 |
| `arms_windoworder/canonical_17_17` and `arms_windoworder/shuffled_17_17` | the window-order control |
| `arms_perwindowseed_17` | the independent-window control |
| `arms_fractionsweep/f{0.25,0.5,0.75}_{static,mobile}` | the defense-activation-fraction arms |
| `arms_trafficsweep/i{1.0,0.5}_{static,mobile}` | the traffic-density sweep |
| `arms_windowsweep/w{10,20}_17` | the window-duration arms |
| `arms_gm10k_17` | the Gauss-Markov mobility control |
| `arms_r34_17feat_2k` | a size-matched campaign, for comparing against a 2,000-run arm rather than the 10,000-run one |
| `arms_r34_17feat_nosplit` | the reported campaign with the ensemble threshold chosen on the whole validation partition instead of its threshold-selection half: the comparison in the threshold and calibration section. Score it with the `nosplit` flagset of `marathon_frozencal.sh` and `PIPE` set to `analysis/frozencal67_pipeline_17` |
| `arms_gm_control_17` | the 2,000-run static anchor at the reported settings, against which the window-length and traffic-density arms are paired |

Each arm must exist before it can be scored. Section 6 lists the launcher that
builds each one.

#### From a campaign tree to an arm

Step 3 scores an arm, and an arm is a bundle rather than a simulation tree. The
launchers of Section 6 write simulation output; this is the step that turns one
of those trees into the input the pipeline reads.

```
cd analysis
DCFM_SIM_ROOT=$REPO/CAMPAIGN_TREE \
DCFM_ARMS_ROOT=$PWD/ARM \
$PY -u make_arm_bundles.py --arm listener
```

Both variables must be given. `DCFM_SIM_ROOT` is the tree the launcher wrote,
and it must contain `features_{static,mobile}/<scenario>/*.csv`; left unset it
points at `simulations_v347/`, the tree admitted with the distance condition,
which is not the reported population (Section 6 builds that one). `DCFM_ARMS_ROOT` is the arm to create; left unset
it points at `analysis/arms`, which is not where any arm named in this guide
lives. The bundle is written to
`ARM/listener/colab_data/wide_{static,mobile}.csv.gz`, which is the path Step 3
reads with the same `DCFM_ARMS_ROOT` and the same `--arm listener`.
`--mode` restricts the build to one configuration and is otherwise unnecessary.

A bundle holds one row per measurement window plus a header, so a campaign of
`N` runs gives `4N + 1` lines:

```
zcat ARM/listener/colab_data/wide_static.csv.gz | wc -l
```

Leave `--limit` alone. It cuts by file-discovery order, which is lexicographic
and unrelated to the manifest, so it is a rehearsal switch and not a way to take a
subset of a campaign. `make_2k_subset.py` in Section 6 is what takes a subset
by manifest order.

#### The four learn flags

Everything after the bare `--` is forwarded verbatim to the pipeline. These four
are part of the definition of the run, and **each one defaults to off**, which is
an older behaviour:

| Flag | Effect |
|---|---|
| `--no-augmentation` | train on the raw training data only |
| `--split-validation` | split the validation partition into two grouped halves, calibrate on one and choose the threshold on the other |
| `--add-linearsvc` | add the linear SVM to the model list |
| `--calibrate-stacking` | tune the ensemble threshold instead of leaving it at 0.5 |

#### Running many arms in one go

`marathon_frozencal.sh` wraps the same command in a queue, two at a time, checks
two gates per run and appends one manifest row per run.

```
cd analysis
PIPE=$PWD/frozencal67_pipeline_17 \
QUEUE_OVERRIDE="ARM_A|campaign ARM_B|campaign" \
bash marathon_frozencal.sh
```

`PIPE` is mandatory. Each queue entry is `arm_root|flagset`, where the flagset is
`campaign` for the four flags above or `nosplit` to omit `--split-validation`.
Pass `--dry-run` as the first argument to print the plan without running it.

**Acceptance gates.** A run counts only if its log contains a
`[frozen-calibration] N/N` line with both numbers equal, and a `FINAL RESULTS`
table. A failed gate stops the queue.

#### The one campaign that is scored differently

The second defense lives in its own tree and its launcher stages the campaign
before scoring it, so it is invoked through that launcher rather than through
`run_arm.py` directly. Underneath it calls the same
`defense_detection_v2.py` with the same four flags.

```
cd "$REPO/fpntbh347"
PIPELINE=$REPO/analysis/frozencal67_pipeline_17 \
ARMS_ROOT=$REPO/fpntbh347/analysis/arms_fpnt_17_frozencal \
bash run_fpnt_arm_all.sh
```

Two defaults will silently give you something else: `PIPELINE` defaults to
unset, and unset makes the launcher fall back to a 33-metric pipeline;
`ARMS_ROOT` defaults to `analysis/arms_fpnt`, an earlier campaign. `ARMS_LIST`
defaults to `listener`, which is the single-vantage arm and the reported one, so
leave it alone. The launcher refuses to start if `PIPELINE` names a directory
with no `defense_detection_v2.py` in it.

The launcher runs in phases, and `PHASE=<name>` runs a single one. Its first
simulating phase, `pools`, calls `build_fpnt_mobile_pool.sh` once per
configuration (`MOB=0`, then `MOB=1`), so that script must sit beside it. It
re-admits the reported campaign's `C_all` seeds, in manifest order, under the
black-hole attack, because a pool admitted under the isolation attack is not
valid for another attack: the attacker is placed by a different allocator and
the same seed lands on a different topology. It keeps a seed when all four of
its windows produce output, until 2,000 are admitted, and records each seed's
own outcome in `admission_log.csv`. The `campaign` phase that follows refuses to
start without the pool manifest.

Yields `analysis/arms_fpnt_17_frozencal/listener/results/{static,mobile}/results.csv`
and a timestamped log at the tree root. Expect 0.9488 static and 0.9394 mobile
for the stacking ensemble.

**Why this second campaign exists.** The first defense was proposed by the
authors of the paper, so a natural question is whether detectability is a
property of that one mechanism rather than of routing-layer defenses in general.
The second defense is an independently proposed trust-based mechanism, evaluated
under a different attack, and it is put through the identical pipeline so that
what differs between the two results is the setting and not the method.

Two analyses belong to that campaign alone. Both are run from the tree root.

The first reports the direction-free single-feature AUC of each observable. It
prints to standard output and writes no file, so redirect it if you want to keep
the figures:

```
cd "$REPO/fpntbh347"
bash fp_importance.sh | tee fp_importance_$(date +%Y%m%d_%H%M%S).log
```

The wrapper runs `fp_importance.py` from beside itself, under `$PY` if you have
exported one and `python3` otherwise, and forwards a bundle directory if you
give it one.

Mean frame size is the strongest single observable, at 0.8759 static and 0.8572
mobile, and three of the five strongest are byte-volume quantities.

The second is the ablation. It exists because this defense attaches metadata to
every topology-control message, so it adds bytes to the control plane. Rebuilding
the bundle with every byte-volume column removed and running the same pipeline on
it separates a result about control-plane behaviour from a result about message
size.

Every variable in the block below must be passed. Each default reproduces the
earlier August behaviour instead: the earlier source bundle, an output root that
overwrites the earlier results, a 33-metric pipeline, one learn flag instead of
four, and 22 jobs.

```
cd "$REPO/fpntbh347"
SET=volume \
SRC=$REPO/fpntbh347/analysis/arms_fpnt_17_frozencal/listener/colab_data \
ARMS_ROOT=$REPO/fpntbh347/analysis/arms_fpnt_17_frozencal_ablate_volume \
FULL_REF=$REPO/fpntbh347/analysis/arms_fpnt_17_frozencal \
PIPELINE=$REPO/analysis/frozencal67_pipeline_17 \
LEARN_FLAGS="--no-augmentation --split-validation --add-linearsvc --calibrate-stacking" \
MAX_JOBS_LEARN=8 OMP_NUM_THREADS=2 \
bash fp_ablate.sh
```

Prefix that same block with `PHASE=build` to rebuild the ablated bundles without
scoring them. `SRC` is the processed-data bundle of the reported campaign, which ships beside it.

Yields
`analysis/arms_fpnt_17_frozencal_ablate_volume/listener/results/{static,mobile}/results.csv`
and a timestamped log at the tree root. Expect 0.9369 static and 0.9238 mobile
for the stacking ensemble, against 0.9488 and 0.9394 with the byte-volume columns
present.

Four arms of this campaign are on disk and two pairs of them agree to four
decimal places, so read the arm name off the path before quoting a figure. The
earlier ablation reads 0.9394 static, which is also the mobile figure of the
reported full arm; the earlier full arm reads 0.9369 mobile, which is also the
static figure of the reported ablation.

Feature selection, model fitting, calibration and threshold selection are all
carried out on this campaign. Nothing is transferred from the first defense, and
because the two defenses face different attacks the two sets of numbers describe
two settings rather than a head-to-head comparison.

### Step 4. Table 4, the confidence intervals

```
cd analysis
$PY -u table4_cluster_ci.py --mode static \
    --results-root arms_r34_17feat_frozencal67/listener/results \
    --n-boot 2000 --seed 42
$PY -u table4_cluster_ci.py --mode mobile \
    --results-root arms_r34_17feat_frozencal67/listener/results \
    --n-boot 2000 --seed 42
```

Resampling is at the simulation-run level, so all four measurement windows of a
run stay together. Writes `table4_ci/table4_ci.csv` under the results root.

Note when reading that file: the static linear SVM row of Table 4 is the
`SVM_linear_C0.01` row of its `rescore` block, not the `results.csv` row. That
block refits the campaign's linear SVM at the `C` that Section 5.2 reports, on the
reported arm, with the 67-column generator.

The value of `C` in that row comes from the sweep Section 5.2 describes, which
answers the reviewer's question directly -- whether an implementation that does
not hit the kernel solver's convergence failure keeps performing at high `C`:

```
cd analysis/arms_r34_17feat
$PY -u linearsvc_csweep.py --mode both
```

`DCFM_PIPELINE` must be the Step 0 value: without it the script falls back to
`arms_r34_17feat/pipeline_17`, whose calibration and column count are not the
campaign's. For each `C` in `[1e-3, 1e6]` it fits a raw `LinearSVC` on the
campaign's own splits with the iteration limit raised tenfold, records whether that
fit converged and its `n_iter_`, and selects `C` by AUC on the validation half among
the converged fits only. The bundle, the feature engineering, the run-grouped split,
the scaler and the feature selection all come from the pipeline itself rather than
from a reimplementation. It retrains none of the other sixteen models and never
touches `results.csv`; everything it writes goes to
`listener/results/<mode>/linearsvc_csweep/`. Expected: `C* = 0.01` under static
conditions and `C* = 1` under mobility, the two values Section 5.2 reports, with the
log printing `Created 67 features (from 18)` for each mode.

### Step 5. Figure 1, the ROC curves

Two steps, a cache and then a plot.

```
cd analysis
R23_PIPELINE=$PWD/frozencal67_pipeline_17 \
  $PY -u build_roc_cache.py --arm arms_r34_17feat_frozencal67
$PY -u plot_roc_17.py \
    --cache-root arms_r34_17feat_frozencal67/listener/results/roc_cache \
    --out arms_r34_17feat_frozencal67/listener/results/roc_cache/roc_curves.png
```

`R23_PIPELINE` and `--arm` are both given explicitly, and both must name the
67-column frozen pipeline and the arm Figure 1 was produced from. Written this
way the command is self-contained and does not rely on either default inside the
script. A mismatch between the two is caught rather than quietly plotted: the
script aborts with `scaler expects N columns, rebuilt frame has M` when the
pipeline and the arm disagree. The plot prints the AUC range over all seventeen
models when it finishes; that printed range is what the figure caption quotes.
Expect 0.8841 to 0.9414 static and 0.9169 to 0.9729 mobile.

### Step 6. The feature-set-size sweep, and Figure 2

```
cd analysis
$PY -u frozencal67_pipeline_17/k_sweep_universal4_v2.py \
    --static-root  PATH/TO/features_static \
    --mobile-root  PATH/TO/features_mobile \
    --hp-results-dir analysis/hp_selected \
    --out-dir      PATH/TO/out/k_sweep_universal4_v2 \
    --k-list 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17 \
    --universal-set AverageAdvertisedLinksPerTCMessage,AverageMprCount,TcMessageRate,AverageHopCount \
    --n-seeds 20
```

Two arguments here are easy to get wrong and both change the result silently:

* `--universal-set` is **required**. The set built into the script is an earlier
  four-feature one, so omitting the flag silently sweeps the wrong anchor.
* `--k-list` defaults to `4,5,9,13,15,17,24,33`, which is the submitted K set.
  Pass `1,...,17` as above.

Note that the anchor names are the data column names. `AverageMprCount` is the
column behind the quantity the paper calls `AdvertisedLinksPerKnownNode`.

Two files in the results tree are called `universal_set.txt` and they are not
the same set. `feature_importance_sensitivity_v2/universal_set.txt` holds the four
features above, the ranking-depth intersection this sweep is anchored on, and it is
what the runner passes when it invokes the sweep for you.
`feature_importance_sensitivity_v2/universal_set_k3/universal_set.txt` holds the three
the sweep then selects, and that is the subset the paper reports. Neither supersedes
the other: the first is the input to the size sweep and the second is its result.

Writes `k_sweep_results.csv` and `k_sweep_summary.txt`. Then:

```
$PY -u plot_figure1_optimal_k_17.py \
    --in-csv  PATH/TO/out/k_sweep_universal4_v2/k_sweep_results.csv \
    --out-dir PATH/TO/out/k_sweep_universal4_v2 \
    --optimal-k 3
```

`--optimal-k` defaults to 4. The reported figure marks 3, so pass it.

To apply the selection rule rather than read the sweep by eye:

```
$PY -u select_optimal_k.py \
    --k-sweep-csv PATH/TO/k_sweep_results.csv \
    --metric acc --out PATH/TO/optimal_k.json
```

`--tolerance` defaults to the standard deviation at the best K, which is the
one-standard-deviation rule the paper uses.

### Step 7. Figure 3, the confusion matrices

```
cd analysis
$PY -u r45_confusion_universal3_t16.py \
    --static-root PATH/TO/features_static \
    --mobile-root PATH/TO/features_mobile \
    --hp-results-dir analysis/hp_selected \
    --out-dir PATH/TO/out/confusion_matrices_universal3_t16 \
    --n-seeds 20
```

Writes `confusion_matrices.png` and one CSV per panel.

### Step 8. The cost-sensitive table

```
cd analysis
$PY -u cost_sensitive_report.py \
    --results arms_r34_17feat_frozencal67/listener/results/static/results.csv \
    --model Stacking_Ensemble --sweep
```

`--sweep` also traces cost against threshold and needs the stored model file.

### Step 9. Deployment cost

```
cd analysis
$PY -u inference_cost_report.py \
    --arm-root arms_r34_17feat_frozencal67 --mode static \
    --model Stacking_Ensemble --repeats 100 \
    --out-dir PATH/TO/out/inference_cost
```

Add `--all-models` for the frontier over all seventeen classifiers.
`--threads 1` pins the process to one core, which gives a less parallel
reference point on the same machine; it is not an emulation of target
hardware, and no latency for other hardware follows from it. Resident memory
is measured separately:

```
$PY -u measure_deployed_rss.py \
    --arm-root arms_r34_17feat_frozencal67 --mode static \
    --model Stacking_Ensemble --calls 100
```

Both measurements also have a launcher that runs the whole set in one log,
and that is how the reported figures were produced. Each takes the arm and the
pipeline from the environment, and each defaults to an earlier campaign, so
give both explicitly:

```
cd analysis
ARM_NAME=arms_r34_17feat_frozencal67 \
  R23_PIPELINE=$PWD/frozencal67_pipeline_17 \
  bash run_r319_cost.sh

ARM_NAME=arms_r34_17feat_frozencal67 \
  R23_PIPELINE=$PWD/frozencal67_pipeline_17 \
  bash run_r319_rss.sh
```

`run_r319_cost.sh` makes four calls to `inference_cost_report.py` -- static and
mobile, each at full width and again on one pinned core -- and writes
`inference_cost_out/<arm>/frontier/`. `run_r319_rss.sh` calls
`measure_deployed_rss.py` for both modes and writes
`inference_cost_out/<arm>/deployed_rss/`. Each leaves a timestamped log beside
the arm. Neither fits anything: every estimator, scaler, feature list and
threshold comes from the arm's stored pickle.

`run_r319_cost.sh` refuses to time anything while the machine's one-minute
load average is above 4, so run it on an idle machine.

#### A note on the latency figures

For inference latency, the canonical stacking-ensemble values reported in the
paper are those in `inference_cost_{mode}_Stacking_Ensemble.csv`, obtained from
the dedicated single-model measurement. The corresponding entries in
`inference_cost_frontier_{mode}.csv` were measured later in the same process as
part of the full 17-model sweep and therefore need not reproduce the dedicated
timing exactly. The frontier is provided for comparative deployment-cost
analysis across classifiers; its stacking entry should not be used to replace
the dedicated latency reported in the paper.

The `_t1` files in the same directory document a single pinned core and are a
separate measurement.

#### A note on the memory figures

The resident-memory values reported in the paper come from
`measure_deployed_rss.py`. It extracts the deployed artefact -- the ensemble,
the scaler, the selected feature names and the threshold -- and measures a fresh
child process that loads that artefact and nothing else, reading the current
resident set from `/proc/self/statm` at the moment of measurement.

The `peak_rss_mb` column in the inference-cost CSVs is a different quantity in
two independent ways, and is not the source of those values. It is
`resource.getrusage(RUSAGE_SELF).ru_maxrss`, a high-water mark over the life of
the measuring process rather than a reading at one point in it; and that process
has loaded the whole results pickle, which carries every stored model, not the
deployed artefact alone. The `results_pkl_mb` column beside it records which
file was loaded. Both are correct measurements, of different things, and neither
contradicts the other.

`measure_deployed_rss.py` also records a `ru_maxrss_inherited_mb` field. Linux
carries the high-water mark across `fork` and does not reset it at `execve`, so
that field describes the parent process rather than the child being measured. It
is named that way for that reason and is not a deployment figure.

### Step 10. The appendix feature table

The appendix of the paper lists all 67 engineered columns, each with a one-line
definition and the configuration that retains it. The pipeline builds those
columns under the canonical names it uses internally -- `analysis/arm_spec.py`
defines that namespace, and every arm is fed the same column names from it so
that the pipeline is identical across arms -- and the paper prints a
different name for twelve of the seventeen base observables: a canonical name is
often the name of the network-wide metric that a single-vantage quantity
replaces, and the two are not the same measurement. `paper_names.py` holds the
mapping between them, and this step applies it.

```
cd analysis/thrbias
$PY -u rename_67.py
```

With no flags it prints the 67 columns grouped by family and writes nothing.
`--tex FILE` writes the appendix table as a LaTeX `longtable`, which is the form
the paper carries; `--out FILE.csv` writes the same rows as CSV.

```
cd analysis/thrbias
$PY -u rename_67.py --tex feature_table.tex
```

It reads `out67/supplementary_features.csv` and `../paper_names.py` by default,
both resolved against the script itself. `--table` and `--names` override either.

Every script in this section resolves its inputs the same way, and the whole chain has
been run from a tree containing nothing but the shipped files: it regenerates the
released feature table byte for byte and reproduces Appendix~C of the paper line for
line.

#### Where the table comes from

`rename_67.py` is the last of four steps, not the whole chain. Each step has its own
script, and the table it hands on is the previous one's output:

```
processed bundle  (listener/colab_data/wide_{static,mobile}.csv.gz)
        |  enumerate_engineered.py <bundle> <mode> eng_<mode>.csv
        v
eng_static.csv, eng_mobile.csv
        |  make_supplementary_table.py eng_static.csv eng_mobile.csv out.csv
        v
out67/supplementary_features.csv        (canonical names, 67 rows)
        |  rename_67.py --tex feature_table.tex
        v
Appendix C of the paper
```

To rebuild the table from the shipped data:

```
cd analysis/thrbias
$PY -u enumerate_engineered.py \
    ../arms_r34_17feat/listener/colab_data/wide_static.csv.gz static eng_static.csv
$PY -u enumerate_engineered.py \
    ../arms_r34_17feat/listener/colab_data/wide_mobile.csv.gz mobile eng_mobile.csv
$PY -u make_supplementary_table.py eng_static.csv eng_mobile.csv out67/supplementary_features.csv
```

`enumerate_engineered.py` imports the pipeline module the reported campaign runs and
applies it to the bundle, so the column set it reports is the one the campaign built,
not a transcription of it. It resolves the pipeline against its own location;
`ENG_PIPELINE` overrides that if you need a different copy. The two `eng_*.csv` files
are intermediates and are not shipped -- the commands above regenerate them.

#### What the table's columns mean

`out67/supplementary_features.csv` carries one row per engineered column and seven
fields:

```
feature      the column's canonical name, as the pipeline emits it
kind         its group: base, log1p, sqrt, power, network-performance
             indicator, or row-wise
formula      how it is computed, in canonical names, e.g.
             TcMessageRate * AverageAdvertisedLinksPerTCMessage
description  a short gloss on the group, not a per-column definition
constant     true where the column holds one value throughout this data
survives_variance_correlation_filters_static
survives_variance_correlation_filters_mobile
```

The names in this file are canonical, not the paper's. Appendix C of the paper uses the
paper's names and carries the per-column definitions; `rename_67.py` maps between the two
through `../paper_names.py`.

The last two fields record one thing: whether the column passed the variance filter and
the |r| > 0.95 correlation filter, in that configuration. They are named for the
computation they record and are **not** defined as final-retention flags -- selection has
a further stage, a ranking across four importance criteria capped at
`n_features_target`.

In the reported campaign the two coincide, so the file can also be read as the
retained-column lists. The cap is 150 while the post-filter sets hold 28 columns under
the static configuration and 32 under mobility, so the ranking stage removes nothing
further. Counting the two fields in the shipped file returns 28 and 32, and `constant` is
true for 7 rows.

Every script in the chain resolves the pipeline against its own location and takes the
same `ENG_PIPELINE` override, and all of them name `frozencal67_pipeline_17` -- the
generator the reported campaign ran. That matters for the check below: pointed at any
other copy of the generator it would verify a feature space the paper does not report.

One thing in the chain is written rather than computed, and it is worth knowing which.
The per-column **formula strings** in `make_supplementary_table.py` are transcribed
from the feature-engineering function rather than parsed out of it, which its own
header explains: a parser would go stale silently. The safeguard is that the script
carries a `--verify` mode which recomputes every non-row-wise formula from a bundle and
compares it against what the pipeline produced, so a drift between the table and the
code is caught rather than assumed away:

```
cd analysis/thrbias
$PY -u make_supplementary_table.py --verify \
    ../arms_r34_17feat/listener/colab_data/wide_static.csv.gz
```

It should report `Created 67 features (from 18)` and then `all formulas reproduce`. Both
lines are worth reading, not just the second: 67 is the reported feature space, so a run
that announces 77 has picked up one of the other pipeline copies and is verifying a space
the paper does not report. The `from 18` is explained below.

So the column set and the retention flags are computed; the wording of each formula is
transcribed and machine-checked against the code.

#### The exhaustive check

`--verify` above samples. The exhaustive check is a separate script, and it is the one
the reproducibility claim rests on: it recomputes **every column with a closed-form
definition** from the base observables and compares each against the column the pipeline
produced. It also reports the structural side -- how many generated columns would have
inputs the observable space does not carry, and whether feature selection retains the
same names either way.

```
cd analysis/thrbias
$PY -u audit_cleanup.py \
    ../arms_r34_17feat/listener/colab_data/wide_static.csv.gz \
    ../arms_r34_17feat/listener/colab_data/wide_mobile.csv.gz
```

For each bundle it should report `structurally undefined : 0`, then
`definitions recomputed : 41 of 67 columns` and `all 41 recomputed definitions match at
<1e-9`. The 26 it does not recompute are not formulas: 17 are the measured observables
and 9 are row-wise statistics aggregated over the whole base set. The other three
row-wise columns -- `row_range`, `row_iqr` and `row_cv` -- are expressions over named
columns and are recomputed with the rest.

It resolves the pipeline exactly as the scripts above do, and takes the same
`ENG_PIPELINE` override.

One line in the pipeline log looks like a contradiction and is not. It reports the
engineered space as built `from 18`, while the observation model has 17 observables.
The eighteenth is `_measurement_duration`, a helper the preprocessing step adds to
turn counts into rates. It is not an observable, it is not a model input, and it is
dropped before the engineered space is formed.

One name is deliberately left alone. `AverageHopCount` keeps its name in the
paper although the mapping would rename it, so the script carries an explicit
override for that one column.

---

## 5. The supporting analyses

All of these read the campaign outputs and none of them changes the campaign.
Run them from `analysis/` with `DCFM_PIPELINE` set as in Step 0.

Their arguments fall into four shapes. Each shape is given once below, followed
by what an individual script adds to it and by the reason to run it: without
that script, the named part of the paper cannot be regenerated.

### Shape A: reads the feature directories and the extended grids

```
$PY -u frozencal67_pipeline_17/SCRIPT \
    --static-root    PATH/TO/features_static \
    --mobile-root    PATH/TO/features_mobile \
    --hp-results-dir analysis/hp_selected \
    --out-dir        PATH/TO/out/SCRIPT_NAME
```

| Script | Adds | What in the paper needs it |
|---|---|---|
| `feature_importance_sensitivity_v2.py` | `--n-seeds`, `--threshold` (stability fraction, 0.8), `--k-min`, `--k-max` | Table 7, the recommended ranking depth per criterion, and the finding that all six criteria agree on the same three observables |
| `instability_ablation_v2.py` | `--cohens-d-csv`, **required**, pointing at the output of `compute_cohens_d_v2.py` below; `--n-seeds`, `--base-seed` | Table 9, and the result that cross-configuration instability does not predict transfer harm |
| `feature_eng_ablation_v2.py` | `--n-seeds`, `--base-seed` | a control on the feature-engineering step. No table depends on it; it ships because the engineering code is published |

`k_sweep_universal4_v2.py` and `r45_confusion_universal3_t16.py` share this shape
too; they are given in full in Steps 6 and 7 because each needs an argument that
changes the result silently.

### Shape B: scores an arm's stored results, one mode at a time

```
$PY -u SCRIPT --mode static --results-root ARM/listener/results
```

Repeat with `--mode mobile`.

| Script | Adds | What in the paper needs it |
|---|---|---|
| `lda_classifier.py` | nothing | one of the two rows in the lower block of Table 4 |
| `hinge_svm_baseline.py` | `--max-iter`; 10000 is the campaign setting, 100000 the tenfold cap, and a non-default value writes to its own output directory | the other row of that block, and the statement that neither hinge fit converged |
| `catboost_retune.py` | `--hp-results-dir`, `--n-iter` (80 is the paper's), `--grid`, `--model {catboost,xgboost}`, `--skip-search` to reuse the previous selection | the control showing that reusing hyperparameters from an earlier round changes accuracy by less than one point |

`table4_cluster_ci.py` shares this shape and is given in Step 4.

### Shape C: takes a whole arm

```
$PY -u SCRIPT --arm ARM
```

| Script | Adds | What in the paper needs it |
|---|---|---|
| `seed_bootstrap_ci.py` | `--n-boot`, `--seed` | quantities reported as a mean over repetitions rather than over runs |
| `separability_run_bootstrap.py` | `--n-boot`, `--seed` | the confidence interval on the difference column of Table 6 |
| `reuse_control.py` | `--mode {static,mobile,both}` | the two control figures in the threshold and calibration section |

`build_roc_cache.py` shares this shape and is given in Step 5.

### Shape D: takes the two feature directories directly

Note the flags here are `--static` and `--mobile`, not the `-root` forms of
Shape A.

```
$PY -u frozencal67_pipeline_17/SCRIPT \
    --static PATH/TO/features_static \
    --mobile PATH/TO/features_mobile \
    --out    PATH/TO/out/SCRIPT_NAME
```

| Script | Adds | What in the paper needs it |
|---|---|---|
| `compute_cohens_d_v2.py` | `--jobs` | Table 5, the per-feature separability table. Its CSV is also the required input of the instability ablation in Shape A |
| `compute_separability_v2.py` | `--jobs`, and two flags spelled with underscores, `--n_bootstrap` and `--n_permutations`, plus `--seed` | Table 6, the joint separability table. It is what shows the higher accuracy under mobility is not explained by greater class separation |

### The remaining six, each with its own shape

| Script | Invocation | What in the paper needs it |
|---|---|---|
| `variance_decomposition_v2.py` | `--results-root ARM/listener/results --out-dir OUT` | the variance ratios quoted in the measurement-procedure section |
| `r33_sourceonly_k.py` | `--direction {s2m,m2s} --rank-on-train --n-seeds 20 --k-list 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17 --out-dir r33_sourceonly_17`, run with `R33_ARM=arms_r34_17feat`, `R33_PIPELINE=$PWD/arms_r34_17feat/pipeline_17` and `R33_EXPECT_COLS=17`. **All of these must be given.** The historical defaults of `R33_PIPELINE` and `R33_ARM` name `analysis/pipeline` and `arms_c10k_stackcal`, neither of which this package ships, and the script resolves `R33_PIPELINE` at import time, before any argument is read. `R33_EXPECT_COLS=17` guards against a silent return to another feature space, `--k-list` otherwise runs to 21, which does not exist among seventeen observables, and `--out-dir` otherwise writes beside an earlier, non-canonical copy | the source-only control, which shows the three-feature subset does not depend on having seen both configurations |
| `select_optimal_k.py` | `--k-sweep-csv K_SWEEP.csv --metric acc --out optimal_k.json`; `--tolerance` defaults to the standard deviation at the best size, which is the one-standard-deviation rule | the choice of three features, stated as a rule rather than as a judgement |
| `windoworder_by_slot_test.py` | `--arms-root arms_windoworder --mode static --arm canonical_17_17 --model Stacking_Ensemble --out-dir OUT`, then again for `shuffled_17_17` and for `--mode mobile`. **`WO_PIPELINE` must be set** to `$PWD/frozencal67_pipeline_17`; its default is `analysis/pipeline`, which this package does not ship | the by-slot half of the window-order control |
| `r41_byslot_chi2.py` | `--in-dir` pointing at the directory the previous script wrote | the chi-square test over those slots |
| `r2_4_thresholds.py` | `both`. This one is not under `analysis/`; it ships at the repository root | the threshold figures of the threshold and calibration section: the post-hoc bound, the band within 0.1 points of it, and the cost of selecting rather than fixing. Its GATE 1 aborts unless it reproduces `results.csv` exactly, so a run that finishes has already checked itself |

### The subset enumeration

Two scripts sit behind the question of whether three features work because of
which features they are, or only because there are three of them. They are
separate jobs and only the first is expensive:

```
cd analysis/arms_r34_17feat
$PY -u r34_subsets.py          # enumerates every subset and scores it
$PY -u r34_top_20seed.py       # re-checks two named subsets at 20 seeds
```

`r34_subsets.py` enumerates every subset of the seventeen observables at each
size, exhaustively to five and 500 sampled per size at six, seven and eight under
`--sample-rng`, which defaults to a fixed value. Each subset is scored in both
transfer directions and in domain, at `--n-seeds` seeds, three by default. That
is 10,901 subsets and about 65,000 fits, so it is hours of work.

**You do not have to run it.** Its output ships:
`r34_subsets/subsets_k1_2_3_4_5_6_7_8_s3.csv`, one row per subset with the
cross-domain and in-domain accuracies of each. The two selection results read
straight off it: the subset ranked first cross-domain at three features, and
separately the subset with the highest in-domain accuracy within each
configuration, which is the same one. `partial.csv` beside it is a resume file
from the run and carries no separate result.

`r34_top_20seed.py` is the cheap half. It takes the selected subset and the
runner-up the enumeration identified and re-scores just those two at twenty
seeds, which is where the paired figures in the response come from. Running it
alone does not reproduce the ranking, only the confirmation of two subsets.

### Cross-criterion agreement

`analysis/thrbias/criterion_agreement.py` reports how often the three selected
features appear together across importance criteria. It reads
`universal_set_members.csv`, the stability record written by
`feature_importance_sensitivity_v2.py`, which carries one row per combination of
importance criterion, stability variant, subset size and threshold. From it the
tool reports three things: how many cells at a given size return exactly the three
features, how often each of the three is retained at that size, and how often each
survives across every size.

It counts agreement and nothing else. It does not re-derive the subset and it
scores nothing, so its output describes how stable the selection is across
criteria, not how well the subset performs.

Two names differ between the stored record and this paper. The record calls one
observable `AverageMprCount`, which this paper calls `AdvertisedLinksPerKnownNode`,
and another `AvgTxPacketSize`, which this paper calls `MeanSniffedFrameSize`. The
stored files keep the earlier names and the tables use the current ones.

No command is given here: every figure this tool reports can be read directly from
`universal_set_members.csv`.

### The paired mechanism measurement

One analysis does not read an arm at all. It reads the campaign trees, and it
asks what the defense does to the traffic a single observer hears rather than
what a classifier does with it. Each accepted seed contributes a baseline run and
a defense-only run from the identical topology, so the comparison is paired with
topology held fixed, and no attack is present in either arm.

```
cd run_artifacts/20260825_r38_mechanism
$PY -u r38_paired_listener.py | tee r38_paired_listener_$(date +%Y%m%d_%H%M%S).log
```

For every run it takes the single listener's row, the same vantage point the
detector's bundle uses for that run, and forms the defense-minus-baseline
difference of three raw quantities received by that listener: advertised links
per topology-control message, the number of those messages, and the OLSR control
bytes. These are measured before any feature is generated.

Those listener rows ship with the package as `r38_listener_static.csv` and
`r38_listener_mobile.csv` in the same directory: one row per run and window
(baseline and defense-only, 20,000 rows per configuration) with the six raw
quantities, written exactly. The analysis reads them from its own directory, or
from `R38_EXTRACT`. They were produced from the raw campaign trees by
`r38_extract_listener.py`, which is the only step that touches
`features_{static,mobile}/<scenario>/observer_metrics-*.csv`; to regenerate
them from your own simulations, point `R38_STATIC` and `R38_MOBILE` at the two
trees and run it (it refuses to overwrite an existing extract). The analysis
writes nothing, so redirect it to keep the figures. An integer argument reads
only that many runs per window, which is a rehearsal and not a subset of the
campaign. Each bootstrap holds a 10,000 by 10,000 index matrix, so allow about
2 GB of memory.

Expect 10,000 paired seeds in each configuration and the figures of Section 6.3:
0.88, 245.1 and 12,885 under static conditions and 0.94, 403.4 and 19,641 under
mobility, each with a 95 % interval that excludes zero. The script compares every
figure with the manuscript's, prints `OK` or `DIFFERS` beside it, and exits
non-zero on any difference.

The resampling unit is the seed, because observers and measurement windows are
nested inside a run; resampling rows instead would treat those as independent.

## 6. The controlled experiments

### Building the simulator

You need this only to regenerate simulation output; every arm this guide scores
ships with its processed bundle. The simulations use ns-3.47 with one added
routing module and one scenario program, both shipped here:

* `src/iolsr/`, the OLSR implementation that carries the node-isolation attack
  and the fictitious-node defense;
* `scratch/iolsr-tests-corrected.cc`, the scenario: deployment, mobility, the
  traffic flow, the four measurement windows, and the metrics of the passive
  observer.

```
cd ns-3.47                                  # a stock ns-3.47 source tree
cp -r PATH/TO/src/iolsr src/
cp PATH/TO/scratch/iolsr-tests-corrected.cc scratch/
./ns3 configure
./ns3 build iolsr-tests-corrected
```

The launchers are run from the root of that tree and call
`build/scratch/ns3.47-iolsr-tests-corrected-default`, the path a default build
produces. The two shipped sources are the ones the reported campaign was built
from: rebuilt this way from a clean tree and run on the first accepted seed of
each configuration, the binary reproduces the campaign's per-run output files
byte for byte, in all four scenarios. Every experiment in this section is a flag of that one program, among
them `--enforceHopFilter`, `--mobilityModel`, `--propagationModel`,
`--rtsCtsThreshold`, `--scenarioOrder`, `--windowSeconds`, `--udpInterval` and
`--defenseActiveFraction`; `--PrintHelp` lists them all with their defaults.

The second defense has its own tree, `fpntbh347/ns-3.47`, with its own
`src/iolsr/` and scenario file, and is built the same way inside that tree.
Build each tree separately: the two scenario files share a name and are not the
same file.

### The reported campaign's population

The campaign runner admits a run only if the network is connected at the end of
stabilization **and** the traffic source is at least three hops from the victim.
The reported campaign keeps the first condition and drops the second, because
the distance condition is a property of the attack rather than of whether the
defense leaves a footprint. Its population is therefore built in three steps:

```
cd "$REPO"
MODE=static TARGET_ACCEPTED=10000 NUM_WORKERS=22 bash run_campaign_v347.sh
MODE=mobile TARGET_ACCEPTED=10000 NUM_WORKERS=22 bash run_campaign_v347.sh
bash run_nohopfilter.sh
bash run_nohopfilter_mobile.sh
STAGE=$PWD/simulations_v347_hopablation_10k        N=10000 POPS=C_all \
    $PY analysis/build_hopablation.py
STAGE=$PWD/simulations_v347_hopablation_mobile_10k N=10000 POPS=C_all \
    $PY analysis/build_hopablation_mobile.py
```

1. `run_campaign_v347.sh` runs candidate seeds under both conditions until 10,000
   are accepted per configuration, writing `simulations_v347/` with one
   `run_status_<mode>.csv` row per attempt and the reason for every rejection.
2. `run_nohopfilter.sh` and `run_nohopfilter_mobile.sh` recover from the worker
   logs the seeds rejected for distance alone, 3,714 static and 5,055 mobile,
   write them to `rejected_seeds_{static,mobile}.txt`, and re-run exactly those
   seeds with `--enforceHopFilter=0` into `simulations_v347_nohopfilter/`.
   Re-running a seed is deterministic, so each re-run is the run the campaign
   rejected. Connectivity is still enforced.
3. `build_hopablation.py` and its mobile sibling stage `C_all`: the first 10,000
   connected seeds in seed order, whatever their distance, as symbolic links
   into the two trees, with the selection in `manifests/C_all.csv`. Nothing is
   copied. All three values must be passed. The defaults build a separate
   2,000-run staging, whose manifests other experiments draw their seeds from,
   so running the script without them would rebuild those manifests.

The same files are the admission record, and one script turns them into the
figures of Section 4.2:

```
cd analysis
$PY admission_tables.py
```

No search was run under the connectivity-only rule, so its record is replayed:
a seed passes if the campaign accepted it or rejected it for distance alone, and
walking the seeds in the order they were probed until the 10,000th pass gives
Table 1's attempt count. Table 2 compares accepted and rejected seeds on the
same seed range, using the topology probe every attempt writes shortly before
admission is decided. The share of admitted runs whose source comes within two
hops of the victim is counted from the `C_all` manifests. Each figure is printed
beside the manuscript's value and marked `OK` or `DIFF`, and the script exits
non-zero on any difference. It reads `simulations_v347/run_status_<mode>.csv`,
`simulations_v347/topology_probes_<mode>.csv`, the two `rejected_seeds` lists and
the two manifests, and writes nothing.

To build an arm from `C_all`, point
`DCFM_SIM_ROOT` at `simulations_v347_hopablation_10k/C_all` for `--mode static`
and at `simulations_v347_hopablation_mobile_10k/C_all` for `--mode mobile`, as in
Step 3. The links are absolute, so copy these trees with the links dereferenced.

### The paired experiments

Every experiment here is paired: one campaign against its own baseline, two
separately trained detectors, scored on the same simulation runs. Each campaign
is produced by a launcher at the tree root, scored by Step 3, and then compared.

| Launcher | The experiment | What in the paper needs it |
|---|---|---|
| `run_propcal.sh`, `run_propmodel.sh`, then `run_prop17.sh` | log-distance path loss with fading, against the range-based baseline, at two calibrations | the propagation rows of Table 10. It is what establishes whether the result depends on the idealised reception model. The first calibrates, the second simulates, and the third re-learns the resulting bundles; the commands are below |
| `run_hidden.sh` | the same campaign with the RTS/CTS handshake enabled, against the reported configuration in which it never fires | the RTS/CTS rows of Table 10. The reported campaign leaves the handshake off, so hidden-terminal collisions are present in it; this is what isolates their contribution. `rehearse_hidden.sh` beside it is a trial run and does not produce the reported arms |
| `run_windoworder_all.sh` (which drives `run_windoworder.sh` and `run_windoworder_learn.sh`), then `run_r41_windoworder_frozencal.sh` | the four scenarios permuted across the four measurement positions, against the fixed order | the window-order control. The main campaign uses a fixed order, so scenario identity is confounded with elapsed time unless this is run |
| `run_perwindowseed.sh` | one independently seeded window per simulation | the independent-window control. The main campaign takes four measurement windows from each simulation run, so those four share one topology; this campaign takes one window per run instead |
| `run_fraction_all.sh` | the same campaign with the defense permitted to operate for only the first fraction of each measurement window | the activation-fraction sensitivity result, which is what shows how much of the signal survives when the defense is active for part of a window rather than all of it. It does not vary traffic |
| `run_traffic_all.sh` | the same campaign at shorter inter-packet intervals, so that one window carries more datagrams | the traffic-density sweep of Section 4.1, which tests whether eighteen packets per window limits what is observable. Density only: the source emits at a fixed interval, so burstiness is not varied |
| `run_windowsweep.sh` | the same campaign at a shorter measurement window | the window-length sweep of Section 4.2, which is what shows that the reported window is an operating point and not a minimum. Paired against the 40 s campaign by `run_r36_bootstrap.sh` |
| `run_campaign_gaussmarkov.sh`, `run_gm10k17_all.sh` | a second mobility process | the Gauss-Markov control, which bounds the cross-domain claim: transfer between the static and random-walk configurations does not carry over to this mobility process |
| `make_2k_subset.py` | the reported campaign cut to its first 2,000 runs, taken in manifest order | the size-matched comparison beside the second defense, so that two campaigns of different size are not compared directly. It writes `arms_r34_17feat_2k` |

The size-matched arm is cut from the reported campaign's own bundle, and a
read-only check runs first:

```
cd analysis
$PY -u check_2k_subset.py       # read-only: are the wanted ids all present
$PY -u make_2k_subset.py        # writes arms_r34_17feat_2k
```

The rule is **the first 2,000 run ids of the `C_all` manifest**, which is the
same set the density and window sweeps feed the simulator, so every point in
those series is a 2,000-run fit of the same campaign. Do not substitute
`make_arm_bundles.py --limit 2000`. That cuts by file-discovery order, which is
lexicographic, and the manifest's first 2,000 ids span 1 to 1,004,049: the
result is a different, arbitrary 2,000 runs, with no warning and no error.

The window sweep re-runs a fixed subset of the accepted seeds at other window
lengths, so the topology is held constant per seed and the comparison is paired.
Run it once per configuration:

```
cd "$REPO"
MODE=static WINDOWS="10 20" N_SEEDS=2000 bash run_windowsweep.sh
MODE=mobile WINDOWS="10 20" N_SEEDS=2000 bash run_windowsweep.sh
```

`WINDOWS` must be given. The script defaults to `16 28 52 64`, which are not the
windows the paper reports and which build arms nothing else reads. `N_SEEDS`
defaults to 2,000, the size the comparison uses. The 40 s condition is not run
here: it already exists as the main campaign, on the same seeds and the same
binary. Output goes to `simulations_v347_windowsweep/w<W>/` and never touches the
main tree, and a run whose output files are already present is skipped, so an
interrupted sweep can be relaunched.

The four comparisons are then reconstructed from the stored models, not refitted:

```
cd analysis
R36_PIPELINE=$PWD/frozencal67_pipeline_17 R36_SUFFIX=_67 bash run_r36_bootstrap.sh
```

It pairs `arms_windowsweep/w10_17` and `w20_17` against the 40 s arm of each
configuration, which is `arms_gm_control_17` under static conditions and
`arms_r34_17feat_2k_w_17` under mobility, and writes the four intervals under
`r36_window_bootstrap_67/` and `r36_window_bootstrap_mobile_67/`, with a timestamped
log beside them. Two settings
inside it are load-bearing. It scores every arm with the same estimator, because
letting each arm be scored by its own winner measures the estimator rather than
the window. And it pins the feature generator to the module the arms were trained
with; a narrower generator produces fewer columns than the stored scaler expects
and the reconstruction is refused rather than silently rescaled.

The traffic-density sweep shortens the inter-packet interval so that the same
40 s window carries more datagrams, with the seeds held fixed per interval:

```
cd "$REPO"
bash run_traffic_all.sh
```

`INTERVALS` defaults to `1.0 0.5`, the 36- and 72-datagram conditions the paper
reports; the 2 s interval is the reported campaign itself and is deliberately not
re-run. `SMOKE_ONLY=1` stops after a 50-seed rehearsal that checks the binary
really emitted 72 datagrams in a window. Output goes to
`simulations_v347_trafficsweep/i<interval>/` and the arms to
`analysis/arms_trafficsweep/i<interval>_<mode>`; the launcher runs the bundling
and the scoring stage itself, so nothing further is needed. It varies density
only. The source emits at a fixed interval, so bursty traffic is outside what
this sweep measures.

The propagation experiment replaces the 190 m reception disk with log-distance
path loss (exponent 3.0) and Nakagami fading behind a 250 m range guard, on the
same seeds, and is three launchers run in order. The first calibrates the
channel, one configuration at a time:

```
cd "$REPO"
MODE=static bash run_propcal.sh
MODE=mobile bash run_propcal.sh
```

For each candidate `--propRefLoss` in `SWEEP` it runs `K` short simulations and
compares the mean OLSR neighbour degree with that of the disk model, computed
from the reported campaign's topology probes, then recommends the value closest
to it. Narrow `SWEEP` and raise `K` (30 was used) to refine around the minimum.
Calibrate each configuration separately: the disk's mean degree is about 6.0
static and 6.6 mobile, so one value cannot match both.

Degree matching gives 46.75 static and 46 mobile. Under fading that value leaves
few deployments with a route from source to victim, so a second operating point,
42, is also run. It preserves route viability and gives a denser graph. The paper
reports both points, so both are run:

```
cd "$REPO"
REFLOSS=46.75 NUM_WORKERS=16 MAX_JOBS=16 bash run_propmodel.sh
REFLOSS=46 MODE=mobile NUM_WORKERS=16 MAX_JOBS=16 bash run_propmodel.sh

REFLOSS=42 NUM_WORKERS=22 MAX_JOBS=16 \
  OUT=$PWD/simulations_v347_propmodel_rl42 \
  ARMS_OUT=$PWD/analysis/arms_propmodel_rl42 \
  BASE_SIM=$PWD/simulations_v347_propmodel_rl42_baseline \
  BASE_ARMS=$PWD/analysis/arms_propmodel_rl42_baseline \
  bash run_propmodel.sh
```

Repeat the last command with `MODE=mobile`. `REFLOSS` has no default and the
script refuses to start without it. The four paths must be given for the second
point, or it writes over the first. Run one instance at a time.

`run_propmodel.sh` draws seeds from the `C_all` manifest in order and admits a
run if a route between victim and source still exists at the end of stabilization,
because the campaign's all-nodes-connected check almost never passes under
fading. It continues until 2,000 runs are admitted. The admission rate it prints
is a reported result: about 18 % static and 21 % mobile at the degree-matched
point, about 61 % at 42. It then builds the range-model baseline from the same
admitted seeds, which already exist in the reported campaign, so the two arms
differ in the channel alone. `STAGES=sim` stops after the simulations.

Its own learning stage predates the reported pipeline. The four bundles are
re-learned by the third launcher, which copies each bundle unchanged into a new
`<arm>_17` root and runs Step 3's scoring on it without simulating anything:

```
cd "$REPO"
bash run_prop17.sh
```

The figures in the paper come from those four arms re-learned under the reported
pipeline:

```
cd analysis
PIPE=$PWD/frozencal67_pipeline_17 \
QUEUE_OVERRIDE="arms_propmodel_17|campaign arms_propmodel_baseline_17|campaign \
 arms_propmodel_rl42_17|campaign arms_propmodel_rl42_baseline_17|campaign" \
bash marathon_frozencal.sh
```

`arms_propmodel_rl42_17` against `arms_propmodel_rl42_baseline_17` is the
route-viability comparison, and `arms_propmodel_17` against
`arms_propmodel_baseline_17` the degree-matched one. Each pair is compared with
`r23_paired_bootstrap.py`, at the end of this section.

The activation-fraction sweep is built the same way:

```
cd "$REPO"
bash run_fraction_all.sh
```

That launcher does not simulate anything itself. It verifies the binary carries
`--defenseActiveFraction`, then calls `run_fractionsweep.sh` twice -- once for a
50-seed smoke test and once per mode for the campaign -- and afterwards bundles
and scores what that produced. The sweep can also be driven on its own:

```
cd "$REPO"
MODE=static bash run_fractionsweep.sh
MODE=mobile bash run_fractionsweep.sh
```

`FRACTIONS` defaults to `0.25 0.5 0.75` and `N_SEEDS` to 2,000. The endpoints are
not simulated here: 0 is the baseline scenario and 1 is the reported campaign,
both already on disk. It writes
`simulations_v347_fractionsweep/f<F>/features_<mode>/<scenario>/` with a log per
run and one `fraction_status.csv`, and skips any run whose twelve output files
already exist. The window stays 40 s and only the defense is switched off inside
it, so the admission criterion fires at the same instants as the main campaign
and the seed set is identical in both modes.

Its own stage-1 accuracies are not the reported ones. Those arms were first
learned in a wider feature space, so they are re-scored with the reported
campaign's detector, frozen:

```
cd analysis/arms_fractionsweep
FRAC_REPORTED=$PWD/../arms_r34_17feat_frozencal67 \
FRAC_PIPELINE=$PWD/../frozencal67_pipeline_17 \
$PY -u fraction_frozen_score.py --mode both --out-dir frozen17_frozencal67
```

`FRAC_REPORTED` names the reported arm whose detector is frozen, and
`FRAC_PIPELINE` the generator that arm was learned with; both defaults name
earlier material, so give both. A generator producing a different column count
from the stored scaler is refused rather than silently rescaled. The script fits nothing. The scaler,
the selected column list, the estimator and its tuned threshold all come from the
reported campaign's stored model, and every cell is restricted to run identifiers
in that model's own test split, so swept seeds the detector was trained on do not
enter the evaluation. It writes
`frozen17_frozencal67/frozen17_{static,mobile}.json`. Expect 411 paired run
identifiers under static conditions and 396 under mobility, and a stored-accuracy
gate of 0.89975 and 0.92975.

The second mobility process is two launchers, and it is the longest job here.
Only the mobile half is simulated. The static half is not re-run: it is the
reported campaign's own static half, copied by manifest order, so the two
campaigns differ in the mobility process and in nothing else.

```
cd "$REPO"
PHASE=campaign bash run_gm10k.sh     # Gauss-Markov mobile runs, to 10,000 accepted
PHASE=copy     bash run_gm10k.sh     # the static half, from the reported campaign
PHASE=bundles  bash run_gm10k.sh     # the bundle the migration reads
nohup bash run_gm10k17_all.sh > /dev/null 2>&1 &
```

`PHASE=campaign` calls `run_campaign_gaussmarkov.sh` with `MODE=mobile` and
`TARGET_ACCEPTED=10000` (`TARGET` and `WORKERS` override the two), refuses to run
unless that launcher disables the distance gate, and afterwards checks that no
run was rejected for distance. It is resume-aware: runs already accepted are
kept and only the shortfall is simulated. `PHASE=copy` takes the first 10,000
runs of the reported static manifest with `copy_subset.py`. `PHASE=bundles`
writes `analysis/arms_gm10k/listener/colab_data/`, which `run_gm10k17_all.sh`
takes as its source and checks by checksum before and after copying it. The
launcher's later phases learn in an earlier, wider feature space and are not
part of the reported result. It writes its own log, `gm10k.log`, so do not
redirect its output into that file. `run_campaign_gaussmarkov.sh` must exist
first; it is generated by `patch_gm_campaign.py`, below.

Yields `simulations_v347_gaussmarkov/` and then `analysis/arms_gm10k_17/`, whose
stage 5 sweep is the seventeen-row table the paper's mobility control rests on.
Expect an in-domain accuracy of about 0.9339 within that configuration, and the
strongly one-sided transfer the paper reports: roughly 0.8330 training on the
static configuration, against 0.5610 in the reverse direction.

The migration is gated at every step and stops with a reason; it writes only
under `arms_gm10k_17` and overwrites nothing beside it.

Read the campaign launcher's body rather than its header comment. That header is
inherited from the script this one was derived from and is stale in three ways:
it gives that script's name, it names `simulations_v347/` as the output, and it
states an acceptance criterion requiring the sender to be at least three hops
from the victim. This campaign writes to `simulations_v347_gaussmarkov/`, guards
that path explicitly, and runs with `--enforceHopFilter=0`, so its population is
the unrestricted one. The body is authoritative.

That header is inherited because the launcher is generated rather than written.
`patch_gm_campaign.py` reads `run_campaign_v347.sh` and rewrites three things:
the output tree, the guard that protects it, and the two flags
`--mobilityModel=gaussmarkov` and `--enforceHopFilter=0`. It is what
`run_gm_arm.sh` runs in its prep phase, and it can be run alone:

```
cd "$REPO"
$PY patch_gm_campaign.py
```

It writes `run_campaign_gaussmarkov.sh` and never modifies the canonical script,
which it only reads. Each of its three edits is anchored on a string that must
occur exactly once; if an anchor is missing or ambiguous it aborts and writes
nothing. Re-running regenerates the file from the canonical source.

The window-order simulations and their first learning are driven by one
launcher, which is how the reported trees were produced:

```
cd "$REPO"
nohup bash run_windoworder_all.sh >> windoworder_all.log 2>&1 &
```

It runs six stages in series: canonical and then shuffled simulations, then
bundles and both learnings, first for the static configuration and then for the
mobile one. Canonical runs first in each configuration because it fixes the seed
set that the shuffled arm must match. Both arms use a 60 s formation lead-in.
Every stage skips itself when its output is complete, so relaunching after an
interruption continues where it stopped. `TARGET` defaults to 2,000;
`STAGES=1,2,3` runs the static configuration only. The steps it performs can
also be run one at a time, as follows.

The window-order control is four steps, and each arm is about an hour. The
canonical arm settles the seed set and the shuffled arm is then restricted to it,
so the two differ in window order and in nothing else:

```
cd "$REPO"
ARM=canonical nohup bash run_windoworder.sh >> wo_canon_static.log 2>&1 &
ARM=shuffled SEEDS_FROM=simulations_v347_windoworder/canonical/manifest_static.csv \
  nohup bash run_windoworder.sh >> wo_shuf_static.log 2>&1 &
```

`DRY=1 TARGET=5 bash run_windoworder.sh` is a rehearsal that prints the commands
and runs nothing. Output goes to `simulations_v347_windoworder/<arm>/`, with a
`manifest_<mode>.csv` per arm. Repeat with `MODE=mobile` for the mobile half.

Pairing, bundling and learning are a separate script, which the campaign prints
as its next step:

```
cd "$REPO"
nohup bash run_windoworder_learn.sh >> windoworder_learn_static.log 2>&1 &
MODE=mobile nohup bash run_windoworder_learn.sh >> windoworder_learn_mobile.log 2>&1 &
```

For every (id, seed) in the shuffled manifest it stages the same seed's canonical
run under the same new id, by symlink, into
`simulations_v347_windoworder_pair/`, then writes
`analysis/arms_windoworder/{canonical,shuffled}/listener/`. The canonical
campaign is read-only throughout and existing results are never overwritten;
`STAGE_ONLY=1` stops after the pairing and bundling. Set `DCFM_PYTHON` to your
interpreter.

The reported cells are re-scored from those bundles under the gated pipeline,
not taken from that script's own output:

```
cd "$REPO"
DRYRUN=1 bash run_r41_windoworder_frozencal.sh    # plan and guards only
DRYRUN=0 bash run_r41_windoworder_frozencal.sh
```

It defaults `PIPELINE` to `analysis/frozencal67_pipeline_17`, backs up the text
artefacts of the four target cells first, and writes
`arms_windoworder/{canonical,shuffled}_17_17` together with
`r41_windoworder_bootstrap_frozencal/` and `r41_byslot_17_frozencal/`.

The per-slot breakdown is a separate read-only pass over a trained arm:

```
cd analysis
$PY -u windoworder_by_slot.py --arms-root arms_r34_17feat_frozencal/listener \
  --mode static --arm canonical
```

It simulates nothing. It re-predicts the arm's own test rows and groups accuracy
by the slot each window was measured in, a control column that is never a model
input, so a flat profile across slots is evidence the classifier is not keying on
window position. Slots follow `_StartTime` under a 60 s formation lead-in:
120, 220, 320 and 420 seconds are slots 0 to 3. `--arm` selects `shuffled` or
`canonical`; `--out` names the CSV, which otherwise lands beside the results.

The independent-window control is one campaign per configuration, and its
default is a rehearsal rather than the run:

```
cd "$REPO"
MODE=static bash run_perwindowseed.sh     # rehearsal: costs it, runs nothing
TIMING=0 MODE=static nohup bash run_perwindowseed.sh > pws_static.log 2>&1 &
TIMING=0 MODE=mobile nohup bash run_perwindowseed.sh > pws_mobile.log 2>&1 &
```

`TIMING` must be set to 0 for the campaign. Left at its default of 1 the script
runs five simulations per phase into a scratch directory, measures the wall
clock, extrapolates the full cost, checks that all four phases yielded the window
it keeps, and stops without writing into the campaign tree. `TARGET` defaults to
2,000 pseudo-runs, which is the size the comparison uses and also a cap the
script refuses to exceed; `NUM_WORKERS` defaults to 22; `SAMPLE_SEED` is fixed,
so the draw from the population is reproducible. The population is `C_all` with
the hop filter off, drawn without replacement from the ten-thousand-run staging
the reported campaign uses, so no topology is ever reused.

Only the first measurement window of each simulation is kept, because the later
windows of a run inherit state from the earlier ones: the scenario raises a flag
when the attack first executes and never clears it. Recomposing existing runs
would remove the shared topology but keep that inherited state, which is why the
campaign is run rather than assembled, and why 2,000 pseudo-runs cost 8,000
simulations per configuration.

Output goes to `simulations_v347_perwindowseed/`, a new tree, together with
`manifest_{static,mobile}.csv`, one row per pseudo-run naming the four seeds it
is composed of. The script refuses to write into an existing campaign tree, reads
the population manifest without rewriting it, and refuses to start while a
learning job is running. Turn that tree into `arms_perwindowseed_17` with the
bundling step under Step 3, then score it there, and expect 0.8888 static and
0.9137 mobile for the stacking ensemble, against 0.8856 and 0.9063 for the
size-matched arm above.

Two further scripts ship without a step of their own, and both are here because
something in the package depends on them:

* `copy_subset.py` is the manifest-order subsetting helper. Nothing in the
  reported results is cut with it, but it is what `run_gm_arm.sh` calls, and its
  header is the clearest statement in the tree of why a subset must be taken by
  manifest row and never by file id. `make_2k_subset.py` above follows the same
  rule and cites it.
* `run_gm_arm.sh` builds a two-thousand-run pair of trees, one Gauss-Markov and
  one control, by calling that helper. Its control arm reaches the same two
  thousand runs as `make_2k_subset.py` does, which is how the manifest rule was
  first verified: the two routes were checked to select identical run ids in both
  modes. The arm it produces is not the one the paper quotes, and it is kept for
  that verification rather than for a figure.

The comparison itself:

```
cd analysis
R23_PIPELINE=$PWD/frozencal67_pipeline_17 \
  $PY -u r23_paired_bootstrap.py \
    --realistic ARM --baseline ITS_OWN_BASELINE \
    --mode static --model Stacking_Ensemble \
    --n-boot 2000 --out-dir PATH/TO/out
```

`R23_PIPELINE` must be given, and it must name the pipeline the two arms were
**learned** under, because each arm's stored scaler accepts only the column count
its own generator emits. The window-order arms of the timeline control were learned
under `frozencal67_pipeline_17`, as above, and so were the window-sweep arms of the
observation-length control. The script's own default names a different
pipeline, so always give it.

This produces the paired difference and its interval, resampled at the
simulation-run level with the same resampled runs applied to both arms. It is
what Table 10 reports.

`--model` forces both arms to be scored with the same estimator. Pass it.
Without it each arm is scored with its own winner, and the difference then mixes
the intervention with the choice of estimator.

## 7. Traps and troubleshooting

Three kinds of thing go wrong here, and only the third announces itself: a setting you did not pass, a file that shares its name with another, and an outright error.

### Settings that must be passed explicitly

Collected in one place, because each one changes the result without an error
message.

| Where | What to pass | If you forget |
|---|---|---|
| any `run_arm.py` call | `DCFM_PIPELINE` | a different, earlier feature space is used |
| `marathon_frozencal.sh` | `PIPE` | the same |
| `k_sweep_universal4_v2.py` | `--universal-set` | the submitted four-feature anchor is swept |
| `k_sweep_universal4_v2.py` | `--k-list 1,...,17` | the submitted K set is swept |
| `plot_figure1_optimal_k_17.py` | `--optimal-k 3` | the figure marks 4 |
| `r23_paired_bootstrap.py` | `--model` | the comparison mixes intervention with estimator choice |
| Step 3 | the four campaign flags | the older pipeline behaviour is used |

### Outputs that share a file name

Three pairs of directories hold files with identical names. Use the one named
here.

* The feature table is the 67-row output under `analysis/thrbias/out67/`.
* The deployment-cost frontier is the one under
  `analysis/inference_cost_out/arms_r34_17feat_frozencal67/frontier/`.
* The confusion-matrix figure is the one under
  `confusion_matrices_universal3_t16/`.

---
### When something fails

**`[fatal] MAX_JOBS env var must be set`.** Prefix the command with
`MAX_JOBS=12`.

**A `PyTorch not available` line.** Normal. The project runs on CPU.

**The campaign finishes but the accuracy is far from the table above.** Check
that `DCFM_PIPELINE` was set and that all four campaign flags were passed after
the bare `--`. Both are silent when omitted.

**The K sweep gives a different best K.** Check `--universal-set` and `--k-list`.
Both default to values from an earlier round of this study.

**A run seems to do nothing.** `run_arm.py` skips work when results already
exist. Pass `--force`.

**Out of memory during the campaign.** Lower `MAX_JOBS` and
`OMP_NUM_THREADS`, and run one configuration at a time.

---

## 8. Checking a reproduction

Open
`arms_r34_17feat_frozencal67/listener/results/{static,mobile}/results.csv`
and find the `Stacking_Ensemble` row:

```
static   accuracy 0.89975    AUC 0.9403051875
mobile   accuracy 0.92975    AUC 0.97294175
```

Then open `k_sweep_summary.txt` and check the rows for K equal to 3 and 17
against the table in Section 1.

A reproduction that matches these, and whose campaign logs show the
`[frozen-calibration] N/N` gate with both numbers equal, has reproduced the
reported results.
