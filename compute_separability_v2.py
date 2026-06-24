#!/usr/bin/env python3
"""
Compute multivariate separability measures for static and mobile configurations
(STRICT-OBSERVABILITY VARIANT — 33 features instead of 36).

VERSION 2 NOTES:
Three features removed because the underlying signal does not travel across
the network: HelloMessageRate, ControlPacketRate, AverageRoutingTableSize.

Measures:
  1. Mahalanobis distance between class centroids (pooled covariance,
     Tikhonov-regularized).
  2. LDA separation ratio (between-class / within-class variance along the
     Fisher discriminant direction).

Statistical validation:
  A. Bootstrap (stratified, with replacement) -> mean, std, 95% percentile CI.
  B. Permutation test on the mobility label -> p-value for the hypothesis
     that mobile > static in joint separability.
  C. Welch's t-test on the two bootstrap distributions -> complementary
     parametric check.
  D. Cohen's d on the difference between bootstrap distributions -> effect size
     of the mobility-induced change.
  E. Histogram plot of the two bootstrap distributions -> visual summary.

Usage:
    python3 compute_separability_v2.py \
        --static ../simulations/features_static \
        --mobile ../simulations/features_mobile \
        --n_bootstrap 100 \
        --n_permutations 10000 \
        --seed 42 \
        --jobs 12 \
        --out ./results/separability
"""

import os
import glob
import argparse
import warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

SCENARIOS = {
    "baseline": 0, "attack_only": 0,
    "defense_only": 1, "defense_vs_attack": 1,
}

METRICS = [
    # Control-plane (TC, MID, HNA flooded; HELLO/ControlPacketRate removed in v2)
    "TcMessageRate", "MidMessageRate", "HnaMessageRate",
    "AverageAdvertisedLinksPerTCMessage",
    "NormalizedRoutingLoad", "RoutingOverheadRatio", "RoutingOverheadBytesRatio",
    # Data-plane
    "PacketDeliveryRatio", "PacketLossRatio", "AverageEndToEndDelay", "AverageJitter",
    "Throughput", "AverageHopCount", "DataPacketRate", "RxTxPacketRatio",
    "FlowCount", "AvgFlowDuration", "FlowDurationStd", "AvgFlowThroughput",
    "AvgFlowDelay", "AvgFlowJitter", "AvgFlowLossRate", "FlowThroughputStd",
    "FlowDelayStd", "FlowJitterStd", "FlowLossRateStd",
    "AvgTxBytesPerFlow", "AvgRxBytesPerFlow", "AvgTxPacketsPerFlow", "AvgRxPacketsPerFlow",
    "AvgTxPacketSize", "AvgRxPacketSize",
    # Inferable from TC (AverageRoutingTableSize removed in v2)
    "AverageMprCount",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_one(csv_path, label, scenario):
    try:
        df = pd.read_csv(csv_path)
        df["defense_active"] = label
        df["scenario"] = scenario
        df["file_source"] = os.path.basename(csv_path)
        return df
    except Exception:
        return None


def load_dataset(data_root, n_jobs=-1):
    # Colab fast-path: reconstruct tall frame from the compact wide bundle.
    if os.environ.get("DCFM_DATA_BUNDLE", "").strip():
        from colab_bundle import load_bundle_tall
        return load_bundle_tall(os.environ["DCFM_DATA_BUNDLE"].strip(), data_root)
    tasks = []
    for scenario, label in SCENARIOS.items():
        scenario_dir = os.path.join(data_root, scenario)
        if not os.path.exists(scenario_dir):
            continue
        for csv_path in glob.glob(os.path.join(scenario_dir, "*.csv")):
            tasks.append((csv_path, label, scenario))
    if not tasks:
        raise ValueError(f"No CSV files found in: {data_root}")
    print(f"  Loading {len(tasks)} files...")
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_load_one)(p, l, s) for p, l, s in tasks
    )
    parts = [r for r in results if r is not None]
    return pd.concat(parts, ignore_index=True)


def to_wide(tall_df):
    tall_df = tall_df[tall_df["Metric"].isin(METRICS)].copy()
    agg = (tall_df
           .groupby(["scenario", "file_source", "Metric"], sort=False)["Value"]
           .mean().reset_index())
    wide = agg.pivot_table(
        index=["scenario", "file_source"], columns="Metric",
        values="Value", aggfunc="mean")
    wide.columns.name = None
    label_map = (tall_df[["scenario", "file_source", "defense_active"]]
                 .drop_duplicates(["scenario", "file_source"])
                 .set_index(["scenario", "file_source"])["defense_active"])
    y = label_map.reindex(wide.index).astype(int).reset_index(drop=True)
    wide = wide.reset_index(drop=True)
    for m in METRICS:
        if m not in wide.columns:
            wide[m] = 0.0
    return wide[METRICS].fillna(0.0), y


# ---------------------------------------------------------------------------
# Core separability measures
# ---------------------------------------------------------------------------

def mahalanobis_distance(X, y, reg_eps=1e-6):
    """
    Mahalanobis distance between class centroids using pooled covariance
    with Tikhonov regularization (+ reg_eps * I) to ensure invertibility.
    D_M = sqrt( (mu1 - mu0)^T * S_pooled^{-1} * (mu1 - mu0) )
    """
    X0 = X[y == 0]
    X1 = X[y == 1]
    mu0 = X0.mean(axis=0)
    mu1 = X1.mean(axis=0)

    n0, n1 = len(X0), len(X1)
    S0 = np.cov(X0.T)
    S1 = np.cov(X1.T)
    S_pooled = ((n0 - 1) * S0 + (n1 - 1) * S1) / (n0 + n1 - 2)
    S_reg = S_pooled + np.eye(S_pooled.shape[0]) * reg_eps

    diff = mu1 - mu0
    try:
        S_inv = np.linalg.inv(S_reg)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S_reg)
    d_sq = diff @ S_inv @ diff
    return float(np.sqrt(max(d_sq, 0)))


def lda_separation(X, y):
    """
    LDA separation ratio (Fisher discriminant ratio):
    ratio of between-class to within-class variance along the Fisher direction.
    """
    lda = LinearDiscriminantAnalysis(n_components=1)
    lda.fit(X, y)
    X_proj = lda.transform(X).ravel()

    X0_proj = X_proj[y == 0]
    X1_proj = X_proj[y == 1]

    mu_overall = X_proj.mean()
    mu0 = X0_proj.mean()
    mu1 = X1_proj.mean()
    n0, n1 = len(X0_proj), len(X1_proj)
    n = len(X_proj)

    S_B = (n0 * (mu0 - mu_overall) ** 2 + n1 * (mu1 - mu_overall) ** 2) / n
    S_W = (np.sum((X0_proj - mu0) ** 2) + np.sum((X1_proj - mu1) ** 2)) / n

    return float(S_B / S_W) if S_W > 0 else float("inf")


def _compute_both_measures(X_std, y_arr):
    """Convenience: compute Mahalanobis and LDA on already-standardized X."""
    d_maha = mahalanobis_distance(X_std, y_arr)
    lda_ratio = lda_separation(X_std, y_arr)
    return d_maha, lda_ratio


# ---------------------------------------------------------------------------
# Bootstrap (mean, std, 95% percentile CI)
# ---------------------------------------------------------------------------

def _bootstrap_iteration(X_raw_values, y_arr, seed):
    """
    One bootstrap iteration:
      1. Stratified resample with replacement (preserves class balance).
      2. StandardScaler fit on the resampled data only (no leakage).
      3. Compute Mahalanobis + LDA ratio.
    """
    rng = np.random.default_rng(seed)
    idx0 = np.where(y_arr == 0)[0]
    idx1 = np.where(y_arr == 1)[0]
    n0, n1 = len(idx0), len(idx1)

    boot_idx0 = rng.choice(idx0, size=n0, replace=True)
    boot_idx1 = rng.choice(idx1, size=n1, replace=True)
    boot_idx = np.concatenate([boot_idx0, boot_idx1])

    X_boot = X_raw_values[boot_idx]
    y_boot = y_arr[boot_idx]

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_boot)
    return _compute_both_measures(X_std, y_boot)


def run_bootstrap(X_raw, y, name, n_bootstrap=100, base_seed=42, n_jobs=12):
    print(f"\n{'=' * 70}")
    print(f"BOOTSTRAP  ::  {name}  ({n_bootstrap} iterations)")
    print(f"{'=' * 70}")
    y_arr = y.values if hasattr(y, "values") else np.asarray(y)
    X_vals = X_raw.values if isinstance(X_raw, pd.DataFrame) else np.asarray(X_raw)

    # Point estimate on the full dataset (sanity check vs. previous run)
    scaler = StandardScaler()
    X_full_std = scaler.fit_transform(X_vals)
    d_maha_point, lda_ratio_point = _compute_both_measures(X_full_std, y_arr)
    print(f"  Point estimate (full dataset):")
    print(f"    Mahalanobis distance: {d_maha_point:.4f}")
    print(f"    LDA separation ratio: {lda_ratio_point:.4f}")

    # Bootstrap
    seeds = [base_seed + i for i in range(n_bootstrap)]
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(_bootstrap_iteration)(X_vals, y_arr, s) for s in seeds
    )
    maha_vals = np.array([r[0] for r in results])
    lda_vals = np.array([r[1] for r in results])

    print(f"  Bootstrap ({n_bootstrap} iterations):")
    print(f"    Mahalanobis: mean = {maha_vals.mean():.4f}, std = {maha_vals.std(ddof=1):.4f}")
    print(f"    LDA ratio  : mean = {lda_vals.mean():.4f}, std = {lda_vals.std(ddof=1):.4f}")
    print(f"    Mahalanobis 95%% CI: [{np.percentile(maha_vals, 2.5):.4f}, "
          f"{np.percentile(maha_vals, 97.5):.4f}]")
    print(f"    LDA ratio   95%% CI: [{np.percentile(lda_vals, 2.5):.4f}, "
          f"{np.percentile(lda_vals, 97.5):.4f}]")

    return {
        "maha_point": d_maha_point,
        "lda_point": lda_ratio_point,
        "maha_mean": float(maha_vals.mean()),
        "maha_std": float(maha_vals.std(ddof=1)),
        "maha_ci_lo": float(np.percentile(maha_vals, 2.5)),
        "maha_ci_hi": float(np.percentile(maha_vals, 97.5)),
        "lda_mean": float(lda_vals.mean()),
        "lda_std": float(lda_vals.std(ddof=1)),
        "lda_ci_lo": float(np.percentile(lda_vals, 2.5)),
        "lda_ci_hi": float(np.percentile(lda_vals, 97.5)),
        "maha_values": maha_vals,
        "lda_values": lda_vals,
    }


# ---------------------------------------------------------------------------
# Permutation test on the mobility label
# ---------------------------------------------------------------------------

def _permutation_iteration(X_combined, y_combined, mobility_combined, seed):
    """
    One permutation iteration:
      1. Shuffle the mobility label across all samples (keeping class labels fixed).
      2. Partition into "static" and "mobile" groups based on shuffled mobility.
      3. Standardize each group separately.
      4. Compute Mahalanobis and LDA for each, return the mobile - static differences.
    """
    rng = np.random.default_rng(seed)
    mobility_perm = rng.permutation(mobility_combined)

    mask_s = mobility_perm == 0
    mask_m = mobility_perm == 1

    X_s = X_combined[mask_s]
    y_s = y_combined[mask_s]
    X_m = X_combined[mask_m]
    y_m = y_combined[mask_m]

    # Standardize each pseudo-group independently (matches how we compute on real data)
    Xs_std = StandardScaler().fit_transform(X_s)
    Xm_std = StandardScaler().fit_transform(X_m)

    d_maha_s, lda_s = _compute_both_measures(Xs_std, y_s)
    d_maha_m, lda_m = _compute_both_measures(Xm_std, y_m)

    return (d_maha_m - d_maha_s), (lda_m - lda_s)


def run_permutation_test(X_s_raw, y_s, X_m_raw, y_m,
                         obs_diff_maha, obs_diff_lda,
                         n_permutations=10000, base_seed=12345, n_jobs=12):
    print(f"\n{'=' * 70}")
    print(f"PERMUTATION TEST  ({n_permutations} permutations)")
    print(f"{'=' * 70}")
    print(f"  H0: mobility has no effect on joint separability")
    print(f"  H1: mobile configuration has higher joint separability than static")

    X_s_vals = X_s_raw.values if isinstance(X_s_raw, pd.DataFrame) else np.asarray(X_s_raw)
    X_m_vals = X_m_raw.values if isinstance(X_m_raw, pd.DataFrame) else np.asarray(X_m_raw)
    y_s_arr = y_s.values if hasattr(y_s, "values") else np.asarray(y_s)
    y_m_arr = y_m.values if hasattr(y_m, "values") else np.asarray(y_m)

    X_combined = np.vstack([X_s_vals, X_m_vals])
    y_combined = np.concatenate([y_s_arr, y_m_arr])
    mobility_combined = np.concatenate([
        np.zeros(len(y_s_arr), dtype=int),
        np.ones(len(y_m_arr), dtype=int),
    ])

    print(f"  Combined dataset: n = {len(y_combined)}")
    print(f"  Observed difference (mobile - static):")
    print(f"    Mahalanobis: {obs_diff_maha:+.4f}")
    print(f"    LDA ratio  : {obs_diff_lda:+.4f}")
    print(f"  Running {n_permutations} permutations in parallel (n_jobs={n_jobs})...")

    seeds = [base_seed + i for i in range(n_permutations)]
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
        delayed(_permutation_iteration)(X_combined, y_combined, mobility_combined, s)
        for s in seeds
    )
    perm_diffs_maha = np.array([r[0] for r in results])
    perm_diffs_lda = np.array([r[1] for r in results])

    # One-sided p-value: probability that a random split produces at least
    # as large a (mobile - static) difference as the observed one.
    # +1 in numerator/denominator = classical Fisher correction (avoids p=0).
    p_maha = (1 + np.sum(perm_diffs_maha >= obs_diff_maha)) / (1 + n_permutations)
    p_lda = (1 + np.sum(perm_diffs_lda >= obs_diff_lda)) / (1 + n_permutations)

    print(f"\n  Permutation p-values (one-sided, H1: mobile > static):")
    print(f"    Mahalanobis: p = {p_maha:.6f}")
    print(f"    LDA ratio  : p = {p_lda:.6f}")
    print(f"  Null distribution (mobile - static under H0):")
    print(f"    Mahalanobis: mean = {perm_diffs_maha.mean():+.4f}, "
          f"std = {perm_diffs_maha.std(ddof=1):.4f}, "
          f"max = {perm_diffs_maha.max():+.4f}")
    print(f"    LDA ratio  : mean = {perm_diffs_lda.mean():+.4f}, "
          f"std = {perm_diffs_lda.std(ddof=1):.4f}, "
          f"max = {perm_diffs_lda.max():+.4f}")
    print(f"  Ratio of observed to null max:")
    print(f"    Mahalanobis: {obs_diff_maha / perm_diffs_maha.max():.2f}x")
    print(f"    LDA ratio  : {obs_diff_lda / perm_diffs_lda.max():.2f}x")
    n_exceed_maha = int(np.sum(perm_diffs_maha >= obs_diff_maha))
    n_exceed_lda = int(np.sum(perm_diffs_lda >= obs_diff_lda))
    print(f"  Permutations exceeding observed difference:")
    print(f"    Mahalanobis: {n_exceed_maha} / {n_permutations}")
    print(f"    LDA ratio  : {n_exceed_lda} / {n_permutations}")

    return {
        "obs_diff_maha": float(obs_diff_maha),
        "obs_diff_lda": float(obs_diff_lda),
        "p_maha": float(p_maha),
        "p_lda": float(p_lda),
        "perm_diffs_maha": perm_diffs_maha,
        "perm_diffs_lda": perm_diffs_lda,
    }


# ---------------------------------------------------------------------------
# Welch's t-test and effect size on bootstrap distributions
# ---------------------------------------------------------------------------

def welch_and_effect_size(boot_s, boot_m, label):
    """
    Welch's t-test on the two bootstrap distributions, plus Cohen's d as
    effect size of the mobility-induced change.
    """
    t_stat, p_val = stats.ttest_ind(boot_m, boot_s, equal_var=False, alternative="greater")
    mean_diff = boot_m.mean() - boot_s.mean()
    pooled_std = np.sqrt(0.5 * (boot_m.var(ddof=1) + boot_s.var(ddof=1)))
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else float("inf")
    print(f"  {label}:")
    print(f"    Welch's t = {t_stat:.3f}, p (one-sided) = {p_val:.3e}")
    print(f"    Cohen's d on bootstrap distributions = {cohens_d:.3f}")
    return {"t_stat": float(t_stat), "p_value": float(p_val),
            "cohens_d": float(cohens_d), "mean_diff": float(mean_diff)}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_bootstrap_distributions(res_s, res_m, perm_res, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Top-left: Mahalanobis bootstrap distributions
    ax = axes[0, 0]
    ax.hist(res_s["maha_values"], bins=25, alpha=0.6, label="Static", color="#1f77b4")
    ax.hist(res_m["maha_values"], bins=25, alpha=0.6, label="Mobile", color="#ff7f0e")
    ax.axvline(res_s["maha_mean"], color="#1f77b4", linestyle="--", linewidth=1)
    ax.axvline(res_m["maha_mean"], color="#ff7f0e", linestyle="--", linewidth=1)
    ax.set_xlabel("Mahalanobis distance")
    ax.set_ylabel("Count (bootstrap iterations)")
    ax.set_title("Bootstrap distribution: Mahalanobis")
    ax.legend()

    # Top-right: LDA bootstrap distributions
    ax = axes[0, 1]
    ax.hist(res_s["lda_values"], bins=25, alpha=0.6, label="Static", color="#1f77b4")
    ax.hist(res_m["lda_values"], bins=25, alpha=0.6, label="Mobile", color="#ff7f0e")
    ax.axvline(res_s["lda_mean"], color="#1f77b4", linestyle="--", linewidth=1)
    ax.axvline(res_m["lda_mean"], color="#ff7f0e", linestyle="--", linewidth=1)
    ax.set_xlabel("LDA separation ratio")
    ax.set_ylabel("Count (bootstrap iterations)")
    ax.set_title("Bootstrap distribution: LDA ratio")
    ax.legend()

    # Bottom-left: permutation null for Mahalanobis diff
    ax = axes[1, 0]
    ax.hist(perm_res["perm_diffs_maha"], bins=40, color="gray", alpha=0.7,
            label="Null (H0)")
    ax.axvline(perm_res["obs_diff_maha"], color="red", linestyle="--",
               linewidth=2, label=f"Observed = {perm_res['obs_diff_maha']:+.3f}")
    ax.set_xlabel("mobile - static (Mahalanobis)")
    ax.set_ylabel("Count (permutations)")
    ax.set_title(f"Permutation null  (p = {perm_res['p_maha']:.4f})")
    ax.legend()

    # Bottom-right: permutation null for LDA diff
    ax = axes[1, 1]
    ax.hist(perm_res["perm_diffs_lda"], bins=40, color="gray", alpha=0.7,
            label="Null (H0)")
    ax.axvline(perm_res["obs_diff_lda"], color="red", linestyle="--",
               linewidth=2, label=f"Observed = {perm_res['obs_diff_lda']:+.3f}")
    ax.set_xlabel("mobile - static (LDA ratio)")
    ax.set_ylabel("Count (permutations)")
    ax.set_title(f"Permutation null  (p = {perm_res['p_lda']:.4f})")
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved to: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--static", required=True)
    parser.add_argument("--mobile", required=True)
    parser.add_argument("--jobs", type=int, default=12,
                        help="Number of parallel workers (default: 12).")
    parser.add_argument("--n_bootstrap", type=int, default=100,
                        help="Bootstrap iterations (default: 100).")
    parser.add_argument("--n_permutations", type=int, default=10000,
                        help="Permutation iterations (default: 10000).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="./results/separability")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Load data
    print("Loading static dataset...")
    X_s, y_s = to_wide(load_dataset(args.static, n_jobs=args.jobs))
    print(f"  Static: {X_s.shape}")
    print("\nLoading mobile dataset...")
    X_m, y_m = to_wide(load_dataset(args.mobile, n_jobs=args.jobs))
    print(f"  Mobile: {X_m.shape}")

    # --- A. Bootstrap on each configuration -----------------------------------
    res_s = run_bootstrap(X_s, y_s, "Static",
                          n_bootstrap=args.n_bootstrap,
                          base_seed=args.seed, n_jobs=args.jobs)
    res_m = run_bootstrap(X_m, y_m, "Mobile",
                          n_bootstrap=args.n_bootstrap,
                          base_seed=args.seed + 100_000, n_jobs=args.jobs)

    # --- B. Welch + effect size on bootstrap distributions --------------------
    print(f"\n{'=' * 70}")
    print("WELCH'S T-TEST AND EFFECT SIZE ON BOOTSTRAP DISTRIBUTIONS")
    print(f"{'=' * 70}")
    print(f"  H1: mobile > static")
    welch_maha = welch_and_effect_size(res_s["maha_values"], res_m["maha_values"],
                                       "Mahalanobis")
    welch_lda = welch_and_effect_size(res_s["lda_values"], res_m["lda_values"],
                                      "LDA ratio")

    # --- C. Permutation test on the mobility label ----------------------------
    obs_diff_maha = res_m["maha_point"] - res_s["maha_point"]
    obs_diff_lda = res_m["lda_point"] - res_s["lda_point"]
    perm_res = run_permutation_test(
        X_s, y_s, X_m, y_m,
        obs_diff_maha=obs_diff_maha,
        obs_diff_lda=obs_diff_lda,
        n_permutations=args.n_permutations,
        base_seed=args.seed + 1_000_000,
        n_jobs=args.jobs,
    )

    # --- D. Plot --------------------------------------------------------------
    plot_path = os.path.join(args.out, "separability_distributions.png")
    plot_bootstrap_distributions(res_s, res_m, perm_res, plot_path)

    # --- E. Persist all outputs -----------------------------------------------
    df_boot = pd.DataFrame({
        "iteration": np.arange(args.n_bootstrap),
        "static_mahalanobis": res_s["maha_values"],
        "mobile_mahalanobis": res_m["maha_values"],
        "static_lda_ratio": res_s["lda_values"],
        "mobile_lda_ratio": res_m["lda_values"],
    })
    df_boot.to_csv(os.path.join(args.out, "separability_bootstrap.csv"), index=False)

    df_perm = pd.DataFrame({
        "iteration": np.arange(args.n_permutations),
        "perm_diff_mahalanobis": perm_res["perm_diffs_maha"],
        "perm_diff_lda_ratio": perm_res["perm_diffs_lda"],
    })
    df_perm.to_csv(os.path.join(args.out, "separability_permutations.csv"), index=False)

    # Null distribution statistics (for p-value transparency)
    n_exceed_maha = int(np.sum(perm_res["perm_diffs_maha"] >= obs_diff_maha))
    n_exceed_lda = int(np.sum(perm_res["perm_diffs_lda"] >= obs_diff_lda))
    null_max_maha = float(perm_res["perm_diffs_maha"].max())
    null_max_lda = float(perm_res["perm_diffs_lda"].max())

    summary_rows = [
        {"measure": "Mahalanobis distance",
         "static_point": res_s["maha_point"], "mobile_point": res_m["maha_point"],
         "static_mean": res_s["maha_mean"], "static_std": res_s["maha_std"],
         "mobile_mean": res_m["maha_mean"], "mobile_std": res_m["maha_std"],
         "static_ci_lo": res_s["maha_ci_lo"], "static_ci_hi": res_s["maha_ci_hi"],
         "mobile_ci_lo": res_m["maha_ci_lo"], "mobile_ci_hi": res_m["maha_ci_hi"],
         "observed_diff": obs_diff_maha,
         "null_max": null_max_maha,
         "null_mean": float(perm_res["perm_diffs_maha"].mean()),
         "null_std": float(perm_res["perm_diffs_maha"].std(ddof=1)),
         "n_permutations_exceeding_observed": n_exceed_maha,
         "n_permutations_total": args.n_permutations,
         "welch_t": welch_maha["t_stat"], "welch_p_onesided": welch_maha["p_value"],
         "effect_size_cohens_d": welch_maha["cohens_d"],
         "permutation_p_onesided": perm_res["p_maha"]},
        {"measure": "LDA separation ratio",
         "static_point": res_s["lda_point"], "mobile_point": res_m["lda_point"],
         "static_mean": res_s["lda_mean"], "static_std": res_s["lda_std"],
         "mobile_mean": res_m["lda_mean"], "mobile_std": res_m["lda_std"],
         "static_ci_lo": res_s["lda_ci_lo"], "static_ci_hi": res_s["lda_ci_hi"],
         "mobile_ci_lo": res_m["lda_ci_lo"], "mobile_ci_hi": res_m["lda_ci_hi"],
         "observed_diff": obs_diff_lda,
         "null_max": null_max_lda,
         "null_mean": float(perm_res["perm_diffs_lda"].mean()),
         "null_std": float(perm_res["perm_diffs_lda"].std(ddof=1)),
         "n_permutations_exceeding_observed": n_exceed_lda,
         "n_permutations_total": args.n_permutations,
         "welch_t": welch_lda["t_stat"], "welch_p_onesided": welch_lda["p_value"],
         "effect_size_cohens_d": welch_lda["cohens_d"],
         "permutation_p_onesided": perm_res["p_lda"]},
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(args.out, "separability_summary.csv"), index=False)

    # --- F. Final report ------------------------------------------------------
    pm = "\u00b1"  # ± symbol — defined outside f-strings (Python 3.10 compat)
    print(f"\n{'=' * 70}")
    print("FINAL SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Measure':<22} {'Static (mean ' + pm + ' std)':>22} {'Mobile (mean ' + pm + ' std)':>22}")
    print("-" * 70)
    print(f"{'Mahalanobis':<22} "
          f"{res_s['maha_mean']:>8.4f} {pm} {res_s['maha_std']:.4f}   "
          f"{res_m['maha_mean']:>8.4f} {pm} {res_m['maha_std']:.4f}")
    print(f"{'LDA ratio':<22} "
          f"{res_s['lda_mean']:>8.4f} {pm} {res_s['lda_std']:.4f}   "
          f"{res_m['lda_mean']:>8.4f} {pm} {res_m['lda_std']:.4f}")
    print(f"\n  95% percentile CI (bootstrap):")
    print(f"    Mahalanobis: static [{res_s['maha_ci_lo']:.4f}, {res_s['maha_ci_hi']:.4f}]  "
          f"vs  mobile [{res_m['maha_ci_lo']:.4f}, {res_m['maha_ci_hi']:.4f}]")
    print(f"    LDA ratio  : static [{res_s['lda_ci_lo']:.4f}, {res_s['lda_ci_hi']:.4f}]  "
          f"vs  mobile [{res_m['lda_ci_lo']:.4f}, {res_m['lda_ci_hi']:.4f}]")

    maha_overlap = not (res_s["maha_ci_hi"] < res_m["maha_ci_lo"] or
                        res_m["maha_ci_hi"] < res_s["maha_ci_lo"])
    lda_overlap = not (res_s["lda_ci_hi"] < res_m["lda_ci_lo"] or
                       res_m["lda_ci_hi"] < res_s["lda_ci_lo"])
    print(f"\n  CI overlap (static vs mobile):")
    print(f"    Mahalanobis: {'YES' if maha_overlap else 'NO'}")
    print(f"    LDA ratio  : {'YES' if lda_overlap else 'NO'}")

    print(f"\n  Significance (one-sided, H1: mobile > static):")
    print(f"    Mahalanobis: permutation p = {perm_res['p_maha']:.6f} "
          f"({n_exceed_maha}/{args.n_permutations} permutations exceed observed), "
          f"Welch p = {welch_maha['p_value']:.3e}, "
          f"Cohen's d on bootstrap = {welch_maha['cohens_d']:.3f}")
    print(f"    LDA ratio  : permutation p = {perm_res['p_lda']:.6f} "
          f"({n_exceed_lda}/{args.n_permutations} permutations exceed observed), "
          f"Welch p = {welch_lda['p_value']:.3e}, "
          f"Cohen's d on bootstrap = {welch_lda['cohens_d']:.3f}")
    print(f"\n  Observed vs null maximum:")
    print(f"    Mahalanobis: observed = {obs_diff_maha:+.4f}, "
          f"null max = {null_max_maha:+.4f} "
          f"(observed is {obs_diff_maha / null_max_maha:.2f}x null max)")
    print(f"    LDA ratio  : observed = {obs_diff_lda:+.4f}, "
          f"null max = {null_max_lda:+.4f} "
          f"(observed is {obs_diff_lda / null_max_lda:.2f}x null max)")

    print(f"\n  Outputs written to:")
    print(f"    {os.path.join(args.out, 'separability_summary.csv')}")
    print(f"    {os.path.join(args.out, 'separability_bootstrap.csv')}")
    print(f"    {os.path.join(args.out, 'separability_permutations.csv')}")
    print(f"    {os.path.join(args.out, 'separability_distributions.png')}")


if __name__ == "__main__":
    main()