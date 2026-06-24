#!/usr/bin/env python3
"""
Compute Cohen's d class separation for all 33 strict-observable features
under static and mobile configurations.

VERSION 2 NOTES:
Strict-observability variant. Three features removed because the underlying
signal does not travel across the network:
    HelloMessageRate, ControlPacketRate, AverageRoutingTableSize.

UNIVERSAL_5 (the previous Universal Set) included AverageRoutingTableSize
which is now removed. UNIVERSAL_5 marker is kept for reference but commented
out — the new Universal Set will be determined by feature_importance_sensitivity_v2.

Cohen's d = (mu1 - mu0) / pooled_std

Usage:
    python3 compute_cohens_d_v2.py \
        --static ../simulations/features_static \
        --mobile ../simulations/features_mobile \
        --out ./results/cohens_d
"""

import os, glob, argparse, warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

SCENARIOS = {
    "baseline": 0, "attack_only": 0,
    "defense_only": 1, "defense_vs_attack": 1,
}

METRICS = [
    # Control-plane metrics observable from network-traveling traffic
    # (TC, MID, HNA are flooded; HELLO is NOT flooded — only 1-hop)
    "TcMessageRate", "MidMessageRate", "HnaMessageRate",
    "AverageAdvertisedLinksPerTCMessage",
    "NormalizedRoutingLoad", "RoutingOverheadRatio", "RoutingOverheadBytesRatio",
    # Data-plane metrics (observable from traffic)
    "PacketDeliveryRatio", "PacketLossRatio", "AverageEndToEndDelay", "AverageJitter",
    "Throughput", "AverageHopCount", "DataPacketRate", "RxTxPacketRatio",
    "FlowCount", "AvgFlowDuration", "FlowDurationStd", "AvgFlowThroughput",
    "AvgFlowDelay", "AvgFlowJitter", "AvgFlowLossRate", "FlowThroughputStd",
    "FlowDelayStd", "FlowJitterStd", "FlowLossRateStd",
    "AvgTxBytesPerFlow", "AvgRxBytesPerFlow", "AvgTxPacketsPerFlow", "AvgRxPacketsPerFlow",
    "AvgTxPacketSize", "AvgRxPacketSize",
    # Inferable from TC messages (flooded across network)
    "AverageMprCount",
    # Removed in v2 (not observable from network-traveling traffic):
    #   HelloMessageRate     - HELLO is 1-hop broadcast, not flooded
    #   ControlPacketRate    - includes HelloMessageRate
    #   AverageRoutingTableSize - requires HELLO-derived neighbor graph
]

# UNIVERSAL_5 (previous Universal Set, now invalid since AverageRoutingTableSize removed):
# UNIVERSAL_5 = ['AverageAdvertisedLinksPerTCMessage', 'AverageMprCount',
#                'AverageRoutingTableSize', 'FlowThroughputStd', 'DataPacketRate']
#
# The new Universal Set will be determined by feature_importance_sensitivity_v2.
# Until then, we mark only known top features for reference.
UNIVERSAL_PROVISIONAL = [
    "AverageAdvertisedLinksPerTCMessage",
    "AverageMprCount",
    "FlowThroughputStd",
    "DataPacketRate",
]

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
    import os as _os
    if _os.environ.get("DCFM_DATA_BUNDLE", "").strip():
        from colab_bundle import load_bundle_tall
        return load_bundle_tall(_os.environ["DCFM_DATA_BUNDLE"].strip(), data_root)
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

def cohens_d(x0, x1):
    """Cohen's d = (mu1 - mu0) / pooled_std"""
    n0, n1 = len(x0), len(x1)
    mu0, mu1 = x0.mean(), x1.mean()
    s0, s1 = x0.std(ddof=1), x1.std(ddof=1)
    pooled_std = np.sqrt(((n0-1)*s0**2 + (n1-1)*s1**2) / (n0+n1-2))
    if pooled_std == 0:
        return 0.0
    return abs(mu1 - mu0) / pooled_std

def compute_cohens_d_all(X, y, config_name):
    X0 = X[y == 0]
    X1 = X[y == 1]
    results = {}
    for feat in METRICS:
        if feat in X.columns:
            d = cohens_d(X0[feat].values, X1[feat].values)
            results[feat] = round(d, 4)
    mean_d = np.mean(list(results.values()))
    print(f"  {config_name}: mean Cohen's d = {mean_d:.4f}")
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--static", required=True)
    parser.add_argument("--mobile", required=True)
    parser.add_argument("--out", default="./domain_shift_output")
    parser.add_argument("--jobs", type=int, default=-1)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("Loading static dataset...")
    X_s, y_s = to_wide(load_dataset(args.static, n_jobs=args.jobs))

    print("Loading mobile dataset...")
    X_m, y_m = to_wide(load_dataset(args.mobile, n_jobs=args.jobs))

    print("\nComputing Cohen's d...")
    d_static = compute_cohens_d_all(X_s, y_s, "Static")
    d_mobile = compute_cohens_d_all(X_m, y_m, "Mobile")

    # Build results dataframe
    rows = []
    for feat in METRICS:
        rows.append({
            "feature": feat,
            "static_d": d_static.get(feat, 0.0),
            "mobile_d": d_mobile.get(feat, 0.0),
            "delta": round(d_mobile.get(feat, 0.0) - d_static.get(feat, 0.0), 4),
            "universal_provisional": "yes" if feat in UNIVERSAL_PROVISIONAL else "no",
        })

    df = pd.DataFrame(rows).sort_values("static_d", ascending=False)

    # Save full CSV
    csv_path = os.path.join(args.out, "cohens_d_all_features.csv")
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"\n  Full results saved to: {csv_path}")

    # Print summary tables
    print(f"\n{'='*70}")
    print("PROVISIONAL UNIVERSAL FEATURES (Universal-5 minus AverageRoutingTableSize)")
    print(f"{'='*70}")
    print(f"{'Feature':<40} {'Static d':>9} {'Mobile d':>9} {'Delta':>8}")
    print("-"*70)
    df_u7 = df[df["universal_provisional"] == "yes"].sort_values("static_d", ascending=False)
    for _, row in df_u7.iterrows():
        print(f"{row['feature']:<40} {row['static_d']:>9.4f} {row['mobile_d']:>9.4f} {row['delta']:>8.4f}")

    print(f"\n{'='*70}")
    print("DELAY AND JITTER FEATURES")
    print(f"{'='*70}")
    delay_jitter = [f for f in METRICS if any(k in f for k in
                    ["Delay", "Jitter", "delay", "jitter"])]
    print(f"{'Feature':<40} {'Static d':>9} {'Mobile d':>9} {'Delta':>8}")
    print("-"*70)
    df_dj = df[df["feature"].isin(delay_jitter)].sort_values("static_d", ascending=False)
    for _, row in df_dj.iterrows():
        print(f"{row['feature']:<40} {row['static_d']:>9.4f} {row['mobile_d']:>9.4f} {row['delta']:>8.4f}")

    print(f"\n{'='*70}")
    print("ALL FEATURES — sorted by static_d")
    print(f"{'='*70}")
    print(f"{'Feature':<40} {'Static d':>9} {'Mobile d':>9} {'Delta':>8} {'UP':>4}")
    print("-"*70)
    for _, row in df.iterrows():
        u7 = "✓" if row["universal_provisional"] == "yes" else ""
        print(f"{row['feature']:<40} {row['static_d']:>9.4f} {row['mobile_d']:>9.4f} {row['delta']:>8.4f} {u7:>4}")

    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS")
    print(f"{'='*70}")
    print(f"  Mean Cohen's d — Static: {df['static_d'].mean():.4f}")
    print(f"  Mean Cohen's d — Mobile: {df['mobile_d'].mean():.4f}")
    print(f"  Features stronger in mobile (delta > 0): {(df['delta'] > 0).sum()}")
    print(f"  Features weaker in mobile (delta < 0):   {(df['delta'] < 0).sum()}")

if __name__ == "__main__":
    main()