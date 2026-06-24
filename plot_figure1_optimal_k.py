#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_figure1_optimal_k.py
==========================
Reproduce the K-vs-cross-domain-accuracy plot for the Universal-4 setting.

Uses results from k_sweep_universal4_v2.py to produce Figure 1 of the paper:
  - X-axis: number of features (K), nonlinear tick positions for clarity
  - Y-axis: cross-domain accuracy
  - Two lines: Static -> Mobile (blue circles), Mobile -> Static (orange squares)
  - Vertical green dotted line at K=4 marked "K = 4 (optimal)"

Reads from:
  results/k_sweep_universal4/k_sweep_results.csv

Writes:
  results/k_sweep_universal4/figure1_optimal_k.png
  results/k_sweep_universal4/figure1_optimal_k.pdf

Usage:
    python3 plot_figure1_optimal_k.py \
        --in-csv ./results/k_sweep_universal4/k_sweep_results.csv \
        --out-dir ./results/k_sweep_universal4
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Colors matched to the reference Figure 1 from the previous paper version
BLUE = "#1f77b4"     # Static -> Mobile
ORANGE = "#ff7f0e"   # Mobile -> Static
GREEN = "#388e3c"    # K = 4 (optimal) marker


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-csv", required=True,
                    help="Path to k_sweep_results.csv")
    ap.add_argument("--out-dir", required=True,
                    help="Output directory")
    ap.add_argument("--optimal-k", type=int, default=4,
                    help="K value to mark as optimal (default 4)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.in_csv)

    # Pivot to (K -> {S_to_M, M_to_S}) accuracy
    sm = df[df["direction"] == "S_to_M"][["K", "acc_mean", "acc_std"]].sort_values("K")
    ms = df[df["direction"] == "M_to_S"][["K", "acc_mean", "acc_std"]].sort_values("K")

    if len(sm) == 0 or len(ms) == 0:
        raise ValueError("No S_to_M or M_to_S rows found in CSV.")

    Ks = sm["K"].to_numpy()
    sm_acc = sm["acc_mean"].to_numpy()
    sm_std = sm["acc_std"].to_numpy()
    ms_acc = ms["acc_mean"].to_numpy()
    ms_std = ms["acc_std"].to_numpy()

    # X-axis: use integer positions per tick (non-uniform spacing collapsed
    # to evenly-spaced ticks for clarity), matching the previous figure.
    x_positions = np.arange(len(Ks))

    fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=140)

    # Lines + markers (errorbars optional - omitted for clarity, std ~0.005)
    ax.plot(x_positions, sm_acc, marker="o", markersize=8,
            color=BLUE, linewidth=2.0, label=r"Static $\rightarrow$ Mobile")
    ax.plot(x_positions, ms_acc, marker="s", markersize=8,
            color=ORANGE, linewidth=2.0, linestyle="--",
            label=r"Mobile $\rightarrow$ Static")

    # Vertical line + label at the optimal K
    if args.optimal_k in Ks:
        opt_idx = int(np.where(Ks == args.optimal_k)[0][0])
        ax.axvline(x=opt_idx, color=GREEN, linestyle=":", linewidth=1.5, alpha=0.9)
        ymin, ymax = ax.get_ylim() if ax.has_data() else (0.55, 0.92)
        ax.annotate(
            f"K = {args.optimal_k} (optimal)",
            xy=(opt_idx, max(sm_acc[opt_idx], ms_acc[opt_idx])),
            xytext=(opt_idx + 0.3, 0.885),
            fontsize=10, color=GREEN, fontweight="normal",
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=1),
        )

    # Axis formatting
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(k) for k in Ks])
    ax.set_xlabel("Number of features (K)", fontsize=12)
    ax.set_ylabel("Cross-domain accuracy", fontsize=12)
    ax.set_ylim(0.55, 0.92)
    ax.set_yticks(np.arange(0.55, 0.925, 0.05))
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    ax.legend(loc="upper right", fontsize=11, frameon=True,
              edgecolor="#888888")

    plt.tight_layout()

    png_path = out_dir / "figure1_optimal_k.png"
    pdf_path = out_dir / "figure1_optimal_k.pdf"
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")

    # Print summary
    print("\nSummary of values plotted:")
    print(f"{'K':>4} | {'S->M':>8} | {'M->S':>8} | asym")
    print("-" * 40)
    for i, k in enumerate(Ks):
        a = abs(sm_acc[i] - ms_acc[i])
        print(f"{k:>4} | {sm_acc[i]:>.4f}  | {ms_acc[i]:>.4f}  | {a:>.4f}")


if __name__ == "__main__":
    main()
