#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_threshold_decomposition.py
===============================
Figures for the threshold-vs-ranking decomposition (threshold_decomposition_v2.py).

Figure A — Few-shot recalibration curve:
  accuracy(target-test) vs n labeled target RUNS, for K=33 and K=4, with the
  source-threshold (operational) and oracle accuracies as horizontal asymptotes.
  One panel per direction (S->M, M->S).  Reads fewshot_curve.csv.

Figure B — Score-distribution drift (mechanism):
  2x2 grid (rows: K=33, K=4; cols: S->M, M->S).  Each panel overlays the
  predicted-probability histograms of source-test vs target-test, with the
  source threshold t*_src and target-optimal threshold t*_tgt marked, annotated
  with |Delta t*| and W1(logit).  Reads hist_data.npz + mechanism_stats.csv.

Reads from:
  <in-dir>/fewshot_curve.csv
  <in-dir>/hist_data.npz
  <in-dir>/mechanism_stats.csv

Writes:
  <in-dir>/fig_fewshot_recalibration.{png,pdf}
  <in-dir>/fig_score_drift.{png,pdf}

Usage:
    python3 plot_threshold_decomposition.py --in-dir ./results/threshold_decomposition
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE = "#1f77b4"     # K=33  / source-test
ORANGE = "#ff7f0e"   # K=4   / target-test
GREEN = "#388e3c"
GREY = "#666666"
DIRS = [("S_to_M", r"Static $\rightarrow$ Mobile"),
        ("M_to_S", r"Mobile $\rightarrow$ Static")]


# ----------------------------------------------------------------------
def plot_fewshot(in_dir, out_dir):
    df = pd.read_csv(in_dir / "fewshot_curve.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), dpi=140, sharey=False)

    for ax, (dkey, dlabel) in zip(axes, DIRS):
        sub = df[df.direction == dkey]
        for K, color, mark in ((33, BLUE, "o"), (4, ORANGE, "s")):
            s = sub[sub.K == K].sort_values("n_runs")
            if s.empty:
                continue
            n = s["n_runs"].to_numpy()
            acc = s["acc_thresh_mean"].to_numpy()
            std = s["acc_thresh_std"].to_numpy()
            ax.errorbar(n, acc, yerr=std, marker=mark, color=color, lw=2,
                        capsize=3, label=f"K={K}: few-shot threshold")
            # isotonic arm (where available)
            iso = s["acc_iso_mean"].to_numpy()
            if np.isfinite(iso).any():
                m = np.isfinite(iso)
                ax.plot(n[m], iso[m], marker=mark, color=color, lw=1.2,
                        ls=":", alpha=0.8, label=f"K={K}: few-shot + isotonic")
            # reference asymptotes (source-thr = operational, oracle = ceiling)
            src_ref = float(s["acc_source_thr_ref"].iloc[0])
            orc_ref = float(s["acc_oracle_ref"].iloc[0])
            ax.axhline(src_ref, color=color, ls="--", lw=1.2, alpha=0.55)
            ax.axhline(orc_ref, color=color, ls="-", lw=1.0, alpha=0.4)
            x_lab = n[0] * 0.92
            ax.text(x_lab, src_ref, f"source-thr K={K}", color=color,
                    fontsize=7.5, va="bottom", ha="left", alpha=0.9)
            ax.text(x_lab, orc_ref, f"oracle K={K}", color=color,
                    fontsize=7.5, va="bottom", ha="left", alpha=0.9)

        ax.set_xscale("log")
        ax.set_xticks(sorted(sub["n_runs"].unique()))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlim(n[0] * 0.78, n[-1] * 1.15)
        ax.set_xlabel("Labeled target runs used to recalibrate threshold (n)",
                      fontsize=11)
        ax.set_ylabel("Cross-domain accuracy (target-test fold)", fontsize=11)
        ax.set_title(dlabel, fontsize=12)
        ax.grid(True, alpha=0.3, lw=0.5)
        ax.legend(loc="center right", fontsize=8, frameon=True)

    fig.suptitle("Few-shot threshold recalibration: a few labeled target "
                 "scenarios recover most of the cross-domain accuracy",
                 fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("png", "pdf"):
        p = out_dir / f"fig_fewshot_recalibration.{ext}"
        plt.savefig(p, dpi=200, bbox_inches="tight")
        print(f"Saved: {p}")
    plt.close(fig)


# ----------------------------------------------------------------------
def plot_score_drift(in_dir, out_dir):
    npz = np.load(in_dir / "hist_data.npz")
    mech = pd.read_csv(in_dir / "mechanism_stats.csv")
    bins = np.linspace(0, 1, 41)

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), dpi=140,
                             sharex=True, sharey=False)
    for ri, K in enumerate((33, 4)):
        for ci, (dkey, dlabel) in enumerate(DIRS):
            ax = axes[ri][ci]
            p_src = npz[f"{dkey}|K{K}|src_test"]
            p_tgt = npz[f"{dkey}|K{K}|tgt_test"]
            t_src = float(npz[f"{dkey}|K{K}|t_src"])
            t_tgt = float(npz[f"{dkey}|K{K}|t_tgt"])
            ax.hist(p_src, bins=bins, density=True, color=BLUE, alpha=0.55,
                    label="source-test")
            ax.hist(p_tgt, bins=bins, density=True, color=ORANGE, alpha=0.55,
                    label="target-test")
            ax.axvline(t_src, color=BLUE, ls="--", lw=2,
                       label=f"$t^*_{{src}}$={t_src:.2f}")
            ax.axvline(t_tgt, color=ORANGE, ls="-", lw=2,
                       label=f"$t^*_{{tgt}}$={t_tgt:.2f}")

            mr = mech[(mech.direction == dkey) & (mech.K == K)]
            if not mr.empty:
                dt = float(mr["abs_t_shift"].iloc[0])
                w1 = float(mr["w1_logit"].iloc[0])
                ax.text(0.03, 0.95,
                        f"$|\\Delta t^*|$={dt:.2f}\n$W_1$(logit)={w1:.2f}",
                        transform=ax.transAxes, va="top", ha="left", fontsize=10,
                        bbox=dict(boxstyle="round", fc="white", ec=GREY, alpha=0.85))
            ax.set_title(f"{dlabel}  —  K={K}", fontsize=11)
            ax.grid(True, alpha=0.25, lw=0.5)
            if ri == 1:
                ax.set_xlabel("Predicted probability  P(defense active)", fontsize=10)
            if ci == 0:
                ax.set_ylabel("Density", fontsize=10)
            ax.legend(loc="upper right", fontsize=8, frameon=True)

    fig.suptitle("Score-distribution drift under domain shift: large at K=33 "
                 "(threshold lands badly), small at UNIVERSAL-4 (stable)",
                 fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("png", "pdf"):
        p = out_dir / f"fig_score_drift.{ext}"
        plt.savefig(p, dpi=200, bbox_inches="tight")
        print(f"Saved: {p}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", default="./results/threshold_decomposition")
    ap.add_argument("--out-dir", default=None,
                    help="defaults to --in-dir")
    args = ap.parse_args()
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir) if args.out_dir else in_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_fewshot(in_dir, out_dir)
    plot_score_drift(in_dir, out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
