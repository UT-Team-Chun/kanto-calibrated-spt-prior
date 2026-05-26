#!/usr/bin/env python
"""Render reliability diagrams for the threshold classifier outputs.

For each (threshold, protocol) pair × 3 folds, plot raw vs. isotonic-
recalibrated reliability curves on equal-mass 10 bins.

Inputs (Phase 1d outputs):
  data/runs/threshold_classifiers_{random,contig}/
    {label}_pred_fold{k}.npy
    {label}_pred_iso_fold{k}.npy
    {label}_y_fold{k}.npy

Output:
  docs/paper/paper_1_kanto/figures/fig_threshold_reliability.pdf
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

LOG = logging.getLogger("reliability_plots")

THRESHOLDS = ["lt5", "lt10", "lt15", "gte30"]
THRESHOLD_DISPLAY = {
    "lt5":   r"$P(N < 5)$",
    "lt10":  r"$P(N < 10)$",
    "lt15":  r"$P(N < 15)$",
    "gte30": r"$P(N \geq 30)$",
}


def _reliability_bins(p: np.ndarray, y: np.ndarray, n_bins: int = 10):
    """Equal-mass bins. Returns (mean_predicted, fraction_positive, weight)."""
    order = np.argsort(p)
    p_sorted = p[order]
    y_sorted = y[order]
    n = len(p_sorted)
    edges = np.linspace(0, n, n_bins + 1, dtype=int)
    mean_pred, frac_pos, weight = [], [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if hi <= lo:
            continue
        mean_pred.append(float(p_sorted[lo:hi].mean()))
        frac_pos.append(float(y_sorted[lo:hi].mean()))
        weight.append(int(hi - lo))
    return np.array(mean_pred), np.array(frac_pos), np.array(weight)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = argparse.ArgumentParser()
    p.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    p.add_argument("--n-folds", type=int, default=3)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharex=True, sharey=True)
    for col, label in enumerate(THRESHOLDS):
        for row, run_dir in enumerate(a.run_dirs):
            protocol = "contig" if "contig" in run_dir.name else "random"
            ax = axes[row, col]
            raw_p, iso_p, ys = [], [], []
            for k in range(a.n_folds):
                pr = run_dir / f"{label}_pred_fold{k}.npy"
                pi = run_dir / f"{label}_pred_iso_fold{k}.npy"
                yp = run_dir / f"{label}_y_fold{k}.npy"
                if not (pr.exists() and pi.exists() and yp.exists()):
                    continue
                raw_p.append(np.load(pr))
                iso_p.append(np.load(pi))
                ys.append(np.load(yp))
            if not raw_p:
                ax.text(0.5, 0.5, "no data", ha="center", va="center")
                continue
            raw_all = np.concatenate(raw_p)
            iso_all = np.concatenate(iso_p)
            y_all = np.concatenate(ys)
            # diagonal
            ax.plot([0, 1], [0, 1], color="black", lw=0.7, alpha=0.5, ls="--")
            # raw
            mp_r, fp_r, _ = _reliability_bins(raw_all, y_all)
            ax.plot(mp_r, fp_r, "-o", color="#d62728", lw=1.2, ms=4,
                    label="raw")
            # iso
            mp_i, fp_i, _ = _reliability_bins(iso_all, y_all)
            ax.plot(mp_i, fp_i, "-s", color="#1f4e79", lw=1.2, ms=4,
                    label="isotonic")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.3)
            if row == 0:
                ax.set_title(THRESHOLD_DISPLAY[label], fontsize=11)
            if col == 0:
                ax.set_ylabel(f"{protocol}\nfraction positive",
                              fontsize=10)
            if row == 1:
                ax.set_xlabel("mean predicted probability")
            if row == 0 and col == 0:
                ax.legend(loc="upper left", fontsize=8)

    fig.suptitle("Reliability diagrams: raw vs.\\ isotonic-recalibrated CatBoost threshold classifiers",
                 fontsize=12, y=1.00)
    plt.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight")
    plt.close(fig)
    LOG.info("Wrote %s", a.out)


if __name__ == "__main__":
    main()
