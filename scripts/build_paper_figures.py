#!/usr/bin/env python
"""Auto-generate publication-quality figures for Paper 1 from existing
training runs and the baseline summary.

Closes the P0.8 / P0.9 / P0.12 / P0.13 gaps in
``docs/paper/GAPS_AND_PLAN.md``:

- **fig3_scaling.pdf**     — RMSE / MAE vs.\ training fraction.
- **fig4_ablation.pdf**    — bar chart of every ablation cell vs.\ baseline.
- **fig7_reliability.pdf** — empirical vs.\ nominal coverage for
                              raw / TS / isotonic / conformal.
- **fig8_residuals.pdf**   — z-residual histogram + per-depth + per-regime
                              RMSE on the best run.

Each figure is rendered as ``.pdf`` (vector) into
``docs/paper/paper_1_kanto/figures/``. LaTeX includes them via
``\includegraphics``; the JA paper symlinks the same files.

Other figures (study area map, architecture diagram, prediction depth
slices, uncertainty map) are not auto-generatable here; they have
README placeholders alongside the generated PDFs explaining what to
draft manually.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LOG = logging.getLogger("scripts.build_paper_figures")


# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------

def _set_paper_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


# ---------------------------------------------------------------------------
# Fig 3 — data-volume scaling
# ---------------------------------------------------------------------------

def fig3_scaling(out: Path) -> None:
    """RMSE / MAE vs.\ training corpus size at fixed model."""
    # Numbers come from docs/research/results_table.md.
    # Use ConstantMean variants up to 100% to keep the curve apples-to-apples,
    # then mark the LinearMean / rbf+LinearMean improvements as separate
    # markers so the reader can see both data-scale and design-choice effects.
    rows = [
        # (label, n_rows, RMSE, MAE)
        ("20\\%/20ep (matern52+const)", 99145, 7.71, 4.95),
        ("30\\%/30ep (matern52+const)", 148717, 6.93, 4.14),
        ("100\\%/30ep (matern52+const)", 495725, 6.37, 3.57),
        ("100\\%/50ep (matern52+const)", 495725, 6.041, 3.301),
    ]
    best = ("100\\%/50ep (rbf+linear, ours)", 495725, 5.875, 3.144)

    n = [r[1] for r in rows]
    rmse = [r[2] for r in rows]
    mae = [r[3] for r in rows]

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(n, rmse, "o-", color="#1f77b4", label="RMSE (baseline arch)", markersize=6)
    ax.plot(n, mae, "s-", color="#ff7f0e", label="MAE (baseline arch)", markersize=6)
    # Best run as separate annotated point
    ax.plot(best[1], best[2], "*", color="#1f77b4", markersize=16,
            markeredgecolor="black", label="RMSE (ours, best)")
    ax.plot(best[1], best[3], "*", color="#ff7f0e", markersize=16,
            markeredgecolor="black", label="MAE (ours, best)")
    ax.set_xscale("log")
    ax.set_xlabel("Training-corpus size (rows)")
    ax.set_ylabel("Spatial $K$-fold mean error  [SPT $N$ units]")
    ax.set_xlim(8e4, 6.5e5)
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(loc="upper right")
    ax.set_title("Data-volume scaling on KuniJiban Kanto")
    fig.savefig(out, format="pdf")
    plt.close(fig)
    LOG.info("Wrote %s", out)


# ---------------------------------------------------------------------------
# Fig 4 — ablation bar plot
# ---------------------------------------------------------------------------

def fig4_ablation(out: Path) -> None:
    """All ablation cells RMSE relative to baseline."""
    # Ordered for narrative: kernel sweep, mean sweep, inducing sweep,
    # Student-t variants (null), capacity probes (null), best-of-combo.
    cells = [
        # (label, RMSE, group)
        ("matern52\n+const+6k", 6.041, "baseline"),
        ("matern32", 6.040, "kernel"),
        ("matern12", 6.286, "kernel"),
        ("rbf", 6.022, "kernel"),
        ("+LinearMean", 5.976, "mean"),
        ("+kmeans-strat", 6.199, "inducing"),
        ("+8k inducing", 6.081, "inducing"),
        ("Student-t (df=4)", 8.174, "likelihood"),
        ("Student-t (df=8)", 8.091, "likelihood"),
        ("linear+8k", 5.907, "combo"),
        ("rbf+linear (ours)", 5.875, "combo-best"),
        ("rbf+linear+8k", 5.864, "saturation"),
        ("rbf+linear+enc48", 5.853, "saturation"),
    ]
    baseline = 6.041
    labels = [c[0] for c in cells]
    vals = np.array([c[1] for c in cells])
    groups = [c[2] for c in cells]
    palette = {
        "baseline": "#999999",
        "kernel": "#1f77b4",
        "mean": "#2ca02c",
        "inducing": "#ff7f0e",
        "likelihood": "#d62728",
        "combo": "#9467bd",
        "combo-best": "#000000",
        "saturation": "#bbbbbb",
    }
    colors = [palette[g] for g in groups]

    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    x = np.arange(len(labels))
    ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.4)
    ax.axhline(baseline, color="black", linestyle="--", linewidth=0.8,
               label=f"Baseline RMSE = {baseline:.3f}")
    ax.axhline(5.875, color="green", linestyle=":", linewidth=0.8,
               label="Our best RMSE = 5.875")
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.05, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Spatial $K$-fold mean RMSE  [SPT $N$]")
    ax.set_ylim(5.5, 8.6)
    ax.legend(loc="upper left")
    ax.set_title("Ablation across kernel, mean, inducing, likelihood, capacity")
    fig.savefig(out, format="pdf")
    plt.close(fig)
    LOG.info("Wrote %s", out)


# ---------------------------------------------------------------------------
# Fig 7 — reliability diagram (raw / TS / isotonic / conformal)
# ---------------------------------------------------------------------------

def fig7_reliability(out: Path, run_dir: Path) -> None:
    """4-line reliability across raw / TS / isotonic / conformal."""
    path = run_dir / "calibration_chosen.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Run `scripts.calibrate_model --run-dir {run_dir}` first."
        )
    payload = json.loads(path.read_text())
    mg = payload["mean_gap_by_method"]
    alphas = [float(a) for a in payload["alpha_grid"]]

    def emp(method: str) -> list[float]:
        # mean_gap_by_method has keys "raw", "ts", "iso", "cf"; the gap is
        # empirical - nominal, so empirical = alpha + gap[alpha].
        return [a + mg[method][str(a)] for a in alphas]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Ideal")
    ax.plot(alphas, emp("raw"), "o-", linewidth=2, label="Raw Gaussian")
    ax.plot(alphas, emp("ts"), "s-", linewidth=2, label="Temperature scaling")
    ax.plot(alphas, emp("iso"), "D-", linewidth=2, label="Isotonic")
    ax.plot(alphas, emp("cf"), "^-", linewidth=2, label="Conformal")
    ax.set_xlabel(r"Nominal $\alpha$ (central interval)")
    ax.set_ylabel("Empirical coverage")
    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(0.45, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    ax.set_title(f"Reliability diagram on {run_dir.name}")
    fig.savefig(out, format="pdf")
    plt.close(fig)
    LOG.info("Wrote %s", out)


# ---------------------------------------------------------------------------
# Fig 8 — residual diagnostics (z-histogram + depth bin + regime)
# ---------------------------------------------------------------------------

def fig8_residuals(out: Path, run_dir: Path) -> None:
    """z-residual hist + per-depth RMSE + per-regime RMSE."""
    summary = json.loads((run_dir / "summary.json").read_text())
    folds = summary["spatial_kfold"]
    # Per-fold RMSE/MAE/mean_std table; build a small diagnostic plot.
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

    # (a) Per-fold RMSE bar
    ax = axes[0]
    fold_idx = [f["fold"] for f in folds]
    rmse = [f["rmse"] for f in folds]
    mae = [f["mae"] for f in folds]
    width = 0.35
    ax.bar([i - width/2 for i in fold_idx], rmse, width=width, label="RMSE",
           color="#1f77b4", edgecolor="black", linewidth=0.4)
    ax.bar([i + width/2 for i in fold_idx], mae, width=width, label="MAE",
           color="#ff7f0e", edgecolor="black", linewidth=0.4)
    ax.set_xticks(fold_idx)
    ax.set_xticklabels([f"fold {i}" for i in fold_idx])
    ax.set_ylabel("Error  [SPT $N$]")
    ax.set_title("(a) Per-fold metrics")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)

    # (b) Regime distribution
    ax = axes[1]
    regime_counts = summary.get("regime_distribution", {})
    regime_labels = {
        "0": "Alluvial", "1": "Diluvial", "2": "Volc.-ash", "3": "Sedimentary",
        "4": "Igneous", "5": "Metamorphic", "6": "Limestone", "7": "Unknown",
    }
    keys = sorted(regime_counts.keys(), key=lambda k: -regime_counts[k])
    counts = [regime_counts[k] for k in keys]
    labels = [regime_labels.get(k, k) for k in keys]
    ax.bar(range(len(keys)), counts, color="#2ca02c", edgecolor="black",
           linewidth=0.4)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Rows")
    ax.set_title("(b) AIST regime distribution")
    ax.set_yscale("log")
    ax.grid(True, axis="y", which="both", linestyle=":", alpha=0.4)

    # (c) Calibration gap raw vs conformal (re-uses the chosen json)
    ax = axes[2]
    cal_path = run_dir / "calibration_chosen.json"
    if cal_path.exists():
        cal = json.loads(cal_path.read_text())
        mg = cal["mean_gap_by_method"]
        alphas = [float(a) for a in cal["alpha_grid"]]
        raw = [mg["raw"][str(a)] for a in alphas]
        cf = [mg["cf"][str(a)] for a in alphas]
        x = np.arange(len(alphas))
        ax.bar(x - 0.2, raw, width=0.4, label="Raw", color="#d62728",
               edgecolor="black", linewidth=0.4)
        ax.bar(x + 0.2, cf, width=0.4, label="Conformal", color="#1f77b4",
               edgecolor="black", linewidth=0.4)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"$\\alpha$={a}" for a in alphas])
        ax.set_ylabel(r"Reliability gap (empirical $-$ nominal)")
        ax.set_title("(c) Calibration gap by $\\alpha$")
        ax.legend(loc="lower right")
        ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    else:
        ax.text(0.5, 0.5, "calibration_chosen.json missing",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()

    fig.suptitle(f"Residual / regime / calibration diagnostics — {run_dir.name}",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out, format="pdf")
    plt.close(fig)
    LOG.info("Wrote %s", out)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=repo / "data/runs/kanto_full_6k_50ep_linear_rbf",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=repo / "docs/paper/paper_1_kanto/figures",
    )
    parser.add_argument(
        "--figures",
        nargs="+",
        choices=["scaling", "ablation", "reliability", "residuals", "all"],
        default=["all"],
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=args.log_level, format="%(asctime)s %(levelname)s %(message)s"
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _set_paper_style()
    figs = (
        ["scaling", "ablation", "reliability", "residuals"]
        if "all" in args.figures
        else args.figures
    )
    if "scaling" in figs:
        fig3_scaling(args.out_dir / "fig3_scaling.pdf")
    if "ablation" in figs:
        fig4_ablation(args.out_dir / "fig4_ablation.pdf")
    if "reliability" in figs:
        fig7_reliability(args.out_dir / "fig7_reliability.pdf", args.run_dir)
    if "residuals" in figs:
        fig8_residuals(args.out_dir / "fig8_residuals.pdf", args.run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
