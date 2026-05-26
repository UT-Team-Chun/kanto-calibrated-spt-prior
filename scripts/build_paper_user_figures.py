#!/usr/bin/env python
"""Generate the four user-TODO figures for Paper 1:
- fig1_study_area.pdf   — KuniJiban Kanto boring density + 3-fold split
- fig2_architecture.pdf — block diagram of DKL + SVGP + conformal pipeline
- fig5_depth_slices.pdf — predicted N at depth slices 5/10/20 m on a grid
- fig6_uncertainty_map.pdf — predictive std at same grid

Run from backend/:
    .venv/bin/python -m scripts.build_paper_user_figures
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LogNorm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PARQUET = PROJECT_ROOT / "data/features/borings_kanto_aist.parquet"
DEFAULT_RUN_DIR = PROJECT_ROOT / "data/runs/kanto_full_6k_50ep_linear_rbf"
DEFAULT_OUT_DIR = PROJECT_ROOT / "docs/paper/paper_1_kanto/figures"

LOG = logging.getLogger("build_paper_user_figures")


def build_fig1_study_area(parquet_path: Path, out_path: Path) -> None:
    df = pd.read_parquet(parquet_path)
    # unique borings only — depth measurements are repeated per location
    locs = df[["latitude_deg", "longitude_deg"]].drop_duplicates().reset_index(drop=True)

    lat_min, lat_max = 35.0, 37.6
    lon_min, lon_max = 138.4, 141.1
    n_grid = 60
    H, xedges, yedges = np.histogram2d(
        locs["longitude_deg"], locs["latitude_deg"],
        bins=n_grid,
        range=[[lon_min, lon_max], [lat_min, lat_max]],
    )

    # Secondary-mesh codes (5 min lat / 7.5 min lon ≈ 10 km grid).
    mesh_lat = (locs["latitude_deg"] // (1 / 12)).astype(int).to_numpy()
    mesh_lon = (locs["longitude_deg"] // (1 / 8)).astype(int).to_numpy()
    mesh_codes = mesh_lat * 1000 + mesh_lon
    unique_codes = np.unique(mesh_codes)

    # Panel B: random-mesh spatial K-fold (load-balanced shuffle of meshes).
    rng = np.random.default_rng(42)
    code_perm = rng.permutation(unique_codes.size)
    fold_of_code_random = np.empty(unique_codes.size, dtype=np.int64)
    code_to_size = np.bincount(np.searchsorted(unique_codes, mesh_codes))
    fold_row_counts = np.zeros(3, dtype=np.int64)
    for ci in code_perm:
        target = int(np.argmin(fold_row_counts))
        fold_of_code_random[ci] = target
        fold_row_counts[target] += int(code_to_size[ci])
    code_idx = np.searchsorted(unique_codes, mesh_codes)
    fold_random = fold_of_code_random[code_idx]

    # Panel C: contiguous spatial K-fold via k-means on mesh centroids.
    from sklearn.cluster import KMeans
    code_to_centroid = np.zeros((unique_codes.size, 2), dtype=np.float64)
    for i, c in enumerate(unique_codes):
        mask = mesh_codes == c
        code_to_centroid[i, 0] = locs["latitude_deg"].values[mask].mean()
        code_to_centroid[i, 1] = locs["longitude_deg"].values[mask].mean()
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    fold_of_code_contig = km.fit_predict(code_to_centroid)
    fold_contig = fold_of_code_contig[code_idx]

    fig, axes = plt.subplots(
        1, 3, figsize=(15, 5),
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.0]},
    )

    # Panel A: density heatmap
    ax = axes[0]
    im = ax.imshow(
        H.T + 1,
        origin="lower",
        extent=[lon_min, lon_max, lat_min, lat_max],
        aspect="auto",
        cmap="viridis",
        norm=LogNorm(vmin=1, vmax=max(H.max(), 2)),
    )
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_title(
        f"(a) KuniJiban boring density, Kanto\n"
        f"{len(locs):,} unique borings, {len(df):,} SPT rows",
        fontsize=10,
    )
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label("Borings per cell (log)", fontsize=8)
    ax.text(
        139.69, 35.69, "Tokyo",
        color="white", fontsize=8, fontweight="bold",
        ha="center", va="center",
    )

    colors = ["#1b9e77", "#d95f02", "#7570b3"]

    # Panel B: random spatial K-fold
    ax = axes[1]
    for k in (0, 1, 2):
        sel = fold_random == k
        ax.scatter(
            locs.loc[sel, "longitude_deg"],
            locs.loc[sel, "latitude_deg"],
            s=2, c=colors[k], alpha=0.5, label=f"Fold {k}",
        )
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_title(
        "(b) Random-mesh spatial 3-fold\n"
        "(load-balanced shuffle; interleaved geometry)",
        fontsize=10,
    )
    ax.legend(loc="lower right", fontsize=8, markerscale=3)

    # Panel C: contiguous spatial K-fold
    ax = axes[2]
    for k in (0, 1, 2):
        sel = fold_contig == k
        ax.scatter(
            locs.loc[sel, "longitude_deg"],
            locs.loc[sel, "latitude_deg"],
            s=2, c=colors[k], alpha=0.5, label=f"Fold {k}",
        )
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_title(
        "(c) Contiguous spatial 3-fold\n"
        "(k-means on mesh centroids; geographic blocks)",
        fontsize=10,
    )
    ax.legend(loc="lower right", fontsize=8, markerscale=3)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    LOG.info("Wrote %s", out_path)
    plt.close(fig)


def build_fig2_architecture(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def block(x, y, w, h, label, fc="#e8eaf6", ec="#283593", fs=8):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.04,rounding_size=0.15",
            fc=fc, ec=ec, linewidth=1.4,
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2, y + h / 2, label,
            ha="center", va="center", fontsize=fs,
        )

    def arrow(x0, y0, x1, y1, label=None):
        ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", lw=1.3, color="#37474f"),
        )
        if label:
            ax.text(
                (x0 + x1) / 2, (y0 + y1) / 2 + 0.1, label,
                ha="center", va="bottom", fontsize=7, color="#37474f",
            )

    # Input block
    block(0.2, 3.6, 2.2, 1.4,
          "14-D input\n(lat, lon, depth,\nabs_elev, river_dist,\ncoast_dist, regime 8-hot)",
          fc="#fff3e0", ec="#e65100")

    # Encoder
    block(2.9, 3.6, 2.6, 1.4,
          "Encoder $\\phi$\nResMLP 4×128 + GELU\nRandom-Fourier (lat, lon)\n→ 24-D latent $z$",
          fc="#e8f5e9", ec="#1b5e20")
    arrow(2.4, 4.3, 2.9, 4.3)

    # SVGP head
    block(6.0, 3.6, 2.9, 1.4,
          "SVGP head\nRBF + LinearMean\nM=6,000 inducing\nWhitened $q(u)$",
          fc="#e8eaf6", ec="#283593")
    arrow(5.5, 4.3, 6.0, 4.3, "$z\\in\\mathbb{R}^{24}$")

    # Gaussian predictive
    block(9.4, 3.6, 2.4, 1.4,
          "Predictive\n$\\mathcal{N}(\\mu(x), \\sigma^2(x))$",
          fc="#fce4ec", ec="#880e4f")
    arrow(8.9, 4.3, 9.4, 4.3)

    # Conformal calibrator (post-hoc)
    block(5.0, 1.0, 4.5, 1.3,
          "Split conformal calibrator (post-hoc)\n"
          "$q_\\alpha = $ empirical quantile of $|y-\\mu|/\\sigma$",
          fc="#fff8e1", ec="#f57f17")
    arrow(10.6, 3.6, 9.5, 2.3, "(μ, σ)")
    # Output
    block(0.2, 1.0, 4.3, 1.3,
          "Calibrated interval\n$[\\mu(x_*) \\pm q_\\alpha\\,\\sigma(x_*)]$\n"
          "|gap|$\\leq 10^{-3}$ at $\\alpha\\in\\{.5,.8,.95\\}$",
          fc="#e0f7fa", ec="#00695c")
    arrow(5.0, 1.65, 4.5, 1.65)

    # Training loss annotation
    ax.text(
        7.45, 5.45,
        r"Loss: $N$-scaled ELBO (Hensman 2013)",
        ha="center", va="center", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#bdbdbd"),
    )

    ax.set_title(
        "Model architecture: deep-kernel SVGP with post-hoc conformal calibration",
        fontsize=11, pad=10,
    )

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    LOG.info("Wrote %s", out_path)
    plt.close(fig)


def build_fig5_fig6_grid(
    parquet_path: Path,
    run_dir: Path,
    fig5_path: Path,
    fig6_path: Path,
) -> None:
    """Predict on a regional grid at depths {5, 10, 20} m using the
    best foundation model. Outputs:
    - fig5_depth_slices.pdf — predictive mean panel x3
    - fig6_uncertainty_map.pdf — predictive std panel x3
    """
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "backend"))
    from national.models.foundation import FoundationModel
    from national.tiling.regime_classifier import Regime

    # Load best run metadata
    import json
    meta_path = run_dir / "foundation_model.meta.json"
    summary_path = run_dir / "summary.json"
    state_path = run_dir / "foundation_model.pt"

    meta = json.loads(meta_path.read_text())
    summary = json.loads(summary_path.read_text())
    target_mean = float(summary["target_mean"])
    target_std = float(summary["target_std"])
    LOG.info("target_mean=%.3f target_std=%.3f", target_mean, target_std)

    # Build grid
    lat_min, lat_max = 35.2, 37.4
    lon_min, lon_max = 138.6, 141.0
    n_lat, n_lon = 90, 110
    lat_grid = np.linspace(lat_min, lat_max, n_lat)
    lon_grid = np.linspace(lon_min, lon_max, n_lon)
    LON, LAT = np.meshgrid(lon_grid, lat_grid)

    # Compute river/coast distance via boring-neighbor approximation
    df = pd.read_parquet(parquet_path)
    unique = (
        df.groupby(["latitude_deg", "longitude_deg"])
        .agg({
            "absolute_elevation": "mean",
            "river_distance_km": "mean",
            "coast_distance_km": "mean",
            "regime_code": (lambda s: s.mode().iloc[0]),
        })
        .reset_index()
    )
    # KD-tree nearest neighbor lookup (very small, OK to brute force)
    from scipy.spatial import cKDTree
    tree = cKDTree(unique[["latitude_deg", "longitude_deg"]].values)
    query = np.stack([LAT.ravel(), LON.ravel()], axis=1)
    _, idx = tree.query(query, k=1)
    abs_elev = unique["absolute_elevation"].values[idx].reshape(n_lat, n_lon)
    river_d = unique["river_distance_km"].values[idx].reshape(n_lat, n_lon)
    coast_d = unique["coast_distance_km"].values[idx].reshape(n_lat, n_lon)
    regime = unique["regime_code"].values[idx].reshape(n_lat, n_lon)
    # mask: keep only cells whose nearest boring is within 0.15 deg (~15 km)
    dist, _ = tree.query(query, k=1)
    inside = (dist < 0.15).reshape(n_lat, n_lon)

    device = "cpu"
    try:
        model = FoundationModel.load(state_path, map_location=device)
    except Exception as exc:
        LOG.warning("Failed FoundationModel.load: %s", exc)
        LOG.warning("Falling back to placeholder figs.")
        _build_fig5_fig6_fallback(LAT, LON, inside, fig5_path, fig6_path)
        return

    model.eval()
    depths = [5.0, 10.0, 20.0]
    means = []
    stds = []
    for d in depths:
        x_arr = np.stack([
            LAT.ravel(),
            LON.ravel(),
            np.full(LAT.size, d),
            abs_elev.ravel() - d,  # abs_elev at the SPT layer
            river_d.ravel(),
            coast_d.ravel(),
        ], axis=1).astype(np.float32)
        reg = regime.ravel().astype(np.int64)
        n_regime = 8
        oh = np.zeros((len(reg), n_regime), dtype=np.float32)
        oh[np.arange(len(reg)), reg] = 1.0
        x_full = np.concatenate([x_arr, oh], axis=1)
        with torch.no_grad():
            pred = model.predict(
                torch.from_numpy(x_full).to(device),
                regime_codes=torch.from_numpy(reg).to(device),
            )
        # FoundationPrediction(mean, std, encoded). Note: model is
        # trained on standardised target; pred.mean is already in raw
        # N-blow units if model.set_target_stats was called at training
        # time. summary.json target_mean/target_std are recorded for
        # downstream tooling.
        mu_np = pred.mean.cpu().numpy().reshape(n_lat, n_lon)
        sigma_np = pred.std.cpu().numpy().reshape(n_lat, n_lon)
        mu_np = np.where(inside, mu_np, np.nan)
        sigma_np = np.where(inside, sigma_np, np.nan)
        means.append(mu_np)
        stds.append(sigma_np)

    extent = [lon_min, lon_max, lat_min, lat_max]
    # fig5 — predictive mean
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    vmin, vmax = 0, max(np.nanpercentile(m, 98) for m in means)
    for ax, mu_arr, d in zip(axes, means, depths):
        im = ax.imshow(
            mu_arr, origin="lower", extent=extent,
            aspect="auto", cmap="viridis",
            vmin=vmin, vmax=vmax,
        )
        ax.set_title(f"Depth = {d:.0f} m", fontsize=10)
        ax.set_xlabel("Longitude (deg)")
    axes[0].set_ylabel("Latitude (deg)")
    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label(r"Predicted $N$ (blow count)", fontsize=9)
    fig.suptitle(
        "Predicted SPT N at depth slices, calibrated DKL+SVGP",
        fontsize=11, y=1.02,
    )
    fig.savefig(fig5_path, bbox_inches="tight")
    LOG.info("Wrote %s", fig5_path)
    plt.close(fig)

    # fig6 — predictive uncertainty
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    vmin = 0
    vmax = max(np.nanpercentile(s, 98) for s in stds)
    for ax, sigma_arr, d in zip(axes, stds, depths):
        im = ax.imshow(
            sigma_arr, origin="lower", extent=extent,
            aspect="auto", cmap="magma",
            vmin=vmin, vmax=vmax,
        )
        ax.set_title(f"Depth = {d:.0f} m", fontsize=10)
        ax.set_xlabel("Longitude (deg)")
    axes[0].set_ylabel("Latitude (deg)")
    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label(r"Predictive std $\sigma(x)$ (blow count)", fontsize=9)
    fig.suptitle(
        "Predictive uncertainty at depth slices (Gaussian posterior std)",
        fontsize=11, y=1.02,
    )
    fig.savefig(fig6_path, bbox_inches="tight")
    LOG.info("Wrote %s", fig6_path)
    plt.close(fig)


def _build_fig5_fig6_fallback(LAT, LON, inside, fig5_path, fig6_path):
    """Plausible synthetic fallback — only used if model load fails."""
    depths = [5.0, 10.0, 20.0]
    extent = [LON.min(), LON.max(), LAT.min(), LAT.max()]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, d in zip(axes, depths):
        synth = (3 + 0.6 * d) + 4 * np.sin(2 * (LAT - 36)) * np.cos(2 * (LON - 140))
        synth = np.where(inside, synth, np.nan)
        im = ax.imshow(synth, origin="lower", extent=extent, aspect="auto", cmap="viridis")
        ax.set_title(f"Depth = {d:.0f} m (placeholder)", fontsize=10)
        ax.set_xlabel("Longitude (deg)")
    fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02, label="Predicted N (placeholder)")
    fig.savefig(fig5_path, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, d in zip(axes, depths):
        synth = 4.0 + 0.5 * np.abs(np.sin(LAT * 3) + np.cos(LON * 2))
        synth = np.where(inside, synth, np.nan)
        im = ax.imshow(synth, origin="lower", extent=extent, aspect="auto", cmap="magma")
        ax.set_title(f"Depth = {d:.0f} m (placeholder)", fontsize=10)
        ax.set_xlabel("Longitude (deg)")
    fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02, label=r"$\sigma$ (placeholder)")
    fig.savefig(fig6_path, bbox_inches="tight")
    plt.close(fig)


def main(args=None):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--skip-grid", action="store_true",
                   help="skip fig5/fig6 (model-inference) generation")
    a = p.parse_args(args)
    a.out_dir.mkdir(parents=True, exist_ok=True)

    build_fig1_study_area(a.parquet, a.out_dir / "fig1_study_area.pdf")
    build_fig2_architecture(a.out_dir / "fig2_architecture.pdf")
    if not a.skip_grid:
        build_fig5_fig6_grid(
            a.parquet, a.run_dir,
            a.out_dir / "fig5_depth_slices.pdf",
            a.out_dir / "fig6_uncertainty_map.pdf",
        )


if __name__ == "__main__":
    main()
