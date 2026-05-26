#!/usr/bin/env python
"""Apply Mondrian and locally-weighted conformal prediction to an existing
DKL+SVGP (or hybrid CatBoost+SVGP) run, using its saved ``predictions.npz``.

Reads:
  --run-dir/predictions.npz  : pred_mean, pred_std, y_true, regime,
                               baseline_pred, hybrid_mode  (from train_kanto_smoke.py)
  --parquet                  : the row-level parquet (for depth, lat/lon
                               needed for locally-weighted features)
  --fold-assignment {random,contiguous}

Computes:
  * Marginal conformal (reference)
  * Mondrian conformal grouped by:
      - AIST regime
      - depth bin (6 standard bins)
      - predicted-mu quintile
      - predicted-sigma quintile
      - (regime, depth_bin) joint
  * Locally-weighted conformal in two feature spaces:
      - geographic = (lat, lon, depth, regime-one-hot)
      - encoder-latent (optional, requires --model-path)
    × 3 bandwidths (0.05, 0.1, 0.2 in standardised units)

Writes:
  --run-dir/conditional_conformal_<fold-assignment>.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG = logging.getLogger("locally_weighted_conformal")


DEPTH_BIN_EDGES = (0.0, 2.0, 5.0, 10.0, 20.0, 50.0, float("inf"))
REGIME_NAMES = (
    "Alluvial", "Volcanic", "Sedimentary", "Limestone",
    "Granite", "Metamorphic", "Igneous", "Unknown",
)


def _depth_bin(depths: np.ndarray) -> np.ndarray:
    edges = np.asarray(DEPTH_BIN_EDGES, dtype=np.float64)
    return np.clip(np.digitize(depths, edges) - 1, 0, len(edges) - 2)


def _quintile(values: np.ndarray) -> np.ndarray:
    """Return 0..4 quintile labels using empirical percentiles."""
    edges = np.percentile(values, [20, 40, 60, 80])
    return np.digitize(values, edges)


def _cov_width(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> dict[str, float]:
    coverage = float(((y >= lo) & (y <= hi)).mean())
    width = float((hi - lo).mean())
    return {"coverage": coverage, "width": width}


def _interval_score(y: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                    alpha: float) -> float:
    """Winkler / interval score (lower is better)."""
    width = hi - lo
    miss_lo = np.maximum(lo - y, 0.0)
    miss_hi = np.maximum(y - hi, 0.0)
    return float((width + (2.0 / (1.0 - alpha)) * (miss_lo + miss_hi)).mean())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True,
                   help="Directory containing predictions.npz")
    p.add_argument("--parquet", type=Path,
                   default=PROJECT_ROOT / "data/features/borings_kanto_aist.parquet")
    p.add_argument("--fold-assignment", choices=["random", "contiguous"],
                   default="random")
    p.add_argument("--alphas", nargs="+", type=float,
                   default=[0.5, 0.8, 0.95])
    p.add_argument("--bandwidths", nargs="+", type=float,
                   default=[0.05, 0.1, 0.2])
    p.add_argument("--cal-fraction", type=float, default=0.2,
                   help="Per-fold inner mesh-disjoint calibration fraction")
    p.add_argument("--min-group-n", type=int, default=30)
    p.add_argument("--n-folds", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--locally-weighted-sample-cap", type=int, default=2000,
        help="Cap on test-row sample size when computing locally-weighted "
             "intervals to control O(n_test * n_cal) cost. The reported "
             "metrics are the per-row averages over the sample.",
    )
    a = p.parse_args()

    pred_path = a.run_dir / "predictions.npz"
    if not pred_path.exists():
        raise FileNotFoundError(f"No predictions.npz at {pred_path}; "
                                 "re-run train_kanto_smoke.py with the "
                                 "updated CLI that dumps predictions.")
    LOG.info("Loading %s", pred_path)
    arrays = np.load(pred_path)
    pred_mean = arrays["pred_mean"].astype(np.float64)
    pred_std = arrays["pred_std"].astype(np.float64)
    y_true = arrays["y_true"].astype(np.float64)
    regime = arrays["regime"].astype(np.int64)
    baseline_pred = arrays["baseline_pred"].astype(np.float64)
    hybrid_mode = bool(int(arrays["hybrid_mode"][0]))
    LOG.info("Loaded %d test rows (hybrid=%s)", len(y_true), hybrid_mode)

    LOG.info("Loading parquet %s for lat/lon/depth", a.parquet)
    df = pd.read_parquet(a.parquet, columns=[
        "latitude_deg", "longitude_deg", "depth_from_surface",
    ])
    if len(df) != len(y_true):
        raise RuntimeError(
            f"Parquet rows ({len(df)}) and predictions rows ({len(y_true)}) "
            "differ. The smoke trainer must have been run with "
            "--train-fraction 1.0 and no subsetting for this driver to be "
            "applicable."
        )
    lat = df["latitude_deg"].to_numpy(dtype=np.float64)
    lon = df["longitude_deg"].to_numpy(dtype=np.float64)
    depth = df["depth_from_surface"].to_numpy(dtype=np.float64)

    from run_advanced_baselines import assign_folds, secondary_mesh_code

    fold = assign_folds(df, n_folds=a.n_folds, seed=a.seed,
                        assignment=a.fold_assignment)
    codes = np.array([secondary_mesh_code(la, lo) for la, lo in zip(lat, lon)])

    # Subgroup labels (independent of fold)
    depth_bin = _depth_bin(depth)
    mu_quintile = _quintile(pred_mean)
    sigma_quintile = _quintile(pred_std)
    regime_depth_joint = regime * 100 + depth_bin

    from national.evaluation.calibration import ConformalCalibrator

    results: dict = {
        "run_dir": str(a.run_dir),
        "hybrid_mode": hybrid_mode,
        "fold_assignment": a.fold_assignment,
        "alphas": list(a.alphas),
        "bandwidths": list(a.bandwidths),
        "per_fold": [],
    }

    for k in range(a.n_folds):
        LOG.info("==== fold %d ====", k)
        train_mask = fold != k
        test_mask = fold == k

        # Inner mesh-disjoint cal split (matches BoringDataset / smoke
        # convention).
        train_codes = codes[train_mask]
        unique_inner, inv_inner = np.unique(train_codes, return_inverse=True)
        rng = np.random.default_rng(a.seed + 1000 + 10 * k)
        n_cal_meshes = max(1, int(unique_inner.size * a.cal_fraction))
        cal_mesh_idx = rng.choice(unique_inner.size, size=n_cal_meshes, replace=False)
        cal_mesh_mask = np.zeros(unique_inner.size, dtype=bool)
        cal_mesh_mask[cal_mesh_idx] = True
        cal_inner = cal_mesh_mask[inv_inner]
        train_row_indices = np.where(train_mask)[0]
        cal_row_indices = train_row_indices[cal_inner]
        # row indices for cal vs test
        test_idx = np.where(test_mask)[0]
        LOG.info("    n_cal=%d, n_test=%d", len(cal_row_indices), len(test_idx))

        cal_y = y_true[cal_row_indices]
        cal_mu = pred_mean[cal_row_indices]
        cal_sigma = np.maximum(pred_std[cal_row_indices], 1e-3)
        test_y = y_true[test_idx]
        test_mu = pred_mean[test_idx]
        test_sigma = np.maximum(pred_std[test_idx], 1e-3)

        fold_record: dict = {"fold": k,
                              "n_cal": int(len(cal_y)),
                              "n_test": int(len(test_y))}

        cal = ConformalCalibrator()
        cal.fit(cal_y, cal_mu, cal_sigma, alphas=a.alphas)
        for alpha in a.alphas:
            lo, hi = cal.interval(test_mu, test_sigma, alpha)
            cw = _cov_width(test_y, lo, hi)
            fold_record[f"marginal_alpha_{alpha:.2f}"] = {
                **cw,
                "interval_score": _interval_score(test_y, lo, hi, alpha),
            }

        # Mondrian over five groupings
        mondrian_groupings = {
            "regime": (regime[cal_row_indices], regime[test_idx]),
            "depth_bin": (depth_bin[cal_row_indices], depth_bin[test_idx]),
            "mu_quintile": (mu_quintile[cal_row_indices], mu_quintile[test_idx]),
            "sigma_quintile": (sigma_quintile[cal_row_indices], sigma_quintile[test_idx]),
            "regime_depth": (regime_depth_joint[cal_row_indices], regime_depth_joint[test_idx]),
        }
        for name, (g_cal, g_test) in mondrian_groupings.items():
            cal_m = ConformalCalibrator()
            cal_m.fit_mondrian(cal_y, cal_mu, cal_sigma, g_cal,
                                alphas=a.alphas, min_group_n=a.min_group_n)
            for alpha in a.alphas:
                lo, hi = cal_m.interval_mondrian(test_mu, test_sigma, g_test, alpha)
                cw = _cov_width(test_y, lo, hi)
                fold_record[f"mondrian_{name}_alpha_{alpha:.2f}"] = {
                    **cw,
                    "interval_score": _interval_score(test_y, lo, hi, alpha),
                    "n_groups": int(np.unique(g_cal).size),
                    "n_groups_with_per_group_quantile": (
                        int(len(cal_m.quantiles_per_group or {}))
                    ),
                }

        # Conditional coverage on stiff-layer + rare regimes (the
        # headline failure modes from the existing paper) for every method
        # — easier than re-running full Mondrian for each
        n30_mask_test = test_y >= 30.0
        for tag, mask in [
            ("n_ge_30", n30_mask_test),
            ("regime_igneous", regime[test_idx] == 6),
            ("regime_metamorphic", regime[test_idx] == 5),
            ("depth_ge_20", depth[test_idx] >= 20.0),
        ]:
            if mask.sum() < 10:
                continue
            for alpha in a.alphas:
                lo_marg, hi_marg = cal.interval(test_mu, test_sigma, alpha)
                fold_record[f"conditional_{tag}_marginal_alpha_{alpha:.2f}"] = {
                    "coverage": float(((test_y[mask] >= lo_marg[mask])
                                       & (test_y[mask] <= hi_marg[mask])).mean()),
                    "n_subset": int(mask.sum()),
                }
                # And under the Mondrian (regime_depth) variant
                cal_m_rd = ConformalCalibrator()
                cal_m_rd.fit_mondrian(
                    cal_y, cal_mu, cal_sigma,
                    regime_depth_joint[cal_row_indices],
                    alphas=a.alphas, min_group_n=a.min_group_n,
                )
                lo_m, hi_m = cal_m_rd.interval_mondrian(
                    test_mu, test_sigma, regime_depth_joint[test_idx], alpha
                )
                fold_record[f"conditional_{tag}_mondrian_regime_depth_alpha_{alpha:.2f}"] = {
                    "coverage": float(((test_y[mask] >= lo_m[mask])
                                       & (test_y[mask] <= hi_m[mask])).mean()),
                }

        # Locally-weighted on a capped random sample of test rows
        cal_scores = np.abs(cal_y - cal_mu) / cal_sigma
        # Geographic features (standardised)
        geo_cal = np.stack([lat[cal_row_indices], lon[cal_row_indices],
                             depth[cal_row_indices], regime[cal_row_indices]], axis=1)
        geo_test = np.stack([lat[test_idx], lon[test_idx],
                              depth[test_idx], regime[test_idx]], axis=1)
        geo_mean = geo_cal.mean(axis=0)
        geo_std = geo_cal.std(axis=0) + 1e-9
        geo_cal_z = (geo_cal - geo_mean) / geo_std
        geo_test_z = (geo_test - geo_mean) / geo_std

        sample_cap = min(int(a.locally_weighted_sample_cap), len(test_idx))
        sample_idx = rng.choice(len(test_idx), size=sample_cap, replace=False)
        sample_y = test_y[sample_idx]
        sample_mu = test_mu[sample_idx]
        sample_sigma = test_sigma[sample_idx]
        sample_geo_z = geo_test_z[sample_idx]

        for bw in a.bandwidths:
            cal_lw = ConformalCalibrator()
            for alpha in a.alphas:
                lo, hi = cal_lw.interval_locally_weighted(
                    sample_mu, sample_sigma, geo_cal_z, cal_scores,
                    sample_geo_z, alpha=alpha, bandwidth=float(bw),
                )
                cw = _cov_width(sample_y, lo, hi)
                fold_record[f"locally_weighted_geo_bw{bw:g}_alpha_{alpha:.2f}"] = {
                    **cw,
                    "interval_score": _interval_score(sample_y, lo, hi, alpha),
                    "n_sample": sample_cap,
                }

        results["per_fold"].append(fold_record)
        LOG.info("    fold %d done", k)

    out_path = a.run_dir / f"conditional_conformal_{a.fold_assignment}.json"
    out_path.write_text(json.dumps(results, indent=2, default=float))
    LOG.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
