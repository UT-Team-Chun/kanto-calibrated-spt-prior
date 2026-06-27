#!/usr/bin/env python
"""Non-neural baselines on the same spatial K-fold protocol as the
DKL/SVGP foundation model.

Closes the P0.1 / P0.2 / P0.3 / P0.5 gaps in
``docs/paper/GAPS_AND_PLAN.md``: reviewer-required head-to-head
against classical and modern baselines.

Baselines:
- **IDW** (inverse-distance weighting): 3-D over (lat-km, lon-km,
  depth-km). Cheap and parameter-free, the geotechnical-engineering
  default.
- **Ordinary Kriging**: ``sklearn.gaussian_process`` Matern kernel
  fit on a 10-k subsample of training rows per fold. The classical
  geostatistics comparison; full-data kriging on 165-k+ rows is
  intractable (O(N^3) Cholesky).
- **Random Forest**: ``sklearn.ensemble.RandomForestRegressor`` on
  the same 14-D input as our SVGP model (lat, lon, depth, abs_elev,
  river_dist, coast_dist, regime one-hot).
- **HGB**: ``HistGradientBoostingRegressor`` — sklearn's drop-in for
  XGBoost. Same 14-D input.

All baselines use the same spatial K-fold splits seeded by the
foundation model's training (seed=42, mesh_level=2) so the row
indices match what we report as the SVGP headline. Per-fold and
mean RMSE/MAE are stored in ``data/runs/kanto/baselines/summary.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from national.data.boring_dataset import BoringDataset
from national.evaluation.baselines import (
    fit_predict_hgb,
    fit_predict_idw,
    fit_predict_kriging,
    fit_predict_rf,
)
from national.evaluation.spatial_kfold import (
    spatial_kfold_split,
    spatial_kfold_split_buffered,
    spatial_kfold_split_contiguous,
)

LOG = logging.getLogger("scripts.run_baselines")


def _project_xyz_km(lat: np.ndarray, lon: np.ndarray, depth_m: np.ndarray) -> np.ndarray:
    """Lat/lon → local km, plus depth_km. Equirectangular at the mean lat."""
    ref_lat = float(np.mean(lat))
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * max(0.1, np.cos(np.radians(ref_lat)))
    ref_lon = float(np.mean(lon))
    x = (lon - ref_lon) * km_per_deg_lon
    y = (lat - ref_lat) * km_per_deg_lat
    z = depth_m * 0.001  # m → km so the ARD lengthscale is in the same unit
    return np.stack([y, x, z], axis=1).astype(np.float64)


def main(argv: list[str] | None = None) -> int:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parquet",
        type=Path,
        default=repo / "data/features/borings_kanto_aist.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo / "data/runs/kanto/baselines",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--mesh-level", type=int, default=2)
    parser.add_argument(
        "--fold-assignment",
        choices=["random", "contiguous", "buffered-contiguous"],
        default="random",
        help="random mesh-shuffle (default), contiguous geographic blocks, "
             "or contiguous blocks with a 1-mesh buffer ring (R1.3 spatial "
             "extrapolation comparison for kriging/IDW).",
    )
    parser.add_argument(
        "--buffer-meshes", type=int, default=1,
        help="Ring size when --fold-assignment buffered-contiguous.",
    )
    parser.add_argument(
        "--baselines",
        nargs="+",
        choices=["idw", "kriging", "rf", "hgb"],
        default=["idw", "kriging", "rf", "hgb"],
    )
    parser.add_argument(
        "--kriging-subsample",
        type=int,
        default=10_000,
        help="Sub-sample size for kriging (full-data kriging is intractable)",
    )
    parser.add_argument(
        "--rf-trees",
        type=int,
        default=500,
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load BoringDataset to get the same 14-D feature matrix ---
    LOG.info("Loading %s", args.parquet)
    ds = BoringDataset(
        args.parquet,
        feature_columns=["absolute_elevation", "river_distance_km", "coast_distance_km"],
        depth_scale_m=30.0,
        standardize_target=True,
        regime_one_hot=True,
        target_transform="none",
    )
    # Recover the raw 14-D x and the *raw-unit* target so metrics are
    # directly comparable to the foundation-model headline numbers.
    x_full = ds._x.astype(np.float64)                # (N, 14)
    y_full = ds._y_raw.astype(np.float64)            # (N,) raw N value
    lat = ds._x[:, 0].astype(np.float64)
    lon = ds._x[:, 1].astype(np.float64)
    depth = ds._x[:, 2].astype(np.float64) * 30.0    # un-normalise
    xyz_km = _project_xyz_km(lat, lon, depth)         # (N, 3) for IDW/Kriging
    LOG.info("Loaded %d rows, %d features", x_full.shape[0], x_full.shape[1])

    # ---- 2. Same spatial K-fold split as the SVGP run -----------------
    sub_df = pd.DataFrame({
        "latitude_deg": lat,
        "longitude_deg": lon,
        "n_value": y_full,
    })
    if args.fold_assignment == "random":
        folds = spatial_kfold_split(
            sub_df, n_folds=args.n_folds, mesh_level=args.mesh_level, seed=args.seed,
        )
    elif args.fold_assignment == "contiguous":
        folds = spatial_kfold_split_contiguous(
            sub_df, n_folds=args.n_folds, mesh_level=args.mesh_level, seed=args.seed,
        )
    else:  # buffered-contiguous
        folds = spatial_kfold_split_buffered(
            sub_df, n_folds=args.n_folds, mesh_level=args.mesh_level,
            buffer_meshes=args.buffer_meshes, seed=args.seed,
            base_split="contiguous",
        )
    LOG.info("Spatial K-fold (%s): %d folds", args.fold_assignment, len(folds))

    # ---- 3. Per-fold metrics ------------------------------------------
    results: dict[str, list[dict]] = {b: [] for b in args.baselines}
    timings: dict[str, float] = {b: 0.0 for b in args.baselines}

    for fi, (train_idx, test_idx) in enumerate(folds):
        LOG.info("=== fold %d  n_train=%d  n_test=%d ===",
                 fi, len(train_idx), len(test_idx))
        yt_train = y_full[train_idx]
        yt_test = y_full[test_idx]

        # 14-D features for RF / HGB
        x_train = x_full[train_idx]
        x_test = x_full[test_idx]
        # 3-D km for IDW / Kriging
        xyz_train = xyz_km[train_idx]
        xyz_test = xyz_km[test_idx]

        if "idw" in args.baselines:
            t0 = time.perf_counter()
            yhat = fit_predict_idw(xyz_train, yt_train, xyz_test, k=16, power=2.0)
            rmse = float(np.sqrt(np.mean((yhat - yt_test) ** 2)))
            mae = float(np.mean(np.abs(yhat - yt_test)))
            dt = time.perf_counter() - t0
            timings["idw"] += dt
            results["idw"].append({"fold": fi, "rmse": rmse, "mae": mae, "wall_s": dt})
            LOG.info("idw       RMSE=%.3f MAE=%.3f  (%.1fs)", rmse, mae, dt)

        if "kriging" in args.baselines:
            t0 = time.perf_counter()
            yhat, _ystd = fit_predict_kriging(
                xyz_train, yt_train, xyz_test,
                n_subsample=args.kriging_subsample,
                nu=1.5,
                random_state=args.seed,
                predict_std=False,  # point RMSE/MAE only; std is O(N*M^2)
            )
            rmse = float(np.sqrt(np.mean((yhat - yt_test) ** 2)))
            mae = float(np.mean(np.abs(yhat - yt_test)))
            dt = time.perf_counter() - t0
            timings["kriging"] += dt
            results["kriging"].append({"fold": fi, "rmse": rmse, "mae": mae, "wall_s": dt})
            LOG.info("kriging   RMSE=%.3f MAE=%.3f  (%.1fs)", rmse, mae, dt)

        if "rf" in args.baselines:
            t0 = time.perf_counter()
            yhat, _ystd = fit_predict_rf(
                x_train, yt_train, x_test, n_estimators=args.rf_trees,
                random_state=args.seed,
            )
            rmse = float(np.sqrt(np.mean((yhat - yt_test) ** 2)))
            mae = float(np.mean(np.abs(yhat - yt_test)))
            dt = time.perf_counter() - t0
            timings["rf"] += dt
            results["rf"].append({"fold": fi, "rmse": rmse, "mae": mae, "wall_s": dt})
            LOG.info("rf        RMSE=%.3f MAE=%.3f  (%.1fs)", rmse, mae, dt)

        if "hgb" in args.baselines:
            t0 = time.perf_counter()
            yhat = fit_predict_hgb(x_train, yt_train, x_test)
            rmse = float(np.sqrt(np.mean((yhat - yt_test) ** 2)))
            mae = float(np.mean(np.abs(yhat - yt_test)))
            dt = time.perf_counter() - t0
            timings["hgb"] += dt
            results["hgb"].append({"fold": fi, "rmse": rmse, "mae": mae, "wall_s": dt})
            LOG.info("hgb       RMSE=%.3f MAE=%.3f  (%.1fs)", rmse, mae, dt)

    # ---- 4. Aggregate + save ------------------------------------------
    summary = {
        "parquet": str(args.parquet),
        "n_folds": args.n_folds,
        "mesh_level": args.mesh_level,
        "fold_assignment": args.fold_assignment,
        "buffer_meshes": (args.buffer_meshes
                          if args.fold_assignment == "buffered-contiguous" else 0),
        "seed": args.seed,
        "kriging_subsample": args.kriging_subsample,
        "rf_trees": args.rf_trees,
        "baselines": {},
    }
    for b, per_fold in results.items():
        rmse_arr = np.array([r["rmse"] for r in per_fold])
        mae_arr = np.array([r["mae"] for r in per_fold])
        summary["baselines"][b] = {
            "per_fold": per_fold,
            "rmse_mean": float(rmse_arr.mean()),
            "rmse_std": float(rmse_arr.std(ddof=0)),
            "mae_mean": float(mae_arr.mean()),
            "mae_std": float(mae_arr.std(ddof=0)),
            "wall_s_total": float(timings[b]),
        }
    suffix = "" if args.fold_assignment == "random" else f"_{args.fold_assignment.replace('-', '_')}"
    summary_path = args.output_dir / f"summary{suffix}.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    LOG.info("Wrote %s", summary_path)

    # ---- 5. Pretty print to stdout ------------------------------------
    print()
    print(f"=== Baselines on spatial {args.n_folds}-fold (mesh L{args.mesh_level}) ===")
    print(f"  {'Baseline':<10}  RMSE (mean ± std)   MAE (mean ± std)   wall")
    for b, s in summary["baselines"].items():
        print(
            f"  {b:<10}  {s['rmse_mean']:.3f} ± {s['rmse_std']:.3f}      "
            f"{s['mae_mean']:.3f} ± {s['mae_std']:.3f}      "
            f"{s['wall_s_total']:>6.1f}s"
        )
    print()
    print("Foundation model (reference):")
    print(f"  linear_rbf   5.875 ± —          3.144 ± —          (cluster)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
