#!/usr/bin/env python
"""Phase R (review response, RB.2) — recommended-model robustness under
buffered contiguous spatial K-fold.

Reviewer RB.2: GPBoost (recommended for out-of-network raw-N profiles) and
CatBoost (recommended for within-network point estimates) are recommended
deployment components, but buffered CV was only reported for DKL+SVGP. This
runner evaluates the recommended tree/GP regressors under the SAME buffered
machinery that produced the DKL buffered row
(``national.evaluation.spatial_kfold.spatial_kfold_split_buffered`` with
``base_split="contiguous"``), so ``tables/buffered_cv.tex`` can gain model
rows apples-to-apples.

For each (model, buffer in {0, 1}) it reports, per fold and aggregate:
  - point RMSE / MAE (raw N units),
  - the buffer train-shrinkage % (rows removed by the ring vs the
    unbuffered contiguous train set for the same fold),
  - split-conformal 95% coverage (overall + stiff N>=30), reusing the
    mesh-disjoint nested calibration convention of the other phase scripts.

Feature conventions match the existing contiguous cells so buffer-0
reproduces the paper's numbers:
  - gpboost : 5-D tree stack (no lat/lon) + lat/lon -> Vecchia GP residual
              (reproduces run_gpboost_baseline_phase_n.py; contig ~10.744),
  - catboost: 7-D stack incl lat/lon as tabular features
              (reproduces run_tree_conformal_phase_n.py; contig ~13.451).

Outputs:
  data/runs/kanto/buffered_baselines/<model>_contiguous/results.json

Run:
  cd backend
  uv run python -m scripts.run_buffered_baselines --model gpboost  --base-split contiguous
  uv run python -m scripts.run_buffered_baselines --model catboost --base-split contiguous
  # add --quick 80000 for a smoke test
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from national.evaluation.baselines import fit_predict_catboost
from national.evaluation.gpboost_baseline import fit_predict_gpboost
from national.evaluation.leave_region_out_runner import _mesh_disjoint_cal_split
from national.evaluation.spatial_kfold import (
    spatial_kfold_split_buffered,
    spatial_kfold_split_contiguous,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PARQUET = PROJECT_ROOT / "data/features/borings_kanto_aist.parquet"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data/runs/kanto/buffered_baselines"

LOG = logging.getLogger("buffered_baselines")

# Feature stacks (match the existing contiguous cells so buffer-0 reproduces
# the paper's numbers).
GP_FEATURE_COLS = [
    "depth_from_surface", "absolute_elevation",
    "river_distance_km", "coast_distance_km", "regime_code",
]
CAT_FEATURE_COLS = [
    "latitude_deg", "longitude_deg", "depth_from_surface",
    "absolute_elevation", "river_distance_km", "coast_distance_km",
    "regime_code",
]
SPATIAL_COLS = ["latitude_deg", "longitude_deg"]
STIFF_THRESHOLD = 30.0


def conformal_radius(abs_residuals: np.ndarray, alpha: float) -> float:
    """Empirical (n+1) quantile of |residuals| at coverage level alpha."""
    n = len(abs_residuals)
    k = int(np.ceil((n + 1) * alpha))
    k = min(max(k, 1), n)
    return float(np.sort(abs_residuals)[k - 1])


def _predict_cal_test(
    model: str,
    df: pd.DataFrame,
    fit_idx: np.ndarray,
    cal_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    gpboost_kwargs: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit once on ``fit_idx``; return (cal_pred, test_pred)."""
    y_fit = df["n_value"].to_numpy(np.float64)[fit_idx]
    if model == "gpboost":
        x_fit = df[GP_FEATURE_COLS].to_numpy(np.float64)[fit_idx]
        g_fit = df[SPATIAL_COLS].to_numpy(np.float64)[fit_idx]
        q_x = df[GP_FEATURE_COLS].to_numpy(np.float64)[np.concatenate([cal_idx, test_idx])]
        q_g = df[SPATIAL_COLS].to_numpy(np.float64)[np.concatenate([cal_idx, test_idx])]
        all_pred = fit_predict_gpboost(x_fit, y_fit, g_fit, q_x, q_g, **gpboost_kwargs)
    elif model == "catboost":
        x_fit = df[CAT_FEATURE_COLS].to_numpy(np.float64)[fit_idx]
        q_x = df[CAT_FEATURE_COLS].to_numpy(np.float64)[np.concatenate([cal_idx, test_idx])]
        all_pred = fit_predict_catboost(x_fit, y_fit, q_x)
    else:
        raise ValueError(f"Unknown model {model!r}; choose gpboost or catboost.")
    all_pred = np.asarray(all_pred, dtype=np.float64)
    n_cal = len(cal_idx)
    return all_pred[:n_cal], all_pred[n_cal:]


def _evaluate_fold(
    model: str,
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    seed: int,
    cal_fraction: float,
    gpboost_kwargs: dict,
) -> dict:
    lats = df[SPATIAL_COLS[0]].to_numpy()[train_idx]
    lons = df[SPATIAL_COLS[1]].to_numpy()[train_idx]
    fit_mask, cal_mask = _mesh_disjoint_cal_split(
        lats, lons, fraction=cal_fraction, seed=seed,
    )
    fit_idx = train_idx[fit_mask]
    cal_idx = train_idx[cal_mask]

    t0 = time.time()
    cal_pred, test_pred = _predict_cal_test(
        model, df, fit_idx, cal_idx, test_idx, gpboost_kwargs=gpboost_kwargs,
    )
    wall = time.time() - t0

    y_cal = df["n_value"].to_numpy(np.float64)[cal_idx]
    y_test = df["n_value"].to_numpy(np.float64)[test_idx]
    rmse = float(np.sqrt(np.mean((y_test - test_pred) ** 2)))
    mae = float(np.mean(np.abs(y_test - test_pred)))

    abs_cal_res = np.abs(y_cal - cal_pred)
    q95 = conformal_radius(abs_cal_res, 0.95)
    inside = np.abs(y_test - test_pred) <= q95
    stiff = y_test >= STIFF_THRESHOLD
    cov95 = float(np.mean(inside))
    cov95_stiff = float(np.mean(inside[stiff])) if stiff.sum() > 0 else float("nan")

    return {
        "n_fit": int(len(fit_idx)),
        "n_cal": int(len(cal_idx)),
        "n_test": int(len(test_idx)),
        "n_train": int(len(train_idx)),
        "rmse": rmse,
        "mae": mae,
        "coverage_95": cov95,
        "coverage_95_stiff": cov95_stiff,
        "conformal_q95": q95,
        "wall_clock_s": wall,
    }


def _aggregate(per_fold: list[dict]) -> dict:
    rmses = np.array([f["rmse"] for f in per_fold])
    maes = np.array([f["mae"] for f in per_fold])
    cov = np.array([f["coverage_95"] for f in per_fold])
    cov_s = np.array([f["coverage_95_stiff"] for f in per_fold])
    shrink = np.array([f.get("train_shrinkage_pct", 0.0) for f in per_fold])
    return {
        "n_folds": len(per_fold),
        "rmse_mean": float(rmses.mean()),
        "rmse_std": float(rmses.std(ddof=0)),
        "rmse_lo": float(rmses.min()),
        "rmse_hi": float(rmses.max()),
        "mae_mean": float(maes.mean()),
        "mae_std": float(maes.std(ddof=0)),
        "coverage_95_mean": float(cov.mean()),
        "coverage_95_stiff_mean": float(np.nanmean(cov_s)),
        "train_shrinkage_pct_mean": float(shrink.mean()),
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--model", choices=["gpboost", "catboost"], default="gpboost")
    p.add_argument("--base-split", choices=["contiguous", "random"], default="contiguous")
    p.add_argument("--buffer-meshes", type=int, nargs="+", default=[0, 1],
                   help="Ring sizes to evaluate (0 = unbuffered base split).")
    p.add_argument("--n-folds", type=int, default=3)
    p.add_argument("--mesh-level", type=int, default=2)
    p.add_argument("--cal-fraction", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-neighbors", type=int, default=20)
    p.add_argument("--n-boost-iter", type=int, default=300)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--quick", type=int, default=0,
                   help="Subsample to N rows for a smoke run (0 = full).")
    args = p.parse_args(argv)

    LOG.info("Loading %s", args.parquet)
    df = pd.read_parquet(args.parquet)
    if args.quick:
        df = df.sample(int(args.quick), random_state=args.seed).reset_index(drop=True)
    LOG.info("Loaded %d rows; model=%s base_split=%s buffers=%s",
             len(df), args.model, args.base_split, args.buffer_meshes)

    sub_df = df[SPATIAL_COLS + ["n_value"]].copy()
    gpboost_kwargs = {
        "num_neighbors": args.num_neighbors,
        "n_boost_iter": args.n_boost_iter,
        "learning_rate": args.learning_rate,
    }

    # Unbuffered base folds (buffer 0) give the per-fold train sizes used to
    # compute the shrinkage of the buffered folds (same base partition + seed).
    base_folds = (
        spatial_kfold_split_contiguous(sub_df, n_folds=args.n_folds,
                                       mesh_level=args.mesh_level, seed=args.seed)
        if args.base_split == "contiguous"
        else None
    )
    unbuffered_train_sizes = (
        [len(tr) for tr, _ in base_folds] if base_folds is not None else None
    )

    results: dict[str, dict] = {}
    for buffer in args.buffer_meshes:
        if buffer == 0:
            folds = (base_folds if base_folds is not None
                     else spatial_kfold_split_contiguous(
                         sub_df, n_folds=args.n_folds,
                         mesh_level=args.mesh_level, seed=args.seed))
        else:
            folds = spatial_kfold_split_buffered(
                sub_df, n_folds=args.n_folds, mesh_level=args.mesh_level,
                buffer_meshes=buffer, seed=args.seed, base_split=args.base_split,
            )
        per_fold = []
        for k, (train_idx, test_idx) in enumerate(folds):
            res = _evaluate_fold(
                args.model, df, train_idx, test_idx,
                seed=args.seed + 10 * k, cal_fraction=args.cal_fraction,
                gpboost_kwargs=gpboost_kwargs,
            )
            if buffer > 0 and unbuffered_train_sizes is not None:
                base_n = unbuffered_train_sizes[k]
                res["train_shrinkage_pct"] = round(
                    100.0 * (1.0 - res["n_train"] / base_n), 1) if base_n else 0.0
            else:
                res["train_shrinkage_pct"] = 0.0
            res["fold"] = k
            per_fold.append(res)
            LOG.info("  buffer=%d fold=%d rmse=%.3f mae=%.3f cov95=%.3f "
                     "shrink=%.1f%% (%.1fs)", buffer, k, res["rmse"], res["mae"],
                     res["coverage_95"], res["train_shrinkage_pct"],
                     res["wall_clock_s"])
        agg = _aggregate(per_fold)
        results[f"buffer_{buffer}"] = {"per_fold": per_fold, "aggregate": agg}
        LOG.info("buffer=%d aggregate rmse=%.3f ± %.3f mae=%.3f cov95=%.3f "
                 "shrink=%.1f%%", buffer, agg["rmse_mean"], agg["rmse_std"],
                 agg["mae_mean"], agg["coverage_95_mean"],
                 agg["train_shrinkage_pct_mean"])

    out_dir = args.out_dir / f"{args.model}_{args.base_split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "config": {
            "model": args.model, "base_split": args.base_split,
            "buffer_meshes": args.buffer_meshes, "n_folds": args.n_folds,
            "mesh_level": args.mesh_level, "cal_fraction": args.cal_fraction,
            "seed": args.seed, "n_rows": int(len(df)),
            "feature_cols": GP_FEATURE_COLS if args.model == "gpboost" else CAT_FEATURE_COLS,
            "gpboost_kwargs": gpboost_kwargs if args.model == "gpboost" else None,
        },
        "results": results,
    }
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(summary, indent=2))
    LOG.info("Wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
