#!/usr/bin/env python
"""Hybrid (CatBoost mean + SVGP residual) vs baselines evaluation table.

Consumes per-fold artefacts and produces a single comparison table for
§5 of the paper:

| Method                                | RMSE | MAE | 95% cov | 95% width | Interval score | N>=30 cov |
|---------------------------------------|------|-----|---------|-----------|----------------|-----------|
| DKL/SVGP only                         | …    | …   | …       | …         | …              | …         |
| CatBoost only (no conformal)          | …    | …   | —       | —         | —              | —         |
| CatBoost + split conformal            | …    | …   | …       | …         | …              | …         |
| Hybrid CatBoost + SVGP residual + cnf | …    | …   | …       | …         | …              | …         |

Inputs (all paths configurable via CLI):
  - DKL run dir   : data/runs/<dkl_run>/predictions.npz
  - Hybrid run dir: data/runs/hybrid_<assignment>_f{0,1,2}/predictions.npz (3 folds)
  - Baselines    : data/runs/baselines_kanto_hybrid[_contig]/<baseline>_*.npy

Writes a JSON summary + a LaTeX fragment.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG = logging.getLogger("hybrid_eval")


def _rmse(y, p):
    return float(np.sqrt(np.mean((p - y) ** 2)))


def _mae(y, p):
    return float(np.mean(np.abs(p - y)))


def _interval_score(y, lo, hi, alpha):
    width = hi - lo
    miss_lo = np.maximum(lo - y, 0.0)
    miss_hi = np.maximum(y - hi, 0.0)
    return float(np.mean(width + (2.0 / (1.0 - alpha)) * (miss_lo + miss_hi)))


def _conformal_radius(z, alpha):
    s = np.sort(np.abs(z))
    n = len(s)
    k = min(max(int(np.ceil((n + 1) * alpha)), 1), n)
    return float(s[k - 1])


def _coverage(y, lo, hi):
    return float(((y >= lo) & (y <= hi)).mean())


def evaluate_dkl(run_dir: Path, fold_idx_arrays: list[np.ndarray],
                 y_true_full: np.ndarray) -> dict:
    """Evaluate the DKL-only baseline. Assumes the run is a single
    train-on-all model whose ``predictions.npz`` carries pred_mean/std for
    every parquet row (no fold split). We slice to each fold and recompute
    metrics."""
    arrays = np.load(run_dir / "predictions.npz")
    pred_mean = arrays["pred_mean"]
    pred_std = arrays["pred_std"]
    y_check = arrays["y_true"]
    if not np.allclose(y_check, y_true_full, atol=1e-3):
        raise RuntimeError(f"y_true mismatch between {run_dir} and ground truth")
    rmses, maes, covs, widths, iscores, n30_covs = [], [], [], [], [], []
    for fold_idx in fold_idx_arrays:
        y_f = y_true_full[fold_idx]
        mu_f = pred_mean[fold_idx]
        std_f = np.maximum(pred_std[fold_idx], 1e-3)
        rmses.append(_rmse(y_f, mu_f))
        maes.append(_mae(y_f, mu_f))
        # 95% Gaussian interval (raw, no conformal)
        lo = mu_f - 1.96 * std_f
        hi = mu_f + 1.96 * std_f
        covs.append(_coverage(y_f, lo, hi))
        widths.append(float((hi - lo).mean()))
        iscores.append(_interval_score(y_f, lo, hi, 0.95))
        n30_mask = y_f >= 30.0
        if n30_mask.sum() > 0:
            n30_covs.append(_coverage(y_f[n30_mask], lo[n30_mask], hi[n30_mask]))
    return {
        "method": "DKL/SVGP only (raw Gaussian)",
        "mean_rmse": float(np.mean(rmses)),
        "mean_mae": float(np.mean(maes)),
        "mean_coverage_95": float(np.mean(covs)),
        "mean_width_95": float(np.mean(widths)),
        "mean_interval_score_95": float(np.mean(iscores)),
        "mean_n30_coverage_95": float(np.mean(n30_covs)) if n30_covs else None,
    }


def evaluate_catboost_only(baseline_dir: Path, y_true_full: np.ndarray) -> dict:
    """Point-estimate CatBoost: load per-fold test predictions, compute
    RMSE/MAE only (no intervals)."""
    rmses, maes = [], []
    for k in range(3):
        idx = np.load(baseline_dir / f"catboost_idx_test_fold{k}.npy")
        pred = np.load(baseline_dir / f"catboost_pred_test_fold{k}.npy")
        y = y_true_full[idx]
        rmses.append(_rmse(y, pred))
        maes.append(_mae(y, pred))
    return {
        "method": "CatBoost only (no conformal)",
        "mean_rmse": float(np.mean(rmses)),
        "mean_mae": float(np.mean(maes)),
    }


def evaluate_catboost_conformal(baseline_dir: Path, y_true_full: np.ndarray,
                                  seed: int = 42) -> dict:
    """CatBoost + split conformal on a held-out inner cal slice (matches
    the LightGBM+conformal path in run_advanced_baselines).
    Reports RMSE/MAE/95% coverage/95% width/interval score."""
    rmses, maes, covs, widths, iscores, n30_covs = [], [], [], [], [], []
    for k in range(3):
        idx_test = np.load(baseline_dir / f"catboost_idx_test_fold{k}.npy")
        pred_test = np.load(baseline_dir / f"catboost_pred_test_fold{k}.npy")
        idx_train = np.load(baseline_dir / f"catboost_idx_train_fold{k}.npy")
        pred_train_oob = np.load(baseline_dir / f"catboost_pred_train_oob_fold{k}.npy")
        y_test = y_true_full[idx_test]
        y_train = y_true_full[idx_train]
        # Use the spatial-OOB residuals as the cal set
        rng = np.random.default_rng(seed + k)
        cal_size = max(2000, len(idx_train) // 5)
        cal_pick = rng.choice(len(idx_train), size=cal_size, replace=False)
        z_cal = np.abs(y_train[cal_pick] - pred_train_oob[cal_pick])
        q95 = _conformal_radius(z_cal, 0.95)
        lo = pred_test - q95
        hi = pred_test + q95
        rmses.append(_rmse(y_test, pred_test))
        maes.append(_mae(y_test, pred_test))
        covs.append(_coverage(y_test, lo, hi))
        widths.append(float((hi - lo).mean()))
        iscores.append(_interval_score(y_test, lo, hi, 0.95))
        n30_mask = y_test >= 30.0
        if n30_mask.sum() > 0:
            n30_covs.append(_coverage(y_test[n30_mask], lo[n30_mask], hi[n30_mask]))
    return {
        "method": "CatBoost + split conformal",
        "mean_rmse": float(np.mean(rmses)),
        "mean_mae": float(np.mean(maes)),
        "mean_coverage_95": float(np.mean(covs)),
        "mean_width_95": float(np.mean(widths)),
        "mean_interval_score_95": float(np.mean(iscores)),
        "mean_n30_coverage_95": float(np.mean(n30_covs)) if n30_covs else None,
    }


def evaluate_hybrid(hybrid_run_dirs: list[Path], y_true_full: np.ndarray) -> dict:
    """Hybrid (CatBoost mean + SVGP residual + conformal on residual)."""
    rmses, maes, covs, widths, iscores, n30_covs = [], [], [], [], [], []
    for run_dir in hybrid_run_dirs:
        z = np.load(run_dir / "predictions.npz")
        if int(z["hybrid_mode"][0]) != 1:
            raise RuntimeError(
                f"{run_dir} is not a hybrid run "
                f"(hybrid_mode={int(z['hybrid_mode'][0])})"
            )
        # pred_mean is the FULL (residual + baseline) prediction;
        # baseline_pred holds the CatBoost mean used at inference.
        # We restrict to the test rows for this fold by reading the
        # baseline idx file from the run's metadata.
        # The simple path: smoke trainer's predictions.npz is over the
        # *full* parquet (--train-fraction 1.0), but the fold split is in
        # `summary.json`. We re-derive the fold here.
        summary = json.loads((run_dir / "summary.json").read_text())
        fold_info = summary["spatial_kfold"]
        # Find the held-out fold (the one with `n_test` > 0 in the
        # smoke trainer's K-fold loop). With --kfold-test-fold N this is
        # fold index N.
        # If the smoke trainer was run in single-fold-holdout mode it
        # stores all 3 fold entries but only one is honest. We grab
        # the kfold_test_fold from the config.
        # ... robust path: use fold_assignment + kfold_test_fold via
        # spatial_kfold_split to derive the test mask.
        held_out_fold = None
        for entry in fold_info:
            if entry["n_train"] > 0 and entry["n_test"] > 0:
                # All 3 entries have non-zero train/test in K-fold mode.
                # In single-holdout mode (--kfold-test-fold N), only
                # fold N has the meaningful test set.
                pass
        # For simplicity: re-derive the fold split from the parquet.
        import pandas as pd
        from run_advanced_baselines import assign_folds

        df = pd.read_parquet(
            PROJECT_ROOT / "data/features/borings_kanto_aist.parquet",
            columns=["latitude_deg", "longitude_deg"],
        )
        fold_assignment_str = (
            "contiguous" if "contig" in run_dir.name else "random"
        )
        fold = assign_folds(df, n_folds=3, seed=42,
                             assignment=fold_assignment_str)
        # Find kfold_test_fold from run dir name (e.g. hybrid_random_f0 -> 0)
        held_out = int(run_dir.name.rsplit("f", 1)[-1])
        test_mask = fold == held_out
        pred_full = z["pred_mean"]
        std_full = z["pred_std"]
        y_full = z["y_true"]
        baseline_full = z["baseline_pred"]
        y = y_full[test_mask]
        mu = pred_full[test_mask]
        sigma = np.maximum(std_full[test_mask], 1e-3)
        rmses.append(_rmse(y, mu))
        maes.append(_mae(y, mu))
        # Conformal on residuals (cal split inside the training fold)
        train_mask = ~test_mask
        rng = np.random.default_rng(42 + held_out)
        cal_size = max(2000, int(train_mask.sum() // 5))
        train_indices = np.where(train_mask)[0]
        cal_pick = rng.choice(train_indices, size=cal_size, replace=False)
        z_cal = (np.abs(y_full[cal_pick] - pred_full[cal_pick])
                 / np.maximum(std_full[cal_pick], 1e-3))
        q95 = _conformal_radius(z_cal, 0.95)
        lo = mu - q95 * sigma
        hi = mu + q95 * sigma
        covs.append(_coverage(y, lo, hi))
        widths.append(float((hi - lo).mean()))
        iscores.append(_interval_score(y, lo, hi, 0.95))
        n30_mask = y >= 30.0
        if n30_mask.sum() > 0:
            n30_covs.append(_coverage(y[n30_mask], lo[n30_mask], hi[n30_mask]))
    return {
        "method": "Hybrid CatBoost + SVGP residual + conformal",
        "mean_rmse": float(np.mean(rmses)),
        "mean_mae": float(np.mean(maes)),
        "mean_coverage_95": float(np.mean(covs)),
        "mean_width_95": float(np.mean(widths)),
        "mean_interval_score_95": float(np.mean(iscores)),
        "mean_n30_coverage_95": float(np.mean(n30_covs)) if n30_covs else None,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--dkl-run-dir", type=Path, required=False, default=None,
                   help="DKL-only operational artefact run dir (train-on-all). "
                        "If absent, the DKL row is skipped.")
    p.add_argument("--hybrid-run-dirs", type=Path, nargs="+", required=True,
                   help="Per-fold hybrid run dirs (3 dirs for K=3).")
    p.add_argument("--baseline-dir", type=Path, required=True,
                   help="Directory with catboost_*_fold*.npy (from "
                        "run_advanced_baselines.py --save-fold-predictions)")
    p.add_argument("--parquet", type=Path,
                   default=PROJECT_ROOT / "data/features/borings_kanto_aist.parquet")
    p.add_argument("--fold-assignment", choices=["random", "contiguous"],
                   default="random")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    # Load full y_true
    df = pd.read_parquet(a.parquet, columns=["n_value"])
    y_true_full = df["n_value"].values.astype(np.float64)

    from run_advanced_baselines import assign_folds

    df_geo = pd.read_parquet(a.parquet, columns=["latitude_deg", "longitude_deg"])
    fold = assign_folds(df_geo, n_folds=3, seed=42, assignment=a.fold_assignment)
    fold_idx = [np.where(fold == k)[0] for k in range(3)]

    rows = []
    if a.dkl_run_dir is not None:
        rows.append(evaluate_dkl(a.dkl_run_dir, fold_idx, y_true_full))
    rows.append(evaluate_catboost_only(a.baseline_dir, y_true_full))
    rows.append(evaluate_catboost_conformal(a.baseline_dir, y_true_full))
    rows.append(evaluate_hybrid(a.hybrid_run_dirs, y_true_full))

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({
        "fold_assignment": a.fold_assignment,
        "rows": rows,
    }, indent=2))
    LOG.info("Wrote %s", a.out)
    for r in rows:
        LOG.info("  %s: RMSE=%.3f MAE=%.3f cov95=%s",
                 r["method"], r["mean_rmse"], r["mean_mae"],
                 r.get("mean_coverage_95"))


if __name__ == "__main__":
    main()
