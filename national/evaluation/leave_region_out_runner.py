"""Leave-region-out (LRO) evaluation runner — the Phase C exit gate.

For each held-out region (or geological macro-block) the model is fitted on all
*other* regions, post-hoc split-conformal calibrated on a mesh-disjoint nested
calibration subset of the training set, and evaluated on the held-out region.
This measures out-of-network cross-region transfer, which is strictly harder
than the within-Kanto contiguous protocol and — per Paper 1's spatial-lookup
memorisation finding — is the *primary* national metric (random folds let the
encoder memorise (lat, lon), so they overstate generalisation).

Design choices following Paper 1:
- Spatial coordinates feed only the GPBoost Vecchia GP residual, never the tree
  feature stack: lat/lon as tabular features memorise within-network and are
  out-of-distribution for a held-out region.
- Split conformal (distribution-free) provides the interval guarantee; the
  Mondrian (per-regime) variant exposes rare-regime conditional coverage.
- Metrics are reported per regime so dominant alluvial rows cannot mask poor
  performance on rare regimes (volcanic ash, limestone).

The runner is pure (returns a dict); the CLI ``scripts/run_leave_region_out.py``
handles I/O.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd

_LOG = logging.getLogger("leave_region_out_runner")

from national.evaluation.calibration import ConformalCalibrator
from national.evaluation.leave_region_out import (
    DEFAULT_REGIONS,
    GEOLOGICAL_BLOCKS,
    leave_block_out_split,
    leave_region_out_split,
)
from national.evaluation.regime_metrics import per_regime_metrics

# Feature stack matches scripts/run_gpboost_baseline_phase_n.py. lat/lon are
# deliberately excluded (they feed the GP residual, not the tree mean).
DEFAULT_FEATURE_COLS = [
    "depth_from_surface",
    "absolute_elevation",
    "river_distance_km",
    "coast_distance_km",
    "regime_code",
]
DEFAULT_SPATIAL_COLS = ["latitude_deg", "longitude_deg"]
DEFAULT_ALPHAS = (0.50, 0.80, 0.95)
DEFAULT_TARGET = "n_value"
DEFAULT_REGIME_COL = "regime_code"

# Kanto within-network contiguous GPBoost RMSE (Paper 1), used as the gate
# reference. LRO is strictly harder than contiguous, hence the tolerance band.
KANTO_CONTIG_GPBOOST_RMSE = 10.744


def _mesh_disjoint_cal_split(
    lats: np.ndarray,
    lons: np.ndarray,
    *,
    fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split training rows into (fit, cal) masks by whole secondary-mesh cells.

    Calibration cells are disjoint from fit cells so the conformal residuals are
    spatially exchangeable with the held-out region rather than leaking from
    adjacent points.
    """
    from shared.geo.tiles import secondary_mesh_key_array

    # Vectorised secondary-mesh cell keys (per-row Python loop would be far too
    # slow at national scale: ~2.2M rows per fold).
    codes = secondary_mesh_key_array(lats, lons)
    unique_codes, inverse = np.unique(codes, return_inverse=True)
    code_size = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    order = rng.permutation(unique_codes.size)
    target_cal_rows = int(fraction * len(codes))
    selected: set = set()
    running = 0
    for ci in order:
        if running >= target_cal_rows:
            break
        selected.add(unique_codes[ci])
        running += int(code_size[ci])
    cal_mask = np.array([c in selected for c in codes], dtype=bool)
    return ~cal_mask, cal_mask


def _predict_mean_std(
    model: str,
    fit_x: np.ndarray,
    fit_y: np.ndarray,
    fit_coords: np.ndarray,
    q_x: np.ndarray,
    q_coords: np.ndarray,
    *,
    gpboost_kwargs: dict | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Dispatch to a model and return (mean, std-or-None) at query points.

    ``std`` is ``None`` for point-only models; the caller substitutes a global
    calibration-residual scale so conformal reduces to absolute-residual
    scaling.
    """
    if model == "gpboost":
        from national.evaluation.gpboost_baseline import fit_predict_gpboost

        mean, std = fit_predict_gpboost(
            fit_x, fit_y, fit_coords, q_x, q_coords,
            predict_var=True, **(gpboost_kwargs or {}),
        )
        return mean, np.maximum(std, 1e-6)

    if model == "rf":
        from national.evaluation.baselines import fit_predict_rf

        mean, std = fit_predict_rf(fit_x, fit_y, q_x)
        return mean, np.maximum(std, 1e-6)

    from national.evaluation import baselines

    point_models: dict[str, Callable] = {
        "hgb": baselines.fit_predict_hgb,
        "lightgbm": baselines.fit_predict_lightgbm,
        "xgboost": baselines.fit_predict_xgboost,
        "catboost": baselines.fit_predict_catboost,
    }
    if model not in point_models:
        raise ValueError(
            f"Unknown model {model!r}. Choose from: gpboost, rf, "
            f"{', '.join(point_models)}."
        )
    mean = np.asarray(point_models[model](fit_x, fit_y, q_x), dtype=np.float64)
    return mean, None


def _evaluate_fold(
    fold_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    model: str,
    feature_cols: list[str],
    spatial_cols: list[str],
    target_col: str,
    regime_col: str,
    alphas: tuple[float, ...],
    cal_fraction: float,
    seed: int,
    gpboost_kwargs: dict | None,
) -> dict:
    fit_mask, cal_mask = _mesh_disjoint_cal_split(
        train_df[spatial_cols[0]].to_numpy(),
        train_df[spatial_cols[1]].to_numpy(),
        fraction=cal_fraction,
        seed=seed,
    )
    fx = train_df[feature_cols].to_numpy(np.float64)
    fg = train_df[spatial_cols].to_numpy(np.float64)
    fy = train_df[target_col].to_numpy(np.float64)

    fit_x, fit_g, fit_y = fx[fit_mask], fg[fit_mask], fy[fit_mask]
    cal_x, cal_g, cal_y = fx[cal_mask], fg[cal_mask], fy[cal_mask]
    test_x = test_df[feature_cols].to_numpy(np.float64)
    test_g = test_df[spatial_cols].to_numpy(np.float64)
    test_y = test_df[target_col].to_numpy(np.float64)

    # One predict over cal+test so the fit is reused.
    q_x = np.concatenate([cal_x, test_x], axis=0)
    q_g = np.concatenate([cal_g, test_g], axis=0)
    mean_all, std_all = _predict_mean_std(
        model, fit_x, fit_y, fit_g, q_x, q_g, gpboost_kwargs=gpboost_kwargs,
    )
    n_cal = len(cal_x)
    cal_pred, test_pred = mean_all[:n_cal], mean_all[n_cal:]

    if std_all is None:
        # Point-only model: global scale from calibration residuals.
        scale = float(np.sqrt(np.mean((cal_y - cal_pred) ** 2))) or 1.0
        cal_std = np.full(n_cal, scale, dtype=np.float64)
        test_std = np.full(len(test_pred), scale, dtype=np.float64)
    else:
        cal_std, test_std = std_all[:n_cal], std_all[n_cal:]

    cal_regimes = train_df[regime_col].to_numpy()[cal_mask]
    cal = ConformalCalibrator().fit_mondrian(
        cal_y, cal_pred, cal_std, groups=cal_regimes, alphas=alphas,
    )

    rmse = float(np.sqrt(np.mean((test_y - test_pred) ** 2)))
    mae = float(np.mean(np.abs(test_y - test_pred)))
    test_regimes = test_df[regime_col].to_numpy()

    per_alpha = {}
    for alpha in alphas:
        per_alpha[str(alpha)] = {
            "coverage_marginal": cal.coverage(test_y, test_pred, test_std, alpha),
            "coverage_mondrian": cal.coverage_mondrian(
                test_y, test_pred, test_std, test_regimes, alpha
            ),
        }

    regime_table = per_regime_metrics(
        test_df, test_y, test_pred, test_std, regime_column=regime_col,
    )

    return {
        "fold": fold_name,
        "n_fit": int(fit_mask.sum()),
        "n_cal": int(cal_mask.sum()),
        "n_test": int(len(test_df)),
        "rmse": rmse,
        "mae": mae,
        "per_alpha": per_alpha,
        "per_regime": regime_table.to_dict(orient="records"),
    }


def evaluate_region_transfer(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    fold_name: str = "transfer",
    model: str = "gpboost",
    feature_cols: list[str] | None = None,
    spatial_cols: list[str] | None = None,
    target_col: str = DEFAULT_TARGET,
    regime_col: str = DEFAULT_REGIME_COL,
    alphas: tuple[float, ...] = DEFAULT_ALPHAS,
    cal_fraction: float = 0.20,
    seed: int = 42,
    gpboost_kwargs: dict | None = None,
) -> dict:
    """Train on ``train_df``, conformal-calibrate, evaluate on ``test_df``.

    Public wrapper around the per-fold evaluation, used by the data-scaling
    curve (fixed held-out region, growing training set). Returns the same
    per-fold dict shape as :func:`run_leave_region_out`'s ``per_fold`` entries.
    """
    feature_cols = feature_cols or list(DEFAULT_FEATURE_COLS)
    spatial_cols = spatial_cols or list(DEFAULT_SPATIAL_COLS)
    required = set(feature_cols) | set(spatial_cols) | {target_col, regime_col}
    for name, frame in (("train_df", train_df), ("test_df", test_df)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} missing columns: {sorted(missing)}")
    return _evaluate_fold(
        fold_name,
        train_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
        model=model,
        feature_cols=feature_cols,
        spatial_cols=spatial_cols,
        target_col=target_col,
        regime_col=regime_col,
        alphas=alphas,
        cal_fraction=cal_fraction,
        seed=seed,
        gpboost_kwargs=gpboost_kwargs,
    )


def run_leave_region_out(
    df: pd.DataFrame,
    *,
    partition: str = "region",
    model: str = "gpboost",
    feature_cols: list[str] | None = None,
    spatial_cols: list[str] | None = None,
    target_col: str = DEFAULT_TARGET,
    regime_col: str = DEFAULT_REGIME_COL,
    alphas: tuple[float, ...] = DEFAULT_ALPHAS,
    cal_fraction: float = 0.20,
    seed: int = 42,
    gpboost_kwargs: dict | None = None,
    reference_rmse: float = KANTO_CONTIG_GPBOOST_RMSE,
    rmse_rel_tol: float = 0.30,
    coverage_abs_tol: float = 0.05,
    prefectures: list[str] | None = None,
) -> dict:
    """Run leave-region-out (or leave-block-out) evaluation and a gate verdict.

    Args:
        df: borehole-row DataFrame with feature, spatial, target and regime cols.
        partition: ``"region"`` (8 geographic regions) or ``"block"``
            (4 geological macro-blocks).
        model: ``gpboost`` | ``rf`` | ``hgb`` | ``lightgbm`` | ``xgboost`` |
            ``catboost``.
        cal_fraction: fraction of *training* rows reserved for conformal
            calibration (drawn as whole mesh cells).
        reference_rmse / rmse_rel_tol: the gate passes on RMSE when the mean
            held-out RMSE is within ``reference_rmse * (1 + rmse_rel_tol)``
            (LRO is harder than the contiguous protocol that produced the
            reference, hence a tolerance band rather than strict non-regression).
        coverage_abs_tol: the gate passes on coverage when the mean 95% interval
            coverage is within this of the 0.95 nominal.

    Returns:
        ``{"config", "per_fold", "aggregate", "gate"}``.
    """
    feature_cols = feature_cols or list(DEFAULT_FEATURE_COLS)
    spatial_cols = spatial_cols or list(DEFAULT_SPATIAL_COLS)

    required = set(feature_cols) | set(spatial_cols) | {target_col, regime_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns: {sorted(missing)}")

    if partition == "region":
        splitter = leave_region_out_split(df, lat_column=spatial_cols[0], lon_column=spatial_cols[1])
    elif partition == "block":
        splitter = leave_block_out_split(df, lat_column=spatial_cols[0], lon_column=spatial_cols[1])
    elif partition == "prefecture":
        # Within-Kanto leave-one-prefecture-out: usable on the Kanto Parquet
        # before the national Parquet exists (region/block partitions are
        # degenerate on single-region data — training set would be empty).
        from national.evaluation.prefecture_regions import (
            KANTO_PREFECTURES,
            leave_prefecture_out_split,
        )

        splitter = leave_prefecture_out_split(
            df, prefectures=prefectures or list(KANTO_PREFECTURES),
            lat_column=spatial_cols[0], lon_column=spatial_cols[1],
        )
    else:
        raise ValueError(
            f"partition must be 'region', 'block' or 'prefecture', got {partition!r}"
        )

    per_fold = []
    for fold_name, train_idx, test_idx in splitter:
        _LOG.info("LRO fold %r: train=%d test=%d (model=%s) ...",
                  fold_name, len(train_idx), len(test_idx), model)
        res = _evaluate_fold(
            fold_name,
            df.iloc[train_idx].reset_index(drop=True),
            df.iloc[test_idx].reset_index(drop=True),
            model=model,
            feature_cols=feature_cols,
            spatial_cols=spatial_cols,
            target_col=target_col,
            regime_col=regime_col,
            alphas=alphas,
            cal_fraction=cal_fraction,
            seed=seed,
            gpboost_kwargs=gpboost_kwargs,
        )
        _LOG.info("LRO fold %r done: rmse=%.3f mae=%.3f", fold_name,
                  res["rmse"], res["mae"])
        per_fold.append(res)

    if not per_fold:
        raise ValueError("No non-empty folds were produced for this partition.")

    rmses = [f["rmse"] for f in per_fold]
    maes = [f["mae"] for f in per_fold]
    a95 = str(0.95) if 0.95 in alphas else str(alphas[-1])
    cov95 = [f["per_alpha"][a95]["coverage_marginal"] for f in per_fold]
    mean_rmse = float(np.mean(rmses))
    mean_cov95 = float(np.mean(cov95))

    aggregate = {
        "n_folds": len(per_fold),
        "rmse_mean": mean_rmse,
        "rmse_std": float(np.std(rmses)),
        "rmse_worst": float(np.max(rmses)),
        "mae_mean": float(np.mean(maes)),
        f"coverage_{a95}_mean": mean_cov95,
    }

    pass_rmse = mean_rmse <= reference_rmse * (1.0 + rmse_rel_tol)
    pass_cov = abs(mean_cov95 - 0.95) <= coverage_abs_tol
    gate = {
        "reference_contig_rmse": reference_rmse,
        "rmse_rel_tol": rmse_rel_tol,
        "rmse_threshold": reference_rmse * (1.0 + rmse_rel_tol),
        "mean_test_rmse": mean_rmse,
        "pass_rmse": bool(pass_rmse),
        "target_coverage": 0.95,
        "coverage_abs_tol": coverage_abs_tol,
        "mean_coverage_95": mean_cov95,
        "pass_coverage": bool(pass_cov),
        "pass": bool(pass_rmse and pass_cov),
    }

    return {
        "config": {
            "partition": partition,
            "model": model,
            "feature_cols": feature_cols,
            "spatial_cols": spatial_cols,
            "target_col": target_col,
            "regime_col": regime_col,
            "alphas": list(alphas),
            "cal_fraction": cal_fraction,
            "seed": seed,
        },
        "per_fold": per_fold,
        "aggregate": aggregate,
        "gate": gate,
    }


__all__ = [
    "DEFAULT_FEATURE_COLS",
    "DEFAULT_SPATIAL_COLS",
    "DEFAULT_ALPHAS",
    "KANTO_CONTIG_GPBOOST_RMSE",
    "run_leave_region_out",
    "evaluate_region_transfer",
]
