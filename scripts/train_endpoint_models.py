#!/usr/bin/env python
"""Train boring-level engineering-endpoint regressors.

Per the paper pivot, regional priors should expose first-class engineering
targets, not just row-wise N-value RMSE. This script:

1. Reconstructs per-boring profiles from the row-wise SPT Parquet.
2. Computes 4 endpoint targets (see :mod:`national.data.endpoints`):
   ``depth_to_first_N30``, ``soft_thickness_lt5_0_to_10m``,
   ``mean_N_upper_10m``, ``min_N_upper_10m``.
3. Trains CatBoost regressors (point) + LightGBM quantile (P10/P50/P90)
   per endpoint, per fold, under either random or contiguous spatial
   K-fold assignment at the borehole level.
4. Reports MAE / R² / quantile coverage + (for ``depth_to_first_N30``)
   binary AUC for "has stiff layer in upper 30 m".

The ``depth_to_first_N30`` target is right-censored at ``+inf`` for
boreholes that never reach N >= 30 within the surveyed depth. We handle
this by training only on uncensored rows for the point regressor and
reporting binary AUC on the full corpus.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PARQUET = PROJECT_ROOT / "data/features/borings_kanto_aist.parquet"

LOG = logging.getLogger("endpoint_models")

# Boring-level features (no depth — endpoints are profile aggregates).
BORING_FEATURE_COLS = [
    "latitude_deg", "longitude_deg",
    "regime_code", "river_distance_km", "coast_distance_km",
    "absolute_elevation", "n_rows_in_boring",
]


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def _auc(p: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--endpoints",
        nargs="+",
        default=["depth_to_first_N30", "soft_thickness_lt5_0_to_10m",
                 "mean_N_upper_10m", "min_N_upper_10m"],
    )
    p.add_argument("--classifier", choices=["catboost", "lightgbm"],
                   default="catboost")
    p.add_argument("--fold-assignment", choices=["random", "contiguous"],
                   default="random")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--n-folds", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("Loading %s", a.parquet)
    df = pd.read_parquet(a.parquet)
    if a.quick:
        df = df.sample(80_000, random_state=a.seed).reset_index(drop=True)
    LOG.info("Reconstructing per-boring endpoints (%d rows -> ?)", len(df))

    from national.data.endpoints import build_endpoint_dataframe

    endpoints_df = build_endpoint_dataframe(df)
    LOG.info("Endpoint corpus: %d boreholes", len(endpoints_df))

    # Assign folds at the boring level. We use the same secondary-mesh
    # logic as the row-level assignment so a borehole and its row siblings
    # never split across folds.
    import sys as _sys
    _here = Path(__file__).resolve().parent
    if str(_here) not in _sys.path:
        _sys.path.insert(0, str(_here))
    from run_advanced_baselines import assign_folds

    bore_fold = assign_folds(
        endpoints_df, n_folds=a.n_folds, seed=a.seed,
        assignment=a.fold_assignment,
    )
    fold_sizes = [int((bore_fold == k).sum()) for k in range(a.n_folds)]
    LOG.info("Boring-level fold sizes: %s", fold_sizes)

    x_full = endpoints_df[BORING_FEATURE_COLS].values.astype(np.float32)

    from national.evaluation.baselines import (
        fit_predict_catboost,
        fit_predict_lightgbm,
        fit_predict_quantile_lightgbm,
    )

    fit_fn = fit_predict_catboost if a.classifier == "catboost" else fit_predict_lightgbm

    results: dict = {
        "endpoints": [],
        "n_boreholes": int(len(endpoints_df)),
        "fold_sizes": fold_sizes,
        "fold_assignment": a.fold_assignment,
        "classifier": a.classifier,
    }

    for endpoint in a.endpoints:
        LOG.info("==== endpoint %s ====", endpoint)
        y_full = endpoints_df[endpoint].values.astype(np.float32)
        # Filter out rows with un-trainable target values:
        #   +inf  -> right-censored (no observed crossing within survey depth)
        #   NaN   -> empty aggregation interval (e.g. mean_N_upper_10m
        #            on a borehole with no readings in [0, 10] m)
        is_censored = np.isinf(y_full) | np.isnan(y_full)
        # For depth_to_first_N30 we also evaluate the binary "has stiff
        # within 30 m" event using the `has_N30_within_30m` companion
        # column.
        has_event = (
            endpoints_df["has_N30_within_30m"].values.astype(np.int32)
            if endpoint == "depth_to_first_N30"
            else None
        )
        n_censored = int(is_censored.sum())
        LOG.info("    %d censored / %d total (%.1f%%)",
                 n_censored, len(y_full), 100.0 * n_censored / len(y_full))

        per_fold: list[dict] = []
        for k in range(a.n_folds):
            tr_mask = (bore_fold != k) & ~is_censored
            te_mask = (bore_fold == k)
            te_mask_uncen = te_mask & ~is_censored
            tx, ty = x_full[tr_mask], y_full[tr_mask]
            qx_full = x_full[te_mask]
            qx_uncen = x_full[te_mask_uncen]
            qy_uncen = y_full[te_mask_uncen]
            t0 = time.time()

            # Point prediction (CatBoost on uncensored)
            pred_point_test_full = fit_fn(tx, ty, qx_full)
            # Quantile envelope (P10 / P50 / P90)
            preds_q = fit_predict_quantile_lightgbm(
                tx, ty, qx_full, quantiles=(0.1, 0.5, 0.9),
            )
            lo, mid, hi = preds_q[0.1], preds_q[0.5], preds_q[0.9]

            # Save per-row predictions
            np.save(a.out_dir / f"{endpoint}_pred_point_fold{k}.npy",
                    pred_point_test_full.astype(np.float32))
            np.save(a.out_dir / f"{endpoint}_pred_p10_fold{k}.npy", lo.astype(np.float32))
            np.save(a.out_dir / f"{endpoint}_pred_p50_fold{k}.npy", mid.astype(np.float32))
            np.save(a.out_dir / f"{endpoint}_pred_p90_fold{k}.npy", hi.astype(np.float32))
            np.save(a.out_dir / f"{endpoint}_y_fold{k}.npy", y_full[te_mask].astype(np.float32))
            np.save(a.out_dir / f"{endpoint}_test_idx_fold{k}.npy", np.where(te_mask)[0])

            # Metrics on uncensored test rows
            pred_uncen = fit_fn(tx, ty, qx_uncen)
            mae = float(np.mean(np.abs(pred_uncen - qy_uncen)))
            rmse = float(np.sqrt(np.mean((pred_uncen - qy_uncen) ** 2)))
            r2 = _r2(qy_uncen, pred_uncen)
            preds_q_uncen = fit_predict_quantile_lightgbm(
                tx, ty, qx_uncen, quantiles=(0.1, 0.9),
            )
            lo_u, hi_u = preds_q_uncen[0.1], preds_q_uncen[0.9]
            cov_80 = float(np.mean((qy_uncen >= lo_u) & (qy_uncen <= hi_u)))
            width_80 = float(np.mean(hi_u - lo_u))

            row = {
                "fold": k, "n_test": int(te_mask.sum()),
                "n_test_uncensored": int(te_mask_uncen.sum()),
                "mae_uncensored": mae, "rmse_uncensored": rmse, "r2_uncensored": r2,
                "quantile_80_coverage": cov_80,
                "quantile_80_width": width_80,
                "wall_clock_s": time.time() - t0,
            }
            if has_event is not None:
                # For has_N30_within_30m: invert the point regressor's
                # prediction into a binary probability via "P(d <= 30) =
                # 1[predicted_depth <= 30]" — simple but interpretable.
                # A better approach (fit a separate classifier) is
                # follow-up; we report AUC of the predicted-depth itself
                # as a -prediction (smaller depth = more likely event).
                auc = _auc(-pred_point_test_full, has_event[te_mask])
                row["auc_has_event"] = auc
            per_fold.append(row)
            LOG.info(
                "    fold %d: MAE=%.3f RMSE=%.3f R2=%.3f Q80 cov=%.3f width=%.3f (%.1fs)",
                k, mae, rmse, r2, cov_80, width_80, row["wall_clock_s"],
            )

        endpoint_summary = {
            "endpoint": endpoint,
            "n_censored": n_censored,
            "per_fold": per_fold,
            "mean_mae": float(np.mean([f["mae_uncensored"] for f in per_fold])),
            "mean_rmse": float(np.mean([f["rmse_uncensored"] for f in per_fold])),
            "mean_r2": float(np.mean([f["r2_uncensored"] for f in per_fold])),
            "mean_q80_coverage": float(np.mean([f["quantile_80_coverage"] for f in per_fold])),
            "mean_q80_width": float(np.mean([f["quantile_80_width"] for f in per_fold])),
        }
        if has_event is not None:
            endpoint_summary["mean_auc_has_event"] = float(
                np.nanmean([f.get("auc_has_event", float("nan")) for f in per_fold])
            )
        results["endpoints"].append(endpoint_summary)
        LOG.info(
            "  -> mean MAE=%.3f RMSE=%.3f R2=%.3f Q80 cov=%.3f",
            endpoint_summary["mean_mae"], endpoint_summary["mean_rmse"],
            endpoint_summary["mean_r2"], endpoint_summary["mean_q80_coverage"],
        )

    (a.out_dir / "summary.json").write_text(json.dumps(results, indent=2))
    LOG.info("Wrote %s", a.out_dir / "summary.json")


if __name__ == "__main__":
    main()
