#!/usr/bin/env python
"""Run tuned gradient-boosting + quantile baselines on the same spatial
3-fold protocol used by the operational DKL+SVGP model.

Outputs:
  docs/paper/paper_1_kanto/tables/advanced_baselines.tex
  data/runs/baselines_kanto_extended/results.json
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
DEFAULT_OUT_DIR = PROJECT_ROOT / "data/runs/baselines_kanto_extended"

LOG = logging.getLogger("baselines_advanced")

FEATURE_COLS = [
    "latitude_deg", "longitude_deg", "depth_from_surface",
    "absolute_elevation", "river_distance_km", "coast_distance_km",
    "regime_code",  # int, treated as numeric -> gradient boosters handle this
]


def secondary_mesh_code(lat: float, lon: float) -> int:
    p_lat = int(lat * 1.5)
    p_lon = int(lon - 100)
    s_lat = int((lat * 1.5 - p_lat) * 8)
    s_lon = int((lon - 100 - p_lon) * 8)
    return (p_lat * 1000 + p_lon) * 100 + s_lat * 10 + s_lon


def assign_folds(df: pd.DataFrame, n_folds: int = 3, seed: int = 42,
                  assignment: str = "random") -> np.ndarray:
    codes = np.array([secondary_mesh_code(lat, lon) for lat, lon in zip(df["latitude_deg"], df["longitude_deg"])])
    unique_codes, inverse = np.unique(codes, return_inverse=True)
    if assignment == "contiguous":
        from sklearn.cluster import KMeans
        # Centroid lat/lon for each unique secondary mesh
        rows_per_code: dict[int, list[int]] = {}
        for i, c in enumerate(codes):
            rows_per_code.setdefault(int(c), []).append(i)
        centroids = []
        for code in unique_codes:
            idx = rows_per_code[int(code)]
            centroids.append([df["latitude_deg"].iloc[idx].mean(),
                              df["longitude_deg"].iloc[idx].mean()])
        centroids = np.array(centroids)
        km = KMeans(n_clusters=n_folds, random_state=seed, n_init=10)
        fold_of_code = km.fit_predict(centroids).astype(np.int64)
        return fold_of_code[inverse]
    rng = np.random.default_rng(seed)
    order = rng.permutation(unique_codes.size)
    code_to_size = np.bincount(inverse)
    fold_of_code = np.empty(unique_codes.size, dtype=np.int64)
    fold_row_counts = np.zeros(n_folds, dtype=np.int64)
    for code_idx in order:
        target = int(np.argmin(fold_row_counts))
        fold_of_code[code_idx] = target
        fold_row_counts[target] += int(code_to_size[code_idx])
    return fold_of_code[inverse]


def conformal_radius(z: np.ndarray, alpha: float) -> float:
    n = len(z)
    k = int(np.ceil((n + 1) * alpha))
    k = min(max(k, 1), n)
    return float(np.sort(np.abs(z))[k - 1])


def _spatial_oob_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_codes: np.ndarray,
    fit_fn,
    n_inner_folds: int = 3,
    seed: int = 0,
) -> np.ndarray:
    """Compute spatial out-of-bag predictions via mesh-disjoint inner K-fold.

    Random inner OOB would let the booster spatially lookup-leak into row i
    via neighbour rows of the same secondary mesh, which would understate the
    spatial-generalisation residual that the GP must subsequently absorb.
    This helper enforces mesh-disjoint inner folds: row i is predicted by a
    booster that has seen no row sharing i's secondary mesh code.
    """
    unique_codes, code_inverse = np.unique(train_codes, return_inverse=True)
    rng = np.random.default_rng(seed)
    order = rng.permutation(unique_codes.size)
    code_to_size = np.bincount(code_inverse)
    fold_of_code = np.empty(unique_codes.size, dtype=np.int64)
    fold_row_counts = np.zeros(n_inner_folds, dtype=np.int64)
    for code_idx in order:
        target = int(np.argmin(fold_row_counts))
        fold_of_code[code_idx] = target
        fold_row_counts[target] += int(code_to_size[code_idx])
    inner_fold_for_row = fold_of_code[code_inverse]

    oob_pred = np.full(len(train_y), np.nan, dtype=np.float32)
    for f in range(n_inner_folds):
        inner_tr = inner_fold_for_row != f
        inner_te = inner_fold_for_row == f
        if inner_te.sum() == 0:
            continue
        pred_inner = fit_fn(train_x[inner_tr], train_y[inner_tr], train_x[inner_te])
        oob_pred[inner_te] = pred_inner.astype(np.float32)
    if np.isnan(oob_pred).any():
        raise RuntimeError(
            f"_spatial_oob_predict left {int(np.isnan(oob_pred).sum())} rows "
            f"without OOB predictions (inner_fold_distribution: "
            f"{fold_row_counts.tolist()})"
        )
    return oob_pred


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--baselines", nargs="+",
                   default=["lightgbm", "xgboost", "catboost", "qlightgbm"])
    p.add_argument("--quick", action="store_true")
    p.add_argument("--fold-assignment", choices=["random", "contiguous"],
                   default="random")
    p.add_argument("--save-fold-predictions", action="store_true",
                   help="Persist per-fold test predictions AND mesh-disjoint "
                        "spatial-OOB training predictions for hybrid SVGP use")
    p.add_argument("--oob-n-folds", type=int, default=3,
                   help="Inner-K for spatial OOB (only used when "
                        "--save-fold-predictions). Default 3.")
    a = p.parse_args()
    if a.fold_assignment == "contiguous":
        a.out_dir = a.out_dir.parent / (a.out_dir.name + "_contig")
    a.out_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("Loading %s", a.parquet)
    df = pd.read_parquet(a.parquet)
    if a.quick:
        df = df.sample(80_000, random_state=42).reset_index(drop=True)
    fold = assign_folds(df, assignment=a.fold_assignment)
    fold_sizes = [int((fold == k).sum()) for k in range(3)]
    LOG.info("Fold sizes (%s): %s", a.fold_assignment, fold_sizes)

    codes = np.array(
        [secondary_mesh_code(lat, lon)
         for lat, lon in zip(df["latitude_deg"], df["longitude_deg"])]
    )

    x_full = df[FEATURE_COLS].values.astype(np.float32)
    y_full = df["n_value"].values.astype(np.float32)

    from national.evaluation.baselines import (
        fit_predict_lightgbm,
        fit_predict_xgboost,
        fit_predict_catboost,
        fit_predict_quantile_lightgbm,
    )

    results = {}
    for baseline in a.baselines:
        LOG.info("==== %s ====", baseline)
        per_fold = []
        for k in range(3):
            tr_mask = fold != k
            te_mask = fold == k
            tx, ty = x_full[tr_mask], y_full[tr_mask]
            qx, qy = x_full[te_mask], y_full[te_mask]
            t0 = time.time()
            if baseline == "lightgbm":
                pred = fit_predict_lightgbm(tx, ty, qx)
                rmse = float(np.sqrt(np.mean((qy - pred) ** 2)))
                mae = float(np.mean(np.abs(qy - pred)))
                # Split conformal post-hoc (random calibration split inside train)
                rng = np.random.default_rng(42 + k)
                cal_idx = rng.choice(len(tx), size=max(2000, len(tx)//5), replace=False)
                cal_pred = fit_predict_lightgbm(np.delete(tx, cal_idx, axis=0),
                                                 np.delete(ty, cal_idx),
                                                 tx[cal_idx])
                z_cal = np.abs(ty[cal_idx] - cal_pred)
                q95 = conformal_radius(z_cal, 0.95)
                width95 = 2 * q95
                cov95 = float(np.mean(np.abs(qy - pred) <= q95))
                per_fold.append({"fold": k, "rmse": rmse, "mae": mae,
                                 "conformal_width_95": width95,
                                 "conformal_coverage_95": cov95,
                                 "wall_clock_s": time.time() - t0})
                if a.save_fold_predictions:
                    train_codes_k = codes[tr_mask]
                    pred_train_oob = _spatial_oob_predict(
                        train_x=tx, train_y=ty, train_codes=train_codes_k,
                        fit_fn=fit_predict_lightgbm,
                        n_inner_folds=a.oob_n_folds, seed=42 + 100 * k,
                    )
                    np.save(a.out_dir / f"lightgbm_pred_test_fold{k}.npy", pred.astype(np.float32))
                    np.save(a.out_dir / f"lightgbm_pred_train_oob_fold{k}.npy", pred_train_oob)
                    np.save(a.out_dir / f"lightgbm_idx_test_fold{k}.npy", np.where(te_mask)[0])
                    np.save(a.out_dir / f"lightgbm_idx_train_fold{k}.npy", np.where(tr_mask)[0])
                    LOG.info("    saved fold-%d lightgbm test (%d) + OOB train (%d) predictions",
                             k, len(pred), len(pred_train_oob))
            elif baseline == "xgboost":
                pred = fit_predict_xgboost(tx, ty, qx)
                rmse = float(np.sqrt(np.mean((qy - pred) ** 2)))
                mae = float(np.mean(np.abs(qy - pred)))
                per_fold.append({"fold": k, "rmse": rmse, "mae": mae,
                                 "wall_clock_s": time.time() - t0})
            elif baseline == "catboost":
                pred = fit_predict_catboost(tx, ty, qx)
                rmse = float(np.sqrt(np.mean((qy - pred) ** 2)))
                mae = float(np.mean(np.abs(qy - pred)))
                per_fold.append({"fold": k, "rmse": rmse, "mae": mae,
                                 "wall_clock_s": time.time() - t0})
                if a.save_fold_predictions:
                    train_codes_k = codes[tr_mask]
                    pred_train_oob = _spatial_oob_predict(
                        train_x=tx, train_y=ty, train_codes=train_codes_k,
                        fit_fn=fit_predict_catboost,
                        n_inner_folds=a.oob_n_folds, seed=42 + 100 * k,
                    )
                    np.save(a.out_dir / f"catboost_pred_test_fold{k}.npy", pred.astype(np.float32))
                    np.save(a.out_dir / f"catboost_pred_train_oob_fold{k}.npy", pred_train_oob)
                    np.save(a.out_dir / f"catboost_idx_test_fold{k}.npy", np.where(te_mask)[0])
                    np.save(a.out_dir / f"catboost_idx_train_fold{k}.npy", np.where(tr_mask)[0])
                    LOG.info("    saved fold-%d catboost test (%d) + OOB train (%d) predictions",
                             k, len(pred), len(pred_train_oob))
            elif baseline == "qlightgbm":
                preds = fit_predict_quantile_lightgbm(tx, ty, qx, quantiles=(0.025, 0.5, 0.975))
                rmse = float(np.sqrt(np.mean((qy - preds[0.5]) ** 2)))
                mae = float(np.mean(np.abs(qy - preds[0.5])))
                lo, hi = preds[0.025], preds[0.975]
                width95 = float(np.mean(hi - lo))
                cov95 = float(np.mean((qy >= lo) & (qy <= hi)))
                per_fold.append({"fold": k, "rmse": rmse, "mae": mae,
                                 "interval_width_95": width95,
                                 "coverage_95": cov95,
                                 "wall_clock_s": time.time() - t0})
            else:
                LOG.warning("Unknown baseline %s, skip", baseline)
            LOG.info("  fold %d done (%.1fs)", k, time.time() - t0)
        rmses = [r["rmse"] for r in per_fold]
        maes = [r["mae"] for r in per_fold]
        results[baseline] = {
            "per_fold": per_fold,
            "mean_rmse": float(np.mean(rmses)),
            "std_rmse": float(np.std(rmses)),
            "mean_mae": float(np.mean(maes)),
            "std_mae": float(np.std(maes)),
        }
        LOG.info(" -> mean RMSE %.3f +- %.3f", results[baseline]["mean_rmse"],
                 results[baseline]["std_rmse"])

    out_json = a.out_dir / "results.json"
    out_json.write_text(json.dumps(results, indent=2))
    LOG.info("Wrote %s", out_json)

    # Build LaTeX results table
    tex_lines = [
        r"\begin{table}[H]",
        r"  \caption{Tuned gradient-boosting baselines on the same spatial",
        r"           3-fold protocol as the operational DKL+SVGP model.",
        r"           RMSE / MAE in raw \Nblow{} units; conformal- or",
        r"           quantile-derived 95\% interval widths and empirical",
        r"           coverage where applicable.}",
        r"  \label{tab:advanced_baselines}",
        r"  \centering",
        r"  \small",
        r"  \begin{tabular}{lrrrr}",
        r"    \toprule",
        r"    Baseline & RMSE & MAE & 95\% width & 95\% cov.\ \\",
        r"    \midrule",
    ]
    pretty = {
        "lightgbm": "LightGBM (point + split conformal)",
        "xgboost": "XGBoost (point)",
        "catboost": "CatBoost (point)",
        "qlightgbm": "Quantile LightGBM (interval)",
    }
    for base in a.baselines:
        if base not in results:
            continue
        r = results[base]
        rmse_str = f"{r['mean_rmse']:.3f} $\\pm$ {r['std_rmse']:.3f}"
        mae_str = f"{r['mean_mae']:.3f} $\\pm$ {r['std_mae']:.3f}"
        # coverage / width
        widths = [pf.get("conformal_width_95") or pf.get("interval_width_95")
                  for pf in r["per_fold"]]
        covs = [pf.get("conformal_coverage_95") or pf.get("coverage_95")
                for pf in r["per_fold"]]
        if any(w is None for w in widths):
            width_str = "---"
            cov_str = "---"
        else:
            width_str = f"{np.mean(widths):.2f}"
            cov_str = f"{np.mean(covs)*100:.1f}\\%"
        tex_lines.append(
            f"    {pretty.get(base, base)} & {rmse_str} & {mae_str} & {width_str} & {cov_str} \\\\"
        )

    tex_lines += [
        r"    \midrule",
        r"    \textbf{Ours} (\BestKernel{}+\BestMean, conformal)"
        r" & \textbf{\OperationalRMSE} & \textbf{\OperationalMAE}"
        r" & --- & ${\sim}$95\% \\",
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    tex_path = PROJECT_ROOT / "docs/paper/paper_1_kanto/tables/advanced_baselines.tex"
    try:
        tex_path.parent.mkdir(parents=True, exist_ok=True)
        tex_path.write_text("\n".join(tex_lines) + "\n")
        LOG.info("Wrote %s", tex_path)
    except (FileNotFoundError, PermissionError, OSError) as exc:
        # When running inside an image that only includes the backend/
        # (e.g. cluster utens jobs), docs/ is not copied AND the parent
        # may not be creatable on a read-only NAS mount. Skip the
        # paper-table side effect; the predictions and summary.json are
        # already persisted to a.out_dir under data/runs/.
        LOG.info("Skipping %s (%s: %s) — likely a cluster run without "
                 "docs/ in the image, or a read-only filesystem",
                 tex_path, type(exc).__name__, exc)


if __name__ == "__main__":
    main()
