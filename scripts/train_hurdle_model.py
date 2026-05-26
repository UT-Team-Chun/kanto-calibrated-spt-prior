#!/usr/bin/env python
"""Two-stage hurdle model for $N$-value regression.

Stage 1: P(N >= 30 | x) — binary classifier (CatBoost + isotonic
recalibration on a mesh-disjoint inner cal split).
Stage 2a: mu_soft(x) = E[N | N < 30, x] — CatBoost regressor on
the N < 30 subset (~93.6% of rows).
Stage 2b: mu_stiff(x) = E[N | N >= 30, x] — CatBoost regressor on
the N >= 30 subset (~6.4% of rows).

Deployment-time prediction (per row x):
    p = stage1.predict_proba(x)[:, 1]    # iso-recal probability
    mu_hat = (1 - p) * mu_soft + p * mu_stiff
    sigma2_hat = p*sigma2_stiff + (1-p)*sigma2_soft
                 + p*(1-p)*(mu_stiff - mu_soft)**2

Diagnostic: the N >= 30 conditional coverage failure (Mondrian
delivers only +1pp on the stiff-layer subset in the single-regressor
setup) is hypothesised to be a point-estimator failure rather than a
conformal-radius failure. The hurdle model isolates the stiff
component into a dedicated regressor, so a clean improvement on
N >= 30 RMSE / coverage would confirm the diagnosis.

Outputs (per protocol):
  data/runs/hurdle_models_<protocol>/summary.json
  data/runs/hurdle_models_<protocol>/<metric>_fold{k}.npy  (predictions)
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG = logging.getLogger("hurdle")

FEATURE_COLS = [
    "latitude_deg", "longitude_deg", "depth_from_surface",
    "absolute_elevation", "river_distance_km", "coast_distance_km",
    "regime_code",
]


def _fit_catboost_regressor(train_x, train_y, query_x,
                            iterations=1500, learning_rate=0.05, depth=8,
                            random_state=42,
                            eval_x=None, eval_y=None,
                            early_stopping_rounds=None):
    """Fit CatBoost regressor with optional early stopping on (eval_x, eval_y).

    When ``eval_x`` is provided, CatBoost uses it as the early-stopping
    eval set. Without it, the model trains for the full ``iterations``.
    The caller is responsible for ensuring ``eval_x`` is held-out from
    ``train_x``.
    """
    from catboost import CatBoostRegressor, Pool

    model = CatBoostRegressor(
        iterations=iterations, learning_rate=learning_rate, depth=depth,
        random_seed=random_state, verbose=False,
        loss_function="RMSE",
    )
    if eval_x is not None and eval_y is not None and early_stopping_rounds:
        eval_pool = Pool(eval_x.astype(np.float32),
                         eval_y.astype(np.float32))
        model.fit(train_x.astype(np.float32), train_y.astype(np.float32),
                  eval_set=eval_pool,
                  early_stopping_rounds=early_stopping_rounds,
                  use_best_model=True)
    else:
        model.fit(train_x.astype(np.float32), train_y.astype(np.float32))
    pred = model.predict(query_x.astype(np.float32))
    return pred.astype(np.float32), model


def _fit_catboost_classifier(train_x, train_y, query_x,
                              class_weights=None,
                              iterations=1500, learning_rate=0.05, depth=8,
                              random_state=42):
    from catboost import CatBoostClassifier

    model = CatBoostClassifier(
        iterations=iterations, learning_rate=learning_rate, depth=depth,
        random_seed=random_state, verbose=False,
        class_weights=list(class_weights) if class_weights else None,
        loss_function="Logloss",
    )
    model.fit(train_x.astype(np.float32), train_y.astype(np.int32))
    pred = model.predict_proba(query_x.astype(np.float32))[:, 1]
    return pred.astype(np.float32), model


def _conformal_radius(z: np.ndarray, alpha: float) -> float:
    n = len(z)
    k = int(np.ceil((n + 1) * alpha))
    k = min(max(k, 1), n)
    return float(np.sort(np.abs(z))[k - 1])


def _split_inner_cal(train_codes: np.ndarray, seed: int, frac: float = 0.2):
    """Mesh-disjoint inner cal split. Returns boolean cal mask over train rows."""
    unique_codes, inv = np.unique(train_codes, return_inverse=True)
    rng = np.random.default_rng(seed)
    n_cal = max(1, int(unique_codes.size * frac))
    cal_meshes = rng.choice(unique_codes.size, size=n_cal, replace=False)
    cal_mesh_mask = np.zeros(unique_codes.size, dtype=bool)
    cal_mesh_mask[cal_meshes] = True
    return cal_mesh_mask[inv]


def _split_three_way(train_codes: np.ndarray, seed: int,
                      iso_frac: float = 0.15,
                      conf_frac: float = 0.15):
    """Three-way mesh-disjoint split: train (~70%) / iso-cal / conformal-cal.

    Returns (train_mask, iso_cal_mask, conf_cal_mask), all aligned to
    the input ``train_codes`` row order. All three are boolean.
    """
    unique_codes, inv = np.unique(train_codes, return_inverse=True)
    rng = np.random.default_rng(seed)
    order = rng.permutation(unique_codes.size)
    n_iso = max(1, int(unique_codes.size * iso_frac))
    n_conf = max(1, int(unique_codes.size * conf_frac))
    iso_meshes = order[:n_iso]
    conf_meshes = order[n_iso:n_iso + n_conf]
    iso_mesh_mask = np.zeros(unique_codes.size, dtype=bool)
    conf_mesh_mask = np.zeros(unique_codes.size, dtype=bool)
    iso_mesh_mask[iso_meshes] = True
    conf_mesh_mask[conf_meshes] = True
    iso_cal_mask = iso_mesh_mask[inv]
    conf_cal_mask = conf_mesh_mask[inv]
    train_mask = ~(iso_cal_mask | conf_cal_mask)
    return train_mask, iso_cal_mask, conf_cal_mask


def _run_protocol(df: pd.DataFrame, protocol: str, n_folds: int, seed: int,
                   out_dir: Path, stiff_threshold: float = 30.0):
    """Run hurdle pipeline for one protocol. Returns per-fold metric dicts."""
    from sklearn.isotonic import IsotonicRegression

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_advanced_baselines import assign_folds, secondary_mesh_code

    fold = assign_folds(df, n_folds=n_folds, seed=seed, assignment=protocol)
    codes = np.array([secondary_mesh_code(la, lo)
                       for la, lo in zip(df["latitude_deg"], df["longitude_deg"])])
    fold_sizes = [int((fold == k).sum()) for k in range(n_folds)]
    LOG.info("[%s] fold sizes: %s", protocol, fold_sizes)

    x_full = df[FEATURE_COLS].values.astype(np.float32)
    n_value = df["n_value"].values.astype(np.float32)
    stiff_full = (n_value >= stiff_threshold).astype(np.int32)
    stiff_rate = float(stiff_full.mean())
    LOG.info("[%s] corpus stiff rate (N>=%g): %.4f",
             protocol, stiff_threshold, stiff_rate)

    fold_metrics = []
    for k in range(n_folds):
        t0 = time.time()
        tr = fold != k
        te = fold == k
        tx, ty = x_full[tr], n_value[tr]
        qx, qy = x_full[te], n_value[te]
        tr_codes = codes[tr]
        tr_stiff = stiff_full[tr]
        te_stiff = stiff_full[te]

        # Three-way mesh-disjoint split: stage2-train / iso-cal / conformal-cal.
        # The previous version used the same inner cal for both isotonic
        # recalibration and conformal radius, AND trained stage 2 on the
        # full train set (including iso-cal rows), so the conformal
        # residuals on the iso-cal subset were training-fit residuals —
        # systematically too small. Three-way splitting fixes both leaks.
        train2_mask, iso_cal_mask, conf_cal_mask = _split_three_way(
            tr_codes, seed=seed + 1000 + 10 * k,
            iso_frac=0.15, conf_frac=0.15,
        )
        # Stage 1: P(N >= stiff | x), CatBoost classifier + isotonic recal
        cls_x_tr = tx[train2_mask]
        cls_y_tr = tr_stiff[train2_mask]
        cls_x_iso = tx[iso_cal_mask]
        cls_y_iso = tr_stiff[iso_cal_mask]
        pos_rate = float(cls_y_tr.mean())
        class_weights = (1.0, max(1.0, (1.0 - pos_rate) / pos_rate)) \
            if pos_rate < 0.2 else None

        LOG.info("[%s f%d] stage 1 (cls, %d rows; iso-cal %d; conf-cal %d)",
                 protocol, k, int(train2_mask.sum()),
                 int(iso_cal_mask.sum()), int(conf_cal_mask.sum()))
        # Predict raw on (iso-cal, conf-cal, test); fit isotonic on iso-cal
        raw_iso, cls_model = _fit_catboost_classifier(
            cls_x_tr, cls_y_tr, cls_x_iso,
            class_weights=class_weights,
        )
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(raw_iso, cls_y_iso)
        raw_conf = cls_model.predict_proba(
            tx[conf_cal_mask].astype(np.float32)
        )[:, 1]
        p_conf = iso.transform(raw_conf).astype(np.float32)
        raw_te = cls_model.predict_proba(qx.astype(np.float32))[:, 1]
        p_te = iso.transform(raw_te).astype(np.float32)

        # Stage 2a: mu_soft on N < threshold subset of stage2-train.
        # Stage 2b: mu_stiff on N >= threshold subset of stage2-train.
        # Use the same iter=1500 as the original campaign; empirical
        # test stiff RMSE is lower at iter=1500 than at iter=400, so the
        # initial overfit hypothesis (small 14k stiff subset → severe
        # overfit) was not supported by the held-out comparison.
        soft_in_train2 = train2_mask & (tr_stiff == 0)
        LOG.info("[%s f%d] stage 2a: mu_soft on N<%g (%d rows)",
                 protocol, k, stiff_threshold, int(soft_in_train2.sum()))
        mu_soft, soft_model = _fit_catboost_regressor(
            tx[soft_in_train2], ty[soft_in_train2], qx,
        )

        stiff_in_train2 = train2_mask & (tr_stiff == 1)
        LOG.info("[%s f%d] stage 2b: mu_stiff on N>=%g (%d rows)",
                 protocol, k, stiff_threshold, int(stiff_in_train2.sum()))
        mu_stiff, stiff_model = _fit_catboost_regressor(
            tx[stiff_in_train2], ty[stiff_in_train2], qx,
        )

        # Hurdle combination on test.
        # sigma2_soft / sigma2_stiff are estimated below from held-out
        # conf-cal residuals rather than training residuals (the earlier
        # training-fit version systematically under-estimated sigma2 and
        # narrowed the resulting prediction intervals).
        mu_hat = (1 - p_te) * mu_soft + p_te * mu_stiff

        # Single-regressor baseline (for direct comparison) — also on stage2-train
        LOG.info("[%s f%d] baseline single regressor on stage2-train", protocol, k)
        mu_single, single_model = _fit_catboost_regressor(
            tx[train2_mask], ty[train2_mask], qx,
        )

        # Metrics: overall + N>=30 subset
        rmse = float(np.sqrt(np.mean((qy - mu_hat) ** 2)))
        mae = float(np.mean(np.abs(qy - mu_hat)))
        rmse_single = float(np.sqrt(np.mean((qy - mu_single) ** 2)))
        mae_single = float(np.mean(np.abs(qy - mu_single)))
        if te_stiff.sum() > 0:
            rmse_stiff = float(np.sqrt(np.mean((qy[te_stiff == 1] - mu_hat[te_stiff == 1]) ** 2)))
            mae_stiff = float(np.mean(np.abs(qy[te_stiff == 1] - mu_hat[te_stiff == 1])))
            rmse_stiff_single = float(np.sqrt(np.mean((qy[te_stiff == 1] - mu_single[te_stiff == 1]) ** 2)))
        else:
            rmse_stiff = mae_stiff = rmse_stiff_single = float("nan")

        # Split conformal on the held-out conformal-cal split (disjoint
        # from stage2-train and iso-cal).
        conf_y = ty[conf_cal_mask]
        conf_mu_soft = soft_model.predict(tx[conf_cal_mask].astype(np.float32))
        conf_mu_stiff = stiff_model.predict(tx[conf_cal_mask].astype(np.float32))
        conf_mu_hat = (1 - p_conf) * conf_mu_soft + p_conf * conf_mu_stiff
        conf_resid = conf_y - conf_mu_hat
        conf_stiff_mask_arr = conf_y >= stiff_threshold
        conf_stiff = conf_stiff_mask_arr.astype(np.int32)

        # Estimate sigma2_soft / sigma2_stiff on held-out conf-cal residuals
        # stratified by the TRUE stiff label. This replaces the earlier
        # training-fit variance estimate, which was systematically too small
        # because the regressors had been trained on the same rows.
        soft_resid_holdout = conf_y[~conf_stiff_mask_arr] - conf_mu_soft[~conf_stiff_mask_arr]
        stiff_resid_holdout = conf_y[conf_stiff_mask_arr] - conf_mu_stiff[conf_stiff_mask_arr]
        if soft_resid_holdout.size >= 50:
            sigma2_soft = float(np.var(soft_resid_holdout))
        else:
            sigma2_soft = float(np.var(conf_resid))  # fallback
        if stiff_resid_holdout.size >= 50:
            sigma2_stiff = float(np.var(stiff_resid_holdout))
        else:
            sigma2_stiff = float(np.var(conf_resid))  # fallback
        sigma2_hat = (p_te * sigma2_stiff + (1 - p_te) * sigma2_soft
                       + p_te * (1 - p_te) * (mu_stiff - mu_soft) ** 2)
        sigma_hat = np.sqrt(np.maximum(sigma2_hat, 1e-9))

        # Marginal conformal radius
        q95 = _conformal_radius(conf_resid, 0.95)
        cov95 = float(np.mean(np.abs(qy - mu_hat) <= q95))
        if te_stiff.sum() > 0:
            cov95_stiff = float(np.mean(
                np.abs(qy[te_stiff == 1] - mu_hat[te_stiff == 1]) <= q95
            ))
        else:
            cov95_stiff = float("nan")

        # Mondrian (deployable): stratify by p_stiff quintile bucket
        # so each test row gets the radius of its predicted-probability
        # group. More robust than a single p_stiff >= 0.5 cutoff when
        # the rare class has predicted probabilities concentrated near
        # the corpus prior. Falls back to marginal when bucket has < 50
        # cal rows. Use pd.qcut with duplicates='drop' so flat regions of
        # the isotonic-recalibrated probabilities (which often produce
        # many ties) don't collapse multiple buckets into one ambiguous
        # bin via np.searchsorted on duplicate edges.
        N_BUCKETS_REQ = 5
        try:
            _, p_edges_unique = pd.qcut(
                p_conf, q=N_BUCKETS_REQ, retbins=True,
                duplicates="drop",
            )
            p_edges_unique = np.asarray(p_edges_unique, dtype=np.float64)
        except ValueError:
            p_edges_unique = np.array([p_conf.min(), p_conf.max()],
                                       dtype=np.float64)
        # Convert to open-ended cut edges for searchsorted.
        p_edges_unique[0] = -np.inf
        p_edges_unique[-1] = np.inf
        n_buckets_actual = max(1, p_edges_unique.size - 1)
        conf_buckets = np.searchsorted(p_edges_unique[1:-1], p_conf, side="right")
        te_buckets = np.searchsorted(p_edges_unique[1:-1], p_te, side="right")
        bucket_q95: list[float] = []
        for b in range(n_buckets_actual):
            mask_b = conf_buckets == b
            if mask_b.sum() >= 50:
                bucket_q95.append(_conformal_radius(conf_resid[mask_b], 0.95))
            else:
                bucket_q95.append(q95)
        bucket_q95_arr = np.array(bucket_q95, dtype=np.float32)
        # Clamp test bucket indices in case te has values outside conf range.
        te_buckets = np.clip(te_buckets, 0, n_buckets_actual - 1)
        te_q = bucket_q95_arr[te_buckets]
        cov95_mondrian = float(np.mean(np.abs(qy - mu_hat) <= te_q))
        if te_stiff.sum() > 0:
            cov95_mondrian_stiff = float(np.mean(
                np.abs(qy[te_stiff == 1] - mu_hat[te_stiff == 1])
                <= te_q[te_stiff == 1]
            ))
        else:
            cov95_mondrian_stiff = float("nan")
        q95_soft_g = float(bucket_q95[0])     # smallest-p_stiff bucket
        q95_stiff_g = float(bucket_q95[-1])   # largest-p_stiff bucket

        # Diagnostic: conditional radius on TRUE stiff label (not
        # deployable since the label is unknown at prediction time, but
        # tells us the magnitude of stiff vs soft residuals).
        conf_true_stiff = conf_y >= stiff_threshold
        if conf_true_stiff.sum() >= 20 and (~conf_true_stiff).sum() >= 20:
            q95_true_soft = _conformal_radius(conf_resid[~conf_true_stiff], 0.95)
            q95_true_stiff = _conformal_radius(conf_resid[conf_true_stiff], 0.95)
        else:
            q95_true_soft = q95_true_stiff = float("nan")

        # Persist per-fold predictions
        np.save(out_dir / f"mu_hat_fold{k}.npy", mu_hat)
        np.save(out_dir / f"sigma_hat_fold{k}.npy", sigma_hat)
        np.save(out_dir / f"p_stiff_fold{k}.npy", p_te)
        np.save(out_dir / f"mu_single_fold{k}.npy", mu_single)
        np.save(out_dir / f"y_test_fold{k}.npy", qy)
        np.save(out_dir / f"stiff_test_fold{k}.npy", te_stiff)

        fold_metric = {
            "fold": k,
            "rmse_hurdle": rmse,
            "mae_hurdle": mae,
            "rmse_single": rmse_single,
            "mae_single": mae_single,
            "rmse_stiff_hurdle": rmse_stiff,
            "mae_stiff_hurdle": mae_stiff,
            "rmse_stiff_single": rmse_stiff_single,
            "conformal_q95": q95,
            "conformal_q95_lowp_bucket": q95_soft_g,
            "conformal_q95_highp_bucket": q95_stiff_g,
            "conformal_q95_true_soft": q95_true_soft,
            "conformal_q95_true_stiff": q95_true_stiff,
            "conformal_cov95_marginal": cov95,
            "conformal_cov95_stiff": cov95_stiff,
            "conformal_cov95_mondrian": cov95_mondrian,
            "conformal_cov95_mondrian_stiff": cov95_mondrian_stiff,
            "sigma2_soft": sigma2_soft,
            "sigma2_stiff": sigma2_stiff,
            "wall_clock_s": time.time() - t0,
        }
        fold_metrics.append(fold_metric)
        LOG.info("[%s f%d] RMSE hurdle %.3f single %.3f; "
                 "stiff RMSE hurdle %.3f single %.3f; "
                 "cov95 marginal %.3f stiff %.3f; "
                 "Mondrian-by-p_stiff %.3f stiff %.3f",
                 protocol, k, rmse, rmse_single,
                 rmse_stiff, rmse_stiff_single,
                 cov95, cov95_stiff,
                 cov95_mondrian, cov95_mondrian_stiff)

    return fold_metrics, fold_sizes, stiff_rate


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", type=Path,
                   default=PROJECT_ROOT / "data/features/borings_kanto_aist.parquet")
    p.add_argument("--out-root", type=Path,
                   default=PROJECT_ROOT / "data/runs")
    p.add_argument("--protocols", nargs="+",
                   default=["random", "contiguous"])
    p.add_argument("--n-folds", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stiff-threshold", type=float, default=30.0)
    p.add_argument("--quick", action="store_true",
                   help="Subsample 80k rows for sanity-check")
    a = p.parse_args()

    LOG.info("Loading %s", a.parquet)
    df = pd.read_parquet(a.parquet)
    if a.quick:
        df = df.sample(80_000, random_state=a.seed).reset_index(drop=True)
    LOG.info("Total rows: %d", len(df))

    for protocol in a.protocols:
        out_dir = a.out_root / f"hurdle_models_{'random' if protocol == 'random' else 'contig'}"
        out_dir.mkdir(parents=True, exist_ok=True)
        LOG.info("=== Protocol %s -> %s ===", protocol, out_dir)
        fold_metrics, fold_sizes, stiff_rate = _run_protocol(
            df, protocol, a.n_folds, a.seed, out_dir, a.stiff_threshold,
        )
        summary = {
            "protocol": protocol,
            "n_folds": a.n_folds,
            "fold_sizes": fold_sizes,
            "stiff_rate": stiff_rate,
            "stiff_threshold": a.stiff_threshold,
            "fold_metrics": fold_metrics,
            "rmse_hurdle_mean": float(np.mean([m["rmse_hurdle"] for m in fold_metrics])),
            "rmse_single_mean": float(np.mean([m["rmse_single"] for m in fold_metrics])),
            "rmse_stiff_hurdle_mean": float(np.nanmean([m["rmse_stiff_hurdle"] for m in fold_metrics])),
            "rmse_stiff_single_mean": float(np.nanmean([m["rmse_stiff_single"] for m in fold_metrics])),
            "cov95_marginal_mean": float(np.mean([m["conformal_cov95_marginal"] for m in fold_metrics])),
            "cov95_stiff_mean": float(np.nanmean([m["conformal_cov95_stiff"] for m in fold_metrics])),
            "cov95_mondrian_mean": float(np.nanmean([m["conformal_cov95_mondrian"] for m in fold_metrics])),
            "cov95_mondrian_stiff_mean": float(np.nanmean([m["conformal_cov95_mondrian_stiff"] for m in fold_metrics])),
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        LOG.info("=== %s summary written to %s ===", protocol, out_dir / "summary.json")
        LOG.info("    rmse_hurdle %.3f vs single %.3f",
                 summary["rmse_hurdle_mean"], summary["rmse_single_mean"])
        LOG.info("    stiff RMSE hurdle %.3f vs single %.3f",
                 summary["rmse_stiff_hurdle_mean"], summary["rmse_stiff_single_mean"])
        LOG.info("    cov95 marginal %.3f stiff %.3f; "
                 "Mondrian %.3f stiff %.3f",
                 summary["cov95_marginal_mean"], summary["cov95_stiff_mean"],
                 summary["cov95_mondrian_mean"], summary["cov95_mondrian_stiff_mean"])


if __name__ == "__main__":
    main()
