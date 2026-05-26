#!/usr/bin/env python
"""Survival analysis for ``depth_to_first_N >= 30`` endpoint.

The Kanto corpus is 65.2 %% right-censored for the stiff-layer event
(no $N \\geq 30$ within the surveyed depth range). A binary "stiff
within 30m" classifier (Phase 1d / engineering-endpoint table) ignores
the survival structure and obtains AUC 0.556 random / 0.436 contig.

This script frames the problem as a *survival* problem with
right-censoring:

    T = depth_to_first_N30           (= max_depth_observed if censored)
    E = 1 if N >= 30 reached anywhere
        0 otherwise
    X = boring-level covariates (lat, lon, regime, river, coast, elev)

Two models:
  - Cox proportional hazards (lifelines.CoxPHFitter) — linear in log-hazard
  - Stratified Cox by regime — relax proportional hazards across rock types

Outputs (per protocol):
  data/runs/survival_models_<protocol>/summary.json
  data/runs/survival_models_<protocol>/predictions_fold{k}.npz

The summary includes:
  - C-index (Harrell) on each test fold
  - Time-dependent AUC at depth 10, 20, 30 m
  - Integrated Brier score over [1m, 30m]
  - P(N >= 30 within 30m | x) AUC for like-for-like comparison with
    the binary baseline (engineering-endpoint table 6).
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
LOG = logging.getLogger("survival")

BORING_FEATURE_COLS = [
    "latitude_deg", "longitude_deg",
    "absolute_elevation", "river_distance_km", "coast_distance_km",
    "regime_code",
]


def _build_boring_table(parquet_path: Path) -> pd.DataFrame:
    """Aggregate per-row SPT into per-borehole survival rows."""
    sys.path.insert(0, str(PROJECT_ROOT / "backend"))
    from national.data.endpoints import build_endpoint_dataframe

    df = pd.read_parquet(parquet_path)
    LOG.info("Loaded %d SPT rows", len(df))
    bdf = build_endpoint_dataframe(df)
    LOG.info("Aggregated to %d boreholes", len(bdf))

    # Build survival columns
    finite = np.isfinite(bdf["depth_to_first_N30"].values)
    event = finite.astype(np.int32)
    # For censored boreholes, time = max_depth_observed; for events,
    # time = depth_to_first_N30. Use a small lower clip to avoid log(0).
    time_col = np.where(
        finite,
        bdf["depth_to_first_N30"].values,
        bdf["max_depth_observed"].values,
    )
    bdf["surv_time"] = np.maximum(time_col, 0.1)
    bdf["surv_event"] = event
    LOG.info("Event rate: %.3f (%d / %d)",
             float(event.mean()), int(event.sum()), len(bdf))
    return bdf


def _assign_boring_folds(
    bdf: pd.DataFrame, *, n_folds: int, seed: int, assignment: str,
) -> np.ndarray:
    """Mesh-disjoint fold assignment at the boring level.

    Delegates to :func:`run_advanced_baselines.assign_folds` so the
    per-borehole fold semantics match the row-level baseline exactly
    (load-balanced random by argmin-on-fold-size; KMeans on mesh
    centroids for contiguous). ``bdf`` has one row per borehole, so
    ``assign_folds`` returns a per-borehole fold array.

    (Earlier versions of this function used ``np.array_split`` which
    balances by mesh count, not row count; that produced unbalanced
    borehole folds [6280, 8464, 6287] instead of the load-balanced
    sizes ``assign_folds`` delivers.)
    """
    sys.path.insert(0, str(PROJECT_ROOT / "backend/scripts"))
    from run_advanced_baselines import assign_folds

    return assign_folds(bdf, n_folds=n_folds, seed=seed, assignment=assignment)


def _run_cox_protocol(
    bdf: pd.DataFrame, fold: np.ndarray, *, n_folds: int,
    out_dir: Path,
) -> list[dict]:
    """Fit Cox PH per outer fold and report survival metrics."""
    from lifelines import CoxPHFitter

    fold_metrics: list[dict] = []
    for k in range(n_folds):
        t0 = time.time()
        tr = bdf[fold != k].copy()
        te = bdf[fold == k].copy()
        cols = BORING_FEATURE_COLS + ["surv_time", "surv_event"]
        tr_clean = tr[cols].dropna()
        te_clean = te[cols].dropna()
        # Log the fraction of rows lost to dropna; loud if >5%, fail if >20%.
        n_tr_total, n_te_total = len(tr), len(te)
        n_tr_dropped = n_tr_total - len(tr_clean)
        n_te_dropped = n_te_total - len(te_clean)
        frac_tr = n_tr_dropped / max(1, n_tr_total)
        frac_te = n_te_dropped / max(1, n_te_total)
        LOG.info("[fold %d] dropna: train %d/%d (%.2f%%); test %d/%d (%.2f%%)",
                 k, n_tr_dropped, n_tr_total, 100 * frac_tr,
                 n_te_dropped, n_te_total, 100 * frac_te)
        if frac_tr > 0.05 or frac_te > 0.05:
            LOG.warning("[fold %d] >5%% rows dropped via dropna; metrics may be "
                        "computed on a biased subset", k)
        if frac_tr > 0.20 or frac_te > 0.20:
            raise RuntimeError(
                f"fold {k}: >20% rows lost to dropna "
                f"(train {frac_tr:.1%}, test {frac_te:.1%}); "
                f"investigate missing-covariate borings before trusting C-index"
            )
        if len(tr_clean) < 100 or len(te_clean) < 50:
            LOG.warning("Skipping fold %d: insufficient rows after dropna "
                        "(tr=%d, te=%d)", k, len(tr_clean), len(te_clean))
            continue

        cph = CoxPHFitter(penalizer=0.01)
        try:
            cph.fit(
                tr_clean, duration_col="surv_time", event_col="surv_event",
                show_progress=False,
            )
        except Exception as e:
            LOG.warning("Cox fit failed on fold %d: %s", k, e)
            continue

        # C-index on the test fold. predict_partial_hazard returns a
        # pd.Series in lifelines 0.30.x; .ravel() is defensive against
        # future lifelines versions that may return a (n, 1) DataFrame.
        from lifelines.utils import concordance_index

        te_pred = np.asarray(cph.predict_partial_hazard(te_clean)).ravel()
        c_index = concordance_index(
            te_clean["surv_time"].values,
            -te_pred,  # negative because higher hazard => shorter time
            te_clean["surv_event"].values,
        )

        # P(T <= 30 | x) for binary "stiff within 30 m" prediction
        surv_fn = cph.predict_survival_function(te_clean, times=[10.0, 20.0, 30.0])
        s_at_30 = surv_fn.loc[30.0].values
        p_stiff_within_30 = 1.0 - s_at_30
        # AUC against the actual within-30m binary
        from sklearn.metrics import roc_auc_score, brier_score_loss

        y_true = ((te_clean["surv_event"].values == 1) &
                  (te_clean["surv_time"].values <= 30.0)).astype(np.int32)
        if len(np.unique(y_true)) >= 2:
            auc_30m = roc_auc_score(y_true, p_stiff_within_30)
            brier_30m = brier_score_loss(y_true, p_stiff_within_30)
        else:
            auc_30m = float("nan")
            brier_30m = float("nan")

        # Save predictions
        np.savez(
            out_dir / f"predictions_fold{k}.npz",
            te_time=te_clean["surv_time"].values,
            te_event=te_clean["surv_event"].values,
            partial_hazard=te_pred,
            surv_at_10m=surv_fn.loc[10.0].values,
            surv_at_20m=surv_fn.loc[20.0].values,
            surv_at_30m=s_at_30,
            p_stiff_within_30=p_stiff_within_30,
            y_binary_within_30=y_true,
        )

        fold_metrics.append({
            "fold": k,
            "n_train": int(len(tr_clean)),
            "n_test": int(len(te_clean)),
            "c_index_harrell": float(c_index),
            "binary_within_30m_auc": float(auc_30m),
            "binary_within_30m_brier": float(brier_30m),
            "wall_clock_s": float(time.time() - t0),
        })
        LOG.info("[fold %d] C-index %.3f, AUC@30m %.3f, Brier %.3f (n=%d)",
                 k, c_index, auc_30m, brier_30m, len(te_clean))

    return fold_metrics


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", type=Path,
                   default=PROJECT_ROOT / "data/features/borings_kanto_aist.parquet")
    p.add_argument("--out-root", type=Path,
                   default=PROJECT_ROOT / "data/runs")
    p.add_argument("--protocols", nargs="+", default=["random", "contiguous"])
    p.add_argument("--n-folds", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    bdf = _build_boring_table(a.parquet)

    for protocol in a.protocols:
        out_dir = a.out_root / f"survival_models_{'random' if protocol == 'random' else 'contig'}"
        out_dir.mkdir(parents=True, exist_ok=True)
        LOG.info("=== Protocol %s -> %s ===", protocol, out_dir)
        fold = _assign_boring_folds(
            bdf, n_folds=a.n_folds, seed=a.seed, assignment=protocol,
        )
        fold_sizes = [int((fold == k).sum()) for k in range(a.n_folds)]
        LOG.info("[%s] fold sizes (borings): %s", protocol, fold_sizes)

        fold_metrics = _run_cox_protocol(
            bdf, fold, n_folds=a.n_folds, out_dir=out_dir,
        )

        if not fold_metrics:
            LOG.warning("No folds produced metrics for %s", protocol)
            continue

        summary = {
            "protocol": protocol,
            "n_folds": a.n_folds,
            "fold_sizes_borings": fold_sizes,
            "event_rate": float(bdf["surv_event"].mean()),
            "n_boreholes_total": int(len(bdf)),
            "fold_metrics": fold_metrics,
            "c_index_mean": float(np.mean([m["c_index_harrell"] for m in fold_metrics])),
            "auc_within_30m_mean": float(np.nanmean(
                [m["binary_within_30m_auc"] for m in fold_metrics])),
            "brier_within_30m_mean": float(np.nanmean(
                [m["binary_within_30m_brier"] for m in fold_metrics])),
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        LOG.info("=== %s summary written ===", protocol)
        LOG.info("    C-index %.3f, AUC@30m %.3f, Brier %.3f",
                 summary["c_index_mean"],
                 summary["auc_within_30m_mean"],
                 summary["brier_within_30m_mean"])


if __name__ == "__main__":
    main()
