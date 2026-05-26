#!/usr/bin/env python
"""Direct threshold-probability classifiers for SPT N-value exceedance.

Replaces the Gaussian-CDF-from-regression approximation
``P(N < c) = Phi((c - mu) / sigma)`` (currently used to build Figure 8 in
the paper) with **direct binary classifiers** on indicator labels
``I[N < c]`` (for soft layers) and ``I[N >= 30]`` (for the bearing-stratum
proxy). Conformal calibrates intervals, not CDFs, so the regression-CDF
approach is methodologically inconsistent with the rest of the calibration
story. Direct classifiers + isotonic recalibration give an honest
probability map.

Outputs per (threshold, fold-assignment) cell:
  data/runs/<out_dir>/<threshold>_<mode>_fold{k}.npy  (raw probabilities)
  data/runs/<out_dir>/<threshold>_<mode>_fold{k}_iso.npy  (isotonic-recalibrated)
  data/runs/<out_dir>/summary.json  (Brier / ECE / AUC / log-loss per fold)
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

LOG = logging.getLogger("threshold_classifiers")

FEATURE_COLS = [
    "latitude_deg", "longitude_deg", "depth_from_surface",
    "absolute_elevation", "river_distance_km", "coast_distance_km",
    "regime_code",
]


def _expected_calibration_error(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Standard ECE: |P(predicted) - empirical_freq| averaged over equal-mass bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    n = len(p)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        mean_p = float(p[mask].mean())
        emp = float(y[mask].mean())
        ece += (mask.sum() / n) * abs(mean_p - emp)
    return ece


def _brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def _log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p_clip = np.clip(p, 1e-7, 1.0 - 1e-7)
    return float(-np.mean(y * np.log(p_clip) + (1 - y) * np.log(1 - p_clip)))


def _auc(p: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def _fit_catboost_classifier(
    train_x: np.ndarray, train_y: np.ndarray, query_x: np.ndarray,
    *, iterations: int = 1500, learning_rate: float = 0.05, depth: int = 8,
    random_state: int = 42, class_weights: tuple[float, float] | None = None,
) -> np.ndarray:
    from catboost import CatBoostClassifier

    model = CatBoostClassifier(
        iterations=iterations, learning_rate=learning_rate, depth=depth,
        random_seed=random_state, verbose=False,
        class_weights=list(class_weights) if class_weights else None,
        loss_function="Logloss",
    )
    model.fit(np.asarray(train_x, dtype=np.float32),
              np.asarray(train_y, dtype=np.int32))
    proba = model.predict_proba(np.asarray(query_x, dtype=np.float32))
    return proba[:, 1].astype(np.float32)


def _fit_lightgbm_classifier(
    train_x: np.ndarray, train_y: np.ndarray, query_x: np.ndarray,
    *, n_estimators: int = 1500, learning_rate: float = 0.05,
    num_leaves: int = 127, random_state: int = 42,
    is_unbalance: bool = True,
) -> np.ndarray:
    import lightgbm as lgb

    model = lgb.LGBMClassifier(
        n_estimators=n_estimators, learning_rate=learning_rate,
        num_leaves=num_leaves, random_state=random_state,
        is_unbalance=is_unbalance, n_jobs=-1, verbose=-1,
    )
    model.fit(np.asarray(train_x, dtype=np.float32),
              np.asarray(train_y, dtype=np.int32))
    return model.predict_proba(np.asarray(query_x, dtype=np.float32))[:, 1].astype(np.float32)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--thresholds", type=float, nargs="+", default=[5.0, 10.0, 15.0, 30.0])
    p.add_argument("--threshold-modes", choices=["lt", "gte"], nargs="+",
                   default=["lt", "lt", "lt", "gte"],
                   help="Per-threshold comparison: lt = I[N < c], gte = I[N >= c]. "
                        "Must match length of --thresholds.")
    p.add_argument("--classifier", choices=["catboost", "lightgbm"],
                   default="catboost")
    p.add_argument("--fold-assignment", choices=["random", "contiguous"],
                   default="random")
    p.add_argument("--quick", action="store_true",
                   help="Subsample 80k rows for a fast sanity check.")
    p.add_argument("--isotonic", action="store_true", default=True,
                   help="Also apply isotonic recalibration on an inner cal "
                        "split (mesh-disjoint). Default ON.")
    p.add_argument("--n-folds", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    if len(a.threshold_modes) != len(a.thresholds):
        raise SystemExit(
            f"--threshold-modes ({len(a.threshold_modes)}) length must match "
            f"--thresholds ({len(a.thresholds)})"
        )
    a.out_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("Loading %s", a.parquet)
    df = pd.read_parquet(a.parquet)
    if a.quick:
        df = df.sample(80_000, random_state=a.seed).reset_index(drop=True)

    # Re-use the fold assignment helper from run_advanced_baselines so the
    # fold-id semantics are identical to the regression baselines.
    import sys as _sys
    _here = Path(__file__).resolve().parent
    if str(_here) not in _sys.path:
        _sys.path.insert(0, str(_here))
    from run_advanced_baselines import assign_folds, secondary_mesh_code

    fold = assign_folds(df, n_folds=a.n_folds, seed=a.seed,
                        assignment=a.fold_assignment)
    codes = np.array([secondary_mesh_code(lat, lon)
                      for lat, lon in zip(df["latitude_deg"], df["longitude_deg"])])
    fold_sizes = [int((fold == k).sum()) for k in range(a.n_folds)]
    LOG.info("Fold sizes (%s): %s", a.fold_assignment, fold_sizes)

    fit_fn = _fit_catboost_classifier if a.classifier == "catboost" else _fit_lightgbm_classifier
    x_full = df[FEATURE_COLS].values.astype(np.float32)
    n_value = df["n_value"].values.astype(np.float32)

    results: dict = {"thresholds": [], "fold_assignment": a.fold_assignment,
                     "classifier": a.classifier, "n_rows": int(len(df)),
                     "fold_sizes": fold_sizes}

    for thr, mode in zip(a.thresholds, a.threshold_modes):
        label = f"{mode}{thr:.0f}"
        LOG.info("==== threshold %s (mode=%s, c=%s) ====", label, mode, thr)
        if mode == "lt":
            y_full = (n_value < float(thr)).astype(np.int32)
        elif mode == "gte":
            y_full = (n_value >= float(thr)).astype(np.int32)
        else:
            raise ValueError(f"unknown threshold mode: {mode!r}")
        pos_rate = float(y_full.mean())
        LOG.info("    positive class rate (full corpus): %.3f", pos_rate)

        per_fold: list[dict] = []
        for k in range(a.n_folds):
            tr_mask = fold != k
            te_mask = fold == k
            tx, ty = x_full[tr_mask], y_full[tr_mask]
            qx, qy = x_full[te_mask], y_full[te_mask]
            t0 = time.time()

            # Inner mesh-disjoint cal split for isotonic recalibration
            inner_codes = codes[tr_mask]
            unique_inner, inv_inner = np.unique(inner_codes, return_inverse=True)
            rng = np.random.default_rng(a.seed + 1000 + 10 * k)
            cal_meshes = rng.choice(
                unique_inner.size, size=max(1, unique_inner.size // 5),
                replace=False,
            )
            cal_mesh_mask = np.zeros(unique_inner.size, dtype=bool)
            cal_mesh_mask[cal_meshes] = True
            inner_cal_mask = cal_mesh_mask[inv_inner]
            cal_x = tx[inner_cal_mask]
            cal_y = ty[inner_cal_mask]
            train_x = tx[~inner_cal_mask]
            train_y = ty[~inner_cal_mask]

            # Fit on train\cal, predict test (and cal for isotonic)
            class_weights = None
            if a.classifier == "catboost" and pos_rate < 0.2:
                # Rebalance rare class so the classifier doesn't trivialise
                # to all-zero on N>=30. CatBoost's class_weights expects
                # [w_negative, w_positive].
                class_weights = (1.0, max(1.0, (1.0 - pos_rate) / pos_rate))
            pred_test_raw = fit_fn(
                train_x, train_y, qx,
                **({"class_weights": class_weights} if a.classifier == "catboost" else {}),
            )
            pred_cal_raw = fit_fn(
                train_x, train_y, cal_x,
                **({"class_weights": class_weights} if a.classifier == "catboost" else {}),
            )

            # Isotonic recalibration
            from sklearn.isotonic import IsotonicRegression

            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(pred_cal_raw, cal_y)
            pred_test_iso = iso.transform(pred_test_raw).astype(np.float32)

            np.save(a.out_dir / f"{label}_pred_fold{k}.npy", pred_test_raw)
            np.save(a.out_dir / f"{label}_pred_iso_fold{k}.npy", pred_test_iso)
            np.save(a.out_dir / f"{label}_y_fold{k}.npy", qy.astype(np.int32))

            metrics_raw = {
                "brier": _brier(pred_test_raw, qy),
                "log_loss": _log_loss(pred_test_raw, qy),
                "auc": _auc(pred_test_raw, qy),
                "ece": _expected_calibration_error(pred_test_raw, qy),
            }
            metrics_iso = {
                "brier": _brier(pred_test_iso, qy),
                "log_loss": _log_loss(pred_test_iso, qy),
                "auc": _auc(pred_test_iso, qy),
                "ece": _expected_calibration_error(pred_test_iso, qy),
            }
            wall = time.time() - t0
            per_fold.append({
                "fold": k, "n_test": int(te_mask.sum()),
                "n_pos_test": int(qy.sum()),
                "raw": metrics_raw, "isotonic": metrics_iso,
                "wall_clock_s": wall,
            })
            LOG.info(
                "    fold %d: brier=%.4f -> %.4f (iso), AUC=%.3f, ECE=%.4f -> %.4f, %.1fs",
                k, metrics_raw["brier"], metrics_iso["brier"],
                metrics_raw["auc"], metrics_raw["ece"], metrics_iso["ece"], wall,
            )

        threshold_summary = {
            "threshold": float(thr), "mode": mode, "label": label,
            "positive_rate": pos_rate, "per_fold": per_fold,
            "mean_brier_raw": float(np.mean([f["raw"]["brier"] for f in per_fold])),
            "mean_brier_iso": float(np.mean([f["isotonic"]["brier"] for f in per_fold])),
            "mean_auc_raw": float(np.mean([f["raw"]["auc"] for f in per_fold])),
            "mean_ece_raw": float(np.mean([f["raw"]["ece"] for f in per_fold])),
            "mean_ece_iso": float(np.mean([f["isotonic"]["ece"] for f in per_fold])),
        }
        results["thresholds"].append(threshold_summary)
        LOG.info(
            "  -> mean Brier raw=%.4f / iso=%.4f, AUC=%.3f, ECE raw=%.4f / iso=%.4f",
            threshold_summary["mean_brier_raw"], threshold_summary["mean_brier_iso"],
            threshold_summary["mean_auc_raw"], threshold_summary["mean_ece_raw"],
            threshold_summary["mean_ece_iso"],
        )

    (a.out_dir / "summary.json").write_text(json.dumps(results, indent=2))
    LOG.info("Wrote %s", a.out_dir / "summary.json")


if __name__ == "__main__":
    main()
