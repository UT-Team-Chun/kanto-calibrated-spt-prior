"""Per-regime evaluation metrics for national predictions.

The foundation model is regime-aware (FiLM), so a fair quality assessment
must report metrics per regime: dominant regimes (alluvial) can mask poor
performance on rare ones (volcanic ash, limestone). We compute RMSE, MAE,
R^2 and two probabilistic metrics (mean predictive log-likelihood and the
variance of standardized residuals) per ``regime_code`` value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def per_regime_metrics(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    *,
    regime_column: str = "regime_code",
) -> pd.DataFrame:
    """Return a DataFrame with one row per regime code.

    Columns: ``regime_code``, ``n_samples``, ``rmse``, ``mae``, ``r2``,
    ``mean_loglik``, ``z_var``. ``z_var`` is the variance of standardized
    residuals ``(y_true - y_pred) / y_std`` -- 1.0 means perfectly
    calibrated, >1 means over-confident, <1 means over-cautious.

    Args:
        df: must contain ``regime_column``; row order must match
            ``y_true``/``y_pred``/``y_std``.
        y_true / y_pred / y_std: 1-D arrays of equal length.
        regime_column: name of the regime code column.
    """
    if regime_column not in df.columns:
        raise ValueError(f"DataFrame missing {regime_column!r}")
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_std = np.asarray(y_std, dtype=np.float64)
    if not (y_true.shape == y_pred.shape == y_std.shape):
        raise ValueError("Shape mismatch among y_true / y_pred / y_std.")
    if len(df) != len(y_true):
        raise ValueError(
            f"Row count mismatch: df has {len(df)} rows, predictions have {len(y_true)}."
        )
    if (y_std <= 0).any():
        raise ValueError("y_std must be strictly positive.")

    regimes = df[regime_column].to_numpy()
    rows = []
    total_var = float(y_true.var()) + 1e-12  # for R^2 denominator (global var)
    for code in np.unique(regimes):
        mask = regimes == code
        yt = y_true[mask]
        yp = y_pred[mask]
        ys = y_std[mask]
        n = int(mask.sum())
        if n == 0:
            continue
        res = yt - yp
        rmse = float(np.sqrt((res * res).mean()))
        mae = float(np.abs(res).mean())
        # R^2 against the GLOBAL variance: a per-regime mean is a better
        # baseline but for cross-regime comparison we want a common reference.
        r2 = float(1.0 - (res * res).mean() / total_var)
        # Gaussian log-likelihood per point, then averaged.
        ll = -0.5 * np.log(2.0 * np.pi * ys * ys) - 0.5 * (res * res) / (ys * ys)
        mean_ll = float(ll.mean())
        z = res / ys
        z_var = float(z.var())
        rows.append(
            {
                "regime_code": int(code),
                "n_samples": n,
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "mean_loglik": mean_ll,
                "z_var": z_var,
            }
        )
    return pd.DataFrame(rows).sort_values("regime_code").reset_index(drop=True)


__all__ = ["per_regime_metrics"]
