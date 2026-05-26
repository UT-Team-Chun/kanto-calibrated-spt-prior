"""Baseline interpolation and non-neural regression models.

The foundation model must beat these baselines on the spatial K-fold gate
defined in ``docs/architecture.md``. Implementations here favor robustness
and clarity over raw speed -- they only need to run once per evaluation,
not in a training inner loop.
"""

from __future__ import annotations

import numpy as np


def fit_predict_idw(
    train_xyz: np.ndarray,
    train_y: np.ndarray,
    query_xyz: np.ndarray,
    *,
    power: float = 2.0,
    k: int = 16,
) -> np.ndarray:
    """k-NN inverse-distance weighting.

    Distance is Euclidean in whatever coordinate system the inputs use --
    callers typically pass ``[x_utm_m, y_utm_m, depth_m]`` for meaningful
    distances. For degree-coordinate inputs, scale longitude by
    ``cos(lat)`` first.

    Args:
        train_xyz: ``(N, D)`` training coordinates.
        train_y: ``(N,)`` training targets.
        query_xyz: ``(M, D)`` query coordinates.
        power: distance exponent.
        k: number of neighbors used per query (clipped to ``N``).

    Returns:
        ``(M,)`` predicted values.
    """
    from scipy.spatial import cKDTree

    train_xyz = np.asarray(train_xyz, dtype=np.float64)
    train_y = np.asarray(train_y, dtype=np.float64)
    query_xyz = np.asarray(query_xyz, dtype=np.float64)

    if train_xyz.ndim != 2 or query_xyz.ndim != 2:
        raise ValueError(
            f"Expected 2-D coordinate arrays, got shapes "
            f"{train_xyz.shape} and {query_xyz.shape}"
        )
    if train_xyz.shape[1] != query_xyz.shape[1]:
        raise ValueError(
            f"Coordinate dimensions differ: train={train_xyz.shape[1]}, "
            f"query={query_xyz.shape[1]}"
        )

    k_eff = int(min(k, train_xyz.shape[0]))
    tree = cKDTree(train_xyz)
    dist, idx = tree.query(query_xyz, k=k_eff)
    if k_eff == 1:
        dist = dist[:, None]
        idx = idx[:, None]
    # Add tiny epsilon to avoid divide-by-zero when a query is exactly on a train point.
    weights = 1.0 / np.maximum(dist, 1e-12) ** power
    weights /= weights.sum(axis=1, keepdims=True)
    return (train_y[idx] * weights).sum(axis=1)


def fit_predict_kriging(
    train_xyz: np.ndarray,
    train_y: np.ndarray,
    query_xyz: np.ndarray,
    *,
    n_subsample: int = 10000,
    nu: float = 1.5,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Ordinary kriging baseline via ``sklearn.gaussian_process``.

    Full-data kriging on >\\!100\\,000 rows is intractable
    ($O(N^3)$ Cholesky on the kernel matrix). We subsample
    ``n_subsample`` training points (random uniform), fit a constant-mean
    Gaussian process with a Mat\\'ern kernel, and predict at the query
    points. Returns (mean, std).

    This is the standard "classical geostatistics" comparison cell that
    geotechnical reviewers expect. It is *not* equivalent to our SVGP
    foundation model — fewer training points, no learned encoder — but
    establishes the lower bound on what a Gaussian-process-only approach
    delivers on the same spatial fold.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import (
        ConstantKernel,
        Matern,
        WhiteKernel,
    )

    train_xyz = np.asarray(train_xyz, dtype=np.float64)
    train_y = np.asarray(train_y, dtype=np.float64)
    query_xyz = np.asarray(query_xyz, dtype=np.float64)

    rng = np.random.default_rng(random_state)
    n = train_xyz.shape[0]
    if n > n_subsample:
        idx = rng.choice(n, size=n_subsample, replace=False)
        idx.sort()
        train_xyz = train_xyz[idx]
        train_y = train_y[idx]

    # Constant + Matern + white-noise kernel; lengthscales free.
    kernel = (
        ConstantKernel(1.0, (1e-2, 1e3))
        * Matern(length_scale=[1.0] * train_xyz.shape[1], nu=nu)
        + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-3, 1e2))
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        # n_restarts_optimizer=0 — single L-BFGS-B pass at default init.
        # Restarts on a 5k–10k subsample take 30 min+ per fold; the
        # marginal NLL is good enough on the first pass for a baseline.
        n_restarts_optimizer=0,
        random_state=random_state,
    )
    gp.fit(train_xyz, train_y)
    mean, std = gp.predict(query_xyz, return_std=True)
    return mean, std


def fit_predict_hgb(
    train_x: np.ndarray,
    train_y: np.ndarray,
    query_x: np.ndarray,
    *,
    max_iter: int = 500,
    learning_rate: float = 0.05,
    max_depth: int | None = 8,
    random_state: int = 42,
) -> np.ndarray:
    """Histogram Gradient Boosting baseline (sklearn drop-in for XGBoost).

    Uses ``sklearn.ensemble.HistGradientBoostingRegressor`` which has the
    same algorithmic profile as XGBoost / LightGBM but ships with sklearn
    so we don't add a new dependency. Performance is within a few percent
    of XGBoost on most tabular regression tasks.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    train_x = np.asarray(train_x, dtype=np.float32)
    train_y = np.asarray(train_y, dtype=np.float32)
    query_x = np.asarray(query_x, dtype=np.float32)

    model = HistGradientBoostingRegressor(
        max_iter=max_iter,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
        early_stopping=False,
    )
    model.fit(train_x, train_y)
    return model.predict(query_x)


def fit_predict_rf(
    train_x: np.ndarray,
    train_y: np.ndarray,
    query_x: np.ndarray,
    *,
    n_estimators: int = 500,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Random Forest baseline returning (mean, ensemble std).

    The per-tree predictions form an empirical posterior, so the standard
    deviation across trees is a (somewhat optimistic) uncertainty estimate
    suitable as a calibration baseline.
    """
    from sklearn.ensemble import RandomForestRegressor

    train_x = np.asarray(train_x, dtype=np.float64)
    train_y = np.asarray(train_y, dtype=np.float64)
    query_x = np.asarray(query_x, dtype=np.float64)

    forest = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    forest.fit(train_x, train_y)
    per_tree = np.stack([tree.predict(query_x) for tree in forest.estimators_], axis=0)
    return per_tree.mean(axis=0), per_tree.std(axis=0)


def fit_predict_lightgbm(
    train_x: np.ndarray,
    train_y: np.ndarray,
    query_x: np.ndarray,
    *,
    n_estimators: int = 1500,
    learning_rate: float = 0.05,
    num_leaves: int = 127,
    min_data_in_leaf: int = 30,
    random_state: int = 42,
) -> np.ndarray:
    """LightGBM regressor — tuned for tabular SPT regression.

    Returns a point mean only. For interval-prediction, wrap with split
    conformal post-hoc.
    """
    import lightgbm as lgb

    train_x = np.asarray(train_x, dtype=np.float32)
    train_y = np.asarray(train_y, dtype=np.float32)
    query_x = np.asarray(query_x, dtype=np.float32)

    model = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_data_in_leaf=min_data_in_leaf,
        random_state=random_state,
        verbose=-1,
    )
    model.fit(train_x, train_y)
    return model.predict(query_x)


def fit_predict_xgboost(
    train_x: np.ndarray,
    train_y: np.ndarray,
    query_x: np.ndarray,
    *,
    n_estimators: int = 1500,
    learning_rate: float = 0.05,
    max_depth: int = 8,
    random_state: int = 42,
) -> np.ndarray:
    """XGBoost regressor — tuned for tabular SPT regression."""
    import xgboost as xgb

    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
        tree_method="hist",
        n_jobs=-1,
    )
    model.fit(np.asarray(train_x, dtype=np.float32),
              np.asarray(train_y, dtype=np.float32))
    return model.predict(np.asarray(query_x, dtype=np.float32))


def fit_predict_catboost(
    train_x: np.ndarray,
    train_y: np.ndarray,
    query_x: np.ndarray,
    *,
    iterations: int = 1500,
    learning_rate: float = 0.05,
    depth: int = 8,
    random_state: int = 42,
) -> np.ndarray:
    """CatBoost regressor — tuned for tabular SPT regression."""
    from catboost import CatBoostRegressor

    model = CatBoostRegressor(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        random_seed=random_state,
        verbose=False,
    )
    model.fit(np.asarray(train_x, dtype=np.float32),
              np.asarray(train_y, dtype=np.float32))
    return model.predict(np.asarray(query_x, dtype=np.float32))


def fit_predict_quantile_lightgbm(
    train_x: np.ndarray,
    train_y: np.ndarray,
    query_x: np.ndarray,
    *,
    quantiles: tuple[float, ...] = (0.025, 0.5, 0.975),
    n_estimators: int = 1500,
    learning_rate: float = 0.05,
    num_leaves: int = 127,
    random_state: int = 42,
) -> dict[float, np.ndarray]:
    """Quantile LightGBM — fits one model per quantile.

    Returns ``{q: prediction_array}`` so callers can build prediction
    intervals directly (no conformal post-hoc required).
    """
    import lightgbm as lgb

    train_x = np.asarray(train_x, dtype=np.float32)
    train_y = np.asarray(train_y, dtype=np.float32)
    query_x = np.asarray(query_x, dtype=np.float32)

    out: dict[float, np.ndarray] = {}
    for q in quantiles:
        model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=q,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            random_state=random_state,
            verbose=-1,
        )
        model.fit(train_x, train_y)
        out[q] = model.predict(query_x)
    return out


def fit_predict_local_kriging(
    train_x: np.ndarray,
    train_y: np.ndarray,
    query_x: np.ndarray,
    *,
    n_neighbours: int = 100,
    nu: float = 1.5,
    n_restarts: int = 0,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Moving-window local kriging.

    For each query point, fit an independent ordinary kriging model on
    its ``n_neighbours`` spatial nearest neighbours from the training
    set. Returns (mean, std). This is far slower than the subsampled
    global kriging baseline but does not throw away training data.
    """
    from scipy.spatial import cKDTree
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

    train_x = np.asarray(train_x, dtype=np.float64)
    train_y = np.asarray(train_y, dtype=np.float64)
    query_x = np.asarray(query_x, dtype=np.float64)

    tree = cKDTree(train_x)
    mean_out = np.zeros(len(query_x))
    std_out = np.zeros(len(query_x))
    base_kernel = (
        ConstantKernel(1.0, (0.1, 100.0))
        * Matern(length_scale=1.0, length_scale_bounds=(0.05, 100.0), nu=nu)
        + WhiteKernel(noise_level=1.0, noise_level_bounds=(0.01, 100.0))
    )
    for i, q in enumerate(query_x):
        _, idx = tree.query(q, k=min(n_neighbours, len(train_x)))
        gp = GaussianProcessRegressor(
            kernel=base_kernel,
            n_restarts_optimizer=n_restarts,
            normalize_y=True,
            random_state=random_state,
        )
        gp.fit(train_x[idx], train_y[idx])
        m, s = gp.predict(q[None, :], return_std=True)
        mean_out[i] = float(m[0])
        std_out[i] = float(s[0])
    return mean_out, std_out


__all__ = [
    "fit_predict_idw",
    "fit_predict_kriging",
    "fit_predict_rf",
    "fit_predict_hgb",
    "fit_predict_lightgbm",
    "fit_predict_xgboost",
    "fit_predict_catboost",
    "fit_predict_quantile_lightgbm",
    "fit_predict_local_kriging",
]
