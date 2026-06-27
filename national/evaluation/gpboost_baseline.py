"""GPBoost baseline: tree-boosted mean + Vecchia-approximated GP residual.

GPBoost was the strongest *contiguous* (out-of-network) predictor in the Kanto
study (Paper 1, RMSE 10.744) because the Vecchia GP residual over (lat, lon)
absorbs geographic-block structure the tabular tree mean cannot. This module
extracts the fit/predict call so both the Phase-N reproduction script
(``scripts/run_gpboost_baseline_phase_n.py``) and the national leave-region-out
runner (``national/evaluation/leave_region_out_runner.py``) share one
implementation.

The GP residual is fitted on ``train_coords`` (degrees lat/lon); the tree mean
is fitted on the tabular ``train_x`` feature stack. Prediction reuses both.
"""

from __future__ import annotations

import numpy as np

# Hyperparameters matching scripts/run_gpboost_baseline_phase_n.py so the
# extracted function reproduces the paper's GPBoost row exactly.
DEFAULT_TREE_PARAMS: dict[str, object] = {
    "objective": "regression_l2",
    "max_depth": 7,
    "num_leaves": 127,
    "min_data_in_leaf": 30,
    "verbose": -1,
}


def fit_predict_gpboost(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_coords: np.ndarray,
    query_x: np.ndarray,
    query_coords: np.ndarray,
    *,
    cov_function: str = "exponential",
    gp_approx: str = "vecchia",
    num_neighbors: int = 20,
    n_boost_iter: int = 300,
    learning_rate: float = 0.05,
    tree_params: dict | None = None,
    predict_var: bool = False,
):
    """Fit GPBoost (tree mean + Vecchia GP residual) and predict at queries.

    Args:
        train_x: ``(N, F)`` tabular features for the tree mean.
        train_y: ``(N,)`` targets.
        train_coords: ``(N, 2)`` spatial coordinates (lat, lon in degrees) for
            the GP residual.
        query_x / query_coords: ``(M, F)`` / ``(M, 2)`` query features/coords.
        cov_function / gp_approx / num_neighbors: GP residual settings (Vecchia
            neighbourhood size controls the spatial approximation fidelity).
        n_boost_iter / learning_rate / tree_params: tree-mean boosting settings.
        predict_var: if True, also return the predictive standard deviation
            ``sqrt(response_var)``.

    Returns:
        ``mean`` of shape ``(M,)`` if ``predict_var`` is False, else
        ``(mean, std)``.
    """
    import gpboost as gpb

    train_x = np.asarray(train_x, dtype=np.float64)
    train_y = np.asarray(train_y, dtype=np.float64)
    train_coords = np.asarray(train_coords, dtype=np.float64)
    query_x = np.asarray(query_x, dtype=np.float64)
    query_coords = np.asarray(query_coords, dtype=np.float64)

    params = dict(DEFAULT_TREE_PARAMS)
    if tree_params:
        params.update(tree_params)
    params["learning_rate"] = learning_rate

    gp_model = gpb.GPModel(
        gp_coords=train_coords,
        cov_function=cov_function,
        gp_approx=gp_approx,
        num_neighbors=num_neighbors,
    )
    data_train = gpb.Dataset(train_x, train_y)
    booster = gpb.train(
        params=params,
        train_set=data_train,
        num_boost_round=n_boost_iter,
        gp_model=gp_model,
    )
    preds = booster.predict(
        data=query_x,
        gp_coords_pred=query_coords,
        predict_var=predict_var,
    )
    mean = np.asarray(preds["response_mean"], dtype=np.float64)
    if predict_var:
        std = np.sqrt(np.asarray(preds["response_var"], dtype=np.float64))
        return mean, std
    return mean


__all__ = ["DEFAULT_TREE_PARAMS", "fit_predict_gpboost"]
