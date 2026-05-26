"""Parquet-backed PyTorch Dataset over boring records joined with covariates.

The dataset is the unique source of training samples for the SVGP head. It
loads a Parquet table of boring measurements (one row per (location, depth)
pair) and presents a tensor view in the row layout expected by the encoder:

    x = [lat, lon, depth, *feature_columns, *registry_sampled_covariates]

The order matters: ``ResMLPEncoder`` hard-codes the first three columns to
lat/lon/depth, so it MUST be preserved. The remaining columns are arbitrary
continuous covariates that the registry has standardized.

Memory model: the full table is loaded eagerly into a single ``numpy.ndarray``
of float32, plus a small int16 regime vector. This is intentional -- 175 k
borings * ~20 depths * 30 features = ~400 MB, which fits comfortably in RAM
on every target machine (laptop, Miyabi-G node). Lazy iteration would only
help if the data outgrew memory, which it does not.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from national.data.covariate_registry import CovariateRegistry


_REQUIRED_COLUMNS = ("latitude_deg", "longitude_deg", "depth_from_surface", "n_value")
_UNKNOWN_REGIME = 7  # matches Regime.UNKNOWN

TargetTransform = Literal["none", "log1p"]


def apply_target_transform(y: np.ndarray, transform: TargetTransform) -> np.ndarray:
    """Forward transform for the regression target."""
    if transform == "none":
        return y
    if transform == "log1p":
        # SPT N-value is >=0 with a heavy right tail (0 to 100). log1p compresses
        # the dynamic range and matches the Gaussian likelihood much better than
        # raw N. clip(min=0) defends against the few negative-or-NaN rows that
        # the upstream filters miss.
        return np.log1p(np.maximum(y, 0.0))
    raise ValueError(f"Unknown target transform: {transform!r}")


def invert_target_transform(y: np.ndarray, transform: TargetTransform) -> np.ndarray:
    """Inverse of :func:`apply_target_transform` for the *mean* prediction.

    Note: this is the inverse for the *point* prediction, not for the
    Gaussian moments. Callers that need calibrated std in original units
    should use :func:`invert_target_transform_moments`.
    """
    if transform == "none":
        return y
    if transform == "log1p":
        return np.expm1(y)
    raise ValueError(f"Unknown target transform: {transform!r}")


def invert_target_transform_moments(
    mean: np.ndarray,
    std: np.ndarray,
    transform: TargetTransform,
    *,
    estimator: Literal["mean", "median"] = "median",
) -> tuple[np.ndarray, np.ndarray]:
    """Invert a Gaussian in log space to the original units.

    Two ``estimator`` modes are supported for log-style transforms:

    - ``"mean"`` (the *unbiased* expectation under the log-normal model):
      ``E[exp(Y_log) - 1] = exp(mu + sigma^2 / 2) - 1``. This is the right
      choice when downstream consumers want a calibrated expected value
      with full log-normal moments.

    - ``"median"`` (default): ``Median[exp(Y_log) - 1] = exp(mu) - 1``.
      Avoids the ``+ sigma^2 / 2`` bias that inflates predictions when
      the model is uncertain, which is what causes log1p training to
      have a *lower* loss but a *higher* RMSE than the no-transform run.
      This is the right choice for RMSE / MAE reporting on raw N-value.

    Std reporting in original units uses the standard log-normal
    standard deviation regardless of estimator.
    """
    if transform == "none":
        return mean, std
    if transform == "log1p":
        var = std * std
        if estimator == "mean":
            mean_orig = np.exp(mean + 0.5 * var) - 1.0
        elif estimator == "median":
            mean_orig = np.exp(mean) - 1.0
        else:
            raise ValueError(f"Unknown estimator: {estimator!r}")
        # std uses log-normal moments around exp(mu + sigma^2 / 2) regardless
        # of which point estimator we report.
        std_orig = np.sqrt(np.maximum(np.exp(var) - 1.0, 0.0)) * np.exp(mean + 0.5 * var)
        return mean_orig, std_orig
    raise ValueError(f"Unknown target transform: {transform!r}")


class BoringDataset(Dataset[dict]):
    """Parquet-backed boring dataset.

    Args:
        parquet_path: Path to a Parquet file with at minimum the columns
            ``[latitude_deg, longitude_deg, depth_from_surface, n_value]``.
            ``regime_code`` is optional and defaults to ``Regime.UNKNOWN``.
        feature_columns: ordered list of *pre-computed* continuous covariate
            column names. These columns must already exist in the Parquet and
            are placed immediately after the ``lat/lon/depth`` triplet in the
            feature vector.
        registry: optional covariate registry used to sample additional
            continuous covariates at construction time. Sampled values are
            cached so ``__getitem__`` stays O(1).
        depth_scale_m: depth normalization factor (kept on the dataset for
            downstream code; the encoder uses ``FoundationSpec.depth_scale_m``).
        target_column: name of the regression target (default ``"n_value"``).
        standardize_target: if True (default), subtract the empirical mean and
            divide by the empirical std on init and return the standardized y
            from ``__getitem__``. The mean/std are exposed via properties so
            the trainer can call ``model.set_target_stats(...)``.
    """

    def __init__(
        self,
        parquet_path: Path,
        *,
        feature_columns: list[str],
        registry: "CovariateRegistry | None" = None,
        depth_scale_m: float = 30.0,
        target_column: str = "n_value",
        standardize_target: bool = True,
        regime_one_hot: bool = False,
        regime_dim: int = 8,
        target_transform: TargetTransform = "none",
        baseline_pred_npy: Path | None = None,
        baseline_idx_npy: Path | None = None,
        baseline_pred_test_npy: Path | None = None,
        baseline_idx_test_npy: Path | None = None,
    ) -> None:
        path = Path(parquet_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Boring parquet not found at {path}. "
                f"Run the data ingestion pipeline first (see docs/architecture.md)."
            )

        import pandas as pd  # heavy; imported lazily

        df = pd.read_parquet(path)
        missing_required = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
        if target_column != "n_value" and target_column not in df.columns:
            missing_required.append(target_column)
        missing_features = [c for c in feature_columns if c not in df.columns]
        if missing_required or missing_features:
            raise KeyError(
                f"Parquet at {path} is missing columns: required={missing_required!r}, "
                f"features={missing_features!r}"
            )

        # Core columns -> float32 numpy.
        lat = df["latitude_deg"].to_numpy(dtype=np.float32)
        lon = df["longitude_deg"].to_numpy(dtype=np.float32)
        depth = df["depth_from_surface"].to_numpy(dtype=np.float32)
        y = df[target_column].to_numpy(dtype=np.float32)
        # Preserve the original (un-residualised) target before any hybrid
        # subtraction. `_y_raw` is the diagnostic ground truth in raw N
        # units, used by the smoke trainer to compute K-fold RMSE / MAE.
        y_original = y.copy()

        # Hybrid residual target: y_residual = y - baseline_pred.
        # The training side (baseline_pred_npy) is the spatial-OOB CatBoost /
        # LightGBM prediction (no row sharing the same secondary mesh has
        # been seen by the teacher when predicting row i). The test side
        # (baseline_pred_test_npy) is the regular outer-training-fold
        # CatBoost prediction. We union them into a single aligned array so
        # the residual `y - baseline_pred` is defined for every row that
        # any downstream call cares about. Rows not covered by either side
        # retain `y` unchanged with `baseline_pred = 0`; these are typically
        # rows the caller should not be evaluating against (the smoke
        # trainer asserts shape alignment at inference time).
        baseline_aligned = np.zeros(len(y), dtype=np.float32)
        baseline_in_use = np.zeros(len(y), dtype=bool)
        any_baseline_loaded = False
        for pred_path, idx_path, label in (
            (baseline_pred_npy, baseline_idx_npy, "train"),
            (baseline_pred_test_npy, baseline_idx_test_npy, "test"),
        ):
            if pred_path is None and idx_path is None:
                continue
            if pred_path is None or idx_path is None:
                raise ValueError(
                    f"baseline_pred_{label} and baseline_idx_{label} must be "
                    "provided together; got pred="
                    f"{pred_path!r}, idx={idx_path!r}"
                )
            pred_raw = np.load(pred_path).astype(np.float32)
            idx_raw = np.load(idx_path).astype(np.int64)
            if pred_raw.shape[0] != idx_raw.shape[0]:
                raise ValueError(
                    f"baseline_pred_{label} ({pred_raw.shape[0]}) and "
                    f"baseline_idx_{label} ({idx_raw.shape[0]}) length mismatch"
                )
            if baseline_in_use[idx_raw].any():
                overlap = int(baseline_in_use[idx_raw].sum())
                raise ValueError(
                    f"baseline_idx_{label} overlaps with previously-loaded "
                    f"baseline indices in {overlap} rows; train and test "
                    "splits must be disjoint."
                )
            baseline_aligned[idx_raw] = pred_raw
            baseline_in_use[idx_raw] = True
            any_baseline_loaded = True
        if any_baseline_loaded:
            self._baseline_pred_per_row = baseline_aligned
            self._baseline_in_use_per_row = baseline_in_use
            y = (y - baseline_aligned).astype(np.float32)
        else:
            self._baseline_pred_per_row = None
            self._baseline_in_use_per_row = None

        # Regime: int16 if present, default to UNKNOWN otherwise.
        if "regime_code" in df.columns:
            regime = df["regime_code"].to_numpy(dtype=np.int16)
        else:
            regime = np.full((len(df),), _UNKNOWN_REGIME, dtype=np.int16)

        feature_cols_arr = (
            np.stack(
                [df[c].to_numpy(dtype=np.float32) for c in feature_columns], axis=1
            )
            if feature_columns
            else np.empty((len(df), 0), dtype=np.float32)
        )

        # On-the-fly covariate sampling via the registry. We sample ONCE at
        # construction time and cache, to avoid I/O cost in the training loop.
        if registry is not None and registry.continuous_names:
            with torch.no_grad():
                sampled = registry.stack_continuous(
                    torch.from_numpy(lat),
                    torch.from_numpy(lon),
                    torch.from_numpy(depth),
                ).cpu().numpy().astype(np.float32, copy=False)
            registry_names = list(registry.continuous_names)
        else:
            sampled = np.empty((len(df), 0), dtype=np.float32)
            registry_names = []

        # Final feature matrix: [lat, lon, depth, *features, *registry_features,
        # *regime_one_hot].
        pieces = [
            lat[:, None],
            lon[:, None],
            depth[:, None],
            feature_cols_arr,
            sampled,
        ]
        regime_hot_names: list[str] = []
        if regime_one_hot:
            if regime_dim < 1:
                raise ValueError(f"regime_dim must be positive, got {regime_dim}")
            # Clip ensures the index is in-range; UNKNOWN (7) falls in the
            # canonical 8-way taxonomy. Negative codes (rare data errors) map
            # to slot 0 which is harmless.
            clipped = np.clip(regime.astype(np.int64), 0, regime_dim - 1)
            one_hot = np.eye(regime_dim, dtype=np.float32)[clipped]
            pieces.append(one_hot)
            regime_hot_names = [f"regime_oh_{i}" for i in range(regime_dim)]
        self._x = np.concatenate(pieces, axis=1)
        # Force contiguous so torch.from_numpy is zero-copy.
        self._x = np.ascontiguousarray(self._x)
        # `_y_raw` is the *original* target in raw N units, NOT the
        # residual. The smoke trainer's K-fold RMSE uses this to compare
        # against the hybrid prediction (residual mean + baseline mean).
        self._y_raw = y_original
        self._regime = regime
        self._target_transform: TargetTransform = target_transform

        # Apply the (optional) forward transform before standardizing. log1p
        # converts the SPT N distribution from heavy-right-tailed to roughly
        # log-normal, which is far better matched to a Gaussian likelihood.
        y_transformed = apply_target_transform(y, target_transform).astype(np.float32)

        if standardize_target:
            self._target_mean = float(y_transformed.mean())
            self._target_std = float(y_transformed.std()) + 1e-6
        else:
            self._target_mean = 0.0
            self._target_std = 1.0
        self._y = ((y_transformed - self._target_mean) / self._target_std).astype(np.float32)

        self._feature_names: list[str] = (
            ["latitude_deg", "longitude_deg", "depth_from_surface"]
            + list(feature_columns)
            + registry_names
            + regime_hot_names
        )
        self._depth_scale_m = float(depth_scale_m)
        self._n_rows = int(self._x.shape[0])

    # ---- introspection ------------------------------------------------------
    @property
    def target_mean(self) -> float:
        return self._target_mean

    @property
    def target_std(self) -> float:
        return self._target_std

    @property
    def n_features(self) -> int:
        return int(self._x.shape[1])

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)

    @property
    def depth_scale_m(self) -> float:
        return self._depth_scale_m

    @property
    def target_transform(self) -> TargetTransform:
        return self._target_transform

    @property
    def baseline_pred_per_row(self) -> np.ndarray | None:
        """Baseline mean prediction per row (CatBoost / LightGBM teacher).

        ``None`` for non-hybrid runs. For hybrid runs, this is in the
        *original* (unstandardised) target unit; recombine with
        ``mu_total = mu_residual_inverse_transformed + baseline_pred_per_row``.
        """
        return self._baseline_pred_per_row

    @property
    def baseline_in_use_per_row(self) -> np.ndarray | None:
        """Boolean mask of rows for which a baseline prediction was supplied."""
        return self._baseline_in_use_per_row

    # ---- Dataset protocol ---------------------------------------------------
    def __len__(self) -> int:
        return self._n_rows

    def __getitem__(self, index: int) -> dict:
        return {
            "x": torch.from_numpy(self._x[index]),
            "y": torch.tensor(self._y[index]),
            "regime": torch.tensor(int(self._regime[index])),
        }


__all__ = [
    "BoringDataset",
    "TargetTransform",
    "apply_target_transform",
    "invert_target_transform",
    "invert_target_transform_moments",
]
