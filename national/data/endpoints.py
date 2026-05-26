"""Profile-level engineering endpoints derived from per-row SPT data.

Geotechnical practice cares less about row-wise N-value RMSE than about
profile summaries:

* Depth to the first stiff layer (first N >= 30) — bearing-stratum
  candidate detection.
* Soft-layer thickness in the upper 10 m (where N < 5) — settlement &
  liquefaction screening proxy.
* Mean and minimum N in the upper 10 m — weak-layer detection.

This module groups the per-row Parquet ``borings_kanto_aist.parquet`` by
its borehole identity ``(latitude_deg, longitude_deg)`` (the dataset has no
explicit boring_id column) and computes those scalar targets.

Boreholes that never reach N >= 30 within the surveyed depth range are
treated as **right-censored**: their ``depth_to_first_N30`` target is set
to ``np.inf`` and downstream loss / metric code must explicitly handle the
censoring (e.g. by switching to binary AUC for the "has N >= 30 within
30 m" event rather than penalising the unbounded depth value).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd


# Standard depth thresholds for upper-soil aggregation.
UPPER_DEPTH_M = 10.0


@dataclass
class ProfileEndpoints:
    """Per-borehole scalar targets derived from a sorted SPT profile."""

    depth_to_first_N30: float  # inf if never reached
    has_N30_within_30m: int  # binary: 0/1
    soft_thickness_lt5_0_to_10m: float
    mean_N_upper_10m: float  # nan if no rows in 0..10m
    min_N_upper_10m: float  # nan if no rows in 0..10m
    n_rows: int  # depth coverage indicator
    max_depth_observed: float  # for censoring diagnostic


def reconstruct_profiles(
    df: pd.DataFrame,
    *,
    boring_key_cols: tuple[str, ...] = ("latitude_deg", "longitude_deg"),
    depth_col: str = "depth_from_surface",
) -> dict[tuple, pd.DataFrame]:
    """Group rows into per-borehole sorted profiles.

    The Parquet has no boring_id column; we use ``(lat, lon)`` as the
    natural borehole identity. Two distinct boreholes at identical
    coordinates would be merged here; this is a documented data limitation
    (see ``docs/research/lessons.md`` if/when we see warning about it).
    """
    profiles: dict[tuple, pd.DataFrame] = {}
    for key, g in df.groupby(list(boring_key_cols), sort=False):
        profiles[tuple(key) if isinstance(key, tuple) else (key,)] = (
            g.sort_values(depth_col).reset_index(drop=True)
        )
    return profiles


def depth_to_first_threshold(
    profile: pd.DataFrame,
    threshold: float,
    *,
    mode: str = "above",
    depth_col: str = "depth_from_surface",
    n_col: str = "n_value",
) -> float:
    """First depth at which N crosses the threshold (above or below).

    Returns ``np.inf`` if no row in the profile satisfies the condition
    (right-censored observation).
    """
    if mode == "above":
        mask = profile[n_col].values >= float(threshold)
    elif mode == "below":
        mask = profile[n_col].values < float(threshold)
    else:
        raise ValueError(f"mode must be 'above' or 'below', got {mode!r}")
    if not mask.any():
        return float("inf")
    first_idx = int(np.argmax(mask))
    return float(profile[depth_col].iloc[first_idx])


def soft_layer_thickness(
    profile: pd.DataFrame,
    *,
    threshold: float = 5.0,
    depth_min: float = 0.0,
    depth_max: float = UPPER_DEPTH_M,
    depth_col: str = "depth_from_surface",
    n_col: str = "n_value",
) -> float:
    """Total thickness of N < threshold layers within [depth_min, depth_max].

    Uses a trapezoidal heuristic: for each pair of adjacent rows whose
    midpoint falls in the depth band and whose endpoints are both below
    the threshold, contribute ``z_{i+1} - z_i`` to the thickness. Rows
    outside the band do not contribute. This is the closest robust
    estimate when the underlying SPT survey samples discrete depths
    rather than continuously logging.
    """
    z = profile[depth_col].values.astype(np.float64)
    n = profile[n_col].values.astype(np.float64)
    mask_in_band = (z >= depth_min) & (z <= depth_max) & (n < float(threshold))
    if mask_in_band.sum() < 1:
        return 0.0
    # Sort by depth and integrate via trapezoidal-like span
    order = np.argsort(z)
    z_s, mask_s = z[order], mask_in_band[order]
    thickness = 0.0
    for i in range(len(z_s) - 1):
        if mask_s[i] and mask_s[i + 1]:
            thickness += float(z_s[i + 1] - z_s[i])
    return thickness


def aggregate_in_depth_range(
    profile: pd.DataFrame,
    fn: Callable[[np.ndarray], float],
    *,
    z_min: float = 0.0,
    z_max: float = UPPER_DEPTH_M,
    depth_col: str = "depth_from_surface",
    n_col: str = "n_value",
) -> float:
    sub = profile[
        (profile[depth_col] >= float(z_min)) & (profile[depth_col] <= float(z_max))
    ]
    if sub.empty:
        return float("nan")
    return float(fn(sub[n_col].values))


def compute_profile_endpoints(
    profile: pd.DataFrame,
    *,
    depth_col: str = "depth_from_surface",
    n_col: str = "n_value",
) -> ProfileEndpoints:
    """Compute the full set of profile-level endpoints for a single boring."""
    d30 = depth_to_first_threshold(profile, 30.0, mode="above",
                                    depth_col=depth_col, n_col=n_col)
    soft = soft_layer_thickness(profile, threshold=5.0, depth_col=depth_col, n_col=n_col)
    mean_upper = aggregate_in_depth_range(
        profile, np.mean, z_min=0.0, z_max=UPPER_DEPTH_M,
        depth_col=depth_col, n_col=n_col,
    )
    min_upper = aggregate_in_depth_range(
        profile, np.min, z_min=0.0, z_max=UPPER_DEPTH_M,
        depth_col=depth_col, n_col=n_col,
    )
    max_depth = float(profile[depth_col].max()) if len(profile) else 0.0
    return ProfileEndpoints(
        depth_to_first_N30=d30,
        has_N30_within_30m=int(d30 <= 30.0),
        soft_thickness_lt5_0_to_10m=soft,
        mean_N_upper_10m=mean_upper,
        min_N_upper_10m=min_upper,
        n_rows=int(len(profile)),
        max_depth_observed=max_depth,
    )


def build_endpoint_dataframe(
    df: pd.DataFrame,
    *,
    boring_key_cols: tuple[str, ...] = ("latitude_deg", "longitude_deg"),
    extra_static_cols: tuple[str, ...] = (
        "regime_code", "river_distance_km", "coast_distance_km",
        "absolute_elevation",
    ),
) -> pd.DataFrame:
    """Aggregate per-row SPT into a per-borehole endpoint DataFrame.

    ``extra_static_cols`` are passed through using the first row's value for
    each borehole — these are *position*-level covariates (lat/lon-keyed),
    so they are constant within a borehole.
    """
    profiles = reconstruct_profiles(df, boring_key_cols=boring_key_cols)
    rows = []
    for key, profile in profiles.items():
        ep = compute_profile_endpoints(profile)
        row = dict(zip(boring_key_cols, key))
        row.update({
            "depth_to_first_N30": ep.depth_to_first_N30,
            "has_N30_within_30m": ep.has_N30_within_30m,
            "soft_thickness_lt5_0_to_10m": ep.soft_thickness_lt5_0_to_10m,
            "mean_N_upper_10m": ep.mean_N_upper_10m,
            "min_N_upper_10m": ep.min_N_upper_10m,
            "n_rows_in_boring": ep.n_rows,
            "max_depth_observed": ep.max_depth_observed,
        })
        for col in extra_static_cols:
            if col in profile.columns:
                row[col] = profile[col].iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)


__all__ = [
    "ProfileEndpoints",
    "UPPER_DEPTH_M",
    "reconstruct_profiles",
    "depth_to_first_threshold",
    "soft_layer_thickness",
    "aggregate_in_depth_range",
    "compute_profile_endpoints",
    "build_endpoint_dataframe",
]
