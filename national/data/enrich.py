"""End-to-end enrichment: raw boring CSV -> model-ready Parquet.

Joins the existing 2.7 M-row N-value CSV (per-row depth measurements,
175 k unique locations) with the derived covariate columns the
foundation model expects:

- ``river_distance_km``  -- nearest Class-1 river polyline (MLIT W05).
- ``coast_distance_km``  -- nearest coastline polyline (MLIT C23).
- ``absolute_elevation`` -- ``mouth_elevation - spt_start_depth``.
- ``depth_from_surface`` -- alias for ``spt_start_depth``.
- ``regime_code``        -- categorical regime (AIST geology code mapped
  through :class:`national.tiling.regime_classifier.Regime`). Defaults
  to ``Regime.UNKNOWN`` when the geology cache is empty.

Per-unique-location enrichment is computed once and joined back to the
full row set, which is ~15× faster than per-row geometry queries on a
dataset this size.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from national.data.derived.distances import (
    compute_distance_to_lines,
    load_coastlines_from_mlit_dir,
)
from national.tiling.regime_classifier import Regime

LOG = logging.getLogger("national.data.enrich")


@dataclass
class EnrichmentSpec:
    """Inputs and outputs for one enrichment run."""

    borings_csv: Path
    output_parquet: Path
    river_geojson: Path | None = None
    coast_dir: Path | None = None
    aist_geology_cache: Path | None = None
    target_column: str = "n_value"
    bbox: tuple[float, float, float, float] | None = None  # (lat_min, lon_min, lat_max, lon_max)


def enrich(spec: EnrichmentSpec) -> Path:
    """Run the full enrichment and write a single Parquet file."""
    LOG.info("Reading borings from %s", spec.borings_csv)
    df = pd.read_csv(spec.borings_csv)
    df = _normalize_input_columns(df)
    if spec.bbox is not None:
        df = _filter_bbox(df, spec.bbox)
        LOG.info("After bbox filter: %d rows", len(df))

    # 1. Per-row derivations (cheap).
    df["depth_from_surface"] = df["spt_start_depth"]
    df["absolute_elevation"] = df["mouth_elevation"] - df["spt_start_depth"]

    # 2. Per-location derivations (the expensive ones).
    unique_locs = (
        df[["latitude_deg", "longitude_deg"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    LOG.info("%d unique boring locations", len(unique_locs))

    if spec.river_geojson and spec.river_geojson.exists():
        LOG.info("Computing river_distance_km from %s", spec.river_geojson)
        unique_locs["river_distance_km"] = _distance_to_lines_from_geojson(
            unique_locs, spec.river_geojson
        )
    else:
        LOG.warning("river_geojson missing; setting river_distance_km = nan")
        unique_locs["river_distance_km"] = np.float32("nan")

    if spec.coast_dir and spec.coast_dir.exists():
        LOG.info("Computing coast_distance_km from %s", spec.coast_dir)
        unique_locs["coast_distance_km"] = _distance_to_coast(unique_locs, spec.coast_dir)
    else:
        LOG.warning("coast_dir missing; setting coast_distance_km = nan")
        unique_locs["coast_distance_km"] = np.float32("nan")

    unique_locs["regime_code"] = _regime_from_geology(unique_locs, spec.aist_geology_cache)

    # 3. Join back to the full row set.
    df = df.merge(unique_locs, on=["latitude_deg", "longitude_deg"], how="left")
    df = _final_schema(df, target_column=spec.target_column)

    spec.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(spec.output_parquet, index=False)
    LOG.info(
        "Wrote %d rows to %s (mean N=%.2f, std=%.2f)",
        len(df),
        spec.output_parquet,
        float(df[spec.target_column].mean()),
        float(df[spec.target_column].std()),
    )
    return spec.output_parquet


# ---------------------------------------------------------------- internals
_REQUIRED = ("longitude_deg", "latitude_deg", "spt_start_depth", "n_value")


def _normalize_input_columns(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise KeyError(f"Boring CSV missing required columns: {missing!r}")
    if "mouth_elevation" not in df.columns:
        df["mouth_elevation"] = 0.0
    df = df.copy()
    # Trim rows with bad coords or missing elevations (KuniJiban has a small
    # number of NaN / -999 / out-of-range entries). Without these filters
    # downstream BoringDataset -> Cholesky decomposition silently produces
    # an all-NaN kernel matrix on any batch that includes the bad rows.
    df = df.dropna(
        subset=[
            "latitude_deg",
            "longitude_deg",
            "n_value",
            "spt_start_depth",
            "mouth_elevation",
        ]
    )
    df = df[(df["latitude_deg"] > 20) & (df["latitude_deg"] < 50)]
    df = df[(df["longitude_deg"] > 120) & (df["longitude_deg"] < 150)]
    df = df[(df["n_value"] >= 0) & (df["n_value"] <= 100)]
    df = df[(df["mouth_elevation"] > -1000) & (df["mouth_elevation"] < 5000)]
    return df.reset_index(drop=True)


def _filter_bbox(
    df: pd.DataFrame, bbox: tuple[float, float, float, float]
) -> pd.DataFrame:
    lat_min, lon_min, lat_max, lon_max = bbox
    mask = (
        (df["latitude_deg"] >= lat_min)
        & (df["latitude_deg"] <= lat_max)
        & (df["longitude_deg"] >= lon_min)
        & (df["longitude_deg"] <= lon_max)
    )
    return df.loc[mask].reset_index(drop=True)


def _distance_to_lines_from_geojson(
    points_df: pd.DataFrame, geojson_path: Path
) -> np.ndarray:
    import geopandas as gpd

    LOG.info("Loading lines GeoJSON: %s", geojson_path)
    lines = gpd.read_file(geojson_path)
    return compute_distance_to_lines(points_df, lines)


def _distance_to_coast(points_df: pd.DataFrame, coast_dir: Path) -> np.ndarray:
    coast = load_coastlines_from_mlit_dir(coast_dir)
    return compute_distance_to_lines(points_df, coast)


def _regime_from_geology(
    points_df: pd.DataFrame, cache_path: Path | None
) -> np.ndarray:
    """Map per-location AIST legend rows to ``Regime`` ints.

    The cache format produced by
    ``national.data.download.aist_geology.fetch_codes_for_borings`` is a
    Parquet with columns ``lat, lon, symbol, formation_age_ja, group_ja,
    lithology_ja``. We resolve each row via
    :func:`national.data.derived.lithology.regime_from_legend` (rules
    documented at the call site).

    Returns ``Regime.UNKNOWN`` for every boring when the cache is
    missing or has no entry at the location.
    """
    from national.data.derived.lithology import regime_from_legend

    n = len(points_df)
    if cache_path is None or not cache_path.exists():
        return np.full((n,), int(Regime.UNKNOWN), dtype=np.int16)

    LOG.info("Joining AIST geology cache: %s", cache_path)
    cache = pd.read_parquet(cache_path)

    required = {"lat", "lon", "symbol", "formation_age_ja", "group_ja", "lithology_ja"}
    missing = required - set(cache.columns)
    if missing:
        LOG.warning(
            "AIST cache %s is missing columns %s; falling back to UNKNOWN.",
            cache_path,
            sorted(missing),
        )
        return np.full((n,), int(Regime.UNKNOWN), dtype=np.int16)

    # Match on 4-decimal rounded coordinates (cache key precision).
    lat_r = points_df["latitude_deg"].round(4)
    lon_r = points_df["longitude_deg"].round(4)
    cache_r = cache.copy()
    cache_r["lat"] = cache_r["lat"].round(4)
    cache_r["lon"] = cache_r["lon"].round(4)
    joined = pd.DataFrame({"lat": lat_r, "lon": lon_r}).merge(
        cache_r,
        on=["lat", "lon"],
        how="left",
    )
    regime = joined.apply(
        lambda row: int(
            regime_from_legend(
                row.get("symbol"),
                row.get("formation_age_ja"),
                row.get("group_ja"),
                row.get("lithology_ja"),
            )
        ),
        axis=1,
    ).to_numpy(dtype=np.int16)
    n_matched = int((regime != int(Regime.UNKNOWN)).sum())
    LOG.info(
        "Resolved %d / %d boring locations to non-UNKNOWN regimes via AIST.",
        n_matched,
        n,
    )
    return regime


def _final_schema(df: pd.DataFrame, *, target_column: str) -> pd.DataFrame:
    cols = [
        "latitude_deg",
        "longitude_deg",
        "depth_from_surface",
        "absolute_elevation",
        target_column,
        "river_distance_km",
        "coast_distance_km",
        "regime_code",
    ]
    out = df[cols].copy()
    out["latitude_deg"] = out["latitude_deg"].astype(np.float32)
    out["longitude_deg"] = out["longitude_deg"].astype(np.float32)
    out["depth_from_surface"] = out["depth_from_surface"].astype(np.float32)
    out["absolute_elevation"] = out["absolute_elevation"].astype(np.float32)
    out[target_column] = out[target_column].astype(np.float32)
    out["river_distance_km"] = out["river_distance_km"].astype(np.float32)
    out["coast_distance_km"] = out["coast_distance_km"].astype(np.float32)
    out["regime_code"] = out["regime_code"].astype(np.int16)
    return out


__all__ = ["EnrichmentSpec", "enrich"]
