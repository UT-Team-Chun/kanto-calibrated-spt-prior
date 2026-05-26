"""Vectorized distance-to-feature helpers for boring enrichment.

The existing PoC ``river_proximity.distance_calculator`` does per-point
spatial queries; that approach is O(N) Python overhead for ~175 k
boring locations and takes hours. The helpers in this module work in
batch: they reproject the points and the target geometry to UTM 54N
once, then use ``geopandas.sjoin_nearest`` (R-tree backed) so the full
all-Japan sweep finishes in minutes.

Two distinct workflows are supported:

- ``compute_distance_to_lines`` -- nearest distance from each point to
  the closest polyline in a vector dataset (used by coast / river
  enrichment).
- ``load_coastlines_from_mlit_dir`` -- read the per-prefecture MLIT C23
  shapefile tree produced by ``national.data.download.mlit_ksj`` and
  return a single merged GeoDataFrame ready for batch distance queries.

Both reuse the canonical UTM 54N CRS (``EPSG:32654``) which matches the
existing river-proximity implementation, so legacy outputs remain
comparable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

LOG = logging.getLogger("national.data.derived.distances")

UTM_CRS = "EPSG:32654"
WGS84 = "EPSG:4326"
JGD2011 = "EPSG:6668"


def compute_distance_to_lines(
    points_df: pd.DataFrame,
    lines_gdf,
    *,
    lat_column: str = "latitude_deg",
    lon_column: str = "longitude_deg",
) -> np.ndarray:
    """Batch nearest-distance from each point to ``lines_gdf``.

    Args:
        points_df: ordinary DataFrame with ``lat_column`` and ``lon_column``
            in EPSG:4326 degrees.
        lines_gdf: ``geopandas.GeoDataFrame`` of polylines. Any CRS is
            accepted; the function reprojects to UTM 54N internally.

    Returns:
        ``numpy.ndarray`` of shape ``(len(points_df),)`` with distances
        in kilometers. Order matches ``points_df.index``.
    """
    import geopandas as gpd

    if lines_gdf.empty:
        raise ValueError("lines_gdf is empty.")
    lats = points_df[lat_column].to_numpy(dtype=np.float64)
    lons = points_df[lon_column].to_numpy(dtype=np.float64)

    points = gpd.GeoDataFrame(
        {"_idx": np.arange(len(points_df), dtype=np.int64)},
        geometry=gpd.points_from_xy(lons, lats),
        crs=WGS84,
    )
    points = points.to_crs(UTM_CRS)
    lines = (
        lines_gdf.to_crs(UTM_CRS)
        if lines_gdf.crs is not None and str(lines_gdf.crs) != UTM_CRS
        else lines_gdf
    )
    # Keep only the geometry column for the sjoin; some MLIT shapefiles have
    # dozens of attribute columns we don't need.
    # Force the same UTM 54N CRS so sjoin_nearest doesn't warn about mixed CRS.
    lines_geom = gpd.GeoDataFrame(geometry=lines.geometry, crs=UTM_CRS)

    LOG.info(
        "sjoin_nearest: %d points x %d line features (UTM 54N)",
        len(points),
        len(lines_geom),
    )
    joined = gpd.sjoin_nearest(points, lines_geom, distance_col="_dist_m")
    # sjoin_nearest may emit multiple rows per point on ties; collapse.
    joined = joined.sort_values("_dist_m").drop_duplicates(subset="_idx", keep="first")
    joined = joined.sort_values("_idx").reset_index(drop=True)
    return (joined["_dist_m"].to_numpy() / 1000.0).astype(np.float32)


def load_coastlines_from_mlit_dir(
    coast_dir: Path,
    *,
    prefectures: Iterable[str] | None = None,
):
    """Read every C23-XX_<PP>_GML/* shapefile under ``coast_dir`` into one GDF.

    Args:
        coast_dir: directory containing the per-prefecture C23 zips
            (or their unzipped contents). Both layouts are tolerated.
        prefectures: optional iterable of 2-digit prefecture codes to
            include. ``None`` means every code found on disk.

    Returns:
        ``geopandas.GeoDataFrame`` of coastline polylines in the source
        CRS (typically EPSG:6668 for MLIT 2006).
    """
    import geopandas as gpd
    import zipfile

    if not coast_dir.exists():
        raise FileNotFoundError(f"coast_dir not found: {coast_dir}")

    wanted = set(prefectures) if prefectures is not None else None
    gdfs = []
    for entry in sorted(coast_dir.iterdir()):
        if not entry.is_file() or entry.suffix.lower() != ".zip":
            continue
        # Filename pattern: C23-06_<PP>_GML.zip
        stem = entry.stem  # e.g. C23-06_13_GML
        parts = stem.split("_")
        if len(parts) < 3:
            continue
        pref = parts[1]
        if wanted is not None and pref not in wanted:
            continue
        # Read the .shp directly from the zip (geopandas + fiona supports it).
        # Locate the .shp member inside the archive.
        with zipfile.ZipFile(entry) as zf:
            shp_members = [n for n in zf.namelist() if n.lower().endswith(".shp")]
            if not shp_members:
                LOG.warning("No .shp inside %s; skipping.", entry)
                continue
            # fiona/geopandas can read /vsizip/ paths.
            url = f"zip://{entry}!{shp_members[0]}"
            try:
                gdf = gpd.read_file(url)
            except Exception as exc:  # noqa: BLE001
                LOG.warning("Failed to read %s: %s", url, exc)
                continue
            if gdf.empty:
                continue
            if gdf.crs is None:
                # MLIT C23-2006 shapefiles ship without a .prj sidecar inside
                # the zip; bounds in 120-154 lon / 20-46 lat make the datum
                # unambiguous. Assign EPSG:4326 (WGS84) so downstream
                # reprojection works.
                gdf = gdf.set_crs("EPSG:4326")
            gdfs.append(gdf)
            LOG.info("loaded %d features from %s", len(gdf), entry.name)

    if not gdfs:
        raise ValueError(
            f"No coastline features loaded from {coast_dir}. "
            f"Download via `python -m scripts.fetch_covariates` first."
        )

    # All gdfs now have CRS=EPSG:4326 (or whatever the source declared).
    target_crs = gdfs[0].crs
    aligned = [g if g.crs == target_crs else g.to_crs(target_crs) for g in gdfs]
    merged = pd.concat(aligned, ignore_index=True)
    return gpd.GeoDataFrame(merged, crs=target_crs)


__all__ = [
    "compute_distance_to_lines",
    "load_coastlines_from_mlit_dir",
    "UTM_CRS",
    "WGS84",
    "JGD2011",
]
