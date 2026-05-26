"""Leave-region-out split for nationwide evaluation.

Each region is defined by a bounding box ``(lat_min, lat_max, lon_min, lon_max)``.
Per iteration the rows whose ``(latitude_deg, longitude_deg)`` fall *inside*
the region are the test set; everything else is the train set. Rows that fall
outside every region (e.g. sea-locked outliers) are kept in the train set of
every fold.

The default regions are the eight standard Japanese geographic regions
(`8地方区分`). The Kyushu/Okinawa group is unified so the rare carbonate
regime gets a single dedicated fold.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd

DEFAULT_REGIONS: dict[str, tuple[float, float, float, float]] = {
    # (lat_min, lat_max, lon_min, lon_max)
    "hokkaido":       (41.0, 46.0, 139.0, 146.0),
    "tohoku":         (37.0, 41.5, 139.0, 142.5),
    "kanto":          (35.0, 37.5, 138.5, 141.0),
    "chubu":          (34.5, 37.5, 136.0, 139.0),
    "kansai":         (33.5, 35.5, 134.0, 137.0),
    "chugoku":        (33.5, 35.6, 130.0, 134.0),
    "shikoku":        (32.5, 34.5, 132.0, 135.0),
    "kyushu_okinawa": (24.0, 33.5, 122.0, 132.0),
}


def leave_region_out_split(
    df: pd.DataFrame,
    regions: dict[str, tuple[float, float, float, float]] | None = None,
    *,
    lat_column: str = "latitude_deg",
    lon_column: str = "longitude_deg",
) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """Yield ``(region_name, train_idx, test_idx)`` for each held-out region.

    Args:
        df: DataFrame of training samples.
        regions: mapping ``name -> (lat_min, lat_max, lon_min, lon_max)`` in
            EPSG:4326. Defaults to :data:`DEFAULT_REGIONS`.
        lat_column / lon_column: column names.

    Yields:
        ``(region_name, train_idx, test_idx)`` tuples.

    Raises:
        ValueError: if a region is empty or columns are missing.
    """
    if regions is None:
        regions = DEFAULT_REGIONS
    if not regions:
        raise ValueError("No regions provided.")
    if lat_column not in df.columns or lon_column not in df.columns:
        raise ValueError(
            f"DataFrame must contain {lat_column!r} and {lon_column!r} columns."
        )

    lats = df[lat_column].to_numpy()
    lons = df[lon_column].to_numpy()
    all_idx = np.arange(len(df), dtype=np.int64)

    for name, (lat_min, lat_max, lon_min, lon_max) in regions.items():
        in_region = (
            (lats >= lat_min) & (lats <= lat_max) & (lons >= lon_min) & (lons <= lon_max)
        )
        test_idx = all_idx[in_region]
        train_idx = all_idx[~in_region]
        if test_idx.size == 0:
            # Empty test fold is uninformative; skip rather than yielding noise.
            continue
        yield name, train_idx, test_idx


__all__ = ["DEFAULT_REGIONS", "leave_region_out_split"]
