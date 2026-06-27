"""Kanto-prefecture-level leave-one-out spatial validation.

Each borehole is assigned to the **administrative prefecture polygon that
contains it** (point-in-polygon against the public prefecture boundaries). This
gives a genuine leave-one-prefecture-out: for each held-out prefecture the test
set is exactly the boreholes administratively inside it and the train set is
everything else. The seven held-out test sets are pairwise disjoint genuine
prefectures.

The study corpus is a generous bounding box around Kanto, so ~12% of boreholes
fall in *neighbouring* prefectures (Niigata, Shizuoka, Fukushima, Yamanashi,
Nagano) or just offshore. These are labelled :data:`_OTHER` (``"other"``): they
are never a held-out fold and always remain in training (legitimate spatial
context). They are deliberately NOT forced into a Kanto prefecture, so a
held-out fold is a genuine administrative prefecture rather than a
nearest-neighbour region. The seven prefecture test sets therefore cover the
in-prefecture boreholes ($435{,}732$ of the $495{,}725$-row corpus); the rest
are train-only.

The polygon assignment is precomputed once (geopandas point-in-polygon) and
cached as a small ``(latitude_deg, longitude_deg) -> prefecture`` lookup of the
*contained* boreholes at :data:`_POLYGON_LOOKUP_PATH`; at run time we only need a
pandas merge (no geopandas dependency in the image). If the lookup is
unavailable, assignment falls back to nearest prefecture *centre* over the seven
(a degraded mode for callers without the asset).

This replaces the original approximate bounding-box membership, whose boxes
overlapped substantially (e.g. Chiba/Ibaraki) and therefore double-counted
boreholes across folds.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd

KANTO_PREFECTURES: dict[str, tuple[float, float, float, float]] = {
    # (lat_min, lat_max, lon_min, lon_max) -- approximate data-bearing extents,
    # used only to derive the prefecture centres below (the nearest-centre
    # FALLBACK assignment; primary assignment is the polygon lookup).
    "tokyo":    (35.50, 35.92, 139.30, 139.95),
    "kanagawa": (35.13, 35.65, 138.93, 139.83),
    "saitama":  (35.75, 36.30, 138.70, 139.92),
    "chiba":    (34.90, 36.10, 139.70, 140.95),
    "ibaraki":  (35.75, 36.95, 139.65, 140.85),
    "tochigi":  (36.20, 37.15, 139.30, 140.30),
    "gunma":    (36.05, 36.95, 138.40, 139.70),
}

# Prefecture centres = bounding-box centroids. The nearest-centre assignment is
# the FALLBACK used only if the polygon lookup is missing or a row is unmatched
# (longitude scaled by cos(lat) for an approximate great-circle metric).
KANTO_CENTROIDS: dict[str, tuple[float, float]] = {
    name: (0.5 * (lat_min + lat_max), 0.5 * (lon_min + lon_max))
    for name, (lat_min, lat_max, lon_min, lon_max) in KANTO_PREFECTURES.items()
}

_KANTO_MEAN_LAT = 36.0  # cos-scale longitude at the Kanto mean latitude

# Label for boreholes not contained by any of the seven Kanto prefecture
# polygons (neighbouring prefectures / offshore within the study bounding box).
# These are train-only: never a held-out fold, never forced into a prefecture.
_OTHER = "other"

# Precomputed administrative-polygon assignment: a small unique-location lookup
# (latitude_deg, longitude_deg, pref). Bundled as package data next to this
# module so it ships in the utens image via ``COPY backend/`` (no .dockerignore
# whitelist needed) and resolves identically for local runs. The generator
# (point-in-polygon against public prefecture boundaries) also writes a copy to
# data/features/derived/ for the repo's derived-data convention.
_POLYGON_LOOKUP_PATH = (
    Path(__file__).resolve().parent
    / "assets/kanto_prefecture_polygon_assignment.parquet"
)


def assign_nearest_prefecture(
    lats: np.ndarray, lons: np.ndarray,
) -> np.ndarray:
    """Assign each (lat, lon) to the nearest prefecture *centre* (fallback).

    Distance is Euclidean in (lat, cos(lat)*lon) degrees, an approximate
    great-circle metric adequate over the small Kanto extent. Returns a string
    array of prefecture names aligned with the inputs.
    """
    names = list(KANTO_CENTROIDS)
    cen = np.array([KANTO_CENTROIDS[n] for n in names], dtype=np.float64)  # (P, 2)
    coslat = math.cos(math.radians(_KANTO_MEAN_LAT))
    pts = np.stack([np.asarray(lats, np.float64),
                    np.asarray(lons, np.float64) * coslat], axis=1)        # (N, 2)
    cen_scaled = cen.copy()
    cen_scaled[:, 1] *= coslat
    d2 = ((pts[:, None, :] - cen_scaled[None, :, :]) ** 2).sum(axis=2)     # (N, P)
    return np.asarray(names, dtype=object)[d2.argmin(axis=1)]


@functools.lru_cache(maxsize=1)
def _load_polygon_lookup() -> pd.DataFrame | None:
    """Load the precomputed (lat, lon) -> prefecture polygon lookup, or None."""
    try:
        lut = pd.read_parquet(_POLYGON_LOOKUP_PATH)
    except (FileNotFoundError, OSError):
        return None
    return lut[["latitude_deg", "longitude_deg", "pref"]]


def assign_prefecture(
    lats: np.ndarray, lons: np.ndarray,
) -> np.ndarray:
    """Assign each (lat, lon) to its containing administrative Kanto prefecture.

    Uses the precomputed point-in-polygon lookup of the *contained* boreholes.
    Locations not contained by any of the seven Kanto prefecture polygons
    (neighbouring prefectures / offshore within the study bounding box) are
    labelled :data:`_OTHER` --- they are never held out and always train. They
    are NOT forced into a Kanto prefecture. If the lookup asset is unavailable
    we fall back to nearest prefecture *centre* over the seven (degraded mode).
    Returns a string array aligned with the inputs.
    """
    lats = np.asarray(lats, np.float64)
    lons = np.asarray(lons, np.float64)
    lut = _load_polygon_lookup()
    if lut is None:
        return assign_nearest_prefecture(lats, lons)  # degraded: asset absent
    key = pd.DataFrame({"latitude_deg": lats, "longitude_deg": lons})
    merged = key.merge(lut, on=["latitude_deg", "longitude_deg"],
                       how="left", sort=False)
    pref = merged["pref"].to_numpy(dtype=object)
    pref[pd.isnull(pref)] = _OTHER
    return pref


def leave_prefecture_out_split(
    df: pd.DataFrame,
    prefectures: list[str] | None = None,
    *,
    lat_column: str = "latitude_deg",
    lon_column: str = "longitude_deg",
) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """Yield ``(pref_name, train_idx, test_idx)`` for each held-out Kanto
    prefecture under the administrative-polygon assignment.

    Each row is assigned to the prefecture whose boundary polygon contains it;
    rows outside the seven prefectures are labelled ``"other"`` and always fall
    in ``train_idx`` (never held out). For each requested prefecture,
    ``test_idx`` are the rows contained in it and ``train_idx`` is the
    complement (the other six prefectures plus all ``"other"`` rows). The seven
    test sets are pairwise disjoint genuine prefectures.

    Args:
        df: DataFrame of borehole rows.
        prefectures: subset of :data:`KANTO_PREFECTURES` keys to iterate.
            Defaults to all seven Kanto prefectures.
        lat_column / lon_column: column names.
    """
    if prefectures is None:
        prefectures = list(KANTO_PREFECTURES)
    unknown = [p for p in prefectures if p not in KANTO_PREFECTURES]
    if unknown:
        raise ValueError(
            f"Unknown prefecture(s): {unknown}. Available: "
            f"{sorted(KANTO_PREFECTURES)}"
        )
    assigned = assign_prefecture(
        df[lat_column].to_numpy(), df[lon_column].to_numpy(),
    )
    all_idx = np.arange(len(df), dtype=np.int64)
    for pref in prefectures:
        test_idx = all_idx[assigned == pref]
        train_idx = all_idx[assigned != pref]
        if test_idx.size == 0:
            continue
        yield pref, train_idx, test_idx


__all__ = [
    "KANTO_PREFECTURES",
    "KANTO_CENTROIDS",
    "assign_nearest_prefecture",
    "assign_prefecture",
    "leave_prefecture_out_split",
]
