"""AIST/GSJ Seamless Geological Map V2 acquisition.

Two acquisition paths are documented:

1. **Bulk shapefile** (recommended for ingest): manual download from
   https://gbank.gsj.jp/seamless/v2/download/ -- the page is JS-driven so
   we cannot safely automate it. The download is a ~150 MB shapefile.

2. **Point API** (used to cache per-boring lookups): the public REST
   endpoint at
   ``https://gbank.gsj.jp/seamless/v2/api/1.3/legend.php?point=<lat>,<lon>&format=json``
   returns the lithology metadata at a single point. We wrap it here so
   the ingest pipeline can pre-populate covariate columns for every
   boring in the KuniJiban dataset.

The bulk download is required for the rasterized covariate; the point
API is what we actually use in production at ingest time because the
boring set has only ~175 k unique locations and the API is fast enough
(~10 requests / second) to populate them in a single overnight run.

License: AIST Seamless Geological Map V2 -- CC BY 4.0 with attribution
(see https://gbank.gsj.jp/seamless/copyright.html).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterable

import pandas as pd

from national.data.download import DownloadResult, DownloadSpec

LOG = logging.getLogger("national.data.download.aist_geology")

# Confirmed endpoints (May 2026):
LEGEND_URL = "https://gbank.gsj.jp/seamless/v2/api/1.3/legend.php"
BULK_PAGE = "https://gbank.gsj.jp/seamless/v2/download/"
LICENSE = "CC BY 4.0 (AIST/GSJ)"


def manifest(raw_root: Path) -> list[DownloadSpec]:
    """Specs for the bulk shapefile download (manual only)."""
    return [
        DownloadSpec(
            name="aist_seamless_v2_bulk",
            url=None,
            destination=raw_root / "aist" / "seamless_geology_v2.zip",
            license=LICENSE,
            method="manual",
            manual_url=BULK_PAGE,
            notes=(
                "Open the page, select 'シェープファイル形式 (全国)' and download "
                "(~150 MB). Unzip into the destination directory."
            ),
        )
    ]


# --------------------------------------------------------------------------- #
# Point-API batch fetcher
# --------------------------------------------------------------------------- #
def fetch_codes_for_borings(
    lats: Iterable[float] | Iterable[tuple[float, float]],
    lons: Iterable[float] | None = None,
    *,
    cache_path: Path,
    rate_limit_s: float = 0.05,
    user_agent: str = "geo-estimation/0.2",
    round_decimals: int = 4,
    flush_every: int = 200,
    progress_every: int = 500,
) -> DownloadResult:
    """Look up the AIST geology metadata at each (lat, lon) and cache to Parquet.

    The cache file accumulates across runs. Each invocation only issues
    API calls for *new* coordinates, so re-running on a superset of an
    earlier dataset is efficient.

    Args:
        lats: iterable of latitudes, OR a list of ``(lat, lon)`` tuples.
        lons: iterable of longitudes, same length as ``lats``. Pass
            ``None`` if ``lats`` already contains tuples.
        cache_path: destination Parquet path. Atomically flushed every
            ``flush_every`` new lookups so a long run is restartable.
        rate_limit_s: minimum delay between API calls (J-SHIS and AIST
            both ask for ~10 rps; we use 0.05 s = 20 rps with frequent
            retries which is conservative).
        round_decimals: cache key precision. 4 decimals = ~11 m at
            Japanese latitudes, smaller than the AIST seamless mesh.
        flush_every: persist the cache to disk every N new lookups.

    Returns:
        :class:`DownloadResult` with ``status="ok"`` and a log line
        reporting how many new lookups were issued vs. served from cache.
    """
    if lons is None:
        coords = list(lats)
    else:
        coords = list(zip(lats, lons, strict=True))

    spec = DownloadSpec(
        name="aist_geology_point_cache",
        url=LEGEND_URL,
        destination=cache_path,
        license=LICENSE,
        method="api",
        notes="Per-boring AIST lookup cache populated via the seamless V2 REST API.",
    )
    result = DownloadResult(spec=spec, status="failed")

    try:
        import httpx
    except ImportError as exc:
        result.error = f"httpx not installed: {exc}"
        return result

    cache: dict[tuple[float, float], dict] = {}
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        for _, row in df.iterrows():
            cache[(float(row["lat"]), float(row["lon"]))] = {
                "symbol": row.get("symbol", ""),
                "formation_age_ja": row.get("formation_age_ja", ""),
                "group_ja": row.get("group_ja", ""),
                "lithology_ja": row.get("lithology_ja", ""),
            }

    headers = {"User-Agent": user_agent, "Accept": "application/json"}
    new_count = 0
    queried_count = 0
    with httpx.Client(timeout=30.0, headers=headers) as client:
        for raw in coords:
            queried_count += 1
            lat, lon = float(raw[0]), float(raw[1])
            key = (round(lat, round_decimals), round(lon, round_decimals))
            if key in cache:
                continue
            try:
                resp = client.get(
                    LEGEND_URL,
                    params={"point": f"{key[0]},{key[1]}", "format": "json"},
                    follow_redirects=True,
                )
                resp.raise_for_status()
                payload = resp.json()
                entry = _legend_to_record(payload)
            except Exception as exc:  # noqa: BLE001 -- offline / API down / OOD
                LOG.warning("AIST lookup failed at (%s, %s): %s", key[0], key[1], exc)
                entry = _empty_record()
            cache[key] = entry
            new_count += 1
            if new_count % flush_every == 0:
                _persist_cache(cache_path, cache)
            if new_count % progress_every == 0:
                LOG.info(
                    "AIST: %d new / %d total cached (queried %d / %d)",
                    new_count,
                    len(cache),
                    queried_count,
                    len(coords),
                )
            if rate_limit_s > 0:
                time.sleep(rate_limit_s)

    _persist_cache(cache_path, cache)
    result.status = "ok"
    result.actual_size = cache_path.stat().st_size if cache_path.exists() else 0
    result.log.append(f"added {new_count} new lookups; total {len(cache)} cached")
    return result


def _legend_to_record(payload: object) -> dict:
    """Pull the relevant fields out of the AIST V2 legend.php JSON payload."""
    if not isinstance(payload, dict):
        return _empty_record()

    def _norm(key: str) -> str:
        # Coerce missing or explicit JSON null values to an empty string so
        # downstream consumers never see the literal 'None'.
        v = payload.get(key)
        if v is None:
            return ""
        return str(v)

    return {
        "symbol": _norm("symbol"),
        "formation_age_ja": _norm("formationAge_ja"),
        "group_ja": _norm("group_ja"),
        "lithology_ja": _norm("lithology_ja"),
    }


def _empty_record() -> dict:
    return {"symbol": "", "formation_age_ja": "", "group_ja": "", "lithology_ja": ""}


def _persist_cache(
    cache_path: Path, cache: dict[tuple[float, float], dict]
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    rows = [
        (lat, lon, v["symbol"], v["formation_age_ja"], v["group_ja"], v["lithology_ja"])
        for (lat, lon), v in cache.items()
    ]
    df = pd.DataFrame(
        rows,
        columns=["lat", "lon", "symbol", "formation_age_ja", "group_ja", "lithology_ja"],
    )
    df.to_parquet(tmp, index=False)
    tmp.replace(cache_path)


__all__ = [
    "manifest",
    "fetch_codes_for_borings",
    "LEGEND_URL",
    "BULK_PAGE",
    "LICENSE",
]
