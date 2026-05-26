"""AVS30 ``CovariateLoader`` wrapping the existing J-SHIS API client.

The J-SHIS API is rate-limited (~10 requests/sec) so we cache every
(lat, lon) -> AVS30 lookup on disk as Parquet. The cache is keyed by
the 6-decimal-rounded coordinate tuple, which gives ~10 cm precision --
plenty for the 250 m J-SHIS mesh.

For the all-Japan dataset (~175 k unique borings), the first cold pass
takes ~5 h sequential. After that, training and prediction read entirely
from the local cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from national.data.covariate_registry import CovariateSpec


@dataclass(frozen=True)
class _CacheKey:
    lat: float
    lon: float


class AVS30Loader:
    """Look up AVS30 (m/s) from a local cache or the J-SHIS API."""

    spec: CovariateSpec

    def __init__(
        self,
        spec: CovariateSpec,
        *,
        cache_path: Path | None = None,
        round_decimals: int = 4,
    ) -> None:
        self.spec = spec
        self.cache_path = Path(cache_path) if cache_path is not None else None
        self.round_decimals = int(round_decimals)
        self._cache: dict[tuple[float, float], float] = {}
        self._fill_value = float(spec.fill_value) if spec.fill_value is not None else 0.0
        self._client: Any = None  # lazy
        self._dirty = False

    # ---- API ----------------------------------------------------------------
    def sample(
        self,
        lats: torch.Tensor,
        lons: torch.Tensor,
        depths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del depths  # depth-invariant
        self._ensure_cache_loaded()
        lats_np = lats.detach().cpu().numpy().astype(np.float64)
        lons_np = lons.detach().cpu().numpy().astype(np.float64)
        out = np.empty(lats_np.shape, dtype=np.float64)
        for i, (lat, lon) in enumerate(zip(lats_np, lons_np, strict=True)):
            out.flat[i] = self._lookup(float(lat), float(lon))
        if self._dirty:
            self._persist_cache()
        return torch.from_numpy(out.astype(np.float32)).to(device=lats.device)

    def sample_grid(
        self,
        bbox: tuple[float, float, float, float],
        resolution_m: float,
        depth: float | None = None,
    ):
        del depth
        import xarray as xr

        lon_min, lat_min, lon_max, lat_max = bbox
        # AVS30 is on a 250 m mesh, finer than typically needed for the prior.
        deg_per_m_lat = 1.0 / 111_320.0
        mid_lat = 0.5 * (lat_min + lat_max)
        deg_per_m_lon = deg_per_m_lat / max(1e-6, abs(np.cos(np.radians(mid_lat))))
        d_lat = resolution_m * deg_per_m_lat
        d_lon = resolution_m * deg_per_m_lon
        lat_grid = np.arange(lat_min, lat_max + d_lat / 2, d_lat)
        lon_grid = np.arange(lon_min, lon_max + d_lon / 2, d_lon)
        LON, LAT = np.meshgrid(lon_grid, lat_grid, indexing="xy")
        flat_lats = torch.from_numpy(LAT.ravel())
        flat_lons = torch.from_numpy(LON.ravel())
        vals = self.sample(flat_lats, flat_lons).numpy()
        return xr.DataArray(
            vals.reshape(LAT.shape),
            dims=("lat", "lon"),
            coords={"lat": lat_grid, "lon": lon_grid},
            name=self.spec.name,
        )

    # ---- cache management --------------------------------------------------
    def _ensure_cache_loaded(self) -> None:
        if self.cache_path is None or not self.cache_path.exists() or self._cache:
            return
        import pandas as pd

        df = pd.read_parquet(self.cache_path)
        for _, row in df.iterrows():
            self._cache[(float(row["lat"]), float(row["lon"]))] = float(row["avs30_m_s"])

    def _persist_cache(self) -> None:
        if self.cache_path is None:
            return
        import pandas as pd

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            [(lat, lon, val) for (lat, lon), val in self._cache.items()],
            columns=["lat", "lon", "avs30_m_s"],
        )
        df.to_parquet(self.cache_path, index=False)
        self._dirty = False

    def _lookup(self, lat: float, lon: float) -> float:
        key = (round(lat, self.round_decimals), round(lon, self.round_decimals))
        if key in self._cache:
            return self._cache[key]
        try:
            self._ensure_client()
            response = self._client.fetch_avs30(latitude=key[0], longitude=key[1])
            val = float(getattr(response, "avs30", self._fill_value))
        except Exception:  # noqa: BLE001 -- offline / API down / OOD
            val = self._fill_value
        self._cache[key] = val
        self._dirty = True
        return val

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        from poc.alg.jshis_avs30.api_client import JShisAPIClient

        self._client = JShisAPIClient()


__all__ = ["AVS30Loader"]
