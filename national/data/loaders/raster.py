"""Raster ``CovariateLoader`` -- GeoTIFF / Zarr / NetCDF backend.

Internally uses ``rioxarray`` to open the source lazily, reproject to a
canonical CRS (EPSG:4326) on demand, and interpolate at query points.

Sampling semantics:

- Continuous specs (``spec.category != "categorical"``): bilinear interp.
- Categorical specs: nearest-neighbor lookup, integer codes preserved.
- Out-of-domain queries return ``spec.fill_value`` (0.0 for continuous and
  -1 for categorical when ``fill_value`` is ``None``).

Caching: if ``cache_in_memory=True`` (default), the *reprojected* raster is
materialized as a numpy array on first ``sample`` call. Subsequent calls
read from RAM. This is the right tradeoff at all-Japan scale because:

- DEM 10 m all-Japan is ~150 GB raw GeoTIFF but compresses to ~40 GB in Zarr.
- Miyabi-G nodes have 480 GB unified memory each, so the whole stack fits.
- Per-batch random access during training would otherwise dominate latency.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from national.data.covariate_registry import CovariateSpec


class RasterLoader:
    """Sample a covariate from a rasterized source."""

    spec: CovariateSpec

    def __init__(
        self,
        spec: CovariateSpec,
        *,
        cache_in_memory: bool = True,
        target_crs: str = "EPSG:4326",
    ) -> None:
        self.spec = spec
        self.cache_in_memory = bool(cache_in_memory)
        self.target_crs = target_crs
        # Lazy state -- materialized on first ``sample()``.
        self._array: np.ndarray | None = None
        self._lat_axis: np.ndarray | None = None
        self._lon_axis: np.ndarray | None = None
        self._raw: Any = None  # cached rioxarray DataArray when cache_in_memory=False

        # Effective fill value with category-aware fallback.
        if spec.fill_value is not None:
            self._fill_value: float = float(spec.fill_value)
        elif spec.category == "categorical":
            self._fill_value = -1.0
        else:
            self._fill_value = 0.0

    # ---- API ----------------------------------------------------------------
    def sample(
        self,
        lats: torch.Tensor,
        lons: torch.Tensor,
        depths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del depths  # depth-invariant
        if lats.shape != lons.shape:
            raise ValueError(
                f"lats and lons shape mismatch: {tuple(lats.shape)} vs {tuple(lons.shape)}"
            )
        self._ensure_loaded()
        assert self._array is not None
        assert self._lat_axis is not None
        assert self._lon_axis is not None

        lats_np = lats.detach().cpu().numpy().astype(np.float64, copy=False)
        lons_np = lons.detach().cpu().numpy().astype(np.float64, copy=False)
        if self.spec.category == "categorical":
            vals = self._sample_nearest(lats_np, lons_np)
        else:
            vals = self._sample_bilinear(lats_np, lons_np)
        return torch.from_numpy(vals.astype(np.float32)).to(device=lats.device)

    def sample_grid(
        self,
        bbox: tuple[float, float, float, float],
        resolution_m: float,
        depth: float | None = None,
    ):
        del depth
        import xarray as xr  # heavy

        self._ensure_loaded()
        lon_min, lat_min, lon_max, lat_max = bbox

        # Approximate degree spacing for the requested metric resolution.
        deg_per_m_lat = 1.0 / 111_320.0
        mid_lat = 0.5 * (lat_min + lat_max)
        deg_per_m_lon = deg_per_m_lat / max(1e-6, abs(np.cos(np.radians(mid_lat))))
        d_lat = resolution_m * deg_per_m_lat
        d_lon = resolution_m * deg_per_m_lon

        lat_grid = np.arange(lat_min, lat_max + d_lat / 2.0, d_lat)
        lon_grid = np.arange(lon_min, lon_max + d_lon / 2.0, d_lon)
        LON, LAT = np.meshgrid(lon_grid, lat_grid, indexing="xy")
        flat_lats = LAT.ravel()
        flat_lons = LON.ravel()
        if self.spec.category == "categorical":
            vals = self._sample_nearest(flat_lats, flat_lons)
        else:
            vals = self._sample_bilinear(flat_lats, flat_lons)
        grid = vals.reshape(LAT.shape)
        return xr.DataArray(
            grid,
            dims=("lat", "lon"),
            coords={"lat": lat_grid, "lon": lon_grid},
            name=self.spec.name,
        )

    # ---- internals ----------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._array is not None:
            return
        if self.spec.path is None:
            raise ValueError(f"RasterLoader requires spec.path; got None for {self.spec.name!r}")
        if not self.spec.path.exists():
            raise FileNotFoundError(
                f"Raster file not found: {self.spec.path}. "
                f"Run the corresponding ingest script (see docs/architecture.md)."
            )

        # Lazy heavy import.
        import rioxarray  # noqa: F401 (registers the .rio accessor)
        import xarray as xr

        da = xr.open_dataarray(str(self.spec.path))
        # Some sources use the older crs= attribute name; rioxarray attaches .rio.
        try:
            current_crs = da.rio.crs
        except Exception:  # noqa: BLE001
            current_crs = None
        if current_crs is None:
            raise ValueError(
                f"Raster {self.spec.path} has no CRS metadata. Set one with "
                f"``da.rio.write_crs(...)`` in the ingest script."
            )
        if str(current_crs) != self.target_crs:
            da = da.rio.reproject(self.target_crs)

        # Reduce to 2D -- some GeoTIFFs come back as (band, y, x).
        if "band" in da.dims and da.sizes["band"] == 1:
            da = da.isel(band=0)
        elif "band" in da.dims:
            raise ValueError(
                f"Raster {self.spec.path} has multiple bands; pick one before loading."
            )

        # Pull the y (lat) and x (lon) axes.
        # Some rasters use "y"/"x", others "latitude"/"longitude".
        y_name = "y" if "y" in da.dims else ("latitude" if "latitude" in da.dims else None)
        x_name = "x" if "x" in da.dims else ("longitude" if "longitude" in da.dims else None)
        if y_name is None or x_name is None:
            raise ValueError(
                f"Could not find lat/lon dims on {self.spec.path}; "
                f"got dims {da.dims!r}"
            )

        lat_axis = da[y_name].to_numpy().astype(np.float64)
        lon_axis = da[x_name].to_numpy().astype(np.float64)
        if lat_axis[0] > lat_axis[-1]:
            # Most GeoTIFFs are stored north-down; flip to ascending so
            # searchsorted-based interpolation is straightforward.
            da = da.isel({y_name: slice(None, None, -1)})
            lat_axis = lat_axis[::-1]

        array = da.to_numpy()
        # ``rioxarray.reproject`` fills missing pixels with the nodata value;
        # mask them so they don't pollute interpolation.
        nodata = da.rio.nodata if hasattr(da, "rio") else None
        if nodata is not None and np.isfinite(nodata):
            mask = array == nodata
            if mask.any():
                array = array.astype(np.float64, copy=True)
                array[mask] = np.nan

        self._array = array
        self._lat_axis = lat_axis
        self._lon_axis = lon_axis
        if not self.cache_in_memory:
            # The numpy array is already in memory after to_numpy(); the flag
            # is mostly forward-looking for a future Zarr-chunked variant.
            self._raw = da

    def _idx_pair(
        self, axis: np.ndarray, q: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return left index, right index, and fractional position along axis."""
        # axis is strictly ascending. Find insertion positions.
        right = np.searchsorted(axis, q, side="left").clip(1, len(axis) - 1)
        left = right - 1
        denom = axis[right] - axis[left]
        # Avoid division by zero on degenerate axes.
        denom = np.where(denom == 0, 1.0, denom)
        frac = (q - axis[left]) / denom
        return left, right, frac

    def _sample_bilinear(self, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        assert self._array is not None and self._lat_axis is not None and self._lon_axis is not None
        out_of_domain = (
            (lats < self._lat_axis[0])
            | (lats > self._lat_axis[-1])
            | (lons < self._lon_axis[0])
            | (lons > self._lon_axis[-1])
        )
        lats_c = np.clip(lats, self._lat_axis[0], self._lat_axis[-1])
        lons_c = np.clip(lons, self._lon_axis[0], self._lon_axis[-1])
        i0, i1, fy = self._idx_pair(self._lat_axis, lats_c)
        j0, j1, fx = self._idx_pair(self._lon_axis, lons_c)
        v00 = self._array[i0, j0]
        v01 = self._array[i0, j1]
        v10 = self._array[i1, j0]
        v11 = self._array[i1, j1]
        top = v00 * (1.0 - fx) + v01 * fx
        bot = v10 * (1.0 - fx) + v11 * fx
        out = top * (1.0 - fy) + bot * fy
        # Replace NaN/inf and out-of-domain pixels with the fill value.
        out = np.where(np.isfinite(out), out, self._fill_value)
        out = np.where(out_of_domain, self._fill_value, out)
        return out.astype(np.float64)

    def _sample_nearest(self, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        assert self._array is not None and self._lat_axis is not None and self._lon_axis is not None
        out_of_domain = (
            (lats < self._lat_axis[0])
            | (lats > self._lat_axis[-1])
            | (lons < self._lon_axis[0])
            | (lons > self._lon_axis[-1])
        )
        lats_c = np.clip(lats, self._lat_axis[0], self._lat_axis[-1])
        lons_c = np.clip(lons, self._lon_axis[0], self._lon_axis[-1])
        i = np.argmin(np.abs(self._lat_axis[None, :] - lats_c[:, None]), axis=1)
        j = np.argmin(np.abs(self._lon_axis[None, :] - lons_c[:, None]), axis=1)
        out = self._array[i, j]
        out = np.where(np.isfinite(out), out, self._fill_value)
        out = np.where(out_of_domain, self._fill_value, out)
        return out.astype(np.float64)


__all__ = ["RasterLoader"]
