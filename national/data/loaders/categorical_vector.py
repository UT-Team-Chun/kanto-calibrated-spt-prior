"""Vector ``CovariateLoader`` -- shapefile / GeoJSON / GeoPackage backend.

Returns an integer category code per query point via point-in-polygon
matching. The lookup uses ``geopandas.sjoin`` with the GeoSeries STRtree
spatial index, which is O(log N + k) per query and fast enough to call once
per batch even at all-Japan scale.

Typical use:

    spec = CovariateSpec(
        name="lithology_group", source="vector:seamless_geology",
        path=Path("data/processed/lithology_group.gpkg"),
        dtype="int16", normalize="none", category="categorical",
        fill_value=-1,
    )
    loader = CategoricalVectorLoader(spec, code_column="lithology_code")
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from national.data.covariate_registry import CovariateSpec


class CategoricalVectorLoader:
    """Sample integer category codes from a vector polygon dataset."""

    spec: CovariateSpec

    def __init__(
        self,
        spec: CovariateSpec,
        *,
        code_column: str,
        target_crs: str = "EPSG:4326",
    ) -> None:
        self.spec = spec
        self.code_column = code_column
        self.target_crs = target_crs
        self._gdf: Any = None  # geopandas.GeoDataFrame
        self._fill_value = (
            int(spec.fill_value) if spec.fill_value is not None else -1
        )

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
        assert self._gdf is not None

        import geopandas as gpd
        from shapely.geometry import Point

        lats_np = lats.detach().cpu().numpy().astype(np.float64)
        lons_np = lons.detach().cpu().numpy().astype(np.float64)

        points = gpd.GeoDataFrame(
            {"_query_idx": np.arange(len(lats_np), dtype=np.int64)},
            geometry=[Point(lon, lat) for lat, lon in zip(lats_np, lons_np, strict=True)],
            crs=self.target_crs,
        )
        joined = gpd.sjoin(
            points,
            self._gdf[[self.code_column, "geometry"]],
            how="left",
            predicate="within",
        )
        # If a point falls in multiple polygons (overlapping geology layers),
        # take the first match per query index; sjoin returns duplicates.
        joined = joined.drop_duplicates(subset="_query_idx", keep="first").sort_values(
            "_query_idx"
        )
        codes = joined[self.code_column].to_numpy()
        # NaN where the point fell outside every polygon.
        codes = np.where(np.isnan(codes.astype(np.float64)), self._fill_value, codes)
        return torch.from_numpy(codes.astype(np.float32)).to(device=lats.device)

    def sample_grid(
        self,
        bbox: tuple[float, float, float, float],
        resolution_m: float,
        depth: float | None = None,
    ):
        del depth
        import xarray as xr

        lon_min, lat_min, lon_max, lat_max = bbox
        deg_per_m_lat = 1.0 / 111_320.0
        mid_lat = 0.5 * (lat_min + lat_max)
        deg_per_m_lon = deg_per_m_lat / max(1e-6, abs(np.cos(np.radians(mid_lat))))
        d_lat = resolution_m * deg_per_m_lat
        d_lon = resolution_m * deg_per_m_lon

        lat_grid = np.arange(lat_min, lat_max + d_lat / 2.0, d_lat)
        lon_grid = np.arange(lon_min, lon_max + d_lon / 2.0, d_lon)
        LON, LAT = np.meshgrid(lon_grid, lat_grid, indexing="xy")
        flat_lats = torch.from_numpy(LAT.ravel())
        flat_lons = torch.from_numpy(LON.ravel())
        vals = self.sample(flat_lats, flat_lons).numpy()
        grid = vals.reshape(LAT.shape)
        return xr.DataArray(
            grid,
            dims=("lat", "lon"),
            coords={"lat": lat_grid, "lon": lon_grid},
            name=self.spec.name,
        )

    # ---- internals ---------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._gdf is not None:
            return
        if self.spec.path is None:
            raise ValueError(
                f"CategoricalVectorLoader requires spec.path; got None for "
                f"{self.spec.name!r}"
            )
        if not self.spec.path.exists():
            raise FileNotFoundError(
                f"Vector file not found: {self.spec.path}. "
                f"Run the ingest script first (see docs/architecture.md)."
            )

        import geopandas as gpd

        gdf = gpd.read_file(str(self.spec.path))
        if self.code_column not in gdf.columns:
            raise KeyError(
                f"Code column {self.code_column!r} not in {self.spec.path}; "
                f"available: {list(gdf.columns)!r}"
            )
        # Ensure integer codes -- the FiLM regime embedding wants ints.
        gdf[self.code_column] = (
            gdf[self.code_column]
            .astype("Int64")
            .fillna(self._fill_value)
            .astype(np.int64)
        )
        if gdf.crs is None:
            raise ValueError(
                f"Vector {self.spec.path} has no CRS; set one in the ingest script."
            )
        if str(gdf.crs) != self.target_crs:
            gdf = gdf.to_crs(self.target_crs)
        self._gdf = gdf


__all__ = ["CategoricalVectorLoader"]
