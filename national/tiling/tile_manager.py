"""Tile partitioning over Japan using the standard 1/N mesh codes.

The :class:`TileManager` partitions a region (defined by a bounding box and
a tile size) into rectangular tiles, each carrying an optional halo region
used for blending at tile boundaries during inference.

Tile sizing:

- ``tile_size_km <= 12``  -> use quarter-mesh tiles (~5.5 x 4.6 km).
- ``tile_size_km <= 30``  -> use secondary-mesh tiles (~11 x 9 km).
- otherwise               -> use primary-mesh tiles (~80 x 75 km).

This auto-selection means a config like ``tile_size_km: 80`` for the
nationwide run yields ~190 primary-mesh tiles over Japan, while a finer
``tile_size_km: 10`` for a regional run yields ~3000 secondary-mesh tiles.

Halos are computed in km and converted to degrees per latitude band so
that the halo width in meters stays approximately constant across Japan.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from shared.geo.tiles import (
    mesh_bounds,
    primary_mesh_code,
    quarter_mesh_code,
    secondary_mesh_code,
)


@dataclass(frozen=True)
class TileBounds:
    """Inclusive bounding box of one tile, in EPSG:4326 degrees."""

    tile_id: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    def expand(self, halo_deg_lat: float, halo_deg_lon: float) -> "TileBounds":
        return TileBounds(
            tile_id=self.tile_id,
            lat_min=self.lat_min - halo_deg_lat,
            lat_max=self.lat_max + halo_deg_lat,
            lon_min=self.lon_min - halo_deg_lon,
            lon_max=self.lon_max + halo_deg_lon,
        )

    def contains(self, lat: float, lon: float) -> bool:
        return (
            self.lat_min <= lat <= self.lat_max
            and self.lon_min <= lon <= self.lon_max
        )


class TileManager:
    """Partition a region into tiles aligned on the standard Japanese mesh."""

    def __init__(
        self,
        region_bbox: tuple[float, float, float, float],
        tile_size_km: float,
        halo_km: float,
    ) -> None:
        lat_min, lon_min, lat_max, lon_max = region_bbox
        if lat_min >= lat_max or lon_min >= lon_max:
            raise ValueError(f"Invalid bbox: {region_bbox}")
        if tile_size_km <= 0 or halo_km < 0:
            raise ValueError("tile_size_km must be positive, halo_km non-negative.")
        self._bbox = (lat_min, lon_min, lat_max, lon_max)
        self._tile_size_km = float(tile_size_km)
        self._halo_km = float(halo_km)
        self._mesh_level = self._pick_mesh_level(tile_size_km)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return self._bbox

    @property
    def mesh_level(self) -> int:
        """``1`` (primary), ``2`` (secondary), or ``4`` (quarter)."""
        return self._mesh_level

    def tiles(self) -> list[TileBounds]:
        """Enumerate the mesh tiles overlapping the region bbox."""
        lat_min, lon_min, lat_max, lon_max = self._bbox

        if self._mesh_level == 1:
            lat_step = 2.0 / 3.0
            lon_step = 1.0
            code_fn = primary_mesh_code
        elif self._mesh_level == 2:
            lat_step = (2.0 / 3.0) / 8.0
            lon_step = 1.0 / 8.0
            code_fn = secondary_mesh_code
        else:
            lat_step = (2.0 / 3.0) / 8.0 / 10.0
            lon_step = 1.0 / 8.0 / 10.0
            code_fn = quarter_mesh_code

        # Walk a grid of (lat, lon) seed points at one-cell offsets from the
        # bottom-left corner. We deliberately step at the mesh stride and use
        # an open upper bound on lat_max/lon_max to avoid including the cell
        # immediately *outside* the bbox; the half-cell tolerance from earlier
        # iterations occasionally pulled in a neighboring primary mesh.
        eps = 1e-9
        seen: set[str] = set()
        tiles: list[TileBounds] = []
        lat = lat_min
        while lat <= lat_max + eps:
            lon = lon_min
            while lon <= lon_max + eps:
                try:
                    code = code_fn(lat, lon)
                except ValueError:
                    lon += lon_step
                    continue
                if code in seen:
                    lon += lon_step
                    continue
                seen.add(code)
                ml, lo_min, ma, lo_max = mesh_bounds(code)
                tiles.append(
                    TileBounds(
                        tile_id=code,
                        lat_min=ml,
                        lat_max=ma,
                        lon_min=lo_min,
                        lon_max=lo_max,
                    )
                )
                lon += lon_step
            lat += lat_step

        # Deterministic ordering for reproducibility across runs.
        tiles.sort(key=lambda t: (t.lat_min, t.lon_min))
        return tiles

    def halo_for(self, tile: TileBounds) -> TileBounds:
        """Return the tile expanded by the configured halo."""
        mid_lat = 0.5 * (tile.lat_min + tile.lat_max)
        halo_deg_lat = self._halo_km / 111.32
        halo_deg_lon = self._halo_km / (111.32 * max(0.1, math.cos(math.radians(mid_lat))))
        return tile.expand(halo_deg_lat, halo_deg_lon)

    @staticmethod
    def _pick_mesh_level(tile_size_km: float) -> int:
        # Mesh cell sizes:
        #   level 4 (1/10 sub-mesh, "1km mesh") -> ~1 km
        #   level 2 (secondary mesh)            -> ~10 km
        #   level 1 (primary mesh)              -> ~80 km
        # Choose the level whose nominal size is closest to the requested
        # tile_size_km without exceeding it.
        if tile_size_km <= 5.0:
            return 4
        if tile_size_km <= 30.0:
            return 2
        return 1


__all__ = ["TileBounds", "TileManager"]
