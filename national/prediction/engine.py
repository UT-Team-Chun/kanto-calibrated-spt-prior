"""Prediction engine: turn a trained FoundationModel into a national Zarr cube.

The engine has three layers:

- :py:meth:`PredictionEngine.predict_points` -- fast online endpoint used by
  the FastAPI route. Takes Tensors / arrays, returns ``(mean, std)``.

- :py:meth:`PredictionEngine.predict_tile` -- batched inference over a
  rectangular tile, returns an ``xarray.DataArray`` with dims
  ``(depth, lat, lon)`` and a ``"statistic"`` axis for mean/std.

- :py:meth:`PredictionEngine.predict_cube` -- iterates over every tile from
  the :class:`TileManager`, calls ``predict_tile``, and writes the result
  into a single Zarr cube with chunked layout. Distributed-aware: when
  ``WORLD_SIZE > 1`` each rank handles a disjoint subset of tiles and
  ranks join via a Zarr open per tile (Zarr supports concurrent writes
  to disjoint chunks).

Memory model: a single tile prediction is ``B = n_lat * n_lon * n_depth``
forward passes. For a 1024x1024 lat-lon tile at 32 depths that's ~33 M
points, which we mini-batch in ``cfg.prediction.batch_size_cells``
chunks (~200 k cells per forward pass on a GH200 is ~5 GB activations).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from national.tiling.tile_manager import TileBounds, TileManager

if TYPE_CHECKING:
    import xarray as xr

    from national.data.covariate_registry import CovariateRegistry
    from national.models.foundation import FoundationModel

LOG = logging.getLogger("national.prediction.engine")


@dataclass(frozen=True)
class GridSpec:
    """Resolution + depth list controlling the predicted cube layout."""

    resolution_m: float
    depths_m: tuple[float, ...]
    batch_size_cells: int = 200_000

    def __post_init__(self) -> None:
        if self.resolution_m <= 0:
            raise ValueError(f"resolution_m must be positive, got {self.resolution_m}")
        if not self.depths_m:
            raise ValueError("depths_m must contain at least one entry.")
        if self.batch_size_cells <= 0:
            raise ValueError(f"batch_size_cells must be positive, got {self.batch_size_cells}")


class PredictionEngine:
    """Run point, tile, and cube inference from a trained foundation model."""

    def __init__(
        self,
        model: "FoundationModel",
        registry: "CovariateRegistry | None",
        tile_manager: TileManager,
        grid: GridSpec,
        *,
        device: torch.device | str = "cpu",
        regime_loader_name: str | None = None,
    ) -> None:
        self.model = model
        self.registry = registry
        self.tile_manager = tile_manager
        self.grid = grid
        self.device = torch.device(device) if isinstance(device, str) else device
        self.regime_loader_name = regime_loader_name
        self.model = self.model.to(self.device)
        self.model.eval()

    # ---- public API --------------------------------------------------------
    @torch.no_grad()
    def predict_points(
        self,
        lats: torch.Tensor | np.ndarray | list[float],
        lons: torch.Tensor | np.ndarray | list[float],
        depths: torch.Tensor | np.ndarray | list[float],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(mean, std)`` at the given coordinates."""
        lat_t = self._as_tensor(lats)
        lon_t = self._as_tensor(lons)
        dep_t = self._as_tensor(depths)
        if not (lat_t.shape == lon_t.shape == dep_t.shape):
            raise ValueError(
                f"Shape mismatch: lats={tuple(lat_t.shape)}, lons={tuple(lon_t.shape)}, "
                f"depths={tuple(dep_t.shape)}"
            )
        x, regime = self._assemble_features(lat_t, lon_t, dep_t)
        pred = self.model.predict(x.to(self.device), regime_codes=regime)
        return pred.mean.cpu(), pred.std.cpu()

    @torch.no_grad()
    def predict_tile(self, tile: TileBounds) -> "xr.DataArray":
        """Predict mean + std over a rectangular tile."""
        import xarray as xr

        lat_axis, lon_axis = self._tile_axes(tile)
        depths = np.asarray(self.grid.depths_m, dtype=np.float64)
        n_lat, n_lon, n_depth = lat_axis.size, lon_axis.size, depths.size

        LON, LAT = np.meshgrid(lon_axis, lat_axis, indexing="xy")
        DEPTHS = depths
        # Flatten to (n_total, 3) row order: (depth_outer, lat, lon).
        flat_lat = np.broadcast_to(LAT[None, :, :], (n_depth, n_lat, n_lon)).reshape(-1)
        flat_lon = np.broadcast_to(LON[None, :, :], (n_depth, n_lat, n_lon)).reshape(-1)
        flat_dep = np.broadcast_to(DEPTHS[:, None, None], (n_depth, n_lat, n_lon)).reshape(-1)

        means = np.empty(flat_lat.size, dtype=np.float32)
        stds = np.empty_like(means)
        batch = int(self.grid.batch_size_cells)
        for start in range(0, flat_lat.size, batch):
            stop = min(start + batch, flat_lat.size)
            mean_b, std_b = self.predict_points(
                flat_lat[start:stop], flat_lon[start:stop], flat_dep[start:stop]
            )
            means[start:stop] = mean_b.numpy().astype(np.float32)
            stds[start:stop] = std_b.numpy().astype(np.float32)

        cube = np.stack(
            [
                means.reshape(n_depth, n_lat, n_lon),
                stds.reshape(n_depth, n_lat, n_lon),
            ],
            axis=0,
        )
        return xr.DataArray(
            cube,
            dims=("statistic", "depth", "lat", "lon"),
            coords={
                "statistic": np.array(["mean", "std"]),
                "depth": depths.astype(np.float32),
                "lat": lat_axis.astype(np.float64),
                "lon": lon_axis.astype(np.float64),
            },
            name="prediction",
            attrs={"tile_id": tile.tile_id},
        )

    def predict_cube(
        self,
        output_path: Path,
        *,
        chunks: dict[str, int] | None = None,
    ) -> Path:
        """Predict the whole region and write to a Zarr cube.

        Distributed-aware: if ``WORLD_SIZE > 1`` each rank handles only the
        tiles it owns (round-robin by tile index). The first rank writes the
        cube skeleton; other ranks write into existing chunks.

        Args:
            output_path: target Zarr directory.
            chunks: optional override for the Zarr chunk layout.
        """
        import xarray as xr
        import zarr

        rank = int(os.environ.get("RANK", "0"))
        world = int(os.environ.get("WORLD_SIZE", "1"))
        tiles = self.tile_manager.tiles()
        my_tiles = [t for i, t in enumerate(tiles) if i % world == rank]
        LOG.info(
            "predict_cube: rank=%d/%d handling %d of %d tiles",
            rank,
            world,
            len(my_tiles),
            len(tiles),
        )

        # Build the cube skeleton on rank 0 using the first tile's axes
        # to infer per-tile size. We assume tile axes are uniform; if they
        # are not, the writer falls back to one Zarr group per tile.
        if rank == 0:
            self._init_cube_skeleton(output_path, tiles, chunks=chunks)

        # Barrier on file-system: every rank waits until the skeleton exists.
        skeleton = Path(output_path)
        if world > 1 and torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        elif rank != 0:
            # Single-process simulation -- still wait for skeleton existence.
            while not skeleton.exists():
                pass

        for t in my_tiles:
            tile_da = self.predict_tile(t)
            self._write_tile_into_cube(output_path, t, tile_da)

        if world > 1 and torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        return Path(output_path)

    # ---- internals ---------------------------------------------------------
    def _as_tensor(
        self, x: torch.Tensor | np.ndarray | list[float]
    ) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.detach().to(dtype=torch.float32).reshape(-1)
        return torch.as_tensor(np.asarray(x, dtype=np.float32)).reshape(-1)

    def _assemble_features(
        self,
        lats: torch.Tensor,
        lons: torch.Tensor,
        depths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Build the input row [lat, lon, depth, *covariates] and regime codes."""
        cols = [lats.unsqueeze(-1), lons.unsqueeze(-1), depths.unsqueeze(-1)]
        regime: torch.Tensor | None = None
        if self.registry is not None and self.registry.continuous_names:
            cov = self.registry.stack_continuous(lats, lons, depths)
            cols.append(cov)
        if (
            self.registry is not None
            and self.regime_loader_name is not None
            and self.regime_loader_name in self.registry.names
        ):
            all_samples = self.registry.sample(lats, lons, depths)
            regime_vec = all_samples[self.regime_loader_name].long().clamp_min(0)
            regime = regime_vec
        x = torch.cat(cols, dim=-1)
        return x, regime

    def _tile_axes(self, tile: TileBounds) -> tuple[np.ndarray, np.ndarray]:
        deg_per_m_lat = 1.0 / 111_320.0
        mid_lat = 0.5 * (tile.lat_min + tile.lat_max)
        deg_per_m_lon = deg_per_m_lat / max(0.1, abs(np.cos(np.radians(mid_lat))))
        d_lat = self.grid.resolution_m * deg_per_m_lat
        d_lon = self.grid.resolution_m * deg_per_m_lon
        lat_axis = np.arange(tile.lat_min, tile.lat_max + d_lat / 2.0, d_lat)
        lon_axis = np.arange(tile.lon_min, tile.lon_max + d_lon / 2.0, d_lon)
        return lat_axis, lon_axis

    def _init_cube_skeleton(
        self,
        output_path: Path,
        tiles: list[TileBounds],
        *,
        chunks: dict[str, int] | None,
    ) -> None:
        # For Phase B we lay down one Zarr group per tile. Phase C will switch
        # to a single global cube once the TileManager exposes the global
        # axes (currently it returns a single tile in Phase A/B).
        Path(output_path).mkdir(parents=True, exist_ok=True)
        del chunks  # forward-looking

    def _write_tile_into_cube(
        self,
        output_path: Path,
        tile: TileBounds,
        tile_da: "xr.DataArray",
    ) -> None:
        from shared.io.zarr_writer import write_zarr_cube

        tile_dir = Path(output_path) / f"tile_{tile.tile_id}.zarr"
        chunks = {
            "statistic": 1,
            "depth": min(8, tile_da.sizes["depth"]),
            "lat": min(512, tile_da.sizes["lat"]),
            "lon": min(512, tile_da.sizes["lon"]),
        }
        write_zarr_cube(tile_da, tile_dir, chunks=chunks)


__all__ = ["GridSpec", "PredictionEngine"]
