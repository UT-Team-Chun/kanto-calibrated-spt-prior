"""Export prediction artifacts for the MapLibre frontend.

Two helpers:

- :func:`write_cog_slices` -- slice a 3-D Zarr cube (``depth, lat, lon``)
  into one Cloud-Optimized GeoTIFF per depth, suitable for direct
  serving via maplibre-cog-protocol.
- :func:`write_pmtiles_bundle` -- bundle a directory of GeoJSON overlays
  into a single PMTiles file via the ``pmtiles`` CLI. Phase D when the
  CLI dependency is wired.
"""

from __future__ import annotations

import logging
from pathlib import Path

LOG = logging.getLogger("national.prediction.frontend_export")


def write_cog_slices(
    zarr_path: Path,
    output_dir: Path,
    depths: list[float] | None = None,
    *,
    statistic: str = "mean",
) -> list[Path]:
    """Write one Cloud-Optimized GeoTIFF per requested depth.

    Args:
        zarr_path: directory of a Zarr cube written by
            :func:`shared.io.zarr_writer.write_zarr_cube`, with dims
            ``(statistic, depth, lat, lon)``.
        output_dir: target directory; created if absent.
        depths: subset of depths to slice. ``None`` -> every depth.
        statistic: ``"mean"`` or ``"std"``.

    Returns:
        List of written GeoTIFF paths, one per depth.
    """
    import numpy as np
    import rioxarray  # noqa: F401 -- registers the .rio accessor on DataArray
    import xarray as xr

    from shared.io.cog_writer import write_cog

    zarr_path = Path(zarr_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = xr.open_zarr(zarr_path)
    var_name = next(iter(ds.data_vars))
    da = ds[var_name]
    if "statistic" not in da.dims:
        raise ValueError(
            f"Zarr cube at {zarr_path} has no 'statistic' axis; got {da.dims!r}"
        )
    if statistic not in da.coords["statistic"].values.tolist():
        raise ValueError(
            f"Requested statistic={statistic!r} not in {da.coords['statistic'].values!r}"
        )
    da = da.sel(statistic=statistic)

    all_depths = list(map(float, da.coords["depth"].values))
    selected = depths if depths is not None else all_depths
    paths: list[Path] = []
    for d in selected:
        slice_da = da.sel(depth=d, method="nearest")
        out = output_dir / f"{var_name}_{statistic}_d{d:06.2f}.tif"
        slice_da.rio.write_crs("EPSG:4326", inplace=True)
        write_cog(slice_da, out)
        paths.append(out)
        LOG.info("Wrote %s", out)
    return paths


def write_pmtiles_bundle(geojson_dir: Path, output_path: Path) -> Path:
    """Bundle GeoJSON overlays into a PMTiles file via the ``pmtiles`` CLI.

    Deferred to Phase D: the ``pmtiles`` CLI dependency is currently optional
    and only used by the frontend overlay export.
    """
    del geojson_dir, output_path
    raise NotImplementedError(
        "PMTiles bundle export is implemented in Phase D once the pmtiles CLI "
        "is added to the project's `national` extra."
    )


__all__ = ["write_cog_slices", "write_pmtiles_bundle"]
