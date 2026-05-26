"""Concrete ``CovariateLoader`` implementations.

Each loader knows how to load one *kind* of source (raster, vector,
remote API) and exposes the ``CovariateLoader`` protocol so the registry
can compose heterogeneous covariates without caring about format.

Module layout:

- :mod:`national.data.loaders.raster` -- GeoTIFF / Zarr / NetCDF rasters.
- :mod:`national.data.loaders.categorical_vector` -- shapefile / GeoJSON polygons.
- :mod:`national.data.loaders.avs30` -- J-SHIS AVS30 API wrapper with caching.

The :func:`build_loader` helper dispatches a :class:`CovariateSpec` to the
appropriate loader class.
"""

from __future__ import annotations

from national.data.covariate_registry import CovariateLoader, CovariateSpec
from national.data.loaders.avs30 import AVS30Loader
from national.data.loaders.categorical_vector import CategoricalVectorLoader
from national.data.loaders.raster import RasterLoader


def build_loader(spec: CovariateSpec, **kwargs) -> CovariateLoader:  # noqa: ANN003
    """Construct the appropriate loader for ``spec.source``.

    Dispatch convention:

    - source starts with ``"raster:"`` -> :class:`RasterLoader`.
    - source starts with ``"vector:"`` -> :class:`CategoricalVectorLoader`.
    - source starts with ``"api:jshis"`` -> :class:`AVS30Loader`.

    Falls back to :class:`RasterLoader` when ``spec.path`` ends in
    ``.tif/.tiff/.zarr/.nc``, and to :class:`CategoricalVectorLoader` when it
    ends in ``.shp/.geojson/.gpkg``.

    Args:
        spec: covariate specification.
        **kwargs: forwarded to the chosen loader constructor.

    Raises:
        ValueError: if no dispatcher matches.
    """
    src = (spec.source or "").lower()
    path_suffix = (spec.path.suffix.lower() if spec.path is not None else "")

    if src.startswith("api:jshis"):
        return AVS30Loader(spec, **kwargs)
    if src.startswith("raster:") or path_suffix in {".tif", ".tiff", ".zarr", ".nc"}:
        return RasterLoader(spec, **kwargs)
    if src.startswith("vector:") or path_suffix in {".shp", ".geojson", ".gpkg"}:
        return CategoricalVectorLoader(spec, **kwargs)
    raise ValueError(
        f"build_loader: cannot dispatch CovariateSpec(name={spec.name!r}, "
        f"source={spec.source!r}, path={spec.path!r}). Use source prefix "
        f"raster: / vector: / api:jshis or a recognized file extension."
    )


__all__ = [
    "RasterLoader",
    "CategoricalVectorLoader",
    "AVS30Loader",
    "build_loader",
]
