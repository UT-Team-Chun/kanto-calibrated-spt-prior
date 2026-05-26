"""Covariate registry: load, cache, and sample spatial covariates.

The registry presents a single ``sample(lats, lons, depths)`` interface to all
model code, regardless of whether a covariate originates as a GeoTIFF raster,
a Zarr cube, a vector layer, or a remote API. Covariates are described in the
Hydra ``covariates`` config group and resolved at construction time.

This file defines the interface (Phase A). Concrete loaders are added in
Phase B as each covariate ingest script lands.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
import torch

Normalize = Literal["zscore", "minmax", "none"]
Category = Literal["continuous", "categorical", "spatial"]


@dataclass(frozen=True)
class CovariateSpec:
    """Static description of one covariate as it appears in a Hydra config."""

    name: str
    source: str
    path: Path | None
    dtype: str
    normalize: Normalize
    category: Category
    n_categories: int | None = None  # for categorical
    fill_value: float | int | None = None


class CovariateLoader(Protocol):
    """Per-covariate loader. Implementations live next to each ingest script."""

    spec: CovariateSpec

    def sample(
        self,
        lats: torch.Tensor,
        lons: torch.Tensor,
        depths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample this covariate at the given coordinates.

        Args:
            lats: 1-D tensor of latitudes in degrees (EPSG:4326).
            lons: 1-D tensor of longitudes in degrees (EPSG:4326).
            depths: optional 1-D tensor of depths in meters. Only used for
                depth-dependent covariates (currently none, but reserved).

        Returns:
            Tensor of shape ``(N,)`` (continuous) or ``(N,)`` of int codes
            (categorical). Out-of-domain samples receive ``spec.fill_value``.
        """
        ...

    def sample_grid(
        self,
        bbox: tuple[float, float, float, float],
        resolution_m: float,
        depth: float | None = None,
    ):
        """Return a regular grid sample as an ``xarray.DataArray``.

        Bbox order is ``(lon_min, lat_min, lon_max, lat_max)``.
        """
        ...


class CovariateRegistry:
    """Composite registry holding all configured covariates for a run.

    A registry is constructed from a list of :class:`CovariateSpec` objects
    typically produced by ``hydra.utils.instantiate`` over the
    ``covariates`` config group. The registry routes each ``sample`` call to
    the appropriate :class:`CovariateLoader` and stacks the results.
    """

    def __init__(self, loaders: list[CovariateLoader]) -> None:
        if not loaders:
            raise ValueError("CovariateRegistry requires at least one loader.")
        if len({l.spec.name for l in loaders}) != len(loaders):
            raise ValueError("Duplicate covariate names in registry.")
        self._loaders: dict[str, CovariateLoader] = {l.spec.name: l for l in loaders}

    @property
    def names(self) -> list[str]:
        return list(self._loaders.keys())

    @property
    def continuous_names(self) -> list[str]:
        return [n for n, l in self._loaders.items() if l.spec.category != "categorical"]

    @property
    def categorical_names(self) -> list[str]:
        return [n for n, l in self._loaders.items() if l.spec.category == "categorical"]

    def sample(
        self,
        lats: torch.Tensor,
        lons: torch.Tensor,
        depths: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Sample every registered covariate at the given coordinates."""
        return {
            name: loader.sample(lats, lons, depths)
            for name, loader in self._loaders.items()
        }

    def stack_continuous(
        self,
        lats: torch.Tensor,
        lons: torch.Tensor,
        depths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Stack continuous covariates into ``(N, n_continuous)`` tensor."""
        cols = [self._loaders[n].sample(lats, lons, depths) for n in self.continuous_names]
        if not cols:
            return torch.empty((lats.shape[0], 0), dtype=lats.dtype, device=lats.device)
        return torch.stack(cols, dim=1)


def load_registry_from_config(cfg) -> CovariateRegistry:  # noqa: ANN001 (OmegaConf)
    """Hydra entry point. Resolves a covariate group config to a registry.

    The config shape (see ``conf/covariates/core.yaml``) is::

        name: core
        features:
          dem_10m_elevation:
            name: dem_10m_elevation
            source: "raster:gsi"          # or a URL prefixed with "raster:"
            local_path: "${io.raw_root}/gsi/dem_10m.tif"
            dtype: float32
            normalize: zscore
            category: continuous
          surface_geology_v2_code:
            name: surface_geology_v2_code
            source: "vector:aist"
            local_path: "${io.raw_root}/aist/seamless_geology_v2.gpkg"
            code_column: "lithology_code"  # required for categorical vectors
            ...

    Args:
        cfg: a Hydra ``DictConfig`` for the ``covariates`` group (the *value*
            of ``cfg.covariates``, not the root).

    Returns:
        A :class:`CovariateRegistry` with one loader per feature entry.

    Raises:
        ValueError: if a feature spec is malformed.
    """
    from pathlib import Path as _Path

    from national.data.loaders import build_loader

    if "features" not in cfg:
        raise ValueError("covariate config missing 'features' mapping.")
    loaders = []
    for key, feat in cfg.features.items():
        path = feat.get("local_path", None)
        spec = CovariateSpec(
            name=str(feat.get("name", key)),
            source=str(feat.get("source", "")),
            path=_Path(str(path)) if path is not None else None,
            dtype=str(feat.get("dtype", "float32")),
            normalize=str(feat.get("normalize", "none")),  # type: ignore[arg-type]
            category=str(feat.get("category", "continuous")),  # type: ignore[arg-type]
            n_categories=feat.get("n_categories"),
            fill_value=feat.get("fill_value"),
        )
        loader_kwargs = {}
        if spec.category == "categorical" and "code_column" in feat:
            loader_kwargs["code_column"] = str(feat["code_column"])
        loaders.append(build_loader(spec, **loader_kwargs))
    return CovariateRegistry(loaders)


__all__ = [
    "CovariateSpec",
    "CovariateLoader",
    "CovariateRegistry",
    "load_registry_from_config",
]
