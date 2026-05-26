"""National-scale geotechnical foundation model package.

This package implements a regressor-agnostic regional screening
framework for SPT N-values. The companion paper (Okauchi & Chun,
2026, *Computers and Geotechnics*, submitted) instantiates and
validates this framework on the Kanto subset of the public
KuniJiban borehole archive as Phase 1; the package is named
``national`` because it is designed to scale to the full
nation-wide archive in follow-up work.

See ``docs/architecture.md`` for the high-level design. Modules:

- ``national.data`` -- covariate registry and boring datasets.
- ``national.tiling`` -- regional tiles, halos, regime classifier.
- ``national.models`` -- DKL+SVGP foundation model and online conditioner.
- ``national.training`` -- Hydra-driven distributed training driver.
- ``national.prediction`` -- tiled inference engine, Zarr/COG output.
- ``national.evaluation`` -- spatial K-fold, LRO, calibration, baselines.
- ``national.api`` -- FastAPI prediction endpoints.
"""

__all__: list[str] = []
