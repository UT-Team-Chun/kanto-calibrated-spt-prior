#!/usr/bin/env python
"""Build the model-ready boring Parquet.

Reads ``data/outputs/location_n_values.csv`` (2.7 M N-value rows), joins
in the derived covariates (river / coast distance, absolute elevation,
regime), and writes a single Parquet to ``data/features/borings.parquet``
that the :class:`national.data.boring_dataset.BoringDataset` can load.

Example::

    cd backend
    uv run python -m scripts.enrich_borings                # full national set
    uv run python -m scripts.enrich_borings --region kanto # Kanto subset only
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from national.data.enrich import EnrichmentSpec, enrich

LOG = logging.getLogger("scripts.enrich_borings")

# Region presets (lat_min, lon_min, lat_max, lon_max) in EPSG:4326.
REGIONS: dict[str, tuple[float, float, float, float]] = {
    "japan":   (24.0, 122.0, 46.0, 146.5),
    "kanto":   (35.0, 138.5, 37.5, 141.0),
    "tohoku":  (37.0, 139.0, 41.5, 142.5),
    "abukuma": (36.7, 140.0, 37.6, 140.7),
    "kinki":   (33.5, 134.0, 35.5, 137.0),
}


def main(argv: list[str] | None = None) -> int:
    repo = Path(__file__).resolve().parents[2]
    default_csv = repo / "data" / "outputs" / "location_n_values.csv"
    default_out = repo / "data" / "features" / "borings.parquet"
    default_river = repo / "data" / "river" / "class1_rivers_all_japan.geojson"
    default_coast = repo / "data" / "raw" / "mlit" / "C23-06"
    default_aist = repo / "data" / "features" / "derived" / "aist_codes.parquet"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=default_csv)
    parser.add_argument("--output", type=Path, default=default_out)
    parser.add_argument("--river-geojson", type=Path, default=default_river)
    parser.add_argument("--coast-dir", type=Path, default=default_coast)
    parser.add_argument(
        "--aist-cache",
        type=Path,
        default=default_aist,
        help="Parquet from national.data.download.aist_geology (optional).",
    )
    parser.add_argument(
        "--region",
        choices=sorted(REGIONS.keys()),
        help="Limit enrichment to a named bbox preset.",
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("LAT_MIN", "LON_MIN", "LAT_MAX", "LON_MAX"),
        help="Explicit EPSG:4326 bbox.",
    )
    parser.add_argument("--target-column", default="n_value")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    bbox = None
    if args.bbox is not None:
        bbox = tuple(args.bbox)
    elif args.region is not None:
        bbox = REGIONS[args.region]

    spec = EnrichmentSpec(
        borings_csv=args.input,
        output_parquet=args.output,
        river_geojson=args.river_geojson,
        coast_dir=args.coast_dir,
        aist_geology_cache=args.aist_cache,
        target_column=args.target_column,
        bbox=bbox,
    )
    if not spec.borings_csv.exists():
        parser.error(f"Boring CSV not found: {spec.borings_csv}")
    enrich(spec)
    return 0


if __name__ == "__main__":
    sys.exit(main())
