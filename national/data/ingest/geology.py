"""Ingest the AIST/GSJ Seamless Geological Map V2 (シームレス地質図 V2).

Source: https://gbank.gsj.jp/seamless/ -- the all-Japan 1:200,000 layer
distributed as a single shapefile or GeoPackage.

Output: a normalized GeoPackage at ``--output`` containing the columns:

- ``geometry``: polygon in EPSG:4326.
- ``geology_code``: int16, the raw 150-class AIST code.
- ``lithology_group``: int8 -- one of the 7 regimes used by the model
  (see :class:`national.tiling.regime_classifier.Regime`).
- ``age_code``: int8 -- 0=Quaternary, 1=Tertiary, 2=Mesozoic, 3=PreMesozoic,
  4=Unknown.

Plus a sidecar ``--output.with_suffix('.codes.json')`` mapping
geology_code -> human-readable Japanese label.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

LOG = logging.getLogger("national.data.ingest.geology")

DOWNLOAD_URL = "https://gbank.gsj.jp/seamless/"
LICENSE_NOTE = "AIST/GSJ Seamless Geological Map V2 利用規約 (CC BY 4.0 with attribution)"

# Coarse lithology mapping: AIST `geology_code` -> regime int.
# These come from the published lookup table. The values here mirror
# Regime.* in national.tiling.regime_classifier.
_DEFAULT_LITHOLOGY_LOOKUP: dict[int, int] = {
    # The full ~150-entry table is loaded from a sidecar JSON in production;
    # this dict serves as the default skeleton. Populate via
    # --lithology-lookup at runtime.
}


def _load_lithology_lookup(path: Path | None) -> dict[int, int]:
    if path is None:
        return dict(_DEFAULT_LITHOLOGY_LOOKUP)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): int(v) for k, v in raw.items()}


def run(
    input_path: Path,
    output_path: Path,
    *,
    geology_code_column: str = "GEOLOGY",
    age_code_column: str | None = "AGE",
    lithology_lookup_path: Path | None = None,
) -> Path:
    """Read the seamless shapefile, normalize columns, write a GeoPackage."""
    import geopandas as gpd
    import numpy as np

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input vector {input_path} not found. Download the AIST Seamless "
            f"Geological Map V2 from {DOWNLOAD_URL}."
        )
    LOG.info("Reading %s", input_path)
    gdf = gpd.read_file(str(input_path))

    if geology_code_column not in gdf.columns:
        raise KeyError(
            f"Source geology column {geology_code_column!r} not found in "
            f"{input_path}; available columns: {list(gdf.columns)!r}"
        )
    geology_codes = gdf[geology_code_column].astype("Int64").fillna(-1).astype(np.int64).to_numpy()

    lithology_lookup = _load_lithology_lookup(lithology_lookup_path)
    lithology = np.array(
        [lithology_lookup.get(int(c), 7) for c in geology_codes],  # 7 = UNKNOWN
        dtype=np.int8,
    )

    if age_code_column is not None and age_code_column in gdf.columns:
        age_codes = (
            gdf[age_code_column].astype("Int64").fillna(4).astype(np.int8).to_numpy()
        )
    else:
        age_codes = np.full((len(gdf),), 4, dtype=np.int8)

    if gdf.crs is None:
        raise ValueError(f"Source {input_path} has no CRS -- aborting to avoid corrupt output.")
    gdf = gdf.to_crs("EPSG:4326")

    out = gpd.GeoDataFrame(
        {
            "geology_code": geology_codes.astype(np.int16),
            "lithology_group": lithology,
            "age_code": age_codes,
            "geometry": gdf.geometry,
        },
        crs="EPSG:4326",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    LOG.info("Writing %s (%d features)", output_path, len(out))
    out.to_file(output_path, driver="GPKG")

    # Sidecar codes JSON (placeholder; ingest reruns may overwrite).
    codes_json = output_path.with_suffix(".codes.json")
    codes_json.write_text(
        json.dumps(
            {
                "version": 1,
                "source": DOWNLOAD_URL,
                "license": LICENSE_NOTE,
                "geology_code_to_label": {
                    str(int(c)): str(int(c)) for c in np.unique(geology_codes)
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    LOG.info("Wrote sidecar %s", codes_json)
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="Source shapefile/GPKG")
    p.add_argument("--output", type=Path, required=True, help="Output GPKG")
    p.add_argument("--geology-code-column", default="GEOLOGY")
    p.add_argument("--age-code-column", default="AGE")
    p.add_argument(
        "--lithology-lookup",
        type=Path,
        default=None,
        help="JSON mapping geology_code -> regime int (defaults to a blank table).",
    )
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    run(
        args.input,
        args.output,
        geology_code_column=args.geology_code_column,
        age_code_column=args.age_code_column,
        lithology_lookup_path=args.lithology_lookup,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["DOWNLOAD_URL", "LICENSE_NOTE", "run", "main"]
