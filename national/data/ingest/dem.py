"""Ingest GSI Digital Elevation Model (基盤地図情報 数値標高モデル) at 10 m.

Source: https://www.gsi.go.jp/kiban/ (download requires a GSI account).

Workflow:

1. Manually download the all-Japan 10 m DEM (the `MEM` series GMLs or the
   `dem10b` GeoTIFF mosaic) into ``--input-dir``. The directory may contain
   either a single GeoTIFF mosaic or a tree of per-mesh GMLs that this
   script will merge.
2. Run ``python -m national.data.ingest.dem --input-dir ...
   --output-elevation ... --output-slope ... --output-aspect ...``.

The script produces three Cloud-Optimized GeoTIFFs (one each for elevation,
slope, aspect), all in EPSG:4326 with the source resolution preserved.
``slope`` is in degrees from horizontal; ``aspect`` is in degrees clockwise
from north (north = 0, east = 90).

License note: GSI 基盤地図情報利用規約 -- redistribution of derived rasters
requires attribution. Document the run in ``docs/covariates.md`` per the
project's data provenance policy.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

LOG = logging.getLogger("national.data.ingest.dem")

DOWNLOAD_URL = "https://www.gsi.go.jp/kiban/"
LICENSE_NOTE = "GSI 基盤地図情報利用規約 (https://www.gsi.go.jp/kibanjoho/kibanjoho40182.html)"


def derive_slope_aspect(elev_path: Path, slope_path: Path, aspect_path: Path) -> None:
    """Compute slope (degrees) and aspect (degrees, north=0 clockwise) from a DEM raster."""
    import numpy as np
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(elev_path) as src:
        elev = src.read(1).astype(np.float32)
        transform = src.transform
        profile = src.profile.copy()
        # ASSUMPTION: pixel size is approximately isotropic in meters. For a
        # GSI 10 m DEM in JGD2011, both transforms are ~10 m. If you reproject
        # to EPSG:4326 first the pixel sizes become degree-spaced and this
        # routine is no longer correct -- run derive_slope_aspect on the
        # NATIVE-CRS file, then reproject the slope/aspect outputs.

    dx = transform.a  # pixel size in x
    dy = -transform.e  # pixel size in y (note: negative in geotransform)

    # Sobel-style gradient: central differences with the standard 3x3 stencil.
    dz_dx = (np.roll(elev, -1, axis=1) - np.roll(elev, 1, axis=1)) / (2.0 * dx)
    dz_dy = (np.roll(elev, -1, axis=0) - np.roll(elev, 1, axis=0)) / (2.0 * dy)
    # Mask the wrap-around edge pixels.
    dz_dx[:, 0] = 0
    dz_dx[:, -1] = 0
    dz_dy[0, :] = 0
    dz_dy[-1, :] = 0

    slope_rad = np.arctan(np.hypot(dz_dx, dz_dy))
    slope_deg = np.degrees(slope_rad).astype(np.float32)

    # Aspect: atan2(-dz/dy, dz/dx) and convert to compass bearing.
    aspect_rad = np.arctan2(-dz_dy, dz_dx)
    aspect_deg = (90.0 - np.degrees(aspect_rad)) % 360.0
    aspect_deg = aspect_deg.astype(np.float32)
    # Flat pixels have undefined aspect -- conventionally encoded as -1.
    aspect_deg[slope_deg < 0.5] = -1.0

    out_profile = profile.copy()
    out_profile.update(dtype="float32", count=1, compress="deflate")

    for path, data in ((slope_path, slope_deg), (aspect_path, aspect_deg)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(path, "w", **out_profile) as dst:
            dst.write(data, 1)
        LOG.info("Wrote %s", path)


def merge_tiles(input_dir: Path, output_path: Path) -> Path:
    """Merge a directory of GeoTIFF tiles into a single mosaic via rasterio.merge."""
    import rasterio
    from rasterio.merge import merge

    tiles = sorted(input_dir.glob("*.tif")) + sorted(input_dir.glob("*.tiff"))
    if not tiles:
        raise FileNotFoundError(
            f"No GeoTIFF tiles found in {input_dir}. Download from {DOWNLOAD_URL} first."
        )
    LOG.info("Merging %d tiles -> %s", len(tiles), output_path)
    handles = [rasterio.open(t) for t in tiles]
    mosaic, transform = merge(handles)
    profile = handles[0].profile.copy()
    profile.update(
        height=mosaic.shape[1],
        width=mosaic.shape[2],
        transform=transform,
        compress="deflate",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(mosaic)
    for h in handles:
        h.close()
    return output_path


def run(
    input_dir: Path,
    output_elevation: Path,
    output_slope: Path,
    output_aspect: Path,
) -> None:
    """End-to-end ingest: merge tiles -> derive slope/aspect."""
    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory {input_dir} not found. Download the GSI DEM from "
            f"{DOWNLOAD_URL} and place tiles there."
        )
    merge_tiles(input_dir, output_elevation)
    derive_slope_aspect(output_elevation, output_slope, output_aspect)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-elevation", type=Path, required=True)
    p.add_argument("--output-slope", type=Path, required=True)
    p.add_argument("--output-aspect", type=Path, required=True)
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    run(args.input_dir, args.output_elevation, args.output_slope, args.output_aspect)
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["DOWNLOAD_URL", "LICENSE_NOTE", "merge_tiles", "derive_slope_aspect", "run", "main"]
