"""Ingest J-SHIS seismic hazard rasters: PGA475 and PGV475.

Source: https://www.j-shis.bosai.go.jp/map/api/sstrct -- the API supports
both point queries (used by ``AVS30Loader``) and a bulk-download endpoint
that returns the full all-Japan 250 m mesh as a GeoTIFF.

This script consumes the pre-downloaded GeoTIFF (placed under
``--input-dir``) and rewrites it as a Cloud-Optimized GeoTIFF in EPSG:4326
suitable for :class:`national.data.loaders.raster.RasterLoader`.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

LOG = logging.getLogger("national.data.ingest.jshis_hazard")

DOWNLOAD_URL = "https://www.j-shis.bosai.go.jp/map/api/sstrct"
LICENSE_NOTE = "NIED J-SHIS データ利用ポリシー (https://www.j-shis.bosai.go.jp/copyright)"


def _to_epsg4326_cog(src_path: Path, dst_path: Path) -> Path:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT

    LOG.info("Reprojecting %s -> %s (EPSG:4326)", src_path, dst_path)
    with rasterio.open(src_path) as src:
        if src.crs is None:
            raise ValueError(
                f"{src_path} has no CRS; check the download. Expected EPSG:4612 or EPSG:6668."
            )
        with WarpedVRT(src, crs="EPSG:4326", resampling=Resampling.bilinear) as vrt:
            data = vrt.read(1)
            profile = vrt.profile.copy()
    profile.update(
        driver="GTiff",
        compress="deflate",
        tiled=True,
        blockxsize=512,
        blockysize=512,
    )
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(data, 1)
    LOG.info("Wrote %s", dst_path)
    return dst_path


def run(
    input_pga: Path,
    input_pgv: Path,
    output_pga: Path,
    output_pgv: Path,
) -> None:
    """Convert raw J-SHIS PGA/PGV GeoTIFFs to project-canonical COGs."""
    for src in (input_pga, input_pgv):
        if not src.exists():
            raise FileNotFoundError(
                f"Missing J-SHIS raster {src}. Download PGA475/PGV475 from {DOWNLOAD_URL}."
            )
    _to_epsg4326_cog(input_pga, output_pga)
    _to_epsg4326_cog(input_pgv, output_pgv)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-pga", type=Path, required=True)
    p.add_argument("--input-pgv", type=Path, required=True)
    p.add_argument("--output-pga", type=Path, required=True)
    p.add_argument("--output-pgv", type=Path, required=True)
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    run(args.input_pga, args.input_pgv, args.output_pga, args.output_pgv)
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["DOWNLOAD_URL", "LICENSE_NOTE", "run", "main"]
