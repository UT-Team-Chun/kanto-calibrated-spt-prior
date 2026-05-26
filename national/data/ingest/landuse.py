"""Ingest MLIT 国土数値情報 LU25 (Land Use 100 m mesh).

Source: https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-L03-b.html
(LU25 / L03-b 100 m mesh, ~12 classes).

The MLIT distribution is a shapefile per 1/2 mesh. This script merges all
shapefiles in ``--input-dir`` into a single rasterized GeoTIFF at 100 m
resolution in EPSG:4326, with integer class codes.

Class codes (per MLIT L03-b spec):

    1 = 田              (rice paddy)
    2 = その他の農用地  (other agricultural)
    5 = 森林            (forest)
    6 = 荒地            (wasteland)
    7 = 建物用地        (built-up)
    9 = 道路            (road)
   10 = 鉄道            (railway)
   11 = その他の用地    (other)
   14 = 河川地及び湖沼  (river / lake)
   15 = 海浜            (beach)
   16 = 海水域          (sea)
   17 = ゴルフ場        (golf course)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

LOG = logging.getLogger("national.data.ingest.landuse")

DOWNLOAD_URL = "https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-L03-b.html"
LICENSE_NOTE = "MLIT 国土数値情報利用規約 (CC BY-like with attribution)"

LANDUSE_CLASS_LABELS: dict[int, str] = {
    1: "rice_paddy",
    2: "other_agricultural",
    5: "forest",
    6: "wasteland",
    7: "built_up",
    9: "road",
    10: "railway",
    11: "other",
    14: "water",
    15: "beach",
    16: "sea",
    17: "golf",
}


def run(
    input_dir: Path,
    output_path: Path,
    *,
    resolution_deg: float = 1.0 / 1200.0,  # ~92 m at mid-latitude
    class_column: str = "L03_b_002",
) -> Path:
    """Merge LU25 shapefiles and rasterize into a single GeoTIFF."""
    import geopandas as gpd
    import numpy as np
    import pandas as pd
    import rasterio
    from rasterio.features import rasterize
    from rasterio.transform import from_origin

    shps = sorted(input_dir.glob("**/*.shp"))
    if not shps:
        raise FileNotFoundError(
            f"No shapefiles under {input_dir}. Download LU25 from {DOWNLOAD_URL}."
        )
    LOG.info("Merging %d shapefiles -> %s", len(shps), output_path)
    gdfs = []
    for shp in shps:
        gdf = gpd.read_file(str(shp))
        if class_column not in gdf.columns:
            raise KeyError(
                f"Class column {class_column!r} not in {shp}; available: "
                f"{list(gdf.columns)!r}"
            )
        gdf = gdf[[class_column, "geometry"]].copy()
        gdf[class_column] = (
            gdf[class_column].astype("Int64").fillna(0).astype(np.int16)
        )
        gdfs.append(gdf)
    merged = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)
    if merged.crs is None:
        raise ValueError("Source CRS missing -- aborting.")
    merged = merged.to_crs("EPSG:4326")

    lon_min, lat_min, lon_max, lat_max = merged.total_bounds
    n_lat = int(np.ceil((lat_max - lat_min) / resolution_deg))
    n_lon = int(np.ceil((lon_max - lon_min) / resolution_deg))
    transform = from_origin(lon_min, lat_max, resolution_deg, resolution_deg)

    LOG.info("Rasterizing -> %d x %d at %.6f deg", n_lat, n_lon, resolution_deg)
    shapes = list(zip(merged.geometry, merged[class_column].astype(np.int16), strict=True))
    raster = rasterize(
        shapes,
        out_shape=(n_lat, n_lon),
        transform=transform,
        fill=0,
        dtype="int16",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=n_lat,
        width=n_lon,
        count=1,
        dtype="int16",
        crs="EPSG:4326",
        transform=transform,
        compress="deflate",
    ) as dst:
        dst.write(raster, 1)
    LOG.info("Wrote %s", output_path)
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resolution-deg", type=float, default=1.0 / 1200.0)
    p.add_argument("--class-column", default="L03_b_002")
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    run(
        args.input_dir,
        args.output,
        resolution_deg=args.resolution_deg,
        class_column=args.class_column,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["DOWNLOAD_URL", "LICENSE_NOTE", "LANDUSE_CLASS_LABELS", "run", "main"]
