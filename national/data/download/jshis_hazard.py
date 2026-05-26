"""J-SHIS bulk hazard download (PGA475 / PGV475).

The J-SHIS bulk download portal at https://www.j-shis.bosai.go.jp/labs/
publishes the national seismic hazard rasters as GeoTIFFs. Direct URLs
are versioned and occasionally rotate, so we default to the manual
download path and only attempt automation when given an explicit URL.

The AVS30 *point* API is unchanged and is wrapped separately by the
runtime loader at :mod:`national.data.loaders.avs30`.

License: NIED J-SHIS data terms (https://www.j-shis.bosai.go.jp/copyright).
"""

from __future__ import annotations

from pathlib import Path

from national.data.download import DownloadSpec

LICENSE = "NIED J-SHIS データ利用ポリシー"
BULK_PORTAL = "https://www.j-shis.bosai.go.jp/labs/"


def manifest(raw_root: Path) -> list[DownloadSpec]:
    """Manual download specs for PGA475 and PGV475."""
    return [
        DownloadSpec(
            name="jshis_pga475_bulk",
            url=None,
            destination=raw_root / "jshis" / "pga475.tif",
            license=LICENSE,
            method="manual",
            manual_url=BULK_PORTAL,
            notes=(
                "Browse 確率論的地震動予測地図 (主要活断層帯) and download "
                "PGA475 GeoTIFF. Save as the destination filename."
            ),
        ),
        DownloadSpec(
            name="jshis_pgv475_bulk",
            url=None,
            destination=raw_root / "jshis" / "pgv475.tif",
            license=LICENSE,
            method="manual",
            manual_url=BULK_PORTAL,
            notes="Same portal; download PGV475 GeoTIFF.",
        ),
    ]


__all__ = ["manifest", "LICENSE", "BULK_PORTAL"]
