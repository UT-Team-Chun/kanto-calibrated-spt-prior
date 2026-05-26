"""GSI Digital Elevation Model (10 m) download notes.

GSI's 基盤地図情報数値標高モデル distribution at https://www.gsi.go.jp/kiban/
requires a free GSI account and a per-mesh selection. The bulk download is
not safely automatable without storing user credentials, so this module
documents the workflow and tracks the expected destination layout.

The full all-Japan 10 m DEM is ~150 GB raw. For Phase B development, the
Kanto-only subset (~5 GB) is sufficient.

License: GSI 基盤地図情報利用規約 -- attribution required, derived rasters
may be redistributed with notice (see
https://www.gsi.go.jp/kibanjoho/kibanjoho40182.html).
"""

from __future__ import annotations

from pathlib import Path

from national.data.download import DownloadSpec

LICENSE = "GSI 基盤地図情報利用規約"
PORTAL = "https://www.gsi.go.jp/kiban/"


def manifest(raw_root: Path) -> list[DownloadSpec]:
    """Manual placeholder spec; the real artifact is per-mesh GeoTIFFs."""
    return [
        DownloadSpec(
            name="gsi_dem10m_all_japan",
            url=None,
            destination=raw_root / "gsi" / "dem_10m",
            license=LICENSE,
            method="manual",
            manual_url=PORTAL,
            notes=(
                "Register for a free GSI account, then download 10 m DEM mesh "
                "tiles for the regions of interest. Place each per-mesh "
                "GeoTIFF under data/raw/gsi/dem_10m/. "
                "Run national.data.ingest.dem.merge_tiles to build a single mosaic."
            ),
        )
    ]


__all__ = ["manifest", "LICENSE", "PORTAL"]
