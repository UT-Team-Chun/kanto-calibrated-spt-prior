"""MLIT 国土数値情報 (KSJ) downloads.

MLIT distributes a stable but heterogeneous set of nationwide GIS datasets
via the KSJ catalogue at https://nlftp.mlit.go.jp/ksj/. URL conventions
vary by dataset family:

- **Per-prefecture** (e.g. C23 coastline, W05 rivers)::

      https://nlftp.mlit.go.jp/ksj/gml/data/<CODE>/<CODE>-<YY>/<CODE>-<YY>_<PP>_GML.zip

  ``<PP>`` is the 2-digit prefecture code (01..47); ``<YY>`` is the
  2-digit fiscal year. Some prefectures' "latest" year differs from
  others -- W05 has Tokyo at FY08 but Hokkaido at FY09, so we let the
  caller pin a per-prefecture year.

- **Per-mesh** (e.g. L03-b LU25)::

      https://nlftp.mlit.go.jp/ksj/gml/data/<CODE>/<CODE>-<YY>/<CODE>-<YY>_<MESH>-<DATUM>_GML.zip

  ``<MESH>`` is the primary-mesh code (e.g. 5339 for central Tokyo);
  ``<DATUM>`` is ``jgd`` (JGD2011) or ``tky`` (Tokyo Datum). The L03-b
  catalogue has ~3,800 mesh codes covering Japan, so a full-Japan auto
  download is many small requests; we expose a CLI flag to limit by
  prefecture (which maps to a primary-mesh range).

License: MLIT 国土数値情報利用規約 (https://nlftp.mlit.go.jp/ksj/other/agreement.html).
"""

from __future__ import annotations

import logging
from pathlib import Path

from national.data.download import DownloadSpec

LOG = logging.getLogger("national.data.download.mlit_ksj")

LICENSE = "MLIT 国土数値情報利用規約 (https://nlftp.mlit.go.jp/ksj/other/agreement.html)"

# Known-good release years per dataset.
DEFAULT_YEAR_C23 = "06"  # Coastline -- the only complete national year.
DEFAULT_YEAR_W05 = "07"  # Rivers -- the oldest complete national year.
DEFAULT_YEAR_L03B = "16"  # LU25 -- 100 m mesh, 12 classes; the latest national release.
DEFAULT_DATUM_L03B = "jgd"  # JGD2011 (world geodetic system).

# 47 prefecture codes.
ALL_PREFECTURES = tuple(f"{i:02d}" for i in range(1, 48))


def _per_prefecture_url(code: str, year: str, prefecture: str) -> str:
    return (
        f"https://nlftp.mlit.go.jp/ksj/gml/data/{code}/{code}-{year}/"
        f"{code}-{year}_{prefecture}_GML.zip"
    )


def _per_mesh_url(code: str, year: str, mesh: str, datum: str) -> str:
    return (
        f"https://nlftp.mlit.go.jp/ksj/gml/data/{code}/{code}-{year}/"
        f"{code}-{year}_{mesh}-{datum}_GML.zip"
    )


def coast_manifest(
    raw_root: Path,
    *,
    year: str = DEFAULT_YEAR_C23,
    prefectures: tuple[str, ...] = ALL_PREFECTURES,
) -> list[DownloadSpec]:
    """C23 海岸線 (per-prefecture).

    The dataset is distributed as 47 small ZIPs (~1 MB each). ``prefectures``
    accepts an explicit list to allow regional fetches.
    """
    return [
        DownloadSpec(
            name=f"mlit_coast_C23_{year}_{pref}",
            url=_per_prefecture_url("C23", year, pref),
            destination=raw_root / "mlit" / f"C23-{year}" / f"C23-{year}_{pref}_GML.zip",
            license=LICENSE,
            method="http",
            notes=(
                "Unzip into data/raw/mlit/C23-<year>/ for ingest.coast.run(). "
                f"Prefecture code {pref}."
            ),
        )
        for pref in prefectures
    ]


def landuse_manifest(
    raw_root: Path,
    *,
    year: str = DEFAULT_YEAR_L03B,
    datum: str = DEFAULT_DATUM_L03B,
    meshes: tuple[str, ...] = (),
) -> list[DownloadSpec]:
    """L03-b 土地利用 LU25 (per primary-mesh).

    ``meshes`` accepts an explicit list of 4-digit primary mesh codes
    (e.g. ``("5339", "5340", "5240")`` for the Kanto plain). Empty
    ``meshes`` returns no specs because the all-Japan inventory is large
    enough that callers should always opt-in to specific meshes.
    """
    if not meshes:
        LOG.info(
            "landuse_manifest called without meshes; pass meshes=(...) to fetch "
            "specific primary-mesh tiles."
        )
    return [
        DownloadSpec(
            name=f"mlit_landuse_L03b_{year}_{mesh}_{datum}",
            url=_per_mesh_url("L03-b", year, mesh, datum),
            destination=raw_root
            / "mlit"
            / f"L03-b-{year}"
            / f"L03-b-{year}_{mesh}-{datum}_GML.zip",
            license=LICENSE,
            method="http",
            notes=(
                f"Primary mesh {mesh}, {datum.upper()} datum. "
                "Unzip and run national.data.ingest.landuse.run()."
            ),
        )
        for mesh in meshes
    ]


def rivers_manifest(
    raw_root: Path,
    *,
    year: str = DEFAULT_YEAR_W05,
    prefectures: tuple[str, ...] = ALL_PREFECTURES,
) -> list[DownloadSpec]:
    """W05 河川 (per-prefecture).

    Default year ``07`` works for every prefecture. Mind that
    ``data/river/class1_rivers_all_japan.geojson`` is already a merged
    filtered output of these files; redownload only if MLIT updates the
    source.
    """
    return [
        DownloadSpec(
            name=f"mlit_rivers_W05_{year}_{pref}",
            url=_per_prefecture_url("W05", year, pref),
            destination=raw_root / "mlit" / f"W05-{year}" / f"W05-{year}_{pref}_GML.zip",
            license=LICENSE,
            method="http",
            notes=(
                "Already filtered into data/river/class1_rivers_all_japan.geojson "
                "for the existing PoC; redownload only on source update. "
                f"Prefecture {pref}."
            ),
        )
        for pref in prefectures
    ]


# Convenience subsets ------------------------------------------------------- #
KANTO_PRIMARY_MESHES = (
    "5339",  # 東京周辺
    "5340",  # 千葉/茨城東部
    "5239",  # 神奈川南
    "5240",  # 房総南
    "5440",  # 茨城北
    "5439",  # 群馬南
    "5438",  # 群馬西
)
"""Primary mesh codes covering the Kanto plain. Used by ``landuse_manifest`` for
a regional bring-up of LU25 before scaling to all-Japan."""


__all__ = [
    "coast_manifest",
    "landuse_manifest",
    "rivers_manifest",
    "DEFAULT_YEAR_C23",
    "DEFAULT_YEAR_L03B",
    "DEFAULT_YEAR_W05",
    "DEFAULT_DATUM_L03B",
    "ALL_PREFECTURES",
    "KANTO_PRIMARY_MESHES",
    "LICENSE",
]
