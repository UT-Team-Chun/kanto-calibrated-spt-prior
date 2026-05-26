"""Kanto-prefecture-level leave-one-out spatial validation.

The bounding boxes below are deliberate approximations: they target
the dominant data-bearing area of each prefecture rather than the
political boundary, because (i) we don't ship a polygon source in
this repository and (ii) the foundation model is evaluated on
hold-out *boreholes* which are concentrated near built infrastructure
within each prefecture.

Bounding-box order follows the convention adopted by
:mod:`national.evaluation.leave_region_out`:
``(lat_min, lat_max, lon_min, lon_max)``.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd

from national.evaluation.leave_region_out import leave_region_out_split

KANTO_PREFECTURES: dict[str, tuple[float, float, float, float]] = {
    # (lat_min, lat_max, lon_min, lon_max)
    "tokyo":    (35.50, 35.92, 139.30, 139.95),
    "kanagawa": (35.13, 35.65, 138.93, 139.83),
    "saitama":  (35.75, 36.30, 138.70, 139.92),
    "chiba":    (34.90, 36.10, 139.70, 140.95),
    "ibaraki":  (35.75, 36.95, 139.65, 140.85),
    "tochigi":  (36.20, 37.15, 139.30, 140.30),
    "gunma":    (36.05, 36.95, 138.40, 139.70),
}


def leave_prefecture_out_split(
    df: pd.DataFrame,
    prefectures: list[str] | None = None,
    *,
    lat_column: str = "latitude_deg",
    lon_column: str = "longitude_deg",
) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """Yield ``(pref_name, train_idx, test_idx)`` for each held-out
    Kanto prefecture.

    Args:
        df: DataFrame of borehole rows.
        prefectures: subset of :data:`KANTO_PREFECTURES` keys to
            iterate. Defaults to ``["tokyo", "tochigi", "chiba"]`` as
            the three geomorphologically distinct representatives
            (urban-dense alluvial, mountainous volcanic-influenced,
            coastal Quaternary mix).
        lat_column / lon_column: column names.

    Yields:
        ``(prefecture_name, train_idx, test_idx)`` triples, each from
        the underlying :func:`leave_region_out_split`.
    """
    if prefectures is None:
        prefectures = ["tokyo", "tochigi", "chiba"]
    unknown = [p for p in prefectures if p not in KANTO_PREFECTURES]
    if unknown:
        raise ValueError(
            f"Unknown prefecture(s): {unknown}. Available: "
            f"{sorted(KANTO_PREFECTURES)}"
        )
    chosen = {p: KANTO_PREFECTURES[p] for p in prefectures}
    yield from leave_region_out_split(
        df, regions=chosen, lat_column=lat_column, lon_column=lon_column,
    )


__all__ = ["KANTO_PREFECTURES", "leave_prefecture_out_split"]
