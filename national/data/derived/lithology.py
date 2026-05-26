"""Map AIST seamless-geology legend payloads to the project ``Regime`` enum.

The AIST V2 legend has ~150 distinct ``symbol`` codes and a few hundred
distinct ``lithology_ja`` strings. We don't need that resolution -- the
foundation model's FiLM block expects an 8-way coarse partition. The
mapping rules below assemble a regime from three input fields (formation
age, rock group, lithology description) so the LUT remains stable even
if AIST changes its symbol vocabulary.

Order of evaluation (first match wins):

1. ``lithology_ja`` contains "石灰岩" -> LIMESTONE.
2. ``lithology_ja`` contains "火山灰" or "テフラ" -> VOLCANIC_ASH.
3. ``formation_age_ja`` contains "完新世" -> ALLUVIAL.
4. ``formation_age_ja`` contains "更新世" (Pleistocene) -> DILUVIAL.
5. ``group_ja`` == "堆積岩" -> SEDIMENTARY.
6. ``group_ja`` == "火成岩" -> IGNEOUS.
7. ``group_ja`` == "変成岩" -> METAMORPHIC.
8. Default -> UNKNOWN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from national.tiling.regime_classifier import Regime


_LIMESTONE_KEYWORDS = ("石灰岩",)
_VOLCANIC_ASH_KEYWORDS = ("火山灰", "テフラ", "ローム")
_ALLUVIAL_TOKEN = "完新世"
_DILUVIAL_TOKEN = "更新世"

# Extended AIST group_ja values seen in the wild beyond the textbook three.
# 付加体 = accretionary complex (deformed sedimentary + volcanic mix; treat as
# sedimentary for our purposes since soil engineering cares about strength).
# 火砕岩 = pyroclastic rocks (volcanic origin -> closer to volcanic ash than
# sedimentary for liquefaction purposes).
# 大陸地殻深部 = deep continental crust (metamorphic in practice).
_GROUP_TO_REGIME: dict[str, Regime] = {
    "堆積岩": Regime.SEDIMENTARY,
    "火成岩": Regime.IGNEOUS,
    "変成岩": Regime.METAMORPHIC,
    "付加体": Regime.SEDIMENTARY,
    "火砕岩": Regime.VOLCANIC_ASH,
    "大陸地殻深部": Regime.METAMORPHIC,
}


def regime_from_legend(
    symbol: str | None,
    formation_age_ja: str | None,
    group_ja: str | None,
    lithology_ja: str | None,
) -> Regime:
    """Resolve one AIST legend tuple to a :class:`Regime`.

    Tolerates ``None`` and ``NaN`` (float) inputs since the upstream
    pandas left-join against the cache emits NaN for unknown locations.
    """

    def _norm(value: object) -> str:
        if value is None:
            return ""
        # pandas yields a float NaN on a missing left-join row; bool(NaN) is True.
        if isinstance(value, float) and value != value:  # NaN check
            return ""
        return str(value).strip()

    sym = _norm(symbol)
    age = _norm(formation_age_ja)
    group = _norm(group_ja)
    lith = _norm(lithology_ja)

    if not sym and not age and not group and not lith:
        return Regime.UNKNOWN

    if any(tok in lith for tok in _LIMESTONE_KEYWORDS):
        return Regime.LIMESTONE
    if any(tok in lith for tok in _VOLCANIC_ASH_KEYWORDS):
        return Regime.VOLCANIC_ASH
    if _ALLUVIAL_TOKEN in age:
        return Regime.ALLUVIAL
    if _DILUVIAL_TOKEN in age:
        return Regime.DILUVIAL
    if group in _GROUP_TO_REGIME:
        return _GROUP_TO_REGIME[group]
    return Regime.UNKNOWN


def regime_codes_for_aist_cache(cache_df: pd.DataFrame) -> np.ndarray:
    """Apply :func:`regime_from_legend` row-wise to an AIST cache DataFrame."""
    required = {"symbol", "formation_age_ja", "group_ja", "lithology_ja"}
    missing = required - set(cache_df.columns)
    if missing:
        raise KeyError(f"AIST cache missing columns: {sorted(missing)}")
    codes = cache_df.apply(
        lambda row: int(
            regime_from_legend(
                row["symbol"],
                row["formation_age_ja"],
                row["group_ja"],
                row["lithology_ja"],
            )
        ),
        axis=1,
    )
    return codes.to_numpy(dtype=np.int16)


__all__ = ["regime_from_legend", "regime_codes_for_aist_cache"]
