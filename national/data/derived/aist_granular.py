"""Granular AIST features: geological era + macro lithology, derived from the
AIST seamless-geology legend strings.

Motivation
----------

The existing :mod:`national.data.derived.lithology` collapses the AIST V2
legend to an 8-way :class:`Regime` (ALLUVIAL / DILUVIAL / VOLCANIC_ASH /
SEDIMENTARY / IGNEOUS / METAMORPHIC / LIMESTONE / UNKNOWN). At Kanto scale
that resolution was adequate because the rare regimes had a handful of rows
each. At national scale, the AIST cache exposes:

* ``formation_age_ja`` -- **92 unique values** (e.g. "新生代 第四紀 完新世",
  "中生代 後期白亜紀 セノマニアン期〜サントニアン期").
* ``lithology_ja`` -- **180 unique values** (e.g. "段丘堆積物", "花崗岩 塊状 島弧・大陸",
  "海成層 砂岩泥岩互層").

These are too granular for direct one-hot encoding into a SVGP input, but the
8-way regime throws away geologically meaningful structure. This module bins
each field to a stable, model-friendly cardinality:

* :class:`AistEra` -- **11-way** geological era (Holocene / Late-Pleistocene /
  Middle-Pleistocene / Early-Pleistocene / Pliocene / Miocene / Paleogene /
  Cretaceous / Pre-Cretaceous / Other / UNKNOWN). Captures the time-since-
  deposition axis that drives consolidation and N-value.
* :class:`AistLithoMacro` -- **15-way** lithology macro group (Alluvial-fan-
  fluvial / Terrace / Marine-sediment / Volcanic-pyroclastic / Volcanic-lava /
  Granitic / Sedimentary-rock / Metamorphic / Limestone / Reclaimed / Loess /
  Accretionary / Other / Mixed / UNKNOWN). Captures depositional-environment
  structure.

Token matching is deliberately permissive (substring + Japanese keyword
catalogue) so that AIST legend variations across DTD versions still resolve.
First-match-wins ordering matches the precedent in
:mod:`national.data.derived.lithology`.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np
import pandas as pd


# ============================================================
# AistEra: 11-way geological era classification
# ============================================================


class AistEra(IntEnum):
    """Geological era buckets for SVGP regime modulation.

    Ordered roughly youngest -> oldest so that one-hot embedding distances
    are monotone in geological time. UNKNOWN is the sentinel for missing /
    unrecognised entries (matches the :class:`Regime.UNKNOWN` convention).
    """

    HOLOCENE = 0           # 完新世 (~12 ka - present)
    LATE_PLEISTOCENE = 1   # 後期更新世
    MIDDLE_PLEISTOCENE = 2 # 中期更新世 / チバニアン期
    EARLY_PLEISTOCENE = 3  # 前期更新世 / ジェラシアン期
    PLIOCENE = 4           # 鮮新世
    MIOCENE = 5            # 中新世
    PALEOGENE = 6          # 古第三紀 (Paleocene/Eocene/Oligocene)
    CRETACEOUS = 7         # 白亜紀
    PRE_CRETACEOUS = 8     # ジュラ紀 / 三畳紀 / 古生代 / 先カンブリア時代
    OTHER = 9              # Recognised AIST text but doesn't fit above
    UNKNOWN = 10           # NaN / empty / unrecognised


# Era keyword catalogue. Each entry is a list of substring tokens; first list
# whose any-token matches wins. Ordering is engineered to handle AIST's
# common multi-stage span strings (e.g. "中新世〜鮮新世", "アルビアン期〜
# 暁新世 セランディアン期"): for spans crossing era boundaries we resolve to
# the OLDER end (the more consolidated stratum), which is the conservative
# call for soil-strength prediction. Two exceptions take precedence:
#
# 1. ``HOLOCENE`` is checked first because (a) it dominates the boring-data
#    distribution (~73 k of 171 k cache rows) and (b) a Holocene component
#    in a span implies the youngest, weakest soil controls engineering
#    behaviour.
# 2. ``LATE_PLEISTOCENE`` is checked next because the substring "後期更新世"
#    must beat the generic "更新世" rule below.
#
# All remaining rules run oldest -> youngest so that mixed-stage spans
# resolve to the older era.
_ERA_RULES: list[tuple[AistEra, tuple[str, ...]]] = [
    # Special-case: Holocene wins any span containing it (dominant + youngest).
    (AistEra.HOLOCENE, ("完新世",)),
    # Special-case: late-Pleistocene before generic Pleistocene (substring trap).
    (AistEra.LATE_PLEISTOCENE, ("後期更新世",)),
    # ------------------------------------------------------------
    # Oldest-first rules below (resolve mixed spans to older era).
    # ------------------------------------------------------------
    # Pre-Cretaceous (Jurassic, Triassic, Paleozoic, Precambrian)
    (
        AistEra.PRE_CRETACEOUS,
        ("ジュラ紀", "三畳紀", "古生代", "先カンブリア時代",
         "ペルム紀", "石炭紀", "デボン紀", "シルル紀", "オルドビス紀", "カンブリア紀",
         "古第二紀"),
    ),
    # Cretaceous
    (
        AistEra.CRETACEOUS,
        ("白亜紀", "セノマニアン期", "チューロニアン期", "コニアシアン期",
         "サントニアン期", "カンパニアン期", "マーストリヒチアン期",
         "アプチアン期", "アルビアン期", "バレミアン期", "オーテリビアン期",
         "バランギニアン期", "ベリアシアン期", "ハウテリビアン期"),
    ),
    # Generic Mesozoic (likely Jurassic/Triassic if not Cretaceous above).
    (AistEra.PRE_CRETACEOUS, ("中生代",)),
    # Paleogene (Paleocene / Eocene / Oligocene)
    (
        AistEra.PALEOGENE,
        ("古第三紀", "暁新世", "始新世", "漸新世",
         "セランディアン期", "ダニアン期", "サネチアン期",
         "イーペル期", "ルテシアン期", "バートン期", "プリアボン期",
         "ルペル期", "シャッティアン期"),
    ),
    # Miocene (Burdigalian / Langhian / Serravallian / Tortonian / Messinian)
    (
        AistEra.MIOCENE,
        ("中新世", "アキタニアン期", "バーディガリアン期", "ランギアン期",
         "セラバリアン期", "トートニアン期", "メッシニアン期"),
    ),
    # Pliocene
    (AistEra.PLIOCENE, ("鮮新世", "ピアセンジアン期", "ザンクリアン期")),
    # Early Pleistocene (Gelasian)
    (AistEra.EARLY_PLEISTOCENE, ("前期更新世", "ジェラシアン期")),
    # Middle Pleistocene (Chibanian)
    (AistEra.MIDDLE_PLEISTOCENE, ("中期更新世", "チバニアン期")),
    # Generic "更新世" (Pleistocene unspecified, none of the specific stages
    # above matched) -> middle as best guess.
    (AistEra.MIDDLE_PLEISTOCENE, ("更新世",)),
]


def era_from_age(formation_age_ja: object) -> AistEra:
    """Resolve an AIST ``formation_age_ja`` string to an :class:`AistEra`.

    Tolerates ``None`` / NaN / empty string -> ``UNKNOWN``. Unrecognised but
    non-empty strings -> ``OTHER``. First-rule-match wins; rule order encodes
    geological-time precedence (younger before older).

    Args:
        formation_age_ja: AIST cache field, e.g. "新生代 第四紀 完新世" or
            ``None`` / float NaN from a pandas left-join miss.

    Returns:
        AistEra integer code.
    """
    if formation_age_ja is None:
        return AistEra.UNKNOWN
    if isinstance(formation_age_ja, float) and formation_age_ja != formation_age_ja:
        return AistEra.UNKNOWN
    text = str(formation_age_ja).strip()
    if not text:
        return AistEra.UNKNOWN
    for era, tokens in _ERA_RULES:
        if any(tok in text for tok in tokens):
            return era
    return AistEra.OTHER


# ============================================================
# AistLithoMacro: 15-way macro lithology classification
# ============================================================


class AistLithoMacro(IntEnum):
    """Macro-lithology buckets for SVGP regime modulation.

    Groups AIST's 180 lithology strings by depositional environment and
    engineering-relevant strength behaviour. UNKNOWN is for missing /
    unrecognised entries.
    """

    ALLUVIAL_FAN_FLUVIAL = 0    # 谷底平野/扇状地/河川/自然堤防/海岸平野堆積物
    TERRACE = 1                  # 段丘堆積物
    MARINE_SEDIMENT = 2          # 海成層 (砂岩/泥岩/砂岩泥岩互層)
    VOLCANIC_PYROCLASTIC = 3     # 大規模火砕流/溶岩・火砕岩 (デイサイト・流紋岩・安山岩)
    VOLCANIC_LAVA = 4            # Pure 溶岩 without pyroclastic component
    GRANITIC = 5                 # 花崗岩 / 花崗閃緑岩 / トーナル岩
    SEDIMENTARY_ROCK = 6         # 一般堆積岩 (砂岩 / 泥岩 / 礫岩 not classified above)
    METAMORPHIC = 7              # 変成岩 (片麻岩 / 片岩 / ホルンフェルス等)
    LIMESTONE = 8                # 石灰岩
    RECLAIMED = 9                # 盛り土 / 埋立地 / 干拓地
    LOESS = 10                   # ローム / 火山灰 / テフラ
    ACCRETIONARY = 11            # 付加体
    BRACKISH_MIXED = 12          # 汽水成層 / 非海成・海成混合
    OTHER = 13                   # Recognised AIST text but doesn't fit above
    UNKNOWN = 14                 # NaN / empty / unrecognised


# Lithology keyword catalogue. First match wins; order is engineering-priority
# (limestone / reclaimed / volcanic-ash are flagged first since they are
# strongly distinct regimes for liquefaction / bearing analysis).
_LITHO_RULES: list[tuple[AistLithoMacro, tuple[str, ...]]] = [
    # Distinct hazard regimes flagged first
    (AistLithoMacro.LIMESTONE, ("石灰岩",)),
    (AistLithoMacro.RECLAIMED, ("盛り土", "埋立地", "干拓地", "人工")),
    (AistLithoMacro.LOESS, ("火山灰", "テフラ", "ローム")),
    # Volcanic - pyroclastic emphasis (because it dominates by row count)
    (
        AistLithoMacro.VOLCANIC_PYROCLASTIC,
        ("火砕", "デイサイト", "流紋岩", "凝灰岩", "凝灰", "安山岩"),
    ),
    # Volcanic - pure lava without pyroclastic above
    (AistLithoMacro.VOLCANIC_LAVA, ("溶岩", "玄武岩")),
    # Granitic family
    (
        AistLithoMacro.GRANITIC,
        ("花崗", "閃緑", "トーナル", "斑れい", "はんれい"),
    ),
    # Terrace before alluvial (more specific)
    (AistLithoMacro.TERRACE, ("段丘",)),
    # Alluvial fan / fluvial / coastal plain
    (
        AistLithoMacro.ALLUVIAL_FAN_FLUVIAL,
        ("谷底平野", "扇状地", "崖錐", "河川", "自然堤防",
         "海岸平野", "三角州", "後背湿地", "氾濫原"),
    ),
    # Brackish / mixed before marine
    (
        AistLithoMacro.BRACKISH_MIXED,
        ("汽水", "海成・非海成混合", "非海成・海成混合"),
    ),
    # Marine sediments
    (AistLithoMacro.MARINE_SEDIMENT, ("海成層", "海成", "海底堆積")),
    # Accretionary complex (must come BEFORE the generic sedimentary rule,
    # since accretionary lithology strings typically embed "砂岩泥岩" tokens).
    (AistLithoMacro.ACCRETIONARY, ("付加体",)),
    # Metamorphic (before generic sedimentary; some metamorphic strings carry
    # "shale-like" tokens that would falsely match sedimentary).
    (
        AistLithoMacro.METAMORPHIC,
        ("片麻", "片岩", "ホルンフェルス", "結晶質", "変成", "ミグマタイト",
         "蛇紋岩", "角閃岩", "エクロジャイト"),
    ),
    # Generic sedimentary rocks (not yet captured)
    (
        AistLithoMacro.SEDIMENTARY_ROCK,
        ("砂岩", "泥岩", "礫岩", "頁岩", "シルト岩", "チャート", "石炭"),
    ),
]


def litho_macro_from_lithology(lithology_ja: object) -> AistLithoMacro:
    """Resolve an AIST ``lithology_ja`` string to an :class:`AistLithoMacro`.

    Tolerates ``None`` / NaN / empty string -> ``UNKNOWN``. Unrecognised but
    non-empty strings -> ``OTHER``. First-rule-match wins; rule order is
    chosen so distinct engineering regimes (LIMESTONE / RECLAIMED / LOESS /
    VOLCANIC) are flagged before falling through to the more generic
    sedimentary / marine / alluvial families.

    Args:
        lithology_ja: AIST cache field, e.g. "海成層 砂岩泥岩互層" or
            ``None`` / float NaN.

    Returns:
        AistLithoMacro integer code.
    """
    if lithology_ja is None:
        return AistLithoMacro.UNKNOWN
    if isinstance(lithology_ja, float) and lithology_ja != lithology_ja:
        return AistLithoMacro.UNKNOWN
    text = str(lithology_ja).strip()
    if not text:
        return AistLithoMacro.UNKNOWN
    for macro, tokens in _LITHO_RULES:
        if any(tok in text for tok in tokens):
            return macro
    return AistLithoMacro.OTHER


# ============================================================
# Vectorised cache-level applicators
# ============================================================


def granular_codes_for_aist_cache(cache_df: pd.DataFrame) -> pd.DataFrame:
    """Apply era + litho_macro binning row-wise over an AIST cache.

    Args:
        cache_df: DataFrame with columns ``formation_age_ja`` and
            ``lithology_ja`` (extra columns ignored). Other AIST cache
            columns from
            :func:`national.data.download.aist_geology.fetch_codes_for_borings`
            (lat/lon/symbol/group_ja) are not required.

    Returns:
        DataFrame with two int16 columns: ``aist_era_code``,
        ``aist_litho_macro_code``. Same row order as input.

    Raises:
        KeyError: if either required column is missing.
    """
    required = {"formation_age_ja", "lithology_ja"}
    missing = required - set(cache_df.columns)
    if missing:
        raise KeyError(f"AIST cache missing columns: {sorted(missing)}")
    era_codes = cache_df["formation_age_ja"].map(
        lambda v: int(era_from_age(v))
    ).to_numpy(dtype=np.int16)
    litho_codes = cache_df["lithology_ja"].map(
        lambda v: int(litho_macro_from_lithology(v))
    ).to_numpy(dtype=np.int16)
    return pd.DataFrame(
        {
            "aist_era_code": era_codes,
            "aist_litho_macro_code": litho_codes,
        },
        index=cache_df.index,
    )


# ============================================================
# Cardinality introspection helpers (for one-hot dim sizing)
# ============================================================


N_ERA_CODES: int = len(AistEra)
"""Total number of :class:`AistEra` codes (including ``UNKNOWN``).

Used by ``BoringDataset`` to size the one-hot encoding. As of writing this
is 11; the value is exposed as a constant so downstream model code does not
have to import the enum just to size a tensor.
"""

N_LITHO_MACRO_CODES: int = len(AistLithoMacro)
"""Total number of :class:`AistLithoMacro` codes (including ``UNKNOWN``).

As of writing this is 15.
"""


__all__ = [
    "AistEra",
    "AistLithoMacro",
    "era_from_age",
    "litho_macro_from_lithology",
    "granular_codes_for_aist_cache",
    "N_ERA_CODES",
    "N_LITHO_MACRO_CODES",
]
