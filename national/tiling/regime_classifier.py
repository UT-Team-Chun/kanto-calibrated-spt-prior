"""Map a surface-geology code to a coarse lithology regime.

The DKL+SVGP head is modulated per regime (FiLM-style). Regimes are a coarser
partition than the ~150-class surface geology so that each regime has enough
data for stable per-regime hyperparameter learning.

Phase A: defines the regime taxonomy. The actual code->regime LUT is loaded in
Phase B once the surface geology ingest script is written.
"""

from __future__ import annotations

from enum import IntEnum
from pathlib import Path


class Regime(IntEnum):
    """Lithology regime labels used to modulate the SVGP head."""

    ALLUVIAL = 0
    DILUVIAL = 1
    VOLCANIC_ASH = 2
    SEDIMENTARY = 3
    IGNEOUS = 4
    METAMORPHIC = 5
    LIMESTONE = 6  # 主に沖縄
    UNKNOWN = 7


def regime_from_geology_code(code: int, lookup: dict[int, int] | None = None) -> Regime:
    """Map a surface-geology integer code to a :class:`Regime`.

    Args:
        code: int code from the surface-geology ingest (see ``codes.json``).
        lookup: optional override; defaults to the bundled lookup table.

    Returns:
        Regime label. Falls back to ``Regime.UNKNOWN`` if not in the LUT.
    """
    if lookup is None:
        raise NotImplementedError(
            "Default geology code lookup is loaded in Phase B from "
            "data/features/geology_codes.json by the surface-geology ingest."
        )
    return Regime(lookup.get(int(code), int(Regime.UNKNOWN)))


def load_lookup(path: Path) -> dict[int, int]:
    """Load the code -> regime lookup JSON."""
    raise NotImplementedError("Implemented in Phase B alongside ingest/geology.py.")


__all__ = ["Regime", "regime_from_geology_code", "load_lookup"]
