from __future__ import annotations

from enum import StrEnum


class SolverPreset(StrEnum):
    """Predefined solver configurations that bundle initialization and optimization strategies.

    Members
    -------

        - DEFAULT:   Alias for SMART.
        - RANDOM:    Random initialization + random swap optimization. Fast but lower quality.
        - GUIDED:    Distance-guided swaps, biased towards removing low-separation and adding high-separation vectors.
        - SMART:     Adaptive swap-based optimization that learns effective swap sizes and candidate strategies.
        - THOROUGH:  Like SMART but with wider parameter ranges. Best for long runs (minutes to hours).
    """

    DEFAULT = "default"
    RANDOM = "random"
    GUIDED = "guided"
    SMART = "smart"
    THOROUGH = "thorough"

    def resolve_alias(self) -> SolverPreset:
        if self == SolverPreset.DEFAULT:
            return SolverPreset.SMART
        return self

    def __lt__(self, other: SolverPreset) -> bool:
        order = {
            SolverPreset.RANDOM: 0,
            SolverPreset.GUIDED: 1,
            SolverPreset.SMART: 2,
            SolverPreset.THOROUGH: 3,
        }
        return str(order.get(self, str(self))) < str(order.get(other, other))

    @classmethod
    def all_sorted(cls) -> list[SolverPreset]:
        """Return list of unique (as in 'resolve_alias'), sorted presets."""
        return sorted({p.resolve_alias() for p in SolverPreset})
