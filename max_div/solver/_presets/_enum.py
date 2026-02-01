from __future__ import annotations

from enum import StrEnum


class SolverPreset(StrEnum):
    DEFAULT = "default"
    RANDOM = "random"
    GUIDED = "guided"
    SMART = "smart"
    THOROUGH = "thorough"

    def resolve_alias(self) -> SolverPreset:
        if self == SolverPreset.DEFAULT:
            return SolverPreset.SMART
        else:
            return self

    def __lt__(self, other: SolverPreset) -> bool:
        order = {
            SolverPreset.RANDOM: 0,
            SolverPreset.GUIDED: 1,
            SolverPreset.SMART: 2,
            SolverPreset.THOROUGH: 3,
        }
        return str(order.get(self, str(self))) < str(order.get(other, other))
