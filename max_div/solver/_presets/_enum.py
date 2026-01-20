from enum import StrEnum


class SolverPreset(StrEnum):
    DEFAULT = "default"  # currently an alias for SMART
    GUIDED = "guided"
    SMART = "smart"
