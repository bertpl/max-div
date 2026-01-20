from enum import StrEnum


class SolverPreset(StrEnum):
    DEFAULT = "default"  # currently an alias for SMART
    RANDOM = "random"
    GUIDED = "guided"
    SMART = "smart"
    THOROUGH = "thorough"
