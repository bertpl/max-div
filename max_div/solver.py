from ._core.solver import (
    InitializationStrategy,
    MaxDivSolver,
    MaxDivSolverBuilder,
    OptimizationStrategy,
    SolverPreset,
    TargetDuration,
    TargetIterationCount,
    TargetTimeDuration,
    hours,
    iterations,
    minutes,
    seconds,
)

__all__ = [
    "InitializationStrategy",
    "MaxDivSolver",
    "MaxDivSolverBuilder",
    "OptimizationStrategy",
    "SolverPreset",
    "TargetDuration",
    "TargetIterationCount",
    "TargetTimeDuration",
    "hours",
    "iterations",
    "minutes",
    "seconds",
]


# --- module patching ---------------------------------------------------------
# This ensures that the user sees all re-exported names as belonging to this module, rather than their original
from max_div._core._utils.api import patch_modules  # noqa: E402

patch_modules(globals(), __all__, __name__)
del patch_modules
