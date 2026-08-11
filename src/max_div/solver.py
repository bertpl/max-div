"""Public API for building and running Maximum Diversity solvers."""

from ._core._warnings import ParallelSolvingWarning
from ._core.solver import (
    ConstraintPenalty,
    DistanceStorage,
    InitializationStrategy,
    MaxDivSolver,
    MaxDivSolverBuilder,
    OptimizationStrategy,
    ParallelMaxDivSolution,
    ParallelMaxDivSolver,
    ParallelMaxDivSolverBuilder,
    SolverPreset,
    TargetDuration,
    TargetIterationCount,
    TargetTimeDuration,
    Verbosity,
    WorkerConfig,
    WorkerSummary,
    hours,
    iterations,
    minutes,
    seconds,
)

__all__ = [
    "ConstraintPenalty",
    "DistanceStorage",
    "InitializationStrategy",
    "MaxDivSolver",
    "MaxDivSolverBuilder",
    "OptimizationStrategy",
    "ParallelMaxDivSolution",
    "ParallelMaxDivSolver",
    "ParallelMaxDivSolverBuilder",
    "ParallelSolvingWarning",
    "SolverPreset",
    "TargetDuration",
    "TargetIterationCount",
    "TargetTimeDuration",
    "Verbosity",
    "WorkerConfig",
    "WorkerSummary",
    "hours",
    "iterations",
    "minutes",
    "seconds",
]


# --- module patching ---------------------------------------------------------
# This ensures that the user sees all re-exported names as belonging to this module, rather than their original
from max_div._core._utils.api import patch_modules

patch_modules(globals(), __all__, __name__)
del patch_modules
