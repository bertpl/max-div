"""Solver package: pipeline, strategies, presets, and scoring.

General usage of dimensions:
n: number of initial items to choose from ('universe')
d: dimensionality of the vectors
k: number of items to be selected
m: number of (group) constraints imposed on the problem.
"""

from ._builders import MaxDivSolverBuilder, ParallelMaxDivSolverBuilder
from ._constraint_penalty import ConstraintPenalty
from ._distance_storage import DistanceStorage
from ._duration import TargetDuration, TargetIterationCount, TargetTimeDuration, hours, iterations, minutes, seconds
from ._parallel import ParallelMaxDivSolution, ParallelMaxDivSolver, WorkerConfig, WorkerSummary
from ._presets import SolverPreset
from ._progress_reporting import Verbosity
from ._score import Score
from ._solution import MaxDivSolution
from ._solver import MaxDivSolver
from ._strategies import InitializationStrategy, OptimizationStrategy
