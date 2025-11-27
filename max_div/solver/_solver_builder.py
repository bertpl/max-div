from typing import Self

import numpy as np

from ._constraints import Constraint
from ._distance import DistanceMetric
from ._diversity import DiversityMetric
from ._solver import MaxDivSolver
from ._strategies import SolverStrategy, StrategyType


class MaxDivSolverBuilder:
    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self):
        """
        Initialize the MaxDivSolverBuilder.
        """
        self._vectors: np.ndarray | None = None
        self._distance_metric: DistanceMetric = DistanceMetric.L2_EUCLIDEAN
        self._diversity_metric: DiversityMetric = DiversityMetric.GEOMEAN_SEPARATION
        self._selection_size: int | None = None
        self._constraints: list[Constraint] = []
        self._strategies: list[SolverStrategy] = [
            SolverStrategy.init_random(),  # Default initialization strategy
        ]

    # -------------------------------------------------------------------------
    #  Builder API
    # -------------------------------------------------------------------------
    def with_vectors(self, vectors: np.ndarray) -> Self:
        if vectors.ndim != 2:
            raise ValueError("Vectors must be a 2D numpy array.")
        if vectors.shape[0] < 2:
            raise ValueError("At least two vectors are required to compute diversity.")
        if vectors.shape[1] == 0:
            raise ValueError("Vectors must have at least one dimension.")
        if vectors.dtype != np.float32:
            raise ValueError("Vectors must be of type np.float32.")
        self._vectors = vectors
        return self

    def with_distance_metric(self, distance_metric: DistanceMetric) -> Self:
        self._distance_metric = distance_metric
        return self

    def with_diversity_metric(self, diversity_metric: DiversityMetric) -> Self:
        self._diversity_metric = diversity_metric
        return self

    def with_selection_size(self, selection_size: int) -> Self:
        if selection_size < 2:
            raise ValueError("selection_size must be at least 2.")
        self._selection_size = selection_size
        return self

    def set_initialization_strategy(self, strategy: SolverStrategy) -> Self:
        if strategy.type != StrategyType.INITIALIZATION:
            raise ValueError("The provided strategy is not an initialization strategy.")
        self._strategies[0] = strategy
        return self

    def add_optimization_strategy(self, strategy: SolverStrategy) -> Self:
        if strategy.type != StrategyType.OPTIMIZATION:
            raise ValueError("The provided strategy is not an optimization strategy.")
        self._strategies.append(strategy)
        return self

    def add_optimization_strategies(self, strategies: list[SolverStrategy]) -> Self:
        for strategy in strategies:
            self.add_optimization_strategy(strategy)
        return self

    def with_constraint(self, constraint: Constraint) -> Self:
        self._constraints.append(constraint)
        return self

    def with_constraints(self, constraints: list[Constraint]) -> Self:
        for con in constraints:
            self.with_constraint(con)
        return self

    # -------------------------------------------------------------------------
    #  Build
    # -------------------------------------------------------------------------
    def _is_buildable(self) -> tuple[bool, str]:
        if self._vectors is None:
            return False, "with_vectors() must be called before build()."
        if self._selection_size is None:
            return False, "with_selection_size() must be called before build()."
        if self._selection_size > self._vectors.shape[0]:
            return False, "selection_size cannot be greater than the number of available vectors."
        return True, ""

    def build(self) -> MaxDivSolver:
        ok, msg = self._is_buildable()
        if not ok:
            raise ValueError(f"Cannot build MaxDivSolver: {msg}")
        return MaxDivSolver(
            vectors=self._vectors,
            distance_metric=self._distance_metric,
            diversity_metric=self._diversity_metric,
            selection_size=self._selection_size,
            constraints=self._constraints,
            strategies=self._strategies,
        )
