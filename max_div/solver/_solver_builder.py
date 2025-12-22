from typing import Self

import numpy as np

from ._constraints import Constraint
from ._distance import DistanceMetric
from ._diversity import DiversityMetric
from ._problem import MaxDivProblem
from ._solver import MaxDivSolver
from ._solver_step import InitializationStep, OptimizationStep, SolverStep
from ._strategies import InitializationStrategy


class MaxDivSolverBuilder:
    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, problem: MaxDivProblem):
        """
        Initialize the MaxDivSolverBuilder.
        """

        # --- problem properties ----------------
        self._vectors: np.ndarray = problem.vectors
        self._k: int = problem.k
        self._distance_metric: DistanceMetric = problem.distance_metric
        self._diversity_metric: DiversityMetric = problem.diversity_metric
        self._constraints: list[Constraint] = problem.constraints

        # --- solver configuration --------------
        self._diversity_tie_breakers: list[DiversityMetric | object] = []
        self._default_diversity_tie_breakers: bool = True
        self._solver_steps: list[SolverStep] = [
            InitializationStep(InitializationStrategy.random_one_shot()),  # Default initialization strategy
        ]
        self._seed = 42

    # -------------------------------------------------------------------------
    #  Builder API
    # -------------------------------------------------------------------------
    def with_diversity_tie_breakers(self, diversity_tie_breakers: list[DiversityMetric]) -> Self:
        self._diversity_tie_breakers = diversity_tie_breakers
        self._default_diversity_tie_breakers = False
        return self

    def with_default_diversity_tie_breakers(self) -> Self:
        self._diversity_tie_breakers = []
        self._default_diversity_tie_breakers = True
        return self

    def set_initialization_strategy(self, init_strategy: InitializationStrategy) -> Self:
        self._solver_steps[0] = InitializationStep(init_strategy)
        return self

    def add_solver_step(self, solver_step: OptimizationStep) -> Self:
        if not isinstance(solver_step, OptimizationStep):
            raise TypeError("Only OptimizationStep instances can be added as solver steps.")
        self._solver_steps.append(solver_step)
        return self

    def add_solver_steps(self, solver_steps: list[OptimizationStep]) -> Self:
        for solver_step in solver_steps:
            self.add_solver_step(solver_step)
        return self

    def with_seed(self, seed: int) -> Self:
        self._seed = seed
        return self

    # -------------------------------------------------------------------------
    #  Build
    # -------------------------------------------------------------------------
    def _determine_diversity_tie_breakers(self) -> list[DiversityMetric]:
        if not self._default_diversity_tie_breakers:
            # custom tie-breakers provided by the user
            return self._diversity_tie_breakers
        else:
            # default tie-breakers based on the main diversity metric
            if self._diversity_metric == DiversityMetric.min_separation():
                return [
                    DiversityMetric.approx_geomean_separation(),
                    DiversityMetric.non_zero_separation_frac(),
                ]
            elif (self._diversity_metric == DiversityMetric.geomean_separation()) or (
                self._diversity_metric == DiversityMetric.approx_geomean_separation()
            ):
                return [DiversityMetric.non_zero_separation_frac()]
            else:
                return []

    def build(self) -> MaxDivSolver:
        return MaxDivSolver(
            vectors=self._vectors,
            k=self._k,
            distance_metric=self._distance_metric,
            diversity_metric=self._diversity_metric,
            diversity_tie_breakers=self._determine_diversity_tie_breakers(),
            constraints=self._constraints,
            solver_steps=self._solver_steps,
            seed=self._seed,
        )
