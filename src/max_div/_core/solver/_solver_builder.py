from typing import TYPE_CHECKING, Self

from max_div._core.metrics import DiversityMetric
from max_div._core.problem import MaxDivProblem

from ._constraint_penalty import ConstraintPenalty
from ._distance_storage import (
    DistanceStorage,
    build_distance_store,
    resolve_distance_storage,
    total_physical_memory_bytes,
)
from ._duration import TargetDuration
from ._presets import SolverPreset, get_preset_strategies
from ._solver import MaxDivSolver
from ._solver_config import SolverConfig
from ._solver_step import InitializationStep, OptimizationStep, SolverStep
from ._strategies import InitializationStrategy

if TYPE_CHECKING:
    from max_div._core.constraints import Constraint


class MaxDivSolverBuilder:
    """Builder for configuring and creating [`MaxDivSolver`][max_div.solver.MaxDivSolver] instances.

    Provides a fluent API for setting up initialization and optimization strategies,
    diversity tie-breakers, random seed, and solver presets. The simplest usage is
    to call `with_preset` with a time budget, then `build`.
    """

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, problem: MaxDivProblem) -> None:
        """:param problem: The MaxDivProblem to solve."""
        # --- problem ---------------------------
        self._problem = problem

        # --- problem properties ----------------
        self._n: int = problem.n
        self._k: int = problem.k
        self._diversity_metric: DiversityMetric = problem.diversity_metric
        self._constraints: list[Constraint] = problem.constraints

        # --- solver configuration --------------
        self._diversity_tie_breakers: list[DiversityMetric] = []
        self._default_diversity_tie_breakers: bool = True
        self._solver_steps: list[SolverStep] = [
            InitializationStep(InitializationStrategy.random_one_shot()),  # Default initialization strategy
        ]
        self._seed = 42
        self._constraint_penalty: ConstraintPenalty = ConstraintPenalty.LINEAR
        self._distance_storage: DistanceStorage = DistanceStorage.AUTO

    # -------------------------------------------------------------------------
    #  Builder API
    # -------------------------------------------------------------------------
    def with_diversity_tie_breakers(self, diversity_tie_breakers: list[DiversityMetric]) -> Self:
        """Set custom diversity tie-breaker metrics, overriding the defaults."""
        self._diversity_tie_breakers = diversity_tie_breakers
        self._default_diversity_tie_breakers = False
        return self

    def with_default_diversity_tie_breakers(self) -> Self:
        """Reset to automatically chosen tie-breakers based on the main diversity metric."""
        self._diversity_tie_breakers = []
        self._default_diversity_tie_breakers = True
        return self

    def set_initialization_strategy(self, init_strategy: InitializationStrategy) -> Self:
        """Set the initialization strategy for the first solver step."""
        self._solver_steps[0] = InitializationStep(init_strategy)
        return self

    def add_solver_step(self, solver_step: OptimizationStep) -> Self:
        """Append an optimization step to the solver pipeline."""
        if not isinstance(solver_step, OptimizationStep):
            raise TypeError("Only OptimizationStep instances can be added as solver steps.")
        self._solver_steps.append(solver_step)
        return self

    def add_solver_steps(self, solver_steps: list[OptimizationStep]) -> Self:
        """Append multiple optimization steps to the solver pipeline."""
        for solver_step in solver_steps:
            self.add_solver_step(solver_step)
        return self

    def with_seed(self, seed: int) -> Self:
        """Set the random seed for reproducibility (default: 42)."""
        self._seed = seed
        return self

    def with_constraint_penalty(self, penalty: ConstraintPenalty) -> Self:
        """Set how constraint violations are penalized in the feasibility score (default: LINEAR)."""
        self._constraint_penalty = penalty
        return self

    def with_distance_storage(self, storage: DistanceStorage) -> Self:
        """Set how pairwise distances are stored during search (default: DistanceStorage.AUTO)."""
        self._distance_storage = storage
        return self

    # -------------------------------------------------------------------------
    #  Builder API - PRESETS
    # -------------------------------------------------------------------------
    def with_preset(
        self,
        target_duration: TargetDuration,
        preset: SolverPreset = SolverPreset.DEFAULT,
    ) -> Self:
        """Configure the builder with specified preset settings (overriding any previous settings).

        This sets:
          - Appropriate initialization strategy (most accurate strategy+settings taking est. <5% of total time)
          - Appropriate optimization strategy
          - Default diversity tie-breakers.

        Please make sure to set diversity metric prior to calling this method, as it influences the choices.

        :param target_duration: Target duration for the init+optim phases (either in time or iterations).
                                       --> rule of thumb for #iterations : 10-100x 'k' should be a good starting point.
        :param preset: Preset to use (default: SolverPreset.DEFAULT)
        """
        # --- apply main preset logic -----------
        init_strategy, optim_steps = get_preset_strategies(
            preset=preset,
            target_duration=target_duration,
        )

        # --- configure solver steps ------------
        self._solver_steps = [
            InitializationStep(init_strategy),
            *optim_steps,
        ]

        # --- diversity tie-breakers ---
        self.with_default_diversity_tie_breakers()

        # --- we're done ---
        return self

    # -------------------------------------------------------------------------
    #  Build
    # -------------------------------------------------------------------------
    def _determine_diversity_tie_breakers(self) -> list[DiversityMetric]:
        if not self._default_diversity_tie_breakers:
            # custom tie-breakers provided by the user
            return self._diversity_tie_breakers
        # default tie-breakers based on the main diversity metric
        if self._diversity_metric == DiversityMetric.MIN_SEPARATION:
            return [
                DiversityMetric.APPROX_GEOMEAN_SEPARATION,
                DiversityMetric.NON_ZERO_SEPARATION_FRAC,
            ]
        if (self._diversity_metric == DiversityMetric.GEOMEAN_SEPARATION) or (
            self._diversity_metric == DiversityMetric.APPROX_GEOMEAN_SEPARATION
        ):
            return [DiversityMetric.NON_ZERO_SEPARATION_FRAC]
        return []

    def build(self) -> MaxDivSolver:
        """Build the distance store this configuration calls for, and a solver reading it."""
        resolved, config = self.resolve_storage_and_config()
        return config.build_solver(build_distance_store(self._problem, resolved))

    def resolve_storage_and_config(self) -> tuple[DistanceStorage, SolverConfig]:
        """Return the backend this configuration resolves to, and the solver config over it.

        `build` resolves the backend and builds the store in one call.  Keeping the two apart lets
        a caller produce the distances once and assemble a solver per worker over them, where
        `build` would produce a store per solver.
        """
        resolved = resolve_distance_storage(self._problem, self._distance_storage, total_physical_memory_bytes())
        label = resolved.value + (" (auto)" if self._distance_storage == DistanceStorage.AUTO else "")
        return resolved, SolverConfig(
            n=self._n,
            k=self._k,
            diversity_metric=self._diversity_metric,
            diversity_tie_breakers=self._determine_diversity_tie_breakers(),
            constraints=self._constraints,
            solver_steps=self._solver_steps,
            seed=self._seed,
            constraint_penalty=self._constraint_penalty,
            distance_storage_label=label,
        )
