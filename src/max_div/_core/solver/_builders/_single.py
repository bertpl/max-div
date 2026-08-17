"""A single-solver builder configures and builds one solver over a problem."""

from typing import Self

from max_div._core.problem import MaxDivProblem
from max_div._core.solver._distance_storage import DistanceStorage, build_distance_store
from max_div._core.solver._duration import TargetDuration
from max_div._core.solver._presets import SolverPreset, get_preset_strategies
from max_div._core.solver._solver import MaxDivSolver
from max_div._core.solver._solver_config import SolverConfig
from max_div._core.solver._solver_step import InitializationStep, OptimizationStep, SolverStep
from max_div._core.solver._strategies import InitializationStrategy

from ._base import SolverBuilderBase


class MaxDivSolverBuilder(SolverBuilderBase):
    """Builder for configuring and creating [`MaxDivSolver`][max_div.solver.MaxDivSolver] instances.

    Provides a fluent API for setting up initialization and optimization strategies,
    diversity tie-breakers, random seed, and solver presets. The simplest usage is
    to call `with_preset` with a time budget, then `build`.
    """

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self, problem: MaxDivProblem) -> None:
        """Initialize the builder for `problem` with a default initialization strategy."""
        super().__init__(problem)
        self._solver_steps: list[SolverStep] = [
            InitializationStep(InitializationStrategy.random_one_shot()),  # Default initialization strategy
        ]

    # -------------------------------------------------------------------------
    #  Builder API
    # -------------------------------------------------------------------------
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

        Args:
            target_duration: Target duration for the init+optim phases (either in time or iterations).
                --> rule of thumb for #iterations : 10-100x 'k' should be a good starting point.
            preset: Preset to use (default: SolverPreset.DEFAULT)
        """
        # --- apply main preset logic ------------
        init_strategy, optim_steps = get_preset_strategies(
            preset=preset,
            target_duration=target_duration,
        )

        # --- configure solver steps -------------
        self._solver_steps = [
            InitializationStep(init_strategy),
            *optim_steps,
        ]

        # --- diversity tie-breakers -------------
        self.with_default_diversity_tie_breakers()

        # --- we're done -------------------------
        return self

    # -------------------------------------------------------------------------
    #  Build
    # -------------------------------------------------------------------------
    def build(self) -> MaxDivSolver:
        """Build the distance store this configuration calls for, and a solver reading it."""
        resolved, config = self.prepare_storage_and_config()
        return config.build_solver(build_distance_store(self._problem, resolved))

    def prepare_storage_and_config(self) -> tuple[DistanceStorage, SolverConfig]:
        """Return the backend this configuration selects, and the solver config over it.

        `build` selects the backend and builds the store in one call.  Keeping the two apart lets a
        caller produce the distances once and assemble a solver per worker over them, where `build`
        would produce a store per solver.
        """
        resolved, label = self._select_storage()
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
