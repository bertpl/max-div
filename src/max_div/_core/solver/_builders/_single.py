"""A single-solver builder configures and builds one solver over a problem."""

from functools import partial
from typing import Self

from max_div._core.distance_storage import DistanceStoragePlan, InProcessDataMatrixReader
from max_div._core.problem import MaxDivProblem
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

    # --------------------------------------------------------------------------
    #  Constructor
    # --------------------------------------------------------------------------
    def __init__(self, problem: MaxDivProblem) -> None:
        """Initialize the builder for `problem` with a default initialization strategy."""
        super().__init__(problem)
        self._solver_steps: list[SolverStep] = [
            InitializationStep(InitializationStrategy.random_selection()),
        ]
        # set_initialization_strategy sets this field and with_preset clears it, so build() can tell
        # the user's initialization from a preset's
        self._user_init_strategy: InitializationStrategy | None = None

    # --------------------------------------------------------------------------
    #  Builder API
    # --------------------------------------------------------------------------
    def set_initialization_strategy(self, init_strategy: InitializationStrategy) -> Self:
        """Set the initialization strategy for the first solver step."""
        self._solver_steps[0] = InitializationStep(init_strategy)
        self._user_init_strategy = init_strategy
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

    # --------------------------------------------------------------------------
    #  Builder API - PRESETS
    # --------------------------------------------------------------------------
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

        `with_end_to_end_budget` makes `target_duration` bound the whole solve, setup included.
        """
        self._target_duration = target_duration

        # --- apply main preset logic ------------
        init_strategy, optim_steps = get_preset_strategies(
            preset=preset,
            target_duration=target_duration,
            has_constraints=bool(self._constraints),
        )

        # --- configure solver steps -------------
        self._solver_steps = [
            InitializationStep(init_strategy),
            *optim_steps,
        ]
        self._user_init_strategy = None  # the preset replaced the user's initialization

        # --- diversity tie-breakers -------------
        self.with_default_diversity_tie_breakers()

        # --- we're done -------------------------
        return self

    # --------------------------------------------------------------------------
    #  Build
    # --------------------------------------------------------------------------
    def build(self) -> MaxDivSolver:
        """Return a solver that produces its data matrices when it solves.

        How each distance is stored is decided here, but no data matrix is produced: `solve`
        produces them, so their cost is part of the solve and a large matrix is not held between
        building the solver and running it.

        Raises:
            ValueError: If `with_initial_selection` was used and `set_initialization_strategy` was called
                with no `with_preset` call after it, the storage type is LAZY on a distance-input
                problem, or the data matrices cannot fit in physical memory at all.
        """
        storage_plan, config = self.prepare_storage_and_config()
        return config.build_solver(
            data_matrix_reader_provider=partial(InProcessDataMatrixReader, storage_plan.data_matrix_producers)
        )

    def prepare_storage_and_config(self) -> tuple[DistanceStoragePlan, SolverConfig]:
        """Return the distance storage plan of a single solve, and the solver config over its resolved objectives.

        When the plan checks that the data matrices fit in physical memory, it assumes a single solve,
        which produces the data matrices in this process and does not copy the user's array; a caller
        that publishes the data matrices into shared memory needs a plan that counts that copy.

        Raises:
            ValueError: If `with_initial_selection` was used and `set_initialization_strategy` was called
                with no `with_preset` call after it, the storage type is LAZY on a distance-input
                problem, or the data matrices cannot fit in physical memory at all.
        """
        storage_plan = self._decide_storage_plan(are_adopted_arrays_copied=False)
        return storage_plan, SolverConfig(
            n=self._n,
            k=self._k,
            diversity_objectives=storage_plan.diversity_objectives,
            constraints=self._constraints,
            solver_steps=self._resolve_solver_steps(),
            seed=self._seed,
            constraint_penalty=self._constraint_penalty,
            distance_storage=storage_plan.distance_storage,
            e2e_budget=self._resolve_e2e_budget(),
            intermediate_selections_enabled=self._intermediate_selections_enabled,
        )

    def _resolve_solver_steps(self) -> list[SolverStep]:
        """Return the solver steps to run, the first one replaced if an initialization override is set.

        The override is the strategy that `_resolve_init_strategy_override` returns, if any.
        """
        init_strategy = self._resolve_init_strategy_override(self._hot_start_strategy, self._user_init_strategy)
        if init_strategy is None:
            return self._solver_steps
        else:
            return [InitializationStep(init_strategy), *self._solver_steps[1:]]
