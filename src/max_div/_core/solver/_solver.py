from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

import numpy as np

from max_div._core._utils import Timer, deterministic_hash
from max_div._core.constraints import Constraint
from max_div._core.constraints.constraints import _np_con_count_satisfied
from max_div._core.metrics import DistanceMetric, DiversityObjective
from max_div._core.metrics._distance import DistanceStore

from ._constraint_penalty import ConstraintPenalty
from ._distance_storage import DistanceStorageTypes
from ._duration import E2eBudget, Elapsed
from ._progress_reporting import ProgressReporter, Verbosity
from ._score_checkpoint import ScoreCheckpoint
from ._solution import MaxDivSolution
from ._solve_timeline import SolveTimeline
from ._solver_state import SolverState
from ._solver_step import REPORTING_BATCH_SECONDS, SolverStep, SolverStepResult
from ._step_identity import SolverStepIdentity

# The solver state initialization is reported and recorded as step 0 under this name.
INIT_STEP_NAME = "Init SolverState"

if TYPE_CHECKING:
    from ._parallel import WorkerCoordinator


class MaxDivSolver:
    """Solver that combines a maximum diversity problem with a solver configuration.

    Use [`MaxDivSolverBuilder`][max_div.solver.MaxDivSolverBuilder] to create instances --
    it provides convenient defaults, presets and validation.
    """

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(
        self,
        n: int,
        stores_by_distance_provider: Callable[[], Mapping[DistanceMetric | None, DistanceStore]],
        k: int,
        diversity_objectives: list[DiversityObjective],
        constraints: list[Constraint],
        solver_steps: list[SolverStep],
        seed: int = 42,
        constraint_penalty: ConstraintPenalty = ConstraintPenalty.LINEAR,
        distance_storage: DistanceStorageTypes = DistanceStorageTypes(),  # noqa: B008 -- frozen, safe as a default
        batch_seconds: float = REPORTING_BATCH_SECONDS,
        e2e_budget: E2eBudget | None = None,
        intermediate_selections_enabled: bool = False,
    ) -> None:
        """Initialize the MaxDivSolver with the given configuration.

        Args:
            n: (int) The number of items in the problem ('universe').
            stores_by_distance_provider: called at the start of each `solve` to obtain the
                distance -> store mapping to read from, so `build` stays lean and the stores are
                built inside `solve`.
            k: (int) The number of items to be selected from the input set ('universe').
            diversity_objectives: the primary objective first, then the tie-breakers, scored in
                that order.
            constraints: (list[Constraint]) A list of m constraints to try to satisfy during solving.
            solver_steps: (list[SolverStep]) A list of solver steps to execute,
                the first of which needs to be an InitializationStep,
                while all latter ones need to be OptimizationSteps.
            seed: (int) Random seed for the solver.
            constraint_penalty: (ConstraintPenalty) How constraint violations are penalized (default: LINEAR).
            distance_storage: (DistanceStorageTypes) Each store's distance and its resolved storage type.
            batch_seconds: (float) Targeted wall-clock size of one optimization batch.
            e2e_budget: (E2eBudget | None) Wall-clock budget for the whole solve — distance
                computation and initialization included; each optimization step receives whatever
                remains.
                An unstarted budget starts counting when `solve` starts; the parallel solver
                hands its workers a budget already counting from its own solve start.
            intermediate_selections_enabled: (bool) Whether every score checkpoint also carries the
                selection held at that moment, at k integers per checkpoint (default: False).
        """
        # --- problem description ----------------
        self._n = n
        self._stores_by_distance_provider = stores_by_distance_provider
        self._distance_storage = distance_storage
        self._k = k
        self._diversity_objectives = diversity_objectives
        self._constraints = constraints

        # --- solver config ----------------------
        self._solver_steps = solver_steps
        self._seed = seed
        self._constraint_penalty = constraint_penalty
        self._batch_seconds = batch_seconds
        self._e2e_budget = e2e_budget
        self._intermediate_selections_enabled = intermediate_selections_enabled

    # -------------------------------------------------------------------------
    #  API
    # -------------------------------------------------------------------------
    def solve(
        self,
        verbosity: int | Verbosity = Verbosity.PROGRESS_BAR,
        coordinator: "WorkerCoordinator | None" = None,
        *,
        progress_reporter: ProgressReporter | None = None,
    ) -> MaxDivSolution:
        """Solve the maximum diversity problem with the given configuration.

        Args:
            verbosity: (int | Verbosity) The verbosity level, as a `Verbosity` member or its
                plain integer value; see `Verbosity` for the levels.
            coordinator: a `WorkerCoordinator` the solver calls at each batch boundary.
            progress_reporter: a ready-made reporter to report into, overriding `verbosity`; this
                is how a parallel worker reports to its parent process.

        Returns:
            A MaxDivSolution object representing the solution found.
        """
        # --- Init -------------------------------
        solve_timeline = SolveTimeline()  # the solve-wide time axis starts here
        e2e_budget = self._e2e_budget.started() if self._e2e_budget else None
        for step in self._solver_steps:
            step.set_e2e_budget(e2e_budget)

        # --- progress reporting -----------------
        if progress_reporter is None:
            progress_reporter = ProgressReporter.from_verbosity(verbosity)

        # --- solver steps -----------------------
        n_steps = len(self._solver_steps)
        progress_reporter.set_step_count(n_steps + 1)  # the solver state initialization is reported too, as step 0
        step_seeds = [deterministic_hash((self._seed, i)) for i in range(n_steps)]

        # --- solver state -----------------------
        init_step_identity = SolverStepIdentity(0, INIT_STEP_NAME)
        solve_timeline.record_step_start()  # the solver state initialization is the timeline's step 0
        with Timer() as timer:
            progress_reporter.solver_step_started(init_step_identity)
            stores_by_distance = self._stores_by_distance_provider()
            state = SolverState.new(
                n=self._n,
                stores_by_distance=stores_by_distance,
                k=self._k,
                diversity_objectives=self._diversity_objectives,
                constraints=self._constraints,
                penalty_quadratic=(self._constraint_penalty == ConstraintPenalty.QUADRATIC),
            )
            if self._k == self._n:
                # k == n forces every item into the selection; adopt it here so the solve can skip
                # every solver step below -- no strategy ever sees the degenerate case, and the
                # solve returns immediately (spending the budget would only propose swaps that
                # cannot change a full selection).
                state.add_many(np.arange(self._n, dtype=np.int32))
            progress_reporter.solver_step_finished(None, state)

        solve_timeline.record_step_result(
            SolverStepResult(
                score_checkpoints=[
                    ScoreCheckpoint.new(
                        init_step_identity,
                        Elapsed(t_elapsed_sec=timer.t_elapsed_sec(), n_iterations=0),
                        state,
                        coordinator,
                        includes_selection=self._intermediate_selections_enabled,
                    )
                ]
            )
        )

        # --- forced full selection --------------
        if self._k == self._n:
            return self._construct_final_solution(state, solve_timeline)

        # --- Main loop --------------------------
        for step_index, (step_seed, step) in enumerate(zip(step_seeds, self._solver_steps), start=1):
            step_identity = SolverStepIdentity(step_index, step.name())
            progress_reporter.solver_step_started(step_identity)
            step.set_seed(step_seed)
            # `elapsed_before_step` is where the step starts on the solve-wide axis: the step passes it to the
            # coordinator, and the solve timeline shifts the step's own checkpoints onto that axis by the same amount
            elapsed_before_step = solve_timeline.record_step_start()
            try:
                step_result = step.run(
                    state,
                    step_identity,
                    progress_reporter,
                    coordinator,
                    self._batch_seconds,
                    elapsed_before_step=elapsed_before_step,
                    intermediate_selections_enabled=self._intermediate_selections_enabled,
                )
                solve_timeline.record_step_result(step_result)
            finally:
                # release all Savepoint objects: they hold cyclic references via the SolverState, which
                # cause out-of-memory when left in place; in a finally, so a step that raises still
                # releases them
                state.release_savepoints()

        # --- Construct result -------------------
        return self._construct_final_solution(state, solve_timeline)

    # -------------------------------------------------------------------------
    #  Internal
    # -------------------------------------------------------------------------
    def _construct_final_solution(self, state: SolverState, solve_timeline: SolveTimeline) -> MaxDivSolution:
        """Construct the final MaxDivSolution from the state and the solve timeline."""
        # --- constraint satisfaction ------------
        n_constraints = state.m
        n_constraints_satisfied = _np_con_count_satisfied(state.con_values)

        # --- construct solution -----------------
        return MaxDivSolution(
            i_selected=state.selected_index_array.copy(),
            score_checkpoints=solve_timeline.checkpoints,
            step_durations=solve_timeline.step_durations,
            diversity_objective_labels=[objective.label for objective in self._diversity_objectives],
            n_constraints=int(n_constraints),
            n_constraints_satisfied=n_constraints_satisfied,
            distance_storage=self._distance_storage,
        )
