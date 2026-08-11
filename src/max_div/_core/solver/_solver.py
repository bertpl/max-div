from typing import TYPE_CHECKING

from max_div._core._utils import Timer, deterministic_hash, ljust_str_list
from max_div._core.constraints import Constraint
from max_div._core.constraints._constraints import _np_con_count_satisfied
from max_div._core.metrics import DiversityMetric
from max_div._core.metrics._distance import DistanceStore

from ._constraint_penalty import ConstraintPenalty
from ._duration import Elapsed
from ._progress_reporting import ProgressReporter, Verbosity
from ._solution import MaxDivSolution
from ._solver_state import SolverState
from ._solver_step import SolverStep, SolverStepResult

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
        store: DistanceStore,
        k: int,
        diversity_metric: DiversityMetric,
        diversity_tie_breakers: list[DiversityMetric],
        constraints: list[Constraint],
        solver_steps: list[SolverStep],
        seed: int = 42,
        constraint_penalty: ConstraintPenalty = ConstraintPenalty.LINEAR,
        distance_storage_label: str = "",
    ) -> None:
        """Initialize the MaxDivSolver with the given configuration.

        :param n: (int) The number of items in the problem ('universe').
        :param store: (DistanceStore) Pairwise-distance storage the solver reads from.
        :param k: (int) The number of items to be selected from the input set ('universe').
        :param diversity_metric: (DiversityMetric) The diversity metric to use.
        :param diversity_tie_breakers: (list[DiversityMetric]) A list of diversity tie-breaker metrics to use.
        :param constraints: (list[Constraint]) A list of m constraints to try to satisfy during solving.
        :param solver_steps: (list[SolverStep]) A list of solver steps to execute,
                                       the first of which needs to be an InitializationStep,
                                       while all latter ones need to be OptimizationSteps.
        :param seed: (int) Random seed for the solver.
        :param constraint_penalty: (ConstraintPenalty) How constraint violations are penalized (default: LINEAR).
        :param distance_storage_label: (str) Resolved distance-storage backend, reported in the solution summary.
        """
        # --- problem description -------------------------
        self._n = n
        self._store = store
        self._distance_storage_label = distance_storage_label
        self._k = k
        self._diversity_metric = diversity_metric
        self._constraints = constraints

        # --- solver config -------------------------------
        self._diversity_tie_breakers = diversity_tie_breakers
        self._solver_steps = solver_steps
        self._seed = seed
        self._constraint_penalty = constraint_penalty

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

        :param verbosity: (int | Verbosity) The verbosity level, as a `Verbosity` member or its
                          plain integer value; see `Verbosity` for the levels.
        :param coordinator: a `WorkerCoordinator` the solver calls at each batch boundary.
        :param progress_reporter: a ready-made reporter to report into, overriding `verbosity`; this
                                  is how a portfolio worker reports to its parent process.
        :return: A MaxDivSolution object representing the solution found.
        """
        # --- Init ----------------------------------------

        # --- progress reporting ---
        if progress_reporter is None:
            progress_reporter = ProgressReporter.from_verbosity(verbosity)

        # --- solver steps ---
        n_steps = len(self._solver_steps)
        step_names = self._get_step_names()  # includes solver state init step (hence length n_steps+1)
        step_seeds = [deterministic_hash((self._seed, i)) for i in range(n_steps)]
        step_results: dict[str, SolverStepResult] = {}

        # --- solver state ---
        with Timer() as timer:
            progress_reporter.solver_step_started(step_names[0])
            state = SolverState.new(
                n=self._n,
                store=self._store,
                k=self._k,
                diversity_metric=self._diversity_metric,
                diversity_tie_breakers=self._diversity_tie_breakers,
                constraints=self._constraints,
                penalty_quadratic=(self._constraint_penalty == ConstraintPenalty.QUADRATIC),
            )
            progress_reporter.solver_step_finished(None, state)

        # init step results with solver state initialization as virtual step 0
        step_results[step_names[0].strip()] = SolverStepResult(
            score_checkpoints=[
                (
                    Elapsed(t_elapsed_sec=timer.t_elapsed_sec(), n_iterations=0),
                    state.score,
                )
            ]
        )

        # --- Main loop -----------------------------------
        for step_name, step_seed, step in zip(step_names[1:], step_seeds, self._solver_steps):
            progress_reporter.solver_step_started(step_name)
            step.set_seed(step_seed)
            step_results[step_name.strip()] = step.run(state, progress_reporter, coordinator)

        # --- Construct result ----------------------------
        return self._construct_final_solution(state, step_results)

    # -------------------------------------------------------------------------
    #  Internal
    # -------------------------------------------------------------------------
    def _get_step_names(self) -> list[str]:
        """Return list of numbered step names, left aligned to be of equal length."""
        names = ["Init SolverState"] + [s.name() for s in self._solver_steps]
        n_steps = len(self._solver_steps)
        return ljust_str_list([f"step {i}/{n_steps} - {name}" for i, name in enumerate(names)])

    def _construct_final_solution(
        self, state: SolverState, step_results: dict[str, SolverStepResult]
    ) -> MaxDivSolution:
        """Construct the final MaxDivSolution from the current state & step results."""
        # --- collect step durations --------------------
        step_durations = {step_name: result.elapsed for step_name, result in step_results.items()}

        # --- aggregate score checkpoints -----------------
        score_checkpoints = []
        elapsed_from_previous_steps = Elapsed(t_elapsed_sec=0.0, n_iterations=0)
        for step_name, result in step_results.items():
            for elapsed, score in result.score_checkpoints:
                score_checkpoints.append(
                    (
                        step_name,
                        elapsed_from_previous_steps + elapsed,
                        score,
                    )
                )

            # Update elapsed_from_previous_steps to include this step's total elapsed time
            elapsed_from_previous_steps += result.elapsed

        # --- constraint satisfaction ----------------------
        n_constraints = state.m
        n_constraints_satisfied = _np_con_count_satisfied(state.con_values)

        # --- construct solution --------------------------
        return MaxDivSolution(
            i_selected=state.selected_index_array.copy(),
            score_checkpoints=score_checkpoints,
            step_durations=step_durations,
            n_constraints=int(n_constraints),
            n_constraints_satisfied=n_constraints_satisfied,
            distance_storage=self._distance_storage_label,
        )
