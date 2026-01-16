import numpy as np

from max_div.internal.benchmarking import Timer
from max_div.internal.formatting import ljust_str_list
from max_div.internal.utils import deterministic_hash
from max_div.random import Constraint

from ._distance import DistanceMetric
from ._diversity import DiversityMetric
from ._duration import Elapsed
from ._progress_reporting import ProgressReporter
from ._solution import MaxDivSolution
from ._solver_state import SolverState
from ._solver_step import SolverStep, SolverStepResult


class MaxDivSolver:
    """
    Class that represents a combination of...
      - a maximum diversity problem (potentially with fairness constraints)
      - a solver configuration for that problem

    The class allows solving the said problem with the said configuration, resulting in a MaxDivSolution object.

    It is STRONGLY recommended to use the MaxDivSolverBuilder class to create instances of this class,
      since it provides convenient defaults, presets and validation of the configuration.
    """

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(
        self,
        vectors: np.ndarray,
        k: int,
        distance_metric: DistanceMetric,
        diversity_metric: DiversityMetric,
        diversity_tie_breakers: list[DiversityMetric],
        constraints: list[Constraint],
        solver_steps: list[SolverStep],
        seed: int = 42,
    ):
        """
        Initialize the MaxDivSolver with the given configuration.

        :param vectors: (n x d ndarray) A set of n vectors in d dimensions.
        :param k: (int) The number of vectors to be selected from the input set ('universe').
        :param distance_metric: (DistanceMetric) The distance metric to use.
        :param diversity_metric: (DiversityMetric) The diversity metric to use.
        :param diversity_tie_breakers: (list[DiversityMetric]) A list of diversity tie-breaker metrics to use.
        :param constraints: (list[Constraint]) A list of m constraints to try to satisfy during solving.
        :param solver_steps: (list[SolverStep]) A list of solver steps to execute,
                                       the first of which needs to be an InitializationStep,
                                       while all latter ones need to be OptimizationSteps.
        :param seed: (int) Random seed for the solver.
        """

        # --- problem description -------------------------
        self._vectors = vectors
        self._k = k
        self._distance_metric = distance_metric
        self._diversity_metric = diversity_metric
        self._constraints = constraints

        # --- solver config -------------------------------
        self._diversity_tie_breakers = diversity_tie_breakers
        self._solver_steps = solver_steps
        self._seed = seed

    # -------------------------------------------------------------------------
    #  API
    # -------------------------------------------------------------------------
    def solve(self, verbosity: int = 10) -> MaxDivSolution:
        """
        Solve the maximum diversity problem with the given configuration.
        :param verbosity: (int) The verbosity level.
                             0 = silent,
                            10 = tqdm progress bar per solver step
                            2x = progress table with iteration count, metrics, elapsed time, ...
                                   20  -->  slowest updates  (spacing increasing with 10%)
                                   21  -->  slower  updates  (spacing increasing with  5%)
                                   22  -->  faster  updates  (spacing increasing with  2%)
                                   23  -->  fastest updates  (spacing increasing with  1%)

                                   25  -->  debug mode       (1% spacing + debug info column)
        :return: A MaxDivSolution object representing the solution found.
        """
        # --- Init ----------------------------------------

        # --- progress reporting ---
        match verbosity:
            case 0:
                progress_reporter = ProgressReporter.silent()
            case 10:
                progress_reporter = ProgressReporter.tqdm()
            case 20 | 21 | 22 | 23:
                progress_reporter = ProgressReporter.tabular(
                    c_slowdown=[1.10, 1.05, 1.02, 1.01][verbosity - 20],
                    debug_info=False,
                )
            case 25:
                # same as 23, but with debug_info enabled
                progress_reporter = ProgressReporter.tabular(
                    c_slowdown=1.01,
                    debug_info=True,
                )
            case _:
                raise ValueError(f"Invalid verbosity level: {verbosity}")

        # --- solver steps ---
        n_steps = len(self._solver_steps)
        step_names = self._get_step_names()  # includes solver state init step (hence length n_steps+1)
        step_seeds = [deterministic_hash((self._seed, i)) for i in range(n_steps)]
        step_results: dict[str, SolverStepResult] = dict()

        # --- solver state ---
        with Timer() as timer:
            progress_reporter.solver_step_started(step_names[0])
            state = SolverState.new(
                vectors=self._vectors,
                k=self._k,
                distance_metric=self._distance_metric,
                diversity_metric=self._diversity_metric,
                diversity_tie_breakers=self._diversity_tie_breakers,
                constraints=self._constraints,
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
            step_results[step_name.strip()] = step.run(state, progress_reporter)

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

    @staticmethod
    def _construct_final_solution(state: SolverState, step_results: dict[str, SolverStepResult]) -> MaxDivSolution:
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

        # --- construct solution --------------------------
        return MaxDivSolution(
            i_selected=state.selected_index_array.copy(),
            score_checkpoints=score_checkpoints,
            step_durations=step_durations,
        )
