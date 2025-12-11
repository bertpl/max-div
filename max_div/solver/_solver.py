import numpy as np

from ..internal.formatting import ljust_str_list
from ._constraints import Constraint
from ._distance import DistanceMetric
from ._diversity import DiversityMetric
from ._duration import Elapsed
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

        # --- state ---------------------------------------
        self._state = SolverState.new(
            vectors=vectors,
            k=k,
            distance_metric=distance_metric,
            diversity_metric=diversity_metric,
            diversity_tie_breakers=diversity_tie_breakers,
            constraints=constraints,
        )

    # -------------------------------------------------------------------------
    #  API
    # -------------------------------------------------------------------------
    def solve(self) -> MaxDivSolution:
        """
        Solve the maximum diversity problem with the given configuration.
        :return: A MaxDivSolution object representing the solution found.
        """
        # --- Init ----------------------------------------
        step_names = self._get_step_names()

        # --- Main loop -----------------------------------
        step_results: dict[str, SolverStepResult] = dict()
        for step_name, step in zip(step_names, self._solver_steps):
            step_results[step_name.strip()] = step.run(self._state, step_name)

        # --- Construct result ----------------------------
        return self._construct_final_solution(step_results)

    # -------------------------------------------------------------------------
    #  Internal
    # -------------------------------------------------------------------------
    def _get_step_names(self) -> list[str]:
        """Return list of numbered step names, left aligned to be of equal length."""
        n_steps = len(self._solver_steps)
        step_names = [f"step {i}/{n_steps} - {s.name()}" for i, s in enumerate(self._solver_steps, start=1)]
        return ljust_str_list(step_names)

    def _construct_final_solution(self, step_results: dict[str, SolverStepResult]) -> MaxDivSolution:
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
            i_selected=self._state.selected_index_array.copy(),
            score_checkpoints=score_checkpoints,
            step_durations=step_durations,
        )
