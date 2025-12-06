import numpy as np

from ._constraints import Constraint
from ._distance import DistanceMetric
from ._diversity import DiversityMetric
from ._solution import MaxDivSolution
from ._solver_state import SolverState
from ._solver_step import SolverStep


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
        distance_metric: DistanceMetric,
        diversity_metric: DiversityMetric,
        selection_size: int,
        constraints: list[Constraint],
        solver_steps: list[SolverStep],
    ):
        """
        Initialize the MaxDivSolver with the given configuration.
        :param vectors: (M x N ndarray) A set of M vectors in N dimensions.
        :param distance_metric: (DistanceMetric) The distance metric to use.
        :param diversity_metric: (str) The diversity metric to use.
        :param selection_size: (int) The number of vectors to be selected from the input set.
        :param constraints: (list) A list of constraints to try to satisfy during solving.
        :param solver_steps: (list) A list of solver steps to execute,
                                       the first of which needs to be an InitializationStep,
                                       while all latter ones need to be OptimizationSteps.
        """

        # --- properties ----------------------------------
        self._vectors = vectors
        self._distance_metric = distance_metric
        self._diversity_metric = diversity_metric
        self._selection_size = selection_size
        self._constraints = constraints
        self._solver_steps = solver_steps

        # --- state ---------------------------------------
        self._state = SolverState.new(
            vectors=vectors,
            target_selection_size=selection_size,
            distance_metric=distance_metric,
            diversity_metric=diversity_metric,
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
        for step_name, step in zip(step_names, self._solver_steps):
            step.run(self._state, step_name)

        # --- Construct result ----------------------------
        return MaxDivSolution(
            i_selected=self._state.selected_index_array.copy(),
        )

    # -------------------------------------------------------------------------
    #  Internal
    # -------------------------------------------------------------------------
    def _get_step_names(self) -> list[str]:
        n_steps = len(self._solver_steps)
        step_names = [f"step {i}/{n_steps} - {s.name}" for i, s in enumerate(self._solver_steps, start=1)]
        max_len = max(len(name) for name in step_names)
        step_names = [name.ljust(max_len + 2) for name in step_names]

        return step_names
