import numpy as np

from ._constraints import Constraint
from ._distance import DistanceMetric
from ._diversity import DiversityMetric
from ._solution import MaxDivSolution
from ._solver_state import SolverState
from ._strategies import SolverStrategy


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
        strategies: list[SolverStrategy],
    ):
        """
        Initialize the MaxDivSolver with the given configuration.
        :param vectors: (M x N ndarray) A set of M vectors in N dimensions.
        :param distance_metric: (DistanceMetric) The distance metric to use.
        :param diversity_metric: (str) The diversity metric to use.
        :param selection_size: (int) The number of vectors to be selected from the input set.
        :param constraints: (list) A list of constraints to try to satisfy during solving.
        :param strategies: (list) A list of solver strategies to use.
        """

        # --- properties ----------------------------------
        self._vectors = vectors
        self._distance_metric = distance_metric
        self._diversity_metric = diversity_metric
        self._selection_size = selection_size
        self._constraints = constraints
        self._strategies = strategies

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
        pass
