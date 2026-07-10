from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy.typing import NDArray

from max_div._core.constraints import Constraint
from max_div._core.metrics import DistanceMetric, DiversityMetric


@dataclass(frozen=True, slots=True)
class MaxDivProblem:
    """Immutable definition of a Maximum Diversity Problem.

    A problem consists of ``n`` vectors in ``d`` dimensions, a target selection size ``k``,
    a distance metric, a diversity metric, and optionally a list of fairness constraints.

    Use the `new` factory method to create instances with validation.
    """

    # --- primary fields ----------------------------------
    vectors: NDArray[np.float32]
    k: int
    distance_metric: DistanceMetric
    diversity_metric: DiversityMetric
    constraints: list[Constraint]

    # --- computed fields --------------------------------
    @property
    def n(self) -> int:
        return self.vectors.shape[0]

    @property
    def d(self) -> int:
        return self.vectors.shape[1]

    @property
    def m(self) -> int:
        return len(self.constraints)

    @property
    def n_constraint_indices(self) -> int:
        return sum([len(con.int_set) for con in self.constraints])

    # --- factory methods ---------------------------------
    @classmethod
    def new(
        cls,
        vectors: np.ndarray,
        k: int,
        distance_metric: DistanceMetric = DistanceMetric.L2_EUCLIDEAN,
        diversity_metric: DiversityMetric = DiversityMetric.GEOMEAN_SEPARATION,
        constraints: list[Constraint] | None = None,
    ) -> Self:
        """Create a new MaxDivProblem with validation.

        :param vectors: 2D numpy array of shape ``(n, d)`` with at least 3 rows.
                        Converted to ``float32`` automatically if needed.
        :param k: Number of vectors to select (must satisfy ``2 <= k <= n``).
        :param distance_metric: Distance metric for pairwise distances.
        :param diversity_metric: Diversity metric to maximize.
        :param constraints: Optional list of fairness constraints.
        """
        # --- validate ----------------
        if vectors.ndim != 2:
            raise ValueError("Vectors must be a 2D numpy array.")
        if vectors.shape[0] < 3:
            raise ValueError("At least 3 vectors are required to formulate a max-div problem.")
        if vectors.shape[1] == 0:
            raise ValueError("Vectors must have at least one dimension.")
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)

        if not (2 <= k <= vectors.shape[0]):
            raise ValueError(f"k must be in range [2, number of vectors (={vectors.shape[0]})]; here: {k}.")

        if constraints is None:
            constraints = []

        # --- build -------------------
        return MaxDivProblem(
            vectors=vectors,
            k=k,
            distance_metric=distance_metric,
            diversity_metric=diversity_metric,
            constraints=constraints,
        )
