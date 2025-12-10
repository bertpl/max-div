from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy.typing import NDArray

from ._constraints import Constraint, Constraints
from ._distance import DistanceMetric
from ._diversity import DiversityMetric


@dataclass(frozen=True, slots=True)
class MaxDivProblem:
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

    # --- factory methods ---------------------------------
    @classmethod
    def new(
        cls,
        vectors: np.ndarray,
        k: int,
        distance_metric: DistanceMetric = DistanceMetric.L2_EUCLIDEAN,
        diversity_metric: DiversityMetric = DiversityMetric.geomean_separation(),
        constraints: list[Constraint] | Constraints | None = None,
    ) -> Self:
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
        elif isinstance(constraints, Constraints):
            constraints = constraints.all()

        # --- build -------------------
        return MaxDivProblem(
            vectors=vectors,
            k=k,
            distance_metric=distance_metric,
            diversity_metric=diversity_metric,
            constraints=constraints,
        )
