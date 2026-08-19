"""k-medoids (FasterPAM) adapter: a coverage-style baseline, not a dispersion method."""

import numpy as np
from numpy.typing import NDArray

from max_div.problem import MaxDivProblem

from ._inputs import square_distances
from .base import SelectionAdapter


class KMedoidsFasterPAM(SelectionAdapter):
    """k-medoids clustering via the kmedoids package's FasterPAM (Rust).

    Medoids are cluster centers, i.e. representativeness — included to show how a
    coverage objective behaves under dispersion metrics, not as a dispersion competitor.
    """

    @property
    def name(self) -> str:
        """Return the tool name as it appears in records and figures."""
        return "kmedoids[FasterPAM]"

    def select(self, problem: MaxDivProblem, seed: int) -> NDArray[np.int64]:
        """Run FasterPAM on the full distance matrix and return the k medoids."""
        import kmedoids

        distances = square_distances(problem)
        result = kmedoids.fasterpam(distances, problem.k, random_state=seed)
        return np.asarray(result.medoids, dtype=np.int64)
