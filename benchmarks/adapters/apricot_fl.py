"""apricot facility-location adapter: submodular selection, a deliberately different objective."""

import numpy as np
from numpy.typing import NDArray

from max_div.problem import MaxDivProblem

from ._inputs import problem_vectors
from .base import SelectionAdapter


class ApricotFacilityLocation(SelectionAdapter):
    """Submodular facility-location selection via apricot (1-1/e guarantee for its own objective).

    Facility location optimizes coverage/representativeness, not dispersion — results pages
    must label it as a different-objective reference, not a like-for-like competitor.
    """

    @property
    def name(self) -> str:
        """Tool name as it appears in records and figures."""
        return "apricot[facility-location]"

    def select(self, problem: MaxDivProblem, seed: int) -> NDArray[np.int64]:
        """Run apricot's greedy facility-location selection (deterministic; seed unused)."""
        from apricot import FacilityLocationSelection

        vectors = problem_vectors(problem).astype(np.float64)
        selector = FacilityLocationSelection(problem.k, metric="euclidean")
        selector.fit(vectors)
        return np.asarray(selector.ranking, dtype=np.int64)
