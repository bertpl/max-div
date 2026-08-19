"""Greedy max-sum insertion baseline: the classical 1/2-approx construction for max-sum diversity."""

import numpy as np
from numpy.typing import NDArray

from max_div.problem import MaxDivProblem

from ._inputs import square_distances
from .base import SelectionAdapter


class GreedyMaxSum(SelectionAdapter):
    """Greedy max-sum insertion (1/2-approx for max-sum).

    Start from the farthest pair, then repeatedly add the item with the largest
    total distance to the current selection.
    """

    @property
    def name(self) -> str:
        """Return the tool name as it appears in records and figures."""
        return "greedy[max-sum]"

    def select(self, problem: MaxDivProblem, seed: int) -> NDArray[np.int64]:
        """Run the greedy insertion (deterministic; seed unused)."""
        distances = square_distances(problem)
        n, k = problem.n, problem.k

        i0, j0 = np.unravel_index(np.argmax(distances), distances.shape)
        selected = [int(i0), int(j0)]
        sum_dist_to_selected = distances[:, i0] + distances[:, j0]
        available = np.ones(n, dtype=bool)
        available[selected] = False

        while len(selected) < k:
            candidate_sums = np.where(available, sum_dist_to_selected, -np.inf)
            nxt = int(np.argmax(candidate_sums))
            selected.append(nxt)
            available[nxt] = False
            sum_dist_to_selected += distances[:, nxt]

        return np.asarray(selected, dtype=np.int64)
