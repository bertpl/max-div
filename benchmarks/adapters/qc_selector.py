"""Wrap qc-selector's greedy max-min and max-sum pickers.

Both run on the full precomputed distance matrix and are seeded through the picker's reference
start index (`seed % n`).
"""

import numpy as np
from numpy.typing import NDArray

from max_div.problem import MaxDivProblem

from ._inputs import square_distances
from .base import SelectionAdapter


class QcSelectorMaxMin(SelectionAdapter):
    """Select greedily for max-min via qc-selector (import name ``selector``)."""

    @property
    def name(self) -> str:
        """Tool name as it appears in records and figures."""
        return "qc-selector[MaxMin]"

    def select(self, problem: MaxDivProblem, seed: int) -> NDArray[np.int64]:
        """Run qc-selector's MaxMin."""
        from selector.methods.distance import MaxMin

        distances = square_distances(problem)
        selected = MaxMin(ref_index=seed % problem.n).select(distances, size=problem.k)
        return np.asarray(selected, dtype=np.int64)


class QcSelectorMaxSum(SelectionAdapter):
    """Select greedily for max-sum via qc-selector's classical insertion construction."""

    @property
    def name(self) -> str:
        """Tool name as it appears in records and figures."""
        return "qc-selector[MaxSum]"

    def select(self, problem: MaxDivProblem, seed: int) -> NDArray[np.int64]:
        """Run qc-selector's MaxSum."""
        from selector.methods.distance import MaxSum

        distances = square_distances(problem)
        selected = MaxSum(ref_index=seed % problem.n).select(distances, size=problem.k)
        return np.asarray(selected, dtype=np.int64)
