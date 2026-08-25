"""qc-selector MaxMin adapter (GPL-3 tool; lives in the opt-in benchmarks-gpl group)."""

import numpy as np
from numpy.typing import NDArray

from max_div.problem import MaxDivProblem

from ._inputs import square_distances
from .base import SelectionAdapter


class QcSelectorMaxMin(SelectionAdapter):
    """Greedy max-min selection via qc-selector (import name ``selector``).

    GPL-3-licensed: installed only via the opt-in ``benchmarks-gpl`` dependency group;
    scenarios must treat this adapter as optional and skip it when the import fails.
    """

    @property
    def name(self) -> str:
        """Tool name as it appears in records and figures."""
        return "qc-selector[MaxMin]"

    def select(self, problem: MaxDivProblem, seed: int) -> NDArray[np.int64]:
        """Run qc-selector's MaxMin on the full distance matrix, seeding via the reference start sample."""
        from selector.methods.distance import MaxMin

        distances = square_distances(problem)
        selected = MaxMin(ref_index=seed % problem.n).select(distances, size=problem.k)
        return np.asarray(selected, dtype=np.int64)
