"""RDKit MaxMinPicker adapter: lazy greedy max-min with a custom distance callback."""

import numpy as np
from numpy.typing import NDArray

from max_div.problem import MaxDivProblem

from ._inputs import problem_vectors
from .base import SelectionAdapter


class RdkitMaxMin(SelectionAdapter):
    """Greedy max-min picking via RDKit's MaxMinPicker (lazy distance evaluation).

    The distance callback re-implements the problem's L2 metric on the problem vectors;
    RDKit's laziness means only the distances the greedy actually needs get computed —
    that is the tool's designed trade-off and is measured as-is.
    """

    @property
    def name(self) -> str:
        """Tool name as it appears in records and figures."""
        return "RDKit[MaxMinPicker]"

    def select(self, problem: MaxDivProblem, seed: int) -> NDArray[np.int64]:
        """Run LazyPick with a euclidean-distance callback over the problem vectors."""
        from rdkit.SimDivFilters import rdSimDivPickers

        vectors = problem_vectors(problem)

        def dist(i: int, j: int) -> float:
            return float(np.linalg.norm(vectors[i] - vectors[j]))

        picker = rdSimDivPickers.MaxMinPicker()
        picks = picker.LazyPick(dist, problem.n, problem.k, seed=seed)
        return np.asarray(list(picks), dtype=np.int64)
