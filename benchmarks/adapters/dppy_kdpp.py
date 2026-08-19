"""DPPy adapter: exact k-DPP sampling over an RBF likelihood kernel."""

import numpy as np
from numpy.typing import NDArray

from max_div.problem import MaxDivProblem

from ._inputs import problem_vectors
from .base import SelectionAdapter


class DppyKDpp(SelectionAdapter):
    """One exact k-DPP sample as the selection — DPPy's standard usage.

    A DPP is a distribution over diverse subsets, not an optimizer, so the selection is a
    single draw. The likelihood kernel is a materialized n x n RBF matrix with the median
    pairwise distance as its bandwidth — the usual default when no domain kernel exists.
    """

    @property
    def name(self) -> str:
        """Tool name as it appears in records and figures."""
        return "DPPy[k-DPP]"

    def select(self, problem: MaxDivProblem, seed: int) -> NDArray[np.int64]:
        """Draw one exact k-DPP sample, seeded through numpy's RNG."""
        from dppy.finite_dpps import FiniteDPP
        from scipy.spatial.distance import pdist, squareform

        vectors = problem_vectors(problem).astype(np.float64)
        sq_dists = squareform(pdist(vectors, metric="sqeuclidean"))
        bandwidth_sq = float(np.median(sq_dists[sq_dists > 0]))
        likelihood = np.exp(-sq_dists / (2.0 * bandwidth_sq))

        dpp = FiniteDPP("likelihood", L=likelihood)
        rng = np.random.RandomState(seed)
        sample = dpp.sample_exact_k_dpp(size=problem.k, random_state=rng)
        return np.sort(np.asarray(sample, dtype=np.int64))
