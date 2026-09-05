"""Convert a problem into the input form an adapter needs: vectors, or a square distance matrix.

The conversion runs inside the adapter's timed `select`, so it counts toward the tool's measured
time, and it uses scipy, not max-div's own distance code, so a competitor's time neither depends
on nor benefits from max-div's implementation.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import pdist, squareform

from max_div.problem import MaxDivProblem, VectorMaxDivProblem


def problem_vectors(problem: MaxDivProblem) -> NDArray[np.float32]:
    """Return the problem's vectors, failing clearly for distance-only problems."""
    if not isinstance(problem, VectorMaxDivProblem):
        raise TypeError(f"{type(problem).__name__} carries no vectors; this adapter needs vector input.")
    return problem.vectors


def square_distances(problem: MaxDivProblem) -> NDArray[np.float64]:
    """Return the full n x n Euclidean distance matrix (O(n^2) memory; fine at benchmark sizes)."""
    if isinstance(problem, VectorMaxDivProblem):
        return squareform(pdist(problem.vectors.astype(np.float64)))
    return squareform(problem.condensed_distances().astype(np.float64))
