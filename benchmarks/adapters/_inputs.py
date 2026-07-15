"""Input-shape helpers shared by adapters: vectors and square distance matrices."""

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import squareform

from max_div.problem import MaxDivProblem, VectorMaxDivProblem


def problem_vectors(problem: MaxDivProblem) -> NDArray[np.float32]:
    """Return the problem's vectors, failing clearly for distance-only problems."""
    if not isinstance(problem, VectorMaxDivProblem):
        raise TypeError(f"{type(problem).__name__} carries no vectors; this adapter needs vector input.")
    return problem.vectors


def square_distances(problem: MaxDivProblem) -> NDArray[np.float64]:
    """Return the full n x n distance matrix (O(n^2) memory; fine at benchmark sizes)."""
    return squareform(problem.condensed_distances().astype(np.float64))
