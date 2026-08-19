"""Uniform-random selection: the quality floor every real tool must clear."""

import numpy as np
from numpy.typing import NDArray

from max_div.problem import MaxDivProblem

from .base import SelectionAdapter


class RandomBaseline(SelectionAdapter):
    """Select k items uniformly at random (constraint-oblivious)."""

    @property
    def name(self) -> str:
        """Return the tool name as it appears in records and figures."""
        return "random"

    def select(self, problem: MaxDivProblem, seed: int) -> NDArray[np.int64]:
        """Draw k distinct indices uniformly at random."""
        rng = np.random.default_rng(seed)
        return rng.choice(problem.n, size=problem.k, replace=False).astype(np.int64)
