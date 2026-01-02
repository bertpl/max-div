import numpy as np
from numpy._typing import NDArray

from max_div.solver._solver_state import SolverState

from ._base import InitializationStrategy


class InitFast(InitializationStrategy):
    """
    Initialize by taking the first 'k' items (indices 0 to k-1).
    This strategy is mainly intended as a baseline method for testing and benchmarking purposes, or for use cases
    where time is very constrained.
    """

    def get_next_samples(self, state: SolverState, k_remaining: int | np.int32) -> NDArray[np.int32]:
        return np.arange(k_remaining, dtype=np.int32)
