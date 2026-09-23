import numpy as np
from numpy._typing import NDArray

from max_div._core.solver._solver_state import SolverState

from ._base import InitializationStrategy


class InitFast(InitializationStrategy):
    """Initialize by taking the first 'k' items (indices 0 to k-1).

    Internal, not offered by the `InitializationStrategy` factory: the optimization-strategy benchmark
    starts every run from it, so that only the optimization step is measured, and tests use it as a
    deterministic start.
    """

    def get_next_samples(self, state: SolverState, k_remaining: int | np.int32) -> NDArray[np.int32]:
        return np.arange(k_remaining, dtype=np.int32)
