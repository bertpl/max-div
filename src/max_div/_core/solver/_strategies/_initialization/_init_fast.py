import numpy as np
from numpy._typing import NDArray

from max_div._core.solver._solver_state import SolverState

from ._base import InitializationStrategy


class InitFast(InitializationStrategy):
    """Initialize by taking the first 'k' items (indices 0 to k-1).

    This strategy is internal: the `InitializationStrategy` factory methods do not offer it. It is
    deterministic and effectively free, so a solve that starts from it spends its time only on the
    steps after it.
    """

    def get_next_samples(self, state: SolverState, k_remaining: int | np.int32) -> NDArray[np.int32]:
        return np.arange(k_remaining, dtype=np.int32)
