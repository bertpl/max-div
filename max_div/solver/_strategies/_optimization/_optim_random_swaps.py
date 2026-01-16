import numpy as np
from numpy.typing import NDArray

from max_div.random import P_UNIFORM, choice
from max_div.solver._solver_state import SolverState

from ._base import SwapBasedOptimizationStrategy


class OptimRandomSwaps(SwapBasedOptimizationStrategy):
    """
    This optimization strategy simply consists of removing 1 _fully_ random sample from the current solution,
    and replacing it with 1 new _fully_ random sample not currently in the solution.  Problem constraints,
    if present, are fully ignored.

    This strategy is not intended for actual use, but rather as a baseline to compare more advanced strategies against.
    """

    # -------------------------------------------------------------------------
    #  Constructor
    # -------------------------------------------------------------------------
    def __init__(self):
        # don't expose any parameters; this strategy is not intended for actual use.
        super().__init__()

    # -------------------------------------------------------------------------
    #  Implementation
    # -------------------------------------------------------------------------
    def _determine_swap_size(self) -> np.int32:
        return np.int32(1)

    def _samples_to_be_removed(self, state: SolverState, n_to_remove: np.int32) -> NDArray[np.int32]:
        return choice(
            values=state.selected_index_array,
            k=np.int32(1),
            replace=False,
            p=P_UNIFORM,
            rng_state=self._rng_state,
        )

    def _samples_to_be_added(
        self, state: SolverState, n_to_add: np.int32, candidate_samples: NDArray[np.int32]
    ) -> NDArray[np.int32]:
        return choice(
            values=candidate_samples,
            k=np.int32(1),
            replace=False,
            p=P_UNIFORM,
            rng_state=self._rng_state,
        )
