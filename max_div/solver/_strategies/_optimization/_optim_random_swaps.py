import numpy as np
from numpy.typing import NDArray

from max_div.sampling import randint_numba
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
    def _samples_to_be_removed(self, state: SolverState, n: np.int32, seed: np.int64) -> NDArray[np.int32]:
        n_selected = state.selected_index_array.shape[0]
        i_to_remove = randint_numba(n_selected, np.int32(1), False, seed=seed)
        return state.selected_index_array[i_to_remove]

    def _samples_to_be_added(
        self, state: SolverState, n: np.int32, samples_just_removed: NDArray[np.int32], seed: np.int64
    ) -> NDArray[np.int32]:
        n_not_selected = state.not_selected_index_array.shape[0]
        i_to_add = randint_numba(n_not_selected, np.int32(1), False, seed=seed)
        return state.not_selected_index_array[i_to_add]
