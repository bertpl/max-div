from max_div.sampling.con import randint_constrained_robust
from max_div.sampling.uncon import randint_numba
from max_div.solver._solver_state import SolverState

from ._base import InitializationStrategy


class InitDummy(InitializationStrategy):
    """
    Initialize by taking the first 'k' items (indices 0 to k-1).  This strategy is not intended for actual use,
    but mainly as a reference method for testing and benchmarking purposes.
    """

    def initialize(self, state: SolverState):
        for idx in range(state.k):
            state.add(idx)
