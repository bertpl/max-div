from max_div.sampling.con import randint_constrained_robust
from max_div.sampling.uncon import randint_numba
from max_div.solver._solver_state import SolverState

from ._base import InitializationStrategy


class InitRandomOneShot(InitializationStrategy):
    """
    Initialize by taking a single (hence: one-shot) random sample of k items.  This is among the fastest
    initialization strategies, but potentially also with the lowest quality.

    Suggested use: if time constraints are severe or problem dimensions `n` or `k` are very large.

    Parameters:
    - constrained (bool): If `True`, respects problem constraints during initialization.
                          If `False`, constraints are ignored. (default: `True`)
    - uniform (bool): If `True`, samples uniformly at random.
                      If `False`, uses vector separations as sampling weights, sampling well-separated vectors
                                         with higher probability (default: `False`)

    Notes:
        - using separation as sampling weights is a heuristic, not an exactly optimal solution, with known limitations:
            - in 1D problems this heuristic should be probabilistically optimal, but in higher dimensions (the more
              likely scenario) it is not.  E.g. in 2D where vectors have half the separation as in other regions, we
              should sample 4x fewer, not 2x fewer vectors.
            - when multiple vectors (e.g. 5) are identical and hence have 0 separation, we will not sample any of them
              (unless k is high enough), while optimal solutions might in fact contain exactly 1 of them.

    Time Complexity:
       - without constraints: ~O(n)
       - with constraints:    ~O(kn)
    """

    def __init__(self, constrained: bool = True, uniform: bool = False):
        super().__init__()
        self.constrained = constrained
        self.uniform = uniform

    def initialize(self, state: SolverState):
        # --- sample --------------------------------------
        if self.constrained and state.has_constraints:
            # take constraints into account
            if self.uniform:
                samples = randint_constrained_robust(
                    n=state.n,
                    k=state.k,
                    con_values=state.con_values,
                    con_indices=state.con_indices,
                    seed=self.seed,
                )
            else:
                samples = randint_constrained_robust(
                    n=state.n,
                    k=state.k,
                    con_values=state.con_values,
                    con_indices=state.con_indices,
                    p=state.global_separation_array,
                    seed=self.seed,
                )
        else:
            # don't take constraints into account
            if self.uniform:
                samples = randint_numba(
                    n=state.n,
                    k=state.k,
                    replace=False,
                    seed=self.seed,
                )
            else:
                samples = randint_numba(
                    n=state.n,
                    k=state.k,
                    replace=False,
                    p=state.global_separation_array,
                    seed=self.seed,
                )

        # --- update state --------------------------------
        for sample in samples:
            state.add(sample)
