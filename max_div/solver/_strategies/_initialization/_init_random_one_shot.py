import numpy as np
from numpy.typing import NDArray

from max_div.random import P_UNIFORM, randint, randint_constrained
from max_div.solver._solver_state import SolverState

from ._base import InitializationStrategy


class InitRandomOneShot(InitializationStrategy):
    """
    Initialize by taking a single (hence: one-shot) random sample of k items.  This is among the fastest
    initialization strategies, but potentially also with the lowest quality.

    Suggested use: if time constraints are severe or problem dimensions `n` or `k` are very large.

    Parameters:
    - uniform (bool): If `True`, samples uniformly at random.
                      If `False`, uses vector separations as sampling weights, sampling well-separated vectors
                                         with higher probability (default: `False`)
    - ignore_constraints (bool): If `False`, respects problem constraints during initialization, if present.
                                 If `True`, constraints are ignored. (default: `False`)

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

    def __init__(self, uniform: bool = False, ignore_constraints: bool = False):
        name = "InitRandomOneShot(" + ("u" if uniform else "nu") + (",uncon)" if ignore_constraints else ")")
        super().__init__(name)
        self.uniform = uniform
        self.ignore_constraints = ignore_constraints

    def get_next_samples(self, state: SolverState, k_remaining: int | np.int32) -> NDArray[np.int32]:
        # --- sample --------------------------------------
        if state.has_constraints and (not self.ignore_constraints):
            # take constraints into account
            if self.uniform:
                return randint_constrained(
                    n=state.n,
                    k=state.k,
                    con_values=state.con_values,
                    con_indices=state.con_indices,
                    rng_state=self._rng_state,
                )
            else:
                return randint_constrained(
                    n=state.n,
                    k=state.k,
                    con_values=state.con_values,
                    con_indices=state.con_indices,
                    p=state.global_separation_array,
                    rng_state=self._rng_state,
                )
        else:
            # don't take constraints into account
            if self.uniform:
                return randint(
                    n=state.n,
                    k=state.k,
                    replace=False,
                    p=P_UNIFORM,
                    rng_state=self._rng_state,
                )
            else:
                return randint(
                    n=state.n,
                    k=state.k,
                    replace=False,
                    p=state.global_separation_array,
                    rng_state=self._rng_state,
                )
