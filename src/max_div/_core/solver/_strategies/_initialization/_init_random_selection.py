import numpy as np
from numpy.typing import NDArray

from max_div._core._random import P_UNIFORM, randint, randint_constrained
from max_div._core.solver._solver_state import SolverState

from ._base import InitializationStrategy


class InitRandomSelection(InitializationStrategy):
    """Initialize by taking a single random sample of k items; see `InitializationStrategy.random_selection`.

    This is among the fastest initialization strategies, but potentially also with the lowest quality.

    Suggested use: if time constraints are severe or problem dimensions `n` or `k` are very large.

    Parameters:
    - ignore_constraints (bool): If `False`, respects problem constraints during initialization, if present.
                                 If `True`, constraints are ignored. (default: `False`)
    - parallel (bool): If `True`, the single batched tracker update runs over parallel threads;
                       see `DiversityContributionTracker.add_many` for the contract. (default: `False`)

    Time Complexity:
       - without constraints: ~O(n)
       - with constraints:    ~O(kn)
    """

    def __init__(self, ignore_constraints: bool = False, parallel: bool = False) -> None:
        name = "InitRandomSelection" + ("(uncon)" if ignore_constraints else "()")
        super().__init__(name, parallel_batch_add=parallel)
        self.ignore_constraints = ignore_constraints

    def get_next_samples(self, state: SolverState, k_remaining: int | np.int32) -> NDArray[np.int32]:
        # --- sample -----------------------------
        if state.has_constraints and (not self.ignore_constraints):
            # take constraints into account
            return randint_constrained(
                n=state.n,
                k=state.k,
                con_values=state.con_values,
                con_indices=state.con_indices,
                rng_state=self._rng_state,
            )
        else:
            # don't take constraints into account
            return randint(
                n=state.n,
                k=state.k,
                replace=False,
                p=P_UNIFORM,
                rng_state=self._rng_state,
            )
