import numpy as np
from numpy.typing import NDArray

from max_div._core._random import P_UNIFORM, randint, randint_constrained
from max_div._core.solver._solver_state import SolverState

from ._base import InitializationStrategy


class InitRandomOneShot(InitializationStrategy):
    """Initialize by taking a single (hence: one-shot) random sample of k items.

    This is among the fastest initialization strategies, but potentially also with the lowest quality.

    Suggested use: if time constraints are severe or problem dimensions `n` or `k` are very large.
    For very large `n`, prefer `uniform=True` (what the solver presets use): contribution-weighted
    sampling (`uniform=False`) computes every pairwise distance, which with distances computed on
    demand from vectors takes roughly a minute at n ≈ 20,000 and grows quadratically — hours
    before solving starts by n ≈ 100,000 with high-dimensional vectors. With a stored distance
    matrix: seconds up to n ≈ 20,000, with the same quadratic growth.

    Parameters:
    - uniform (bool): If `True`, samples uniformly at random.
                      If `False`, uses global diversity contributions as sampling weights, favoring
                                         high-contribution items with higher probability (default: `False`)
    - ignore_constraints (bool): If `False`, respects problem constraints during initialization, if present.
                                 If `True`, constraints are ignored. (default: `False`)
    - parallel (bool): If `True`, the single batched tracker update runs over parallel threads;
                       see `DiversityContributionTracker.add_many` for the contract. (default: `False`)

    Notes:
        - using the global diversity contribution as sampling weights is a heuristic, not an exactly optimal
          solution, with known limitations:
            - in 1D problems this heuristic should be probabilistically optimal, but in higher dimensions (the more
              likely scenario) it is not.  E.g. in 2D where items have half the separation as in other regions, we
              should sample 4x fewer, not 2x fewer items.
            - when multiple items (e.g. 5) are identical and hence have 0 separation, we will not sample any of them
              (unless k is high enough), while optimal solutions might in fact contain exactly 1 of them.

    Time Complexity:
       - without constraints: ~O(n)
       - with constraints:    ~O(kn)
    """

    def __init__(self, uniform: bool = False, ignore_constraints: bool = False, parallel: bool = False) -> None:
        name = "InitRandomOneShot(" + ("u" if uniform else "nu") + (",uncon)" if ignore_constraints else ")")
        super().__init__(name, parallel_batch_add=parallel)
        self.uniform = uniform
        self.ignore_constraints = ignore_constraints

    def get_next_samples(self, state: SolverState, k_remaining: int | np.int32) -> NDArray[np.int32]:
        # --- sample -----------------------------
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
            return randint_constrained(
                n=state.n,
                k=state.k,
                con_values=state.con_values,
                con_indices=state.con_indices,
                p=state.global_contribution_array,
                rng_state=self._rng_state,
            )
        # don't take constraints into account
        if self.uniform:
            return randint(
                n=state.n,
                k=state.k,
                replace=False,
                p=P_UNIFORM,
                rng_state=self._rng_state,
            )
        return randint(
            n=state.n,
            k=state.k,
            replace=False,
            p=state.global_contribution_array,
            rng_state=self._rng_state,
        )
