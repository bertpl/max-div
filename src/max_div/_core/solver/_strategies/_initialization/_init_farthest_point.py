import numpy as np
from numpy.typing import NDArray

from max_div._core._random import P_UNIFORM, randint
from max_div._core.solver._solver_state import SolverState

from ._base import InitializationStrategy


class InitFarthestPoint(InitializationStrategy):
    """Initialize by farthest-point sampling: a seeded random start item, then greedy picks.

    Each pick adds the not-yet-selected item with the highest diversity contribution wrt the
    current selection:

    - separation-family metrics: the item farthest from its nearest selected neighbor
      (classical farthest-point sampling);
    - `MEAN_PAIRWISE_DISTANCE`: the item with the highest mean distance to the selection
      (the greedy max-sum construction).

    Constraints are ignored by design; feasibility is left to the optimization steps.

    Suggested use: when the strongest possible starting point is desired, e.g. at short time
    budgets.

    Time Complexity:
       - ~O(n * k), times d when distances are computed on demand from vectors.
    """

    def get_next_samples(self, state: SolverState, k_remaining: int | np.int32) -> NDArray[np.int32]:
        if state.n_selected == 0:
            return randint(n=state.n, k=np.int32(1), replace=False, p=P_UNIFORM, rng_state=self._rng_state)
        # both arrays below are ascending-index, so positions align
        contributions = state.not_selected_contribution_array
        return np.array([state.not_selected_index_array[np.argmax(contributions)]], dtype=np.int32)
