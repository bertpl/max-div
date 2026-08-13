import numpy as np
from numpy.typing import NDArray

from max_div._core._math import select_k_max
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

    A `top_k` above one makes every greedy pick sample uniformly among the `top_k` highest
    contributions instead of taking the argmax, so picks vary while every one stays among the best
    candidates. `top_k=1` is the exact argmax and consumes no randomness.

    Suggested use: when the strongest possible starting point is desired, e.g. at short time
    budgets.

    Time Complexity:
       - ~O(n * k), times d when distances are computed on demand from vectors.
    """

    def __init__(self, top_k: int = 1) -> None:
        """Create the strategy.

        :raises ValueError: If `top_k` is below 1.
        """
        super().__init__()
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        self._top_k = top_k

    def get_next_samples(self, state: SolverState, k_remaining: int | np.int32) -> NDArray[np.int32]:
        if state.n_selected == 0:
            return randint(n=state.n, k=np.int32(1), replace=False, p=P_UNIFORM, rng_state=self._rng_state)
        # both arrays below are ascending-index, so positions align
        contributions = state.not_selected_contribution_array
        if self._top_k == 1:
            return np.array([state.not_selected_index_array[np.argmax(contributions)]], dtype=np.int32)
        k_eff = min(self._top_k, len(contributions))
        top_positions = select_k_max(contributions, np.int32(k_eff))
        drawn = randint(n=np.int32(k_eff), k=np.int32(1), replace=False, p=P_UNIFORM, rng_state=self._rng_state)
        return np.array([state.not_selected_index_array[top_positions[drawn[0]]]], dtype=np.int32)
