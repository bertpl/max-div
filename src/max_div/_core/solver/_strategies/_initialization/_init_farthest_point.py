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

    A `random_fraction` above zero widens the random start into a random *prefix*: the first
    `round(random_fraction * k)` items (at least one) are drawn uniformly at random in a single
    batch, and the greedy picks fill the rest. At the endpoints, 0.0 is the pure farthest-point
    construction from a single random seed, and 1.0 is a fully random selection. The prefix trades
    a little of the pure construction's peak quality for diversity among start points.

    Suggested use: when the strongest possible starting point is desired, e.g. at short time
    budgets.

    Time Complexity:
       - ~O(n * k), times d when distances are computed on demand from vectors.
    """

    def __init__(self, random_fraction: float = 0.0) -> None:
        """Create the strategy; `random_fraction` in [0, 1] sets the random-prefix size.

        :raises ValueError: If `random_fraction` is outside [0, 1].
        """
        super().__init__()
        if not (0.0 <= random_fraction <= 1.0):
            raise ValueError(f"random_fraction must be in [0, 1], got {random_fraction}")
        self._random_fraction = random_fraction

    def get_next_samples(self, state: SolverState, k_remaining: int | np.int32) -> NDArray[np.int32]:
        if state.n_selected == 0:
            n_random = min(max(round(self._random_fraction * int(state.k)), 1), int(state.k))
            return randint(n=state.n, k=np.int32(n_random), replace=False, p=P_UNIFORM, rng_state=self._rng_state)
        # both arrays below are ascending-index, so positions align
        contributions = state.not_selected_contribution_array
        return np.array([state.not_selected_index_array[np.argmax(contributions)]], dtype=np.int32)
