import numpy as np
from numpy.typing import NDArray

from max_div._core._random import P_UNIFORM, randint
from max_div._core.solver._solver_state import SolverState

from ._base import InitializationStrategy


class InitFarthestPoint(InitializationStrategy):
    """Initialize by farthest-point sampling: a seeded random start item, then greedy picks.

    After the start item, each step adds the not-yet-selected item with the highest diversity
    contribution wrt the current selection.  Under the separation-family metrics that is the item
    farthest from its nearest selected neighbor — classical farthest-point sampling (FPS); under
    `MEAN_PAIRWISE_DISTANCE` it is the item with the highest mean distance to the selection — the
    greedy max-sum construction.  The random start item keeps runs seed-dependent.

    Constraints are ignored by design: greedy picks are constraint-unaware, and the optimization
    phase repairs feasibility afterwards (the solver presets also ignore constraints during
    initialization).

    Suggested use: when the strongest possible starting point is desired, e.g. at short time
    budgets.  Costs ~O(k * n) contribution reads on top of the tracker updates any initialization
    incurs.
    """

    def get_next_samples(self, state: SolverState, k_remaining: int | np.int32) -> NDArray[np.int32]:
        if state.n_selected == 0:
            return randint(n=state.n, k=np.int32(1), replace=False, p=P_UNIFORM, rng_state=self._rng_state)
        # both arrays below are ascending-index, so positions align
        contributions = state.not_selected_contribution_array
        return np.array([state.not_selected_index_array[np.argmax(contributions)]], dtype=np.int32)
