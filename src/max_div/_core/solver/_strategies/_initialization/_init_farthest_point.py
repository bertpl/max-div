import numpy as np
from numpy.typing import NDArray

from max_div._core._math import select_k_max
from max_div._core._random import P_UNIFORM, randint
from max_div._core.metrics import DiversityObjective
from max_div._core.solver._solver_state import SolverState

from ._base import InitializationStrategy
from ._farthest_point_rounds import are_rounds_supported, draw_round


class InitFarthestPoint(InitializationStrategy):
    """Initialize by farthest-point sampling: a seeded random start item, then greedy picks.

    Each pick adds a not-yet-selected item among the `top_k` highest diversity contributions wrt the
    current selection:

    - separation-family metrics: the items farthest from their nearest selected neighbor
      (classical farthest-point sampling);
    - `MEAN_PAIRWISE_DISTANCE`: the items with the highest mean distance to the selection
      (the greedy max-sum construction).

    The picks are made in one of 2 ways, with the same candidates offered to every pick:

    - **in rounds** (`_farthest_point_rounds`): many picks per pass over the dataset, several times
      faster at large n. Used when `batch_size` is not None and the objective is a single
      separation-family metric.
    - **one item at a time**: one pass over the dataset per pick. Used otherwise, and as long as the
      strategy has not been bound to an objective.

    Constraints are ignored by design; feasibility is left to the optimization steps.

    Parameters:
    - top_k (int): every pick samples uniformly among the `top_k` highest contributions; `top_k=1`
                   is the exact argmax and consumes no randomness. (default: 8)
    - batch_size (int | None): how many candidates a round collects, which bounds how many items
                               it can draw. A deeper pool needs fewer passes over the dataset but
                               refreshes more candidates per draw; above a few hundred the extra
                               refreshing costs more than the saved passes. It cannot affect the
                               selection's quality, only the time spent. `None` picks one item at
                               a time for every objective. (default: 256)

    Time Complexity:
       - ~O(n * k), times d when distances are computed on demand from vectors.
    """

    def __init__(self, top_k: int = 8, batch_size: int | None = 256) -> None:
        """Create the strategy.

        Raises:
            ValueError: If `top_k` is below 1, or `batch_size` is below `top_k` (a round could then
                not offer a full draw).
        """
        super().__init__()
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if batch_size is not None and batch_size < top_k:
            raise ValueError(f"batch_size must be >= top_k ({top_k}), got {batch_size}")
        self._top_k = top_k
        self._batch_size = batch_size
        # the batch size of the rounds once bound to an objective that allows them; None picks one item at a time
        self._round_batch_size: int | None = None

    def bind_objective(self, objective: DiversityObjective) -> None:
        """Draw in rounds when `batch_size` allows it and the objective supports it."""
        self._round_batch_size = self._batch_size if are_rounds_supported(objective) else None

    def get_next_samples(self, state: SolverState, k_remaining: int | np.int32) -> NDArray[np.int32]:
        if state.n_selected == 0:
            # the seeded random start item
            return randint(n=state.n, k=np.int32(1), replace=False, p=P_UNIFORM, rng_state=self._rng_state)
        elif self._round_batch_size is not None:
            return draw_round(state, self._top_k, self._round_batch_size, k_remaining, self._rng_state)
        else:
            return self._pick_one_item(state)

    def _pick_one_item(self, state: SolverState) -> NDArray[np.int32]:
        """Return one item among the `top_k` highest contributions, after one pass over the dataset."""
        # both arrays below are ascending-index, so positions align
        contributions = state.not_selected_contribution_array
        if self._top_k == 1:
            return np.array([state.not_selected_index_array[np.argmax(contributions)]], dtype=np.int32)
        else:
            k_eff = min(self._top_k, len(contributions))
            top_positions = select_k_max(contributions, np.int32(k_eff))
            drawn = randint(n=np.int32(k_eff), k=np.int32(1), replace=False, p=P_UNIFORM, rng_state=self._rng_state)
            return np.array([state.not_selected_index_array[top_positions[drawn[0]]]], dtype=np.int32)
