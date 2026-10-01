import numpy as np
from numpy.typing import NDArray

from max_div._core.solver._solver_state import SolverState
from max_div._core.solver._strategies._sampling import build_add_probabilities, select_items_to_add_with_p

from ._base import InitializationStrategy


class InitGuidedBatches(InitializationStrategy):
    """Initialize in batches, each the best-scoring of `nc` candidate batches favoring items far from the selection.

    Each iteration adds `batch_size` items (fewer when fewer remain to be selected):

    - `nc` candidate batches are drawn, each with probabilities that grow with the unselected items'
      diversity contribution to the current selection; on a constrained problem each batch is drawn
      so that it moves the selection toward satisfying the constraints, taking into account how many
      items remain to be selected after it;
    - each candidate batch is provisionally added and scored, and the one with the best score is added.

    The probabilities grow more selective as the selection fills, from uniform at the start (nothing
    is selected yet, so no item contributes more than another) to strongly favoring the items
    farthest from the selection near the end.

    The 2 parameters span a range from fast to thorough:

    - `batch_size=16, nc=1`: fast; the items of a batch are drawn without regard to each other,
      so 2 items of the same batch can lie close to each other;
    - `batch_size=1, nc=1`: the draw probabilities are recomputed after every added item;
    - `batch_size=1, nc=16`: every item is the best of 16 such draws.

    Each batch is drawn using how many items each constraint still needs, without checking whether
    all those counts can still be met from the items left.  Where constraints overlap (an item
    counts toward several constraints) and require exact counts, the last picks can therefore leave
    some constraints 1 or 2 items away from their required counts; the optimization steps repair
    that shortfall.

    `InitGuidedBatches` is meant for constrained problems.  On an unconstrained problem it works,
    drawing with the diversity-based probabilities alone, but `farthest_point` reaches a higher
    diversity in less time there.

    Time Complexity:
       - O(nc * k / batch_size) candidate batches are evaluated.  Each one costs a provisional add,
         a score evaluation and an undo, each O(n) or more, and on a constrained problem each draw
         also recomputes how much every item helps the constraints.  At large n and k
         `InitGuidedBatches` is far slower than `farthest_point`.
    """

    def __init__(self, batch_size: int = 1, nc: int = 8) -> None:
        """Create the strategy.

        Raises:
            ValueError: If `batch_size` or `nc` is below 1.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if nc < 1:
            raise ValueError(f"nc must be >= 1, got {nc}")
        super().__init__(f"InitGuidedBatches({batch_size},{nc})")
        self._batch_size = batch_size
        self._nc = nc

    def get_next_samples(self, state: SolverState, k_remaining: int | np.int32) -> NDArray[np.int32]:
        # --- draw probabilities -----------------
        n_to_add = np.int32(min(self._batch_size, k_remaining))
        add_candidates = state.not_selected_index_array
        # capped below 1, where the draw would keep only the top-contributing items and lose its randomness
        selectivity_modifier = min(0.9, float(state.n_selected) / float(state.k))
        p = build_add_probabilities(state, add_candidates, selectivity_modifier)

        # --- best of nc candidate batches -------
        best_batch = select_items_to_add_with_p(state, add_candidates, p, n_to_add, self._rng_state)
        if self._nc == 1:
            return best_batch  # the one draw is added as is; scoring it would change nothing
        else:
            best_score_tuple = self._score_tuple_after_adding(state, best_batch)
            for _ in range(self._nc - 1):
                batch = select_items_to_add_with_p(state, add_candidates, p, n_to_add, self._rng_state)
                score_tuple = self._score_tuple_after_adding(state, batch)
                if score_tuple > best_score_tuple:
                    best_score_tuple = score_tuple
                    best_batch = batch
            return best_batch

    # -------------------------------------------------------------------------
    #  Helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _score_tuple_after_adding(state: SolverState, batch: NDArray[np.int32]) -> tuple[float, ...]:
        """Return the state's score tuple with `batch` added; the state is left unchanged."""
        with state.savepoint():
            state.add_many(batch)
            score = state.score
        return score.as_tuple()
