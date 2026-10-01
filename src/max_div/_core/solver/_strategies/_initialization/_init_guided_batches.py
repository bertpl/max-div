import numpy as np
from numpy.typing import NDArray

from max_div._core.solver._solver_state import SolverState
from max_div._core.solver._strategies._sampling import build_add_probabilities, select_items_to_add_with_p

from ._base import InitializationStrategy


class InitGuidedBatches(InitializationStrategy):
    """Initialize in batches, each the best of `nc` draws guided by diversity and steered toward the constraints.

    Each iteration adds `batch_size` items (fewer when fewer remain to be selected):

    - `nc` candidate batches are drawn, each with probabilities that grow with the unselected items'
      diversity contribution to the current selection; on a constrained problem the constrained
      sampler draws each batch so that it moves toward satisfying the constraints, knowing how many
      items remain to be selected after it;
    - each candidate batch is provisionally added and scored, and the one with the best score is added.

    The probabilities grow more selective as the selection fills, from uniform at the start (nothing
    is selected yet, so no item contributes more than another) to strongly favoring the items
    farthest from the selection near the end.

    The 2 parameters span a range from fast to thorough:

    - `batch_size=16, nc=1`: fast; the items of a batch are drawn without regard to each other,
      so they can land in the same gap;
    - `batch_size=1, nc=1`: every item is drawn against the selection so far;
    - `batch_size=1, nc=16`: every item is the best of 16 such draws.

    The constrained sampler steers each batch by the counts still open, without checking whether the
    remaining counts can all be met from the items left.  Where constraints cross each other with
    exact counts, the last picks can therefore leave the selection a few units short of feasible;
    the optimization steps repair that.

    Meant for constrained problems.  On an unconstrained problem it works, drawing without the
    constrained sampler, but `farthest_point` reaches a higher diversity in less time there.

    Time Complexity:
       - every candidate batch costs a provisional add, a score evaluation and an undo, each O(n)
         or more, and on a constrained problem every draw rebuilds the per-item constraint scores:
         ~O(nc * k / batch_size) such rounds, so at large n and k this is far slower than
         `farthest_point`.
    """

    def __init__(self, batch_size: int = 1, nc: int = 8) -> None:
        """Create the strategy.

        Args:
            batch_size: Items added per iteration.
            nc: Candidate batches drawn per iteration, the best of which is added; 1 adds the one
                draw without scoring it.

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
        # --- draw probabilities ---------------------
        n_to_add = np.int32(min(self._batch_size, k_remaining))
        candidates = state.not_selected_index_array
        selectivity_modifier = min(0.9, float(state.n_selected) / float(state.k))  # cap keeps the draw random
        p = build_add_probabilities(state, candidates, selectivity_modifier)

        # --- best of nc candidate batches -----------
        best_batch = select_items_to_add_with_p(state, candidates, p, n_to_add, self._rng_state)
        if self._nc == 1:
            return best_batch  # the one draw is added as is; scoring it would change nothing
        best_score_tuple = self._score_tuple_after_adding(state, best_batch)
        for _ in range(self._nc - 1):
            batch = select_items_to_add_with_p(state, candidates, p, n_to_add, self._rng_state)
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
        """Return the score tuple the state would have with `batch` added; the state is left unchanged."""
        with state.savepoint():
            state.add_many(batch)
            score_tuple = state.score.as_tuple()
        return score_tuple
