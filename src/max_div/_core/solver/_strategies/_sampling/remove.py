import numpy as np
from numpy.typing import NDArray

from max_div._core._math.modify_p_selectivity import DEFAULT_LOW_VALUE, exponential_selectivity
from max_div._core._random import choice
from max_div._core.solver._solver_state import SolverState


def select_items_to_remove(
    state: SolverState, k: np.int32 | int, selectivity_modifier: float, rng_state: NDArray[np.uint64]
) -> NDArray[np.int32]:
    """Select k items to be removed from the current selection in the provided SolverState.

    Args:
        state: (SolverState) The current solver state containing selected items and other relevant information.
        k: (int) number of items to remove from the selection.
        selectivity_modifier: (float) value in [-1, 1] that modifies the selectivity of the
            diversity-contribution-based probabilities used for sampling items to be removed.
            -1: maximally un-selective --> uniform
            0: no modification to the contribution-based probabilities
            +1: maximally selective --> only the items with very lowest contribution are sampled
        rng_state: (NDArray[np.uint64]) The RNG state to be used (and updated in-place) for random sampling

    Returns:
        list of np.int32 indices of the items to be removed from the selection (unique values, unsorted).
    """
    # --- guiding probabilities for removal ---
    p = state.selected_contribution_array  # this creates a copy
    exponential_selectivity(
        p_in=p,
        p_out=p,  # in-place
        modifier=np.float32(selectivity_modifier),
        reverse=True,  # for removal, we want to have items with small diversity contribution have higher probability
        low_value=DEFAULT_LOW_VALUE,
    )

    # --- sample ---
    return choice(
        values=state.selected_index_array,
        k=np.int32(k),
        replace=False,
        p=p,
        rng_state=rng_state,
    )
