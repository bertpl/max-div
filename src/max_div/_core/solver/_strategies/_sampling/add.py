import numpy as np
from numpy.typing import NDArray

from max_div._core._math.modify_p_selectivity import DEFAULT_LOW_VALUE, exponential_selectivity
from max_div._core._random import choice, choice_constrained
from max_div._core.solver._solver_state import SolverState


def build_add_probabilities(
    state: SolverState,
    candidates: NDArray[np.int32],
    selectivity_modifier: float,
) -> NDArray[np.float32]:
    """Return the sampling probabilities of `candidates` for addition, derived from the current state.

    Lets a caller that draws several times from one unchanged state (each trial add reverted before
    the next draw) build the probabilities once.  The result is only valid until the selection changes.

    Each candidate's probability grows with its contribution to the current selection.  With nothing
    selected yet, no candidate has a selected item to be compared with, so every candidate gets
    the same probability.

    Args:
        state: (SolverState) The current solver state containing selected items and other relevant information.
        candidates: (NDArray[np.int32]) array of candidate item indices (must be a subset of not-selected items)
        selectivity_modifier: (float) value in [-1, 1] that modifies the selectivity of the
            diversity-contribution-based probabilities used for sampling items to be added.
            -1: maximally un-selective --> uniform
            0: no modification to the contribution-based probabilities
            +1: maximally selective --> only the items with very lowest contribution are sampled

    Returns:
        (NDArray[np.float32]) one probability per candidate, same order as `candidates`.
    """
    if state.n_selected == 0:
        # only the first draw of an initialization strategy sees an empty selection
        return np.ones(len(candidates), dtype=np.float32)
    else:
        p = state.full_contribution_array[candidates]  # new array; contribution of candidates wrt selected items
        exponential_selectivity(
            p_in=p,
            p_out=p,  # in-place
            modifier=np.float32(selectivity_modifier),
            reverse=False,  # for adding, we want to have items with high diversity contribution have higher probability
            low_value=DEFAULT_LOW_VALUE,
        )
        return p


def select_items_to_add_with_p(
    state: SolverState,
    candidates: NDArray[np.int32],
    p: NDArray[np.float32],
    k: np.int32 | int,
    rng_state: NDArray[np.uint64],
    ignore_constraints: bool = False,
) -> NDArray[np.int32]:
    """Draw k items from `candidates` with the probabilities `p` built by `build_add_probabilities`.

    `p` is read, never written, so the same array can be passed to any number of draws from the same state.

    On a constrained problem (unless `ignore_constraints` is True), the k items are drawn jointly, so that
    together they move toward satisfying the constraints.

    Args:
        state: (SolverState) The current solver state containing selected items and other relevant information.
        candidates: (NDArray[np.int32]) array of candidate item indices to choose from
            (must be a subset of not-selected items; must be of size>=k)
        p: (NDArray[np.float32]) sampling probabilities, one per candidate.
        k: (int) number of items to add to the selection.
        rng_state: (NDArray[np.uint64]) The RNG state to be used (and updated in-place) for random sampling
        ignore_constraints: (bool) If True, constraints are ignored even if present in the SolverState.

    Returns:
        list of np.int32 indices of the items to be added to the selection (unique values, unsorted).
    """
    if (not state.has_constraints) or ignore_constraints:
        # UNCONSTRAINED
        return choice(
            values=candidates,
            k=np.int32(k),
            replace=False,
            p=p,
            rng_state=rng_state,
        )
    else:
        # CONSTRAINED: the k items are added as a group, so jointly they should move toward
        # satisfying the constraints
        return choice_constrained(
            n=state.n,
            values=candidates,
            k=np.int32(k),
            p=p,
            rng_state=rng_state,
            con_values=state.con_values,
            con_indices=state.con_indices,
            eager=False,
            k_context=state.k - state.n_selected,
        )


def select_items_to_add(
    state: SolverState,
    candidates: NDArray[np.int32],
    k: np.int32 | int,
    selectivity_modifier: float,
    rng_state: NDArray[np.uint64],
    ignore_constraints: bool = False,
) -> NDArray[np.int32]:
    """Select k items from 'candidates' to be added to the provided SolverState.

    Build the probabilities once and draw once; `build_add_probabilities` and `select_items_to_add_with_p`
    document the two halves and their arguments.  Candidates must be a subset of the not-selected items.

    Returns:
        list of np.int32 indices of the items to be added to the selection (unique values, unsorted).
    """
    p = build_add_probabilities(state, candidates, selectivity_modifier)
    return select_items_to_add_with_p(state, candidates, p, k, rng_state, ignore_constraints)
