from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from max_div._core._math.modify_p_selectivity import DEFAULT_LOW_VALUE, exponential_selectivity
from max_div._core._random import choice, choice_constrained
from max_div._core.solver._solver_state import SolverState

from ._helpers import remove_sample_from_candidates_and_p


class SamplingType(StrEnum):
    """Context in which 'k' elements are sampled for addition to the selection.

    Sampling can happen in 2 different contexts:
     - we want 'k' items to be added as a 'group' (all or nothing)             --> GROUP
     - we want 'k' candidate items, only one of which (the best) will be added --> CANDIDATES.

    In case of a constrained problem, the expected behavior of how to take constraints into account will differ:
     - GROUP      -> we try to select the 'k' items such that, as a whole, they will satisfy the constraints
     - CANDIDATES -> each candidate is to be seen as the first item that will help moving towards satisfying
                         the constraints  (i.e. k=1, k_context=n-n_selected)
    """

    GROUP = "group"
    CANDIDATES = "candidates"


def build_add_probabilities(
    state: SolverState,
    candidates: NDArray[np.int32],
    selectivity_modifier: float,
    include_within_group_contribution: bool | np.bool_ = True,
) -> NDArray[np.float32]:
    """Return the sampling probabilities of `candidates` for addition, derived from the current state.

    Split off from `select_items_to_add` so a caller that draws several times from one unchanged state
    (each trial add reverted before the next draw) builds the probabilities once.  The result is
    only valid until the selection changes.

    Args:
        state: (SolverState) The current solver state containing selected items and other relevant information.
        candidates: (NDArray[np.int32]) array of candidate item indices (must be a subset of not-selected items)
        selectivity_modifier: (float) value in [-1, 1] that modifies the selectivity of the
            diversity-contribution-based probabilities used for sampling items to be added.
            -1: maximally un-selective --> uniform
            0: no modification to the contribution-based probabilities
            +1: maximally selective --> only the items with very lowest contribution are sampled
        include_within_group_contribution: (bool) flag that influences how sampling probabilities are built.
            True: start from contribution wrt already selected items + within-group contribution
            False: start from contribution wrt already selected items only

    Returns:
        (NDArray[np.float32]) one probability per candidate, same order as `candidates`.
    """
    if state.n_selected == 0:
        # the only option is to look at the global contribution (wrt all other items), as we don't have a selection yet
        # this branch is only taken in the first iteration of initialization strategies
        p = state.global_contribution_for(candidates)  # contribution of candidates wrt all other items
    else:
        # standard path
        p = state.full_contribution_array[candidates]  # new array; contribution of candidates wrt selected items
        if include_within_group_contribution:
            p += state.global_contribution_for(candidates)  # add contribution of candidates wrt all other items

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
    sampling_type: SamplingType = SamplingType.GROUP,
    ignore_constraints: bool = False,
) -> NDArray[np.int32]:
    """Draw k items from `candidates` with the probabilities `p` built by `build_add_probabilities`.

    `p` is read, never written, so one array serves any number of draws from the same state.

    Args:
        state: (SolverState) The current solver state containing selected items and other relevant information.
        candidates: (NDArray[np.int32]) array of candidate item indices to choose from
            (must be a subset of not-selected items; must be of size>=k)
        p: (NDArray[np.float32]) sampling probabilities, one per candidate.
        k: (int) number of items to add to the selection.
        rng_state: (NDArray[np.uint64]) The RNG state to be used (and updated in-place) for random sampling
        sampling_type: (SamplingType) context in which the k items are being sampled (GROUP vs CANDIDATES)
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
    # CONSTRAINED
    if sampling_type == SamplingType.GROUP:
        # these samples are intended to be added as a GROUP, so jointly should try to satisfy constraints
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
    # these samples are intended to be individual CANDIDATES, from which only one will be actually added
    samples = np.empty(k, dtype=np.int32)
    for i in range(k):
        # obtain new sample
        samples[i] = choice_constrained(
            n=state.n,
            values=candidates,
            k=np.int32(1),
            p=p,
            rng_state=rng_state,
            con_values=state.con_values,
            con_indices=state.con_indices,
            eager=False,
            k_context=state.k - state.n_selected,
        )[0]

        # remove sample from candidates & p to prevent duplicates (fresh arrays; the caller's p stays intact)
        candidates, p = remove_sample_from_candidates_and_p(candidates, p, samples[i])

    # return final array of sample candidates
    return samples


def select_items_to_add(
    state: SolverState,
    candidates: NDArray[np.int32],
    k: np.int32 | int,
    selectivity_modifier: float,
    rng_state: NDArray[np.uint64],
    sampling_type: SamplingType = SamplingType.GROUP,
    include_within_group_contribution: bool | np.bool_ = True,
    ignore_constraints: bool = False,
) -> NDArray[np.int32]:
    """Select k items from 'candidates' to be added to the provided SolverState.

    One build of the probabilities followed by one draw; see `build_add_probabilities` and
    `select_items_to_add_with_p` for the two halves.

    Candidates are guaranteed to be a subset of the not-selected items in the SolverState.

    Args:
        state: (SolverState) The current solver state containing selected items and other relevant information.
        candidates: (NDArray[np.int32]) array of candidate item indices to choose from
            (must be a subset of not-selected items; must be of size>=k)
        k: (int) number of items to add to the selection.
        selectivity_modifier: (float) value in [-1, 1] that modifies the selectivity of the
            diversity-contribution-based probabilities used for sampling items to be added.
            -1: maximally un-selective --> uniform
            0: no modification to the contribution-based probabilities
            +1: maximally selective --> only the items with very lowest contribution are sampled
        rng_state: (NDArray[np.uint64]) The RNG state to be used (and updated in-place) for random sampling
        sampling_type: (SamplingType) context in which the k items are being sampled (GROUP vs CANDIDATES)
        include_within_group_contribution: (bool) flag that influences how sampling probabilities are built.
            True: start from contribution wrt already selected items + within-group contribution
            False: start from contribution wrt already selected items only
        ignore_constraints: (bool) If True, constraints are ignored even if present in the SolverState.

    Returns:
        list of np.int32 indices of the items to be added to the selection (unique values, unsorted).
    """
    p = build_add_probabilities(state, candidates, selectivity_modifier, include_within_group_contribution)
    return select_items_to_add_with_p(state, candidates, p, k, rng_state, sampling_type, ignore_constraints)
