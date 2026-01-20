from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from max_div.internal.math.modify_p_selectivity import exponential_selectivity
from max_div.random import choice, choice_constrained
from max_div.solver._solver_state import SolverState

from ._helpers import remove_sample_from_candidates_and_p


class SamplingType(StrEnum):
    """
    When sampling 'k' elements for addition to the selection, this can happen in 2 different contexts:
     - we want 'k' items to be added as a 'group' (all or nothing)             --> GROUP
     - we want 'k' candidate items, only one of which (the best) will be added --> CANDIDATES

    In case of a constrained problem, the expected behavior of how to take constraints into account will differ:
     - GROUP      -> we try to select the 'k' items such that, as a whole, they will satisfy the constraints
     - CANDIDATES -> each candidate is to be seen as the first item that will help moving towards satisfying
                         the constraints  (i.e. k=1, k_context=n-n_selected)
    """

    GROUP = "group"
    CANDIDATES = "candidates"


def select_items_to_add(
    state: SolverState,
    candidates: NDArray[np.int32],
    k: np.int32 | int,
    selectivity_modifier: float,
    rng_state: NDArray[np.uint64],
    sampling_type: SamplingType = SamplingType.GROUP,
    include_within_group_separation: bool = True,
    ignore_constraints: bool = False,
) -> NDArray[np.int32]:
    """
    Select k items from 'candidates' to be added to the provided SolverState.  candidates are guaranteed to be a subset
    of the not-selected items in the SolverState.
    :param state: (SolverState) The current solver state containing selected items and other relevant information.
    :param candidates: (NDArray[np.int32]) array of candidate item indices to choose from
                                                    (must be a subset of not-selected items; must be of size>=k)
    :param k: (int) number of items to add to the selection.
    :param selectivity_modifier: (float) value in [-1, 1] that modifies the selectivity of the separation-based
                                 probabilities used for sampling items to be added.
                                    -1: maximally un-selective --> uniform
                                     0: no modification to the separation-based probabilities
                                    +1: maximally selective --> only the items with very lowest separation are sampled
    :param rng_state: (NDArray[np.uint64]) The RNG state to be used (and updated in-place) for random sampling
    :param sampling_type: (SamplingType) context in which the k items are being sampled (GROUP vs CANDIDATES)
    :param include_within_group_separation: (bool) flag that influences how sampling probabilities are built.
                                            True: start from sep. to already selected items + within-group separation
                                            False: start from sep. to already selected items only
    :param ignore_constraints: (bool) If True, constraints are ignored even if present in the SolverState.
    :return: list of np.int32 indices of the items to be added to the selection (unique values, unsorted).
    """

    # --- prepare probabilities ---------------------------
    if state.n_selected == 0:
        # the only option is to look at separation wrt all other items, as we don't have a selection yet
        # this branch is only taken in the first iteration of initialization strategies
        p = state.global_separation_array[candidates]  # add separation of candidates wrt to all other items
    else:
        # standard path
        p = state.full_separation_array[candidates]  # new array; separation of candidates to selected items
        if include_within_group_separation:
            p += state.global_separation_array[candidates]  # add separation of candidates wrt to all other items

    exponential_selectivity(
        p_in=p,
        p_out=p,  # in-place
        modifier=np.float32(selectivity_modifier),
        reverse=False,  # for adding, we want to have vectors with high separation have higher probability
    )

    # --- actual sampling ---------------------------------
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
        else:
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

                # remove sample from candidates & p to prevent duplicates
                candidates, p = remove_sample_from_candidates_and_p(candidates, p, samples[i])

            # return final array of sample candidates
            return samples
