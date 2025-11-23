"""
Methods for sampling WITH constraints.
"""

import copy
import math

import numba
import numpy as np
from numba import types

from max_div.constraints import Constraint
from max_div.constraints._numba import _np_con_indices, _np_con_max_value, _np_con_min_value
from max_div.sampling import randint_numba


# =================================================================================================
#  randint_constrained
# =================================================================================================
def randint_constrained(
    n: int,
    k: int,
    cons: list[Constraint],
    p: np.ndarray | None = None,
    seed: int | None = None,
) -> list[int]:
    """
    Generate `k` unique random integers from the range `[0, n)` while satisfying given constraints.

    NOTES:

    * there are no guarantees are given that constraints are satisfied; a best-effort attempt will be made, with the
    probability of the result satisfying the constraints increasing the simpler the constraints are.

    * `randint_constrained` is essentially a version of [`randint`][max_div.sampling.uncon.randint] that supports constraints.

    * It is strongly advised to use the fully equivalent function [randint_constrained_numba][max_div.sampling.con.randint_constraint_numba]
      which is 10-100x faster due to its use of numba JIT compilation and efficient numpy-based data structures.

    :param n: range to sample from [0, n)
    :param k: number of unique samples to draw (no replacement)
    :param cons: list of Constraint objects, indicating we want to sample at least `min_count`
                 and at most `max_count` integers from `int_set`, with all values of `int_set` in range `[0, n)`.
    :param p: optional, target probabilities for each integer in `[0, n)`. No guarantees are given if provided,
              but will help guide qualitative preference of sampling algorithm.  Higher p[i] values will increase
              probability of integer `i` being included in the sample, to the extent the constraints allow this.
    :param seed: (int, optional) random seed  (not set if None or 0 are provided)
    :return: list of samples
    """

    # --- initialize --------------------------------------
    samples = []
    k_remaining = k
    cons = copy.deepcopy(cons)  # make sure we don't modify the original ones

    # --- sample ------------------------------------------
    while k_remaining > 0:
        # --- score & thresholds ----------------

        # determine how much each integer would help us satisfy min_count constraints
        #  (at this point, hard-excluding those integers that are already sampled or that would violate max_count)
        score = _compute_score(n, cons, samples, hard_max_constraints=True)

        # determine how much improvement we need to be able to satisfy all min_count constraints
        # with k_remaining samples
        total_score_needed = sum([con.min_count for con in cons if con.min_count > 0])
        score_threshold = math.ceil(total_score_needed / k_remaining)

        if max(score) >= score_threshold:
            # at this point, it still seems possible to satisfy all min_count constraints with the
            # remaining # of samples we have.
            #  --> STRATEGY 1: focus on those samples that help us enough to satisfy all constraints with the
            #                  remaining # of samples we have, and do not sample from any of the others.
            pass
        else:
            # we cannot satisfy all constraints with the k remaining samples.
            #  --> STRATEGY 2: choose samples with best net effect (help achieve min_count vs not violating max_count),
            #                  still hard-excluding already sampled integers.
            score = _compute_score(n, cons, samples, hard_max_constraints=False)
            score_threshold = max(score)  # this could even be negative; we focus on those samples that do least harm

        # --- sample according to strategy ------
        # construct modified probabilities, taking into account scores
        if (p is None) or (p.size == 0):
            p_mod = np.ones(n)  # uniform probabilities
        else:
            p_mod = p.copy()  # avoid modifying input array

        # make sure no p_mod is == 0, such that we will always have some p_mod[i]>0 after setting some to 0
        p_mod = np.maximum(p_mod, 1e-12 * max(p_mod))
        for i in range(n):
            if score[i] < score_threshold:
                p_mod[i] = 0.0  # exclude from sampling

        # one sample from p_mod
        s = int(
            randint_numba(
                n=np.int32(n),
                k=np.int32(1),
                p=p_mod.astype(np.float32),
                replace=False,
                seed=np.int64(seed or 0),
            )[0]
        )

        # --- update stats --------------------------------
        for con in cons:
            if s in con.int_set:
                con.min_count -= 1
                con.max_count -= 1

        samples.append(s)
        k_remaining -= 1

    # --- done ----------------------------------------
    return samples


def _compute_score(
    n: int,
    cons: list[Constraint],
    already_sampled: list[int],
    hard_max_constraints: bool,
) -> np.ndarray:
    """
    Score each integer in `[0, n)` based on how sampling each integer helps toward satisfying the constraints
      - if it helps achieve a min_count that is not satisfied yet:    +1
      - if it would violate a max_count that we already hit:          -1      if hard_max_constraints=False
                                                                      -2**24  if hard_max_constraints=True
      - if we already sampled it:                                     -2**24  if hard_max_constraints=True

    The basic idea behind the scoring is that integers with score <= 0 will not be sampled, if at all possible.

    :param n: range to score [0, n)
    :param cons: list of Constraint objects, indicating we want to sample at least `min_count`
                 and at most `max_count` integers from `int_set`, with all values of `int_set` in range `[0, n)`.
    :param already_sampled: list of integers already sampled
    :param hard_max_constraints: if True, integers that would violate max_count constraints are penalized such that they
                                 will never be sampled.
    :return: array of scores for each integer in `[0, n)`
    """

    # --- init --------------------------------------------
    large_penalty = 2**24
    if hard_max_constraints:
        max_count_penalty = large_penalty
    else:
        max_count_penalty = 1
    scores = np.zeros(n, dtype=np.int32)

    # --- min_count / max_count ---------------------------
    for con in cons:
        if con.min_count > 0:
            for i in con.int_set:
                scores[i] += 1
        if con.max_count <= 0:
            for i in con.int_set:
                scores[i] -= max_count_penalty

    # --- already sampled ---------------------------------
    for i in already_sampled:
        scores[i] -= large_penalty

    return scores


# =================================================================================================
#  randint_constrained_numba
# =================================================================================================
@numba.njit(
    types.int32[:](
        types.int32,
        types.int32[:, :],
        types.int32[:],
        types.int32[:],
        types.boolean,
    )
)
def _compute_score_numba(
    n: np.int32,
    con_values: np.ndarray[np.int32],
    con_indices: np.ndarray[np.int32],
    already_sampled: np.ndarray[np.int32],
    hard_max_constraints: bool,
) -> np.ndarray[np.int32]:
    """
    Numba version of _compute_score.
    Score each integer based on how sampling each integer helps toward satisfying the constraints.

    :param n: range to score [0, n)
    :param con_values: 2D array (n_cons, 2) with min_count and max_count for each constraint
    :param con_indices: 1D array with constraint indices in the format described in _constraints.py
    :param already_sampled: 1D array of integers already sampled (negative values indicate no more samples)
    :param hard_max_constraints: if True, integers that would violate max_count constraints are heavily penalized
    :return: array of scores for each integer
    """
    n_cons = con_values.shape[0]

    # --- init --------------------------------------------
    large_penalty = np.int32(2**24)
    if hard_max_constraints:
        max_count_penalty = large_penalty
    else:
        max_count_penalty = np.int32(1)
    scores = np.zeros(n, dtype=np.int32)

    # --- min_count / max_count ---------------------------
    for i_con in np.arange(n_cons, dtype=np.int32):
        min_val = _np_con_min_value(con_values, i_con)
        max_val = _np_con_max_value(con_values, i_con)
        indices = _np_con_indices(con_indices, i_con)

        if min_val > 0:
            for idx in indices:
                scores[idx] += 1
        if max_val <= 0:
            for idx in indices:
                scores[idx] -= max_count_penalty

    # --- already sampled ---------------------------------
    for i in already_sampled:
        if i >= 0:  # negative values indicate end of valid samples
            scores[i] -= large_penalty

    return scores


@numba.njit(
    types.int32[:](
        types.int32,
        types.int32,
        types.int32[:, :],
        types.int32[:],
        types.float32[:],
        types.int64,
    )
)
def randint_constrained_numba(
    n: np.int32,
    k: np.int32,
    con_values: np.ndarray[np.int32],
    con_indices: np.ndarray[np.int32],
    p: np.ndarray[np.float32] = np.zeros(0, dtype=np.float32),
    seed: np.int64 = 0,
) -> np.ndarray[np.int32]:
    """
    Numba version of randint_constrained, which is 10-100x faster than the pure Python version.

    Generate `k` unique random integers from the range `[0, n)` while satisfying given constraints.

    `con_values` & `con_indices` can be obtained by using the `to_numpy` method of the [Constraints][max_div.constraints.constraints.Constraints] class.

    For benchmark results, see [here](../../../../benchmarks/randint_constrained.md)

    :param n: range to sample from [0, n)
    :param k: number of unique samples to draw (no replacement)
    :param con_values: 2D array (n_cons, 2) with min_count and max_count for each constraint
    :param con_indices: 1D array with constraint indices in the format described in _constraints.py
    :param p: optional, target probabilities for each integer in `[0, n)`
    :param seed: random seed
    :return: array of samples
    """
    # --- initialize --------------------------------------
    samples = np.empty(k, dtype=np.int32)
    k_remaining = k
    n_cons = con_values.shape[0]

    # Make a copy of con_values to track current min/max counts
    con_values_working = con_values.copy()

    sample_idx = np.int32(0)

    # --- sample ------------------------------------------
    while k_remaining > 0:
        # --- score & thresholds ----------------

        # Get already sampled integers
        already_sampled = samples[:sample_idx]

        # determine how much each integer would help us satisfy min_count constraints
        score = _compute_score_numba(n, con_values_working, con_indices, already_sampled, True)

        # determine how much improvement we need to be able to satisfy all min_count constraints
        total_score_needed = np.int32(0)
        for i_con in range(n_cons):
            min_val = _np_con_min_value(con_values_working, np.int32(i_con))
            if min_val > 0:
                total_score_needed += min_val

        score_threshold = np.int32((total_score_needed + k_remaining - 1) // k_remaining)  # ceil division

        max_score = np.int32(-(2**30))
        for s in score:
            if s > max_score:
                max_score = s

        if max_score >= score_threshold:
            # STRATEGY 1: focus on those samples that help satisfy constraints
            pass
        else:
            # STRATEGY 2: choose samples with best net effect
            score = _compute_score_numba(n, con_values_working, con_indices, already_sampled, False)
            max_score = np.int32(-(2**30))
            for s in score:
                if s > max_score:
                    max_score = s
            score_threshold = max_score

        # --- sample according to strategy ------
        # construct modified probabilities
        if p.size == 0:
            p_mod = np.ones(n, dtype=np.float32)
        else:
            p_mod = p.copy()

        # make sure no p_mod is == 0
        max_p = np.float32(0.0)
        for val in p_mod:
            if val > max_p:
                max_p = val
        min_p = np.float32(1e-12 * max_p)
        for i in range(n):
            if p_mod[i] < min_p:
                p_mod[i] = min_p

        # zero out probabilities for scores below threshold
        for i in range(n):
            if score[i] < score_threshold:
                p_mod[i] = np.float32(0.0)

        # sample one integer
        result = randint_numba(n, np.int32(1), False, p_mod, seed)
        s = result[0]

        # --- update stats --------------------------------
        for i_con in range(n_cons):
            indices = _np_con_indices(con_indices, np.int32(i_con))
            for idx in indices:
                if idx == s:
                    # Decrement both min and max count for this constraint
                    con_values_working[i_con, 0] -= 1
                    con_values_working[i_con, 1] -= 1
                    break

        samples[sample_idx] = s
        sample_idx += 1
        k_remaining -= 1

    # --- done ----------------------------------------
    return samples
