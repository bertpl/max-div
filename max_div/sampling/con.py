"""
Methods for sampling WITH constraints.
"""

import numba
import numpy as np
from numba import types
from numpy.typing import NDArray

from max_div.sampling import randint_numba

from ._constraint_helpers import (
    _np_con_indices,
    _np_con_max_value,
    _np_con_min_value,
    _np_con_satisfied,
)


# =================================================================================================
#  randint_constrained
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
def _compute_score(
    n: np.int32,
    con_values: NDArray[np.int32],
    con_indices: NDArray[np.int32],
    already_sampled: NDArray[np.int32],
    hard_max_constraints: bool,
) -> NDArray[np.int32]:
    """
    Score each integer in `[0, n)` based on how sampling each integer helps toward satisfying the constraints
      - if it helps achieve a min_count that is not satisfied yet:    +1
      - if it would violate a max_count that we already hit:          -1      if hard_max_constraints=False
                                                                      -2**24  if hard_max_constraints=True
      - if we already sampled it:                                     -2**24  if hard_max_constraints=True

    The basic idea behind the scoring is that integers with score <= 0 will not be sampled, if at all possible.

    :param n: range to score [0, n)
    :param con_values: 2D array (m, 2) with min_count and max_count for each constraint
    :param con_indices: 1D array with constraint indices in the format described in _constraints.py
    :param already_sampled: 1D array of integers already sampled (negative values indicate no more samples)
    :param hard_max_constraints: if True, integers that would violate max_count constraints are heavily penalized
    :return: array of scores for each integer
    """
    m = con_values.shape[0]

    # --- init --------------------------------------------
    large_penalty = np.int32(2**24)
    if hard_max_constraints:
        max_count_penalty = large_penalty
    else:
        max_count_penalty = np.int32(1)
    scores = np.zeros(n, dtype=np.int32)

    # --- min_count / max_count ---------------------------
    for i_con in np.arange(m, dtype=np.int32):
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


@numba.njit(fastmath=True)
def randint_constrained(
    n: np.int32,
    k: np.int32,
    con_values: NDArray[np.int32],
    con_indices: NDArray[np.int32],
    p: NDArray[np.float32] = np.zeros(0, dtype=np.float32),
    seed: np.int64 = np.int64(0),
    eager: bool = False,
    k_context: np.int32 = np.int32(-1),
    i_forbidden: NDArray[np.int32] = np.empty(0, dtype=np.int32),
) -> NDArray[np.int32]:
    """
    Generate `k` unique random integers from the range `[0, n)` while satisfying given constraints.

    NOTES:

    * no guarantees are given that the solution will satisfy all constraints; a best-effort attempt will be made, with the
    probability of the result satisfying the constraints increasing the simpler the constraints are.

    * `randint_constrained` is essentially a version of [`randint`][max_div.sampling.uncon.randint] that supports constraints.

    * This version is numba-accelerated and uses efficient numpy-based data structures, resulting in 10-100x speedup
      compared to the older pure-Python implementation.

    * `con_values` & `con_indices` can be obtained by using the `to_numpy`
       method of the `Constraints` class.

    *  For benchmark results, see [here](../../../../benchmarks/internal/bm_randint_constrained.md)

    PRIORITIES:

    1) Provide exactly 'k' unique samples (no replacement)
    2) if provided, don't generate samples from i_forbidden   (can be used to indicate already sampled values)
    3) satisfy constraints
    4) if p is provided, don't sample from integers with p=0

    :param n: range to sample from [0, n)
    :param k: number of unique samples to draw (no replacement)
    :param con_values: 2D array (m, 2) with min_count and max_count for each constraint              (never modified!)
    :param con_indices: 1D array with constraint indices in the format described in _constraints.py  (never modified!)
    :param p: optional, target probabilities for each integer in `[0, n)`                            (never modified!)
    :param seed: random seed
    :param eager: if True, the algorithm will try to satisfy as many constraints as early as possible; in some cases
                  increasing the probability of finding a feasible solution, albeit at the cost of sampling diversity
                  and adherence to the provided p-values.
    :param k_context: (int, default=-1) number of total samples - in the bigger context - we want to sample in order to
                        satisfy the constraints.  This informs the algorithm about the urgency of fulfilling
                        constraints, giving it potentially more liberty to pick from a wider range of samples and with
                        potentially higher p-values.

                      Two cases:
                        a) not provided or <=k:  the algorithm assumes k_context = k
                        b) provided and >k:      the algorithm knows that more samples will be drawn later.

    :param i_forbidden: (optional) 1D array of integers in `[0, n)` that must not be sampled         (never modified!)

    :return: array of samples
    """

    # --- parameter validation ----------------------------
    n_forbidden = i_forbidden.shape[0]
    if k > (n - n_forbidden):
        if n_forbidden:
            raise ValueError(
                f"Cannot sample {k} unique integers from [0, {n}) when {n_forbidden} integers are forbidden."
                f"  ({k} > {n}-{n_forbidden})"
            )
        else:
            raise ValueError(f"Cannot sample {k} unique integers from [0, {n}). ({k} > {n})")

    # --- initialize --------------------------------------
    if k_context < k:
        k_context = k
    samples = np.empty(k, dtype=np.int32)
    k_remaining = k_context
    m = con_values.shape[0]

    # Make a copy of con_values to track current min/max counts
    con_values_working = con_values.copy()

    sample_idx = np.int32(0)

    # --- pre-process p -----------------------------------
    # we construct an 'augmented p' aug_p, which is identical to p, except small entries are adjusted to be >0,
    # avoiding issues later on when we exclude certain elements due to constraint-violation, which might otherwise
    # cause all p-values to become zero.
    if p.size == 0:
        # no p provided --> uniform
        p_aug = np.ones(n, dtype=np.float32)
    else:
        # determine p_max
        p_max = np.float32(0.0)
        for i in range(n):
            p_max = max(p_max, p[i])

        # construct p_aug by adding small value to each p
        if p_max == 0.0:
            # all p are zero --> uniform
            p_aug = np.ones(n, dtype=np.float32)
        else:
            p_delta = np.float32(1e-12 * p_max)
            p_aug = p.copy()
            for i in range(n):
                p_aug[i] += p_delta

    # --- sample ------------------------------------------
    for _ in range(k):
        # --- score & thresholds ----------------

        # Get already sampled integers
        # (we include i_forbidden, since they're excluded from sampling, with equal priority as already sampled values)
        if n_forbidden:
            already_sampled = np.concatenate((samples[:sample_idx], i_forbidden))
        else:
            already_sampled = samples[:sample_idx]

        # determine how much each integer would help us satisfy min_count constraints
        score = _compute_score(n, con_values_working, con_indices, already_sampled, True)

        # determine how much improvement we need to be able to satisfy all min_count constraints
        total_score_needed = np.int32(0)
        for i_con in range(m):
            min_val = _np_con_min_value(con_values_working, np.int32(i_con))
            if min_val > 0:
                total_score_needed += min_val

        score_threshold = np.int32((total_score_needed + k_remaining - 1) // k_remaining)  # ceil division

        max_score = np.int32(-(2**30))
        for s in score:
            if s > max_score:
                max_score = s

        if max_score >= score_threshold:
            # at this point, it still seems possible to satisfy all min_count constraints with the
            # remaining # of samples we have.
            #  --> STRATEGY 1: focus on those samples that help us enough to satisfy all constraints with the
            #                  remaining # of samples we have, and do not sample from any of the others.
            if eager:
                # if eager, we only focus on those candidate samples with the highest score
                # (focus on 'best' samples, instead of 'good enough' samples)
                score_threshold = max_score
        else:
            # we cannot satisfy all constraints with the k remaining samples.
            #  --> STRATEGY 2: choose samples with best net effect (help achieve min_count vs not violating max_count),
            #                  still hard-excluding already sampled integers.
            score = _compute_score(n, con_values_working, con_indices, already_sampled, False)
            max_score = np.int32(-(2**30))
            for s in score:
                if s > max_score:
                    max_score = s
            score_threshold = max_score

        # --- sample according to strategy ------

        # zero out probabilities for scores below threshold  (there will always be at least 1 we don't zero out)
        p_mod = p_aug.copy()
        for i in range(n):
            if score[i] < score_threshold:
                p_mod[i] = np.float32(0.0)

        # sample one integer
        result = randint_numba(n, np.int32(1), False, p_mod, seed)
        s = result[0]

        # --- update stats --------------------------------
        for i_con in range(m):
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


# =================================================================================================
#  randint_constrained_robust
# =================================================================================================
@numba.njit
def randint_constrained_robust(
    n: np.int32,
    k: np.int32,
    con_values: NDArray[np.int32],
    con_indices: NDArray[np.int32],
    p: NDArray[np.float32] = np.zeros(0, dtype=np.float32),
    seed: np.int64 = 0,
    n_trials: int = 5,
) -> NDArray[np.int32]:
    """
    Robust version of `randint_constrained` that tries multiple times to find a feasible solution.

    Strategy:
      - attempt 0:              randint(...)
                                        --> this is very cheap, and we might get lucky if constraints are loose
      - attempt 1:              randint_constrained(..., eager=False)
                                        --> try once with constraints
      - attempt 2...n_trials-1: randint_constrained(..., eager=True)
                                        --> try multiple times with eager=True to maximize chance of feasibility

    :param n: range to sample from [0, n)
    :param k: number of unique samples to draw (no replacement)
    :param con_values: 2D array (m, 2) with min_count and max_count for each constraint
    :param con_indices: 1D array with constraint indices in the format described in _constraints.py
    :param p: optional, target probabilities for each integer in `[0, n)`
    :param seed: random seed
    :param n_trials: number of trials to attempt (>=3)
    """

    # --- init ------------------------
    if n_trials < 3:
        raise ValueError(f"n_trials must be at least 3, here: {n_trials}")

    # --- main loop -------------------
    for i in range(n_trials):
        if i == 0:
            # attempt 0: pure random sampling
            samples = randint_numba(n, k, False, p, seed + i)
        elif i == 1:
            # attempt 1: constrained sampling, non-eager
            samples = randint_constrained(n, k, con_values, con_indices, p, seed + i, eager=False)
        else:
            # attempt 2...n_trials-1: constrained sampling, eager
            samples = randint_constrained(n, k, con_values, con_indices, p, seed + i, eager=True)

        # --- check feasibility -------
        if _np_con_satisfied(con_values, con_indices, samples):
            return samples

    # --- failed all attempts -----
    return samples
