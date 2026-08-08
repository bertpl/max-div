"""Numba-accelerated function for constrained sampling.

This function is roughly equivalent with randint, but supports defining constraints via the `con_values` and
`con_indices` parameters.
"""

import numba
import numpy as np
from numpy.typing import NDArray

from max_div._core.constraints._constraints import _np_con_min_value

from ._constraint_score_state import activate_soft_scores, apply_draw, new_constraint_score_state
from ._randint import randint1


# =================================================================================================
#  randint_constrained
# =================================================================================================
@numba.njit(fastmath=True, cache=True)
def randint_constrained(  # noqa: C901 — case-dispatch structure is clearer un-split
    n: np.int32,
    k: np.int32,
    con_values: NDArray[np.int32],
    con_indices: NDArray[np.int32],
    item_con_indices: NDArray[np.int32],
    rng_state: NDArray[np.uint64],
    p: NDArray[np.float32] = np.zeros(0, dtype=np.float32),  # noqa: B008 — numba needs a concrete typed default
    eager: bool = False,
    k_context: np.int32 = np.int32(-1),  # noqa: B008 — numba needs a concrete typed default
    i_forbidden: NDArray[np.int32] = np.empty(0, dtype=np.int32),  # noqa: B008 — numba needs a concrete typed default
) -> NDArray[np.int32]:
    """Generate `k` unique random integers from the range `[0, n)` while satisfying given constraints.

    Notes:
    * no guarantees are given that the solution will satisfy all constraints; a best-effort attempt will be made,
    with the probability of the result satisfying the constraints increasing the simpler & less strict the
    constraints are.

    * `randint_constrained` is essentially a version of randint that supports constraints.

    * This version is numba-accelerated and uses efficient numpy-based data structures, resulting in 10-100x speedup
      compared to equivalent pure-Python implementations.

    * `con_values`, `con_indices` & `item_con_indices` can be obtained by using the `to_numpy`
       method of the `ConstraintList` class.

    * Constraint-satisfaction scores are maintained incrementally across the k draws (see
      `_constraint_score_state.py`), so the per-draw cost does not depend on the total constraint
      membership size.

    *  For benchmark results, see [here](../../../../../docs/benchmarks/internal/bm_randint_constrained.md)

    PRIORITIES that this algorithm adheres to:

     1) Provide exactly 'k' unique samples (no replacement)
     2) if provided, don't generate samples from i_forbidden   (can be used to indicate already sampled values)
     3) satisfy constraints
     4) if p is provided, don't sample from integers with p=0

    :param n: range to sample from [0, n)
    :param k: number of unique samples to draw (no replacement)
    :param con_values: 2D array (m, 2) with min_count and max_count for each constraint              (never modified!)
    :param con_indices: 1D array with constraint indices in the format described in _constraints.py  (never modified!)
    :param item_con_indices: 1D array with the transposed constraint membership (same module)        (never modified!)
    :param p: optional, target probabilities for each integer in `[0, n)`                            (never modified!)
    :param rng_state: (2-element uint64 array) state for random number generation; updated in-place.
                                (use new_rng_state(seed) to construct an initial state)            (modified in-place)
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
        raise ValueError(f"Cannot sample {k} unique integers from [0, {n}). ({k} > {n})")

    # --- initialize --------------------------------------
    if k_context < k:
        k_context = k
    samples = np.empty(k, dtype=np.int32)
    k_remaining = k_context
    m = con_values.shape[0]

    # Score state for this call: prices every integer once, then stays current draw by draw
    # (con_values itself is copied inside, honoring the never-modified contract above)
    state = new_constraint_score_state(n, con_values, con_indices, item_con_indices, i_forbidden)

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

        # how much each integer would help us satisfy min_count constraints (maintained incrementally;
        # already-drawn and forbidden integers sit at the sampled marker)
        score = state.scores

        # determine how much improvement we need to be able to satisfy all min_count constraints
        total_score_needed = np.int32(0)
        for i_con in range(m):
            min_val = _np_con_min_value(state.con_values, np.int32(i_con))
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
            if state.soft_active[0] == 0:
                already_sampled = (
                    np.concatenate((samples[:sample_idx], i_forbidden)) if n_forbidden else samples[:sample_idx]
                )
                activate_soft_scores(state, n, already_sampled)
            score = state.scores_soft
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
        s = randint1(n=n, p=p_mod, rng_state=rng_state)

        # --- update stats --------------------------------
        apply_draw(state, np.int32(s))

        samples[sample_idx] = s
        sample_idx += 1
        k_remaining -= 1

    # --- done ----------------------------------------
    return samples
