from dataclasses import dataclass

import numpy as np

from max_div.sampling import randint_numba


@dataclass()
class Constraint:
    """Constraint indicating we want to sample at least `min_count` and at most `max_count` integers from `int_set`."""

    int_set: set[int]
    min_count: int
    max_count: int


# =================================================================================================
#  randint_constrained
# =================================================================================================
def randint_constrained(
    n: int,
    k: int,
    cons: list[Constraint],
    p: np.ndarray | None = None,
) -> list[int]:
    """
    Generate `k` unique random integers from the range `[0, n)` while satisfying given constraints.

    Note that no guarantees are given that constraints are satisfied; a best-effort attempt will be made, with the
    probability of the result satisfying the constraints increasing the simpler the constraints are.

    :param n: range to sample from [0, n)
    :param k: number of unique samples to draw (no replacement)
    :param cons: list of Constraint objects, indicating we want to sample at least `min_count`
                 and at most `max_count` integers from `int_set`, with all values of `int_set` in range `[0, n)`.
    :param p: optional, target probabilities for each integer in `[0, n)`. No guarantees are given if provided,
              but will help guide qualitative preference of sampling algorithm.  Higher p[i] values will increase
              probability of integer `i` being included in the sample, to the extent the constraints allow this.
    :return: list of samples
    """

    # --- initialize --------------------------------------
    samples = []
    k_remaining = k

    # --- sample ------------------------------------------
    while k_remaining > 0:
        # --- score by heuristics ---------------
        score = _score_by_contribution(n, cons, samples)
        if max(score) > 0:
            # there ARE samples that help us achieve min_count constraints without violating max_count constraints
            #  --> STRATEGY 1: focus on those samples that help us enough to satisfy all constraints with the
            #                  remaining # of samples we have, if possible.
            pass  # TODO: sample according to STRATEGY 1
        else:
            # there are NO samples that help us achieve min_count constraints without violating max_count constraints
            #  --> try to focus on those samples that at least do the least harm to max_count constraints
            score = _score_by_least_harming(n, cons, samples)

            if max(score) > 0:
                # at least there ARE samples that do not harm any max_count constraints
                #  --> STRATEGY 2: focus on those samples that do the least harm to max_count constraints
                pass  # TODO: sample according to STRATEGY 2
            else:
                # there are NO samples that do not harm any max_count constraints
                #  --> try to sample from remaining samples according to provided probabilities (if they're >0)
                if not p:
                    p = np.ones(n)  # uniform probabilities
                else:
                    p = p.copy()  # avoid modifying input array

                # set probabilities of already sampled integers to 0
                for i in samples:
                    p[i] = 0.0

                # check if any probabilities are >0
                if max(p) > 0:
                    # there are samples with non-zero probabilities
                    #  --> STRATEGY 3: sample according to provided probabilities
                    pass  # TODO: sample according to STRATEGY 3
                else:
                    # all remaining samples have zero probability
                    #  --> STRATEGY 4: just sample remaining samples randomly (uniformly)
                    p = np.ones(n)  # uniform probabilities
                    for i in samples:
                        p[i] = 0.0

                    pass  # TODO: sample according to STRATEGY 4

        # --- done ----------------------------------------
        return samples


# =================================================================================================
#  Helpers
# =================================================================================================
def _score_by_contribution(
    n: int,
    cons: list[Constraint],
    already_sampled: list[int],
) -> np.ndarray:
    """
    Score each integer in `[0, n)` by how much it helps satisfy the given constraints + eliminate those that would
    either violate constraints or that we already sampled.

    :param n: range to score [0, n)
    :param cons: list of Constraint objects, indicating we want to sample at least `min_count`
                 and at most `max_count` integers from `int_set`, with all values of `int_set` in range `[0, n)`.
    :param already_sampled: list of integers already sampled
    :return: array of scores for each integer in `[0, n)`
    """

    scores = np.zeros(n)

    # add 1 point to an integer for each constraint it helps satisfy
    for con in cons:
        for i in con.int_set:
            scores[i] += 1

    # zero out scores for integers in sets with max_count 0
    for con in cons:
        if con.max_count == 0:
            for i in con.int_set:
                scores[i] = 0

    # zero out scores for integers we already sampled
    for i in already_sampled:
        scores[i] = 0

    return scores


def _score_by_least_harming(
    n: int,
    cons: list[Constraint],
    already_sampled: list[int],
) -> np.ndarray:
    """
    Score each integer in `[0, n)` by how much it avoids violating the given constraints + eliminate those that would
    violate constraints or that we already sampled.

    :param n: range to score [0, n)
    :param cons: list of Constraint objects, indicating we want to sample at least `min_count`
                 and at most `max_count` integers from `int_set`, with all values of `int_set` in range `[0, n)`.
    :param already_sampled: list of integers already sampled
    :return: array of scores for each integer in `[0, n)`
    """

    scores = np.full(n, fill_value=len(cons))

    # add subtract 1 point to an integer for each constraint it does harm
    for con in cons:
        if con.max_count == 0:
            for i in con.int_set:
                scores[i] -= 1

    # zero out scores for integers we already sampled
    for i in already_sampled:
        scores[i] = 0

    return scores
