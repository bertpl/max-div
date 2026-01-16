import numba
import numpy as np
from numpy.typing import NDArray

from max_div.random._randint import randint_constrained


@numba.njit(fastmath=True, cache=True)
def choice_constrained(
    n: np.int32,
    values: NDArray[np.int32],
    k: np.int32,
    p: NDArray[np.float32],
    rng_state: NDArray[np.uint64],
    con_values: NDArray[np.int32],
    con_indices: NDArray[np.int32],
    eager: bool = False,
    k_context: np.int32 = np.int32(-1),
) -> NDArray[np.int32]:
    """
    This function is to 'randint_constrained' what 'choice' is to 'randint': it samples from a provided 'values' array,
    instead of the range [0, n), while respecting the provided constraints.

    NOTES:
      - `con_values` and `con_indices` can be constructed using ConstraintList(constraints).to_numpy()
      - `con_indices` refers to values of the `values` array, not indices into it.

    :param n: (np.int32) specify the 'encompassing range' [0, n) from which 'values' are drawn and to which
                           `con_indices` are restricted.
    :param values: (NDArray[np.int32]) The array of values to sample from.
    :param k: (np.int32) The number of samples to draw.
    :param p: (NDArray[np.float32]) The probabilities associated with each value. If provided with a (0,)-sized array,
                                    uniform sampling is used.  The constant 'P_UNIFORM' can be used for this purpose.
    :param rng_state: (NDArray[np.uint64]) The RNG state used (and updated in-place) for sampling.
    :param con_values: 2D array (m, 2) with min_count and max_count for each constraint              (never modified!)
    :param con_indices: 1D array with constraint indices in the format described in _constraints.py  (never modified!)
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
    :return: (NDArray[np.int32]) The array of samples, size (k,), taken from the provided 'values' array.
    """

    # --- argument handling -------------------------------
    if not ((p.size == 0) or (p.size == values.size)):
        raise ValueError(f"p array size should be either 0 or same size as 'values' ({values.size}). [here: {p.size}]")
    if k > values.size:
        raise ValueError(f"Cannot sample k={k} values from 'values' array of size {values.size} without replacement.")

    # --- construct p_full --------------------------------
    #  p_full: (n,)-sized array with probabilities for each value in [0, n), setting to 0.0 those not in 'values'
    if p.size == 0:
        # uniform sampling
        p_full = p  # pass on the 0-sized array to randint_constrained, indicating we want uniform sampling
    else:
        # non-uniform sampling
        p_full = np.zeros(n, dtype=np.float32)
        p_full[values] = p

    # --- construct i_forbidden ---------------------------
    #  i_forbidden: (n-len(values),)-sized array with values in [0,n) that are NOT in 'values' and should not be sampled
    is_in_values = np.zeros(n, dtype=np.bool_)
    for v in values:
        is_in_values[v] = True
    i_forbidden = np.empty(n - values.size, dtype=np.int32)
    idx = 0
    for i in range(n):
        if not is_in_values[i]:
            i_forbidden[idx] = i
            idx += 1

    # --- call randint_constrained ------------------------
    i_samples = randint_constrained(
        n=n,
        k=k,
        p=p_full,
        rng_state=rng_state,
        con_values=con_values,
        con_indices=con_indices,
        i_forbidden=i_forbidden,
        eager=eager,
        k_context=k_context,
    )

    # --- return values -----------------------------------
    return i_samples
