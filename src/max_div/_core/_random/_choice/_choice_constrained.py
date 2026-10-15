import numpy as np
from numpy.typing import NDArray

from max_div._core._random._randint import randint_constrained
from max_div._core.jit import lazy_njit


@lazy_njit(
    "int32[::1](int32, int32[::1], int32, float32[::1], uint64[::1], int32[:, ::1], int32[::1], boolean, int32)",
    fastmath=True,
    cache=True,
)
def choice_constrained(
    n: np.int32,
    values: NDArray[np.int32],
    k: np.int32,
    p: NDArray[np.float32],
    rng_state: NDArray[np.uint64],
    con_values: NDArray[np.int32],
    con_indices: NDArray[np.int32],
    eager: bool,
    k_context: np.int32,
) -> NDArray[np.int32]:
    """Sample from a provided 'values' array instead of the range [0, n), respecting the provided constraints.

    This function is to 'randint_constrained' what 'choice' is to 'randint'.

    Notes:
      - `con_values` and `con_indices` can be constructed using to_numpy_constraints(constraints, n)
      - `con_indices` refers to values of the `values` array, not indices into it.

    Args:
        n: (np.int32) specify the 'encompassing range' [0, n) from which 'values' are drawn and to which
            `con_indices` are restricted.
        values: (NDArray[np.int32]) The array of values to sample from.
        k: (np.int32) The number of samples to draw.
        p: (NDArray[np.float32]) The probabilities associated with each value. If provided with a (0,)-sized array,
            uniform sampling is used.  The constant 'P_UNIFORM' can be used for this purpose.
        rng_state: (NDArray[np.uint64]) The RNG state used (and updated in-place) for sampling.
        con_values: 2D array (m, 2) with min_count and max_count for each constraint              (never modified!)
        con_indices: 1D array with constraint indices in the format described in _constraints.py  (never modified!)
        eager: if True, the algorithm will try to satisfy as many constraints as early as possible; in some cases
            increasing the probability of finding a feasible solution, albeit at the cost of sampling diversity
            and adherence to the provided p-values.
        k_context: total number of samples, including the k of this call and any that later calls will draw,
            that must together satisfy the constraints.  This informs the algorithm about the urgency of
            fulfilling constraints, giving it potentially more liberty to pick from a wider range of samples and
            with potentially higher p-values.

                      Two cases:
                        a) <=k (for example -1):  the algorithm assumes k_context = k
                        b) >k:                    the algorithm knows that more samples will be drawn later.

    Returns:
        (NDArray[np.int32]) The array of samples, size (k,), taken from the provided 'values' array.
    """
    # --- argument handling ----------------------
    if not ((p.size == 0) or (p.size == values.size)):
        raise ValueError(f"p array size should be either 0 or same size as 'values' ({values.size}). [here: {p.size}]")
    if k > values.size:
        raise ValueError(f"Cannot sample k={k} values from 'values' array of size {values.size} without replacement.")

    # --- construct p_full -----------------------
    #  p_full: (n,)-sized array with probabilities for each value in [0, n), setting to 0.0 those not in 'values'
    if p.size == 0:
        # uniform sampling
        p_full = p  # pass on the 0-sized array to randint_constrained, indicating we want uniform sampling
    else:
        # non-uniform sampling
        p_full = np.zeros(n, dtype=np.float32)
        p_full[values] = p

    # --- construct i_forbidden ------------------
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

    # --- call randint_constrained ---------------
    return randint_constrained(
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

    # --- return values --------------------------
