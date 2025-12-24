import numpy as np
from numba import njit
from numpy.typing import NDArray

from max_div.internal.math.fast_log_exp import fast_exp2_f32


@njit(fastmath=True, inline="always")
def exponential_selectivity(
    p_in: NDArray[np.float32],
    p_out: NDArray[np.float32],
    modifier: np.float32,
    descending: bool = False,
    low_value: np.float32 = np.float32(0.1),
):
    """
    Takes float32-array p_in of shape (n,) containing non-normalized probabilities in range [p_min, p_max].
    p_out is populated with values exponentially depending on the corresponding p_in values, such that p_out values
    are in range [1.0, low_value**t], where t is computed as in the other modification methods:

            t = (1.0 + modifier) / (1.0 - modifier)

    This then boils down to the following formulas:

     - descending=False (default)    p_out[i] = low_value ** (t * (p_max - p_in[i]) / (p_max - p_min))
     - descending=True               p_out[i] = low_value ** (t * (p_max - p_in[i]) / (p_max - p_min))

    The approximate function fast_exp2(exponent * np.log2(base)) is used to compute the exponentiation efficiently,
      taking into account that np.log2(low_value) can be precomputed outside the loop.

    :param p_in: np.ndarray of shape (n,) containing the original probabilities.
    :param p_out: np.ndarray of shape (n,) to be populated with the transformed probabilities.
    :param modifier: float32 in (-1, 1) indicating how to modify selectivity
    :param descending: (bool, default False) if True, higher p_in values result in lower p_out values
    :param low_value: (float, default 0.1) the lowest value in p_out
    """

    # --- init ----------------------------------
    n = p_in.shape[0]
    p_min = np.float32(np.inf)
    p_max = np.float32(-np.inf)
    for i in range(n):
        pi = p_in[i]
        p_min = min(p_min, pi)
        p_max = max(p_max, pi)
    p_range = p_max - p_min

    # --- corner case ---------------------------
    if p_range == 0.0:
        for i in range(n):
            p_out[i] = np.float32(1.0)
        return

    # --- actual transformation -----------------

    # precompute values
    t = (np.float32(1.0) + modifier) / (np.float32(1.0) - modifier)
    p_range_inv = np.float32(1.0) / p_range
    log2_low_value = np.float32(np.log2(low_value))
    t_times_log2_low_value = t * log2_low_value

    # main loop
    if descending:
        for i in range(n):
            exponent = (p_in[i] - p_min) * p_range_inv
            p_out[i] = fast_exp2_f32(exponent * t_times_log2_low_value)
    else:
        for i in range(n):
            exponent = (p_max - p_in[i]) * p_range_inv
            p_out[i] = fast_exp2_f32(exponent * t_times_log2_low_value)
