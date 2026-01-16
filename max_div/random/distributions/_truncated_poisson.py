import numpy as np
from numba import njit
from numpy.typing import NDArray

from max_div.random import randint1


@njit(fastmath=True, inline="always")
def sample_truncated_poisson(
    min_value: np.int32,
    max_value: np.int32,
    _lambda: np.float32,
    rng_state: NDArray[np.uint64],
):
    """
    Generate single int32 sample from a two-sided truncated Poisson distribution with given min and max values

    Regular Poisson distribution formula:
        P(X=k) = (λ^k * e^(-λ)) / k!

    In this implementation all probabilities for x<min_value or x>max_value are set to 0.  Renormalization is not
      needed in our case, as our sampling methods inherently work with unnormalized probabilities.

    NOTE: we can also omit the constant factor e^(-λ) as it is the same for all k and we don't need normalized p.

    :param min_value: (np.int32) minimum value (inclusive)
    :param max_value: (np.int32) maximum value (inclusive)
    :param _lambda: (np.float32) lambda parameter of the Poisson distribution
    :param rng_state: (np.uint64 array) RNG state for underlying random sampling
    :return: (np.int32) single sample from the truncated Poisson distribution in range [min_value, max_value]
    """

    # --- generate p ------------------
    p = np.zeros(max_value - min_value + 1, dtype=np.float32)
    for k in range(min_value, max_value + 1):
        if k == min_value:
            p[0] = 1.0  # we can start from any arbitrary value, as we don't need normalization
        else:
            p[k - min_value] = p[k - min_value - 1] * (_lambda / np.float32(k))

    # --- sample ----------------------
    return randint1(n=np.int32(p.shape[0]), p=p, rng_state=rng_state) + min_value


@njit(fastmath=True, inline="always")
def truncated_poisson_expected_value(min_value: np.int32, max_value: np.int32, _lambda: np.float32) -> np.float32:
    """
    Compute expected value of a two-sided truncated Poisson distribution with given min, max & lambda values

    :param min_value: (np.int32) minimum value (inclusive)
    :param max_value: (np.int32) maximum value (inclusive)
    :param _lambda: (np.float32) lambda parameter of the Poisson distribution
    :return: (np.float32) expected value of the truncated Poisson distribution in range [min_value, max_value]
    """

    # --- compute p -------------------
    p = np.zeros(max_value - min_value + 1, dtype=np.float32)
    for k in range(min_value, max_value + 1):
        if k == min_value:
            p[0] = 1.0  # we can start from any arbitrary value, as we don't need normalization
        else:
            p[k - min_value] = p[k - min_value - 1] * (_lambda / np.float32(k))

    # --- compute expected value -------
    sum_p_k = np.float32(0.0)
    sum_p = np.float32(0.0)
    for k in range(min_value, max_value + 1):
        sum_p_k += p[k - min_value] * np.float32(k)
        sum_p += p[k - min_value]

    return sum_p_k / sum_p
