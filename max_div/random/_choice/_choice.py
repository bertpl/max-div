import numba
import numpy as np
from numpy.typing import NDArray

from max_div.random._randint import randint, randint1


@numba.njit("int32[:](int32[:], int32, bool, float32[:], uint64[:])", fastmath=True, cache=True)
def choice(
    values: NDArray[np.int32],
    k: np.int32,
    replace: bool,
    p: NDArray[np.float32],
    rng_state: NDArray[np.uint64],
) -> NDArray[np.int32]:
    """
    Identical in behavior to max_div.random.randint, except that it samples fro the 'values' array,
    instead of the range [0, n).

    This method is nearly a drop-in replacement for numpy.random.choice with similar parameters, optimized for speed.

    :param values: ((n,)-sized NDArray[np.int32]) The array of values to sample from.
    :param k: (np.int32) The number of samples to draw.
    :param replace: (bool) Whether to sample with replacement.
    :param p: (NDArray[np.float32]) The probabilities associated with each value. If provided with a (0,)-sized array,
                                    uniform sampling is used.  The constant 'P_UNIFORM' can be used for this purpose.
    :param rng_state: (NDArray[np.uint64]) The RNG state used (and updated in-place) for sampling.
    :return: (NDArray[np.int32]) The array of samples, size (k,), taken from the provided 'values' array.
    """

    # first, sample k values from range [0, n)
    n = np.int32(values.shape[0])
    samples = randint(n, k, replace, p, rng_state)

    # replace indices with actual values
    for i in range(k):
        samples[i] = values[samples[i]]

    # we're done
    return samples


@numba.njit("int32(int32[:], float32[:], uint64[:])", fastmath=True, inline="always", cache=True)
def choice1(
    values: NDArray[np.int32],
    p: NDArray[np.float32],
    rng_state: NDArray[np.uint64],
) -> np.int32:
    """
    Special case of choice for k=1.

    NOTE: this method guarantees that when probabilities are provided, the returned sample will always have p[i]>0.0.

    :param values: ((n,)-sized NDArray[np.int32]) The array of values to sample from.
    :param p: (NDArray[np.float32]) The probabilities associated with each value. If provided with a (0,)-sized array,
                                    uniform sampling is used.  The constant 'P_UNIFORM' can be used for this purpose.
    :param rng_state: (NDArray[np.uint64]) The RNG state used (and updated in-place) for sampling.
    :return: (np.int32) sample taken from 'values' array.
    """
    n = np.int32(values.size)
    i = randint1(n=n, p=p, rng_state=rng_state)
    return values[i]
