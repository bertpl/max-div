import numpy as np
from numpy.typing import NDArray


def randint_python(
    n: np.int32 | int,
    k: np.int32 | int,
    replace: bool,
    p: NDArray[np.float32] | None = None,
    seed: int | None = None,
) -> NDArray[np.int32]:
    """
    This is the non-numba, numpy-based equivalent of randint, purely implemented for speed comparison purposes.
    :param n: (int) upper bound of range to sample from [0, n)
    :param k: (int) number of samples to be returned
    :param replace: (bool) whether sampling is with replacement
    :param p: (np.array[np.float32] | None) optional probability weights for sampling, if None, sampling is uniform.
    :param seed: (int|None) optional random seed for reproducibility.
    :return: np.array[np.int32] of shape (k,) containing the sampled integers.
    """

    # --- argument handling ---------------------------
    if k is None:
        k = 1  # indicates single sample
    if k == 1:
        replace = True  # single sample, replacement makes no difference, so we can fall back to faster methods

    # --- argument validation -------------------------
    if n < 1:
        raise ValueError(f"n must be >=1. (here: {n})")
    if k < 1:
        raise ValueError(f"k must be >=1. (here: {k})")
    if (not replace) and (k > n):
        raise ValueError(f"Cannot sample {k} unique values from range [0, {n}) without replacement.")
    if p is not None:
        if (p.size > 0) and (p.size != n):
            raise ValueError(f"p must be of size n=0 or n={n}. (here: size={p.size})")
        elif p.size == 0:
            p = None  # indicates no probabilities were specified

    # --- sampling ------------------------------------
    if seed:
        np.random.seed(seed)
    if p is not None:
        p = p * (1.0 / np.sum(p))  # normalize probabilities, which numpy requires

    # always specify 'k', even if k==1, to ensure we always return an array
    return np.random.choice(n, size=k, replace=replace, p=p).astype(np.int32)
