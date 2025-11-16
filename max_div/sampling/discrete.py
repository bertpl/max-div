import numba
import numpy as np

from max_div.internal.math.fast_log import fast_log2_f32_poly
from max_div.internal.math.random import GLOBAL_RNG_STATE, rand_float32, rand_int32, set_seed


# =================================================================================================
#  sample_int
# =================================================================================================
def sample_int(
    n: int,
    k: int | None = None,
    replace: bool = True,
    p: np.ndarray[np.float32] | None = None,
    seed: int | None = None,
    use_numba: bool = True,
) -> int | np.ndarray[np.int64]:
    """
    Randomly sample `k` integers from range `[0, n-1]`, optionally with replacement and per-value probabilities.

    Depending on the value of `use_numba`, computations are executed by...

    - `use_numba=False`: see [sample_int_numpy][max_div.sampling.discrete.sample_int_numpy]
    - `use_numba=True`: see [sample_int_numba][max_div.sampling.discrete.sample_int_numba]

    :param n: defines population to sample from as range [0, n-1].  `n` must be >0.
    :param k: The number of integers to sample (>0).  `k=None` indicates a single integer sample.
    :param replace: Whether to sample with replacement.
    :param p: Optional 1D array of probabilities associated with each integer in the range.
              Size must be equal to max_value + 1 and sum to 1.
              (if `None` or size==0, uniform sampling is performed)
    :param seed: Optional random seed for reproducibility. If `None` or 0, no seed is set.
    :param use_numba: Use the custom numba-accelerated implementation, otherwise we use `np.random.choice`.
    :return: `k=None` --> single integer; `k>=1` --> (k,)-sized array with sampled integers.
    """
    n = np.int32(n)
    k = np.int32(k) if k is not None else None

    if not use_numba:
        return sample_int_numpy(n=n, k=k, replace=replace, p=p, seed=seed)

    else:
        # NOTE: minimal validation to make sure numba doesn't fail to compile
        if (p is not None) and p.ndim != 1:
            raise ValueError(f"p must be a 1D array. (here: ndim={p.ndim})")

        # NOTE: we need a few if-clauses, since numba does not support optional arguments

        if k is None:
            # assume k=1 and return an integer
            if p is None:
                return sample_int_numba(n=n, k=1, replace=replace, seed=seed or 0)[0]
            else:
                return sample_int_numba(n=n, k=1, replace=replace, p=p, seed=seed or 0)[0]

        else:
            # k is specified, return array
            if p is None:
                return sample_int_numba(n=n, k=k, replace=replace, seed=seed or 0)
            else:
                return sample_int_numba(n=n, k=k, replace=replace, p=p, seed=seed or 0)


# =================================================================================================
#  sample_int_numpy
# =================================================================================================
def sample_int_numpy(
    n: np.int32,
    k: np.int32 | None = None,
    replace: bool = True,
    p: np.ndarray[float] | None = None,
    seed: np.int64 | None = None,
) -> np.int32 | np.ndarray[np.int32]:
    """
    Randomly sample `k` integers from range `[0, n-1]`, optionally with replacement and per-value probabilities.

    This will always use `np.random.choice` for sampling and is intended to be used to compare against the
    numba-accelerated version.  For production-use, use `sample_int_numba` or `sample_int` with `accelerated=True`.

    :param n: defines population to sample from as range [0, n-1].  `n` must be >0.
    :param k: The number of integers to sample (>0).  `k=None` indicates a single integer sample.
    :param replace: Whether to sample with replacement.
    :param p: Optional 1D array of probabilities associated with each integer in the range.
              Size must be equal to max_value + 1 and sum to 1.
    :param seed: Optional random seed for reproducibility.  If `None` or 0, no seed is set.
    :return: `k=None` --> single integer; `k>=1` --> (k,)-sized array with sampled integers.
    """

    # --- argument handling ---------------------------
    if (k == 1) or (k is None):
        replace = True  # single sample, replacement makes no difference, so we can fall back to faster methods

    # --- argument validation -------------------------
    if n < 1:
        raise ValueError(f"n must be >=1. (here: {n})")
    if k is not None:
        if k < 1:
            raise ValueError(f"k must be >=1. (here: {k})")
        if (not replace) and (k > n):
            raise ValueError(f"Cannot sample {k} unique values from range [0, {n}) without replacement.")
    if p is not None:
        if (p.size > 0) and (p.size != n):
            raise ValueError(f"p must be of size n=0 or n={n}. (here: size={p.size})")
        elif p.size == 0:
            p = None  # indicate no probabilities specified

    # --- sampling ------------------------------------
    if seed:
        np.random.seed(seed)

    if k is None:
        # returns scalar
        return np.int32(np.random.choice(n, size=None, replace=replace, p=p))
    else:
        # returns array
        return np.random.choice(n, size=k, replace=replace, p=p).astype(np.int32)


# =================================================================================================
#  sample_int_numba
# =================================================================================================
@numba.njit(fastmath=True)
def sample_int_numba(
    n: np.int32,
    k: np.int32,
    replace: bool,
    p: np.ndarray[np.float32] = np.zeros(0, dtype=np.float32),
    seed: np.int64 = 0,
) -> np.ndarray[np.int32]:
    """
    Randomly sample `k` integers from range `[0, n-1]`, optionally with replacement and per-value probabilities.

    This is a custom numba, speed-optimized implementation, using a different algorithm depending on the case:

    | `p` specified  | `replace`  | `k`   | Method Used                              | Complexity      |
    |----------------|------------|-------|------------------------------------------|-----------------|
    | No             | `True`     | *any* | `np.random.randint`, uniform sampling    | O(k)            |
    | No             | `False`    | *any* | k-element Fisher-Yates shuffle           | O(n)            |
    | Yes            | *any*      | 1     | Multinomial sampling using CDF           | O(n + log(n))   |
    | Yes            | `True`     | >1    | Multinomial sampling using CDF           | O(n + k log(n)) |
    | Yes            | `False`    | >1    | Efraimidis-Spirakis sampling + exponential key sampling (Gumbel-Max Trick).  | O(n) |

    NOTES:
     - using the np.random.Generator API incurs an extra 3-4 μsec overhead per call compared to using the legacy
       np.random functions. The main reason is that the new interface requires calls through the numpy C-API, while the
       legacy functions are re-implemented in Numba and compiled together with the rest of the numba-accelerated code.
       Instantiating a Generator incurs a ~10 μsec penalty, so should also be avoided to be done repeatedly.
     - given the intended use-case within max_div, it is acceptable that provided probabilities are only approximately
       taken into account.  Therefore, we use float32 representation and use a fast-approx-log function in the
       Efraimidis-Spirakis sampling method.  Overall this can result in <1% deviation from target probabilities, i.e.
         p[3] = 0.1 --> actual frequency in samples = [0.099 to 0.101].

    <br>

    :param n: defines population to sample from as range [0, n-1].  `n` must be >0.
    :param k: The number of integers to sample (>0).  `k=None` indicates a single integer sample.
    :param replace: Whether to sample with replacement.
    :param p: Optional 1D array of probabilities associated with each integer in the range.
              Size must be equal to max_value + 1 and sum to 1.
              NOTE: if size is 0, indicates no probabilities specified.  (=DEFAULT)
                    if size > 0, but not equal to max_value+1, a ValueError is raised.
    :param seed: (default=0) Optional random seed for reproducibility. If `None` or 0, no seed is set.
    :return: (k,)-sized array with sampled integers.
    """

    if n < 1:
        raise ValueError(f"n must be >=1. (here: {n})")
    if k < 1:
        raise ValueError(f"k must be >=1. (here: {k})")
    if k == 1:
        replace = True  # single sample, replacement makes no difference, so we can fall back to faster methods
    elif (not replace) and (k > n):
        raise ValueError(f"Cannot sample {k} unique values from range [0, {n}) without replacement.")

    if seed != 0:
        rng_state = set_seed(seed)
    else:
        rng_state = set_seed(np.random.randint(1, 1_000_000_000_000))

    if p.size == 0:
        if replace:
            # UNIFORM sampling with replacement
            samples = np.empty(k, dtype=np.int32)
            for i in range(k):
                samples[i] = rand_int32(rng_state, 0, n)
            return samples
        else:
            # UNIFORM sampling without replacement using Fisher-Yates shuffle
            population = np.arange(n, dtype=np.int32)  # O(n)
            for i in range(k):  # k x O(1)
                j = rand_int32(rng_state, i, n)
                population[i], population[j] = population[j], population[i]
            return population[:k]  # O(k)

    elif p.size == n:
        if replace:
            # NON-UNIFORM sampling with replacement using CDF
            cdf = np.cumsum(p)  # O(n)
            samples = np.empty(k, dtype=np.int32)  # O(k)
            # note: computing the below in a loop, is faster than writing a np-vectorized one-liner
            for i in range(k):  # k x O(log(n))
                r = rand_float32(rng_state)
                idx = np.searchsorted(cdf, r)
                samples[i] = idx
            return samples
        else:
            # NON-UNIFORM sampling without replacement using Efraimidis-Spirakis + Exponential keys
            # algorithm description:
            #   Efraimidis:       select k elements corresponding to k largest values of  u_i^{1/p_i} (u_i ~ U(0,1))
            #   Gumbel-Max Trick: select k smallest values of  -log(u_i)/p_i  (u_i ~ U(0,1))
            #   Ziggurat:         INVESTIGATE: generate log(u_i) more efficiently, applying the Ziggurat algorithm
            #                            to the exponential distribution, which avoids usage of transcendental
            #                            functions for the majority of the samples.
            #                     (Initial testing surprisingly did not show improvements)
            if k < n:
                keys = np.empty(n, dtype=np.float32)  # O(n)
                # note: computing -np.log(u[i]) does not seem to be noticeably slower than np.random.standard_exponential().
                for i in range(n):  # n x O(1)
                    if p[i] == 0.0:
                        keys[i] = np.inf
                    else:
                        ui = rand_float32(rng_state)
                        # NOTE: we use a fast log2 approximation here for speed; log2 vs log is irrelevant since
                        #       it's just a scaling factor, and we are only interested in the order of the final list
                        keys[i] = -fast_log2_f32_poly(ui, degree=2) / p[i]  # using fast log2 approximation

                # Get indices of k smallest keys
                return np.argpartition(keys, k)[:k].astype(np.int32)  # O(n) average case

            else:
                # corner case: return all elements in random order
                # to this end we perform 1 full Fisher-Yates shuffle
                population = np.arange(n, dtype=np.int32)  # O(n)
                for i in range(n):  # n x O(1)
                    j = rand_int32(rng_state, i, n)
                    population[i], population[j] = population[j], population[i]
                return population[:k]  # O(k)

    else:
        raise ValueError(f"p must be of size 0 (no probabilities) or size n={n}. (here: size={p.size})")
