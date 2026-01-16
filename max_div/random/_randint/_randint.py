"""
Numba-accelerated implementation of a function roughly equivalent to `numpy.random.randint`, but significantly faster
due to use of faster underlying RNG functionality (see `max_div.random.rng`), and optimized algorithms for each case,
making use of a.o. the fact that we only want to sample int32 values here.
"""

import numba
import numpy as np
from numpy.typing import NDArray

from max_div.internal.math.fast_log_exp import fast_log2_f32
from max_div.internal.math.select_k_minmax import select_k_min
from max_div.random.rng import (
    rand_float32,
    rand_int32,
    rand_int32_array,
    rand_nz_float32,
)

_SMALLEST_F32 = np.finfo(np.float32).smallest_subnormal


# to be used to indicate no probabilities specified
P_UNIFORM = np.zeros(0, dtype=np.float32)


# =================================================================================================
#  randint
# =================================================================================================
@numba.njit("int32[:](int32, int32, bool, float32[:], uint64[:])", fastmath=True, cache=True)
def randint(
    n: np.int32,
    k: np.int32,
    replace: bool,
    p: NDArray[np.float32],
    rng_state: NDArray[np.uint64],
) -> NDArray[np.int32]:
    """
    Randomly sample `k` int32 values from range `[0, n-1]`, optionally with replacement and per-value probabilities.
    This function mimics the behavior of `numpy.random.randint` in that it provides values in range [0, n), but offers
    the additional options of sampling without replacement and with per-value probabilities that random.choice offers.

    This implementation uses numba, and is speed-optimized using a different algorithm depending on the case:

    | `p` specified  | `replace`  | `k`   | Method Used                              | Complexity      |
    |----------------|------------|-------|------------------------------------------|-----------------|
    | No             | `True`     | *any* | `np.random.randint`, uniform sampling    | O(k)            |
    | No             | `False`    | *any* | k-element Fisher-Yates shuffle           | O(n)            |
    | Yes            | *any*      | 1     | Multinomial sampling using CDF           | O(n + log(n))   |
    | Yes            | `True`     | >1    | Multinomial sampling using CDF           | O(n + k log(n)) |
    | Yes            | `False`    | >1    | Efraimidis-Spirakis sampling + exponential key sampling (Gumbel-Max Trick).  | O(n) |

    NOTES:

      - For benchmark results, see [here](../../../../benchmarks/internal/bm_randint.md)

      - When providing `p`...

          - we will never return a sample `i` for which `p[i]==0.0`, except when replace=False and there are <k options
            with probability p>0.0.

          - it is not needed to normalize p to sum to 1; any non-negative values are accepted.
            However, we do require that sum(p) > 0, such that it can be guaranteed we always return a sample
            with p[i] > 0.

          - given the intended use-case within max_div, it is acceptable that provided probabilities are only approximately
            taken into account.  Therefore, we use float32 representation and use a fast-approx-log function in the
            Efraimidis-Spirakis sampling method.  Overall this can result in <1% deviation from target probabilities, i.e.
              p[3] = 0.1 --> actual frequency in samples = [0.099 to 0.101].

    ALTERNATIVES CONSIDERED:

      - the updated np.random.Generator API provides options to choose faster underlying rng methods & has updated
        algorithms for certain sampling functions compared to the legacy np.random functions.  However, using the
        np.random.Generator API incurs an extra 3-4 μsec overhead per call compared to using the legacy
        np.random functions. The main reason is that the new interface requires calls through the numpy C-API, while the
        legacy functions are re-implemented in Numba and compiled together with the rest of the numba-accelerated code.
        Also, instantiating a Generator incurs a ~10 μsec penalty, so should also be avoided to be done repeatedly.
        The net effect is that no performance benefit could be obtained through this new API, making this implementation
        still the faster choice with a significant margin.

    <br>

    :param n: defines population to sample from as range [0, n-1].  `n` must be >0.
    :param k: The number of integers to sample (>0).  `k=None` indicates a single integer sample.
    :param replace: Whether to sample with replacement.
    :param p: 1D array of probabilities associated with each integer in the range.
              Size must be equal to max_value + 1, and should have non-negative values. Sum is not require to be 1.
              NOTE: if size is 0, indicates no probabilities specified.  Use the constant P_UNIFORM for this.
                    if size > 0, but not equal to max_value+1, a ValueError is raised.
    :param rng_state: (2-element uint64 array) RNG state to be used (and updated in-place)
                      Use new_rng_state(seed) to generate a new one based on an integer seed.
    :return: (k,)-sized array with sampled np.int32 values.
    """
    if n < 1:
        raise ValueError(f"n must be >=1. (here: {n})")
    if k < 1:
        raise ValueError(f"k must be >=1. (here: {k})")
    if k == 1:
        replace = True  # single sample, replacement makes no difference, so we can fall back to faster methods
    elif (not replace) and (k > n):
        raise ValueError(f"Cannot sample {k} unique values from range [0, {n}) without replacement.")

    if p.size == 0:
        if replace:
            # UNIFORM sampling with replacement
            return rand_int32_array(rng_state, 0, n, k)  # O(k)
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
            p_sum = cdf[-1]
            if p_sum <= 0.0:
                raise ValueError("Sum of probabilities in p must be > 0.0.")
            samples = np.empty(k, dtype=np.int32)  # O(k)
            # notes:
            #  - computing the below in a loop, is faster than writing a np-vectorized one-liner
            #  - implementing & calling a rand_float32_array outside the loop once is not faster
            for i in range(k):  # k x O(log(n))
                # ---------------------
                # NOTE about adding _SMALLEST_F32 below:
                #   a) np.searchsorted will return idx such that cdf[idx-1] < r <= cdf[idx]
                #                               hence, such that cdf[idx-1] < r <= cdf[idx-1] + p[idx]
                #      as a result, _NORMALLY_, it will never return an idx for which p[idx]==0.0.
                #   b) HOWEVER, a corner case exists in case r <= cdf[0], which is an exception where
                #      searchsorted will return idx=0.
                #      This is problematic, in case p[0]==0.0 and r=0.0, since in the case the resulting idx=0
                #         is invalid.
                #
                #  --> therefore we will simply ensure r>0.0 by adding a tiny number, which solves this
                #      corner case entirely.
                #  --> using r = rand_nz_float32(...)*p_sum would not solve this case entirely, in case p_sum if very
                #      small, which might cause the product to underflow and still cause cases of r=0.0.
                # ---------------------
                r = _SMALLEST_F32 + (rand_float32(rng_state) * p_sum)  # ensure r>0.0
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
                # notes:
                #  - computing -np.log(u[i]) does not seem to be noticeably slower than np.random.standard_exponential().
                #  - implementing & calling a rand_float32_array outside the loop once is not faster
                for i in range(n):  # n x O(1)
                    if p[i] <= 0.0:
                        keys[i] = np.inf
                    else:
                        ui = rand_nz_float32(rng_state)  # float in (0.0, 1.0)
                        # NOTE: we use a fast log2 approximation here for speed; log2 vs log is irrelevant since
                        #       it's just a scaling factor, and we are only interested in the order of the final list
                        keys[i] = -fast_log2_f32(ui) / p[i]  # using fast log2 approximation

                # Get indices of k smallest keys
                if k <= (10 + n // 20):
                    return select_k_min(keys, np.int32(k))  # most efficient for small k and k/n
                else:
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
        raise ValueError(
            f"p must be of size 0 (uniform sampling) or size n={n} (non-uniform sampling). (here: size={p.size})"
        )


# =================================================================================================
#  randint1
# =================================================================================================
@numba.njit("int32(int32, float32[:], uint64[:])", fastmath=True, cache=True)
def randint1(
    n: np.int32,
    p: NDArray[np.float32],
    rng_state: NDArray[np.uint64],
) -> np.int32:
    """
    This is a dedicated, simplified and faster version of randint for the special case k=1.

    Differences:
      - hard assumption of k=1, eliminating this argument
      - no need for 'replace' argument, as this is irrelevant for k=1
      - directly returns a single int32, instead of an array of shape (1,)

    NOTE: this method guarantees that when probabilities are provided, the returned sample will always have p[i]>0.0.

    :param n: defines population to sample from as range [0, n-1].  `n` must be >0.
    :param p: 1D array of probabilities associated with each integer in the range.
              Size must be equal to max_value + 1, and should have non-negative values. Sum is not require to be 1.
              NOTE: if size is 0, indicates no probabilities specified.  Use the constant P_UNIFORM for this.
                    if size > 0, but not equal to max_value+1, a ValueError is raised.
    :param rng_state: (2-element uint64 array) RNG state to be used (and updated in-place)
                      Use new_rng_state(seed) to generate a new one based on an integer seed.
    :return: (np.int32) generated sample.
    """
    if n < 1:
        raise ValueError(f"n must be >=1. (here: {n})")

    if p.size == 0:
        # UNIFORM sampling
        return rand_int32(rng_state, 0, n)

    elif p.size == n:
        # NON-UNIFORM sampling
        cdf = np.cumsum(p)  # O(n)
        p_sum = cdf[-1]
        if p_sum <= 0.0:
            raise ValueError("Sum of probabilities in p must be > 0.0.")

        # sample r in (0.0, p_sum) -> see docs of randint for rationale of avoiding r==0.0
        r = _SMALLEST_F32 + (rand_nz_float32(rng_state) * p_sum)

        # return single result
        return np.int32(np.searchsorted(cdf, r))

    else:
        raise ValueError(
            f"p must be of size 0 (uniform sampling) or size n={n} (non-uniform sampling). (here: size={p.size})"
        )
