import numpy as np

from max_div.internal.compat import numba


# =================================================================================================
#  sample_int
# =================================================================================================
def sample_int(
    n: int,
    k: int | None = None,
    replace: bool = True,
    p: np.ndarray[float] | None = None,
    seed: int | None = None,
    use_numba: bool = True,
) -> int | np.ndarray[np.int64]:
    """
    Randomly sample `k` integers from range `[0, n-1]`, optionally with replacement and per-value probabilities.

    Different implementation is used, depending on the case:

    | `use_numba`    | `p` specified  | `replace`  | `k`  | Method Used                              | Complexity      |
    |----------------|----------------|------------|------|------------------------------------------|-----------------|
    |  No            | Any            | Any        | Any  | `np.random.choice`                       | depends         |
    |  Yes           | No             | True       | Any  | `np.random.randint`, uniform sampling    | O(k)            |
    |  Yes           | No             | False      | Any  | k-element Fisher-Yates shuffle           | O(n)            |
    |  Yes           | Yes            | Any        | 1    | Multinomial sampling using CDF           | O(n + log(n))   |
    |  Yes           | Yes            | True       | >1   | Multinomial sampling using CDF           | O(n + k log(n)) |
    |  Yes           | Yes            | False      | >1   | Efraimidis-Spirakis sampling + exponential key sampling (Gumbel-Max Trick) using the Ziggurat algorithm.  | O(n) |

    :param n: defines population to sample from as range [0, n-1].  `n` must be >0.
    :param k: The number of integers to sample (>0).  `k=None` indicates a single integer sample.
    :param replace: Whether to sample with replacement.
    :param p: Optional 1D array of probabilities associated with each integer in the range.
              Size must be equal to max_value + 1 and sum to 1.
    :param seed: Optional random seed for reproducibility.
    :param use_numba: Use the self-implemented algorithm (which is `numba`-accelerated if `numba` is installed)
    :return: `k=None` --> single integer; `k>=1` --> (k,)-sized array with sampled integers.
    """

    # -------------------------------------------------------------------------
    #  Don't try using numba
    # -------------------------------------------------------------------------
    if not use_numba:
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

        # --- sampling ------------------------------------
        if seed is not None:
            np.random.seed(seed)

        if k is None:
            # returns scalar
            return np.random.choice(n, size=None, replace=replace, p=p)
        else:
            # returns array
            return np.random.choice(n, size=k, replace=replace, p=p)

    # -------------------------------------------------------------------------
    #  Try using numba
    # -------------------------------------------------------------------------
    else:
        # --- argument handling ---------------------------
        k_orig = k  # remember, to know what return value we need
        if (k == 1) or (k is None):
            k = 1  # make sure we always have an integer for the numba function
            replace = True  # single sample, replacement makes no difference, so we can fall back to faster methods
        if p is None:
            use_p = False
            p = np.zeros(0, dtype=np.float64)  # dummy value
        else:
            use_p = True
        if seed is None:
            use_seed = False
            seed = 42  # dummy value
        else:
            use_seed = True

        # --- argument validation -------------------------
        if n < 1:
            raise ValueError(f"n must be >=1. (here: {n})")
        if k < 1:
            raise ValueError(f"k must be >=1. (here: {k})")
        if (not replace) and (k > n):
            raise ValueError(f"Cannot sample {k} unique values from range [0, {n}) without replacement.")
        if use_p:
            if (p.ndim != 1) or (p.size != n):
                raise ValueError(f"p must be a 1D array of size {n}. (here: shape={p.shape})")

        # --- call numba function -------------------------
        samples = sample_int_numba(n=n, k=k, replace=replace, use_p=use_p, p=p, use_seed=use_seed, seed=seed)
        if k_orig is None:
            return samples[0]
        else:
            return samples


@numba.njit
def sample_int_numba(
    n: int,
    k: int,
    replace: bool,
    use_p: bool = False,
    p: np.ndarray[float] = np.zeros(0),
    use_seed: bool = False,
    seed: int = 42,
) -> np.ndarray[int]:
    """
    See native python version docstring.  We split the implementation into a core numba-decorated function and
    a Python front-end, because numba-decorated functions do not allow for multiple possible return types.
    """
    if use_seed:
        np.random.seed(seed)

    if (k == n) and (not replace):
        # corner case: return all elements in random order
        population = np.arange(n, dtype=np.int64)
        np.random.shuffle(population)
        return population

    if not use_p:
        if replace:
            # UNIFORM sampling with replacement
            return np.random.choice(n, size=k)  # O(k)
        else:
            # UNIFORM sampling without replacement using Fisher-Yates shuffle
            population = np.arange(n, dtype=np.int64)  # O(n)
            for i in range(k):  # k x O(1)
                j = np.random.randint(i, n)
                population[i], population[j] = population[j], population[i]
            return population[:k]  # O(k)
    else:
        if replace:
            # NON-UNIFORM sampling with replacement using CDF
            cdf = np.empty(n)  # O(n)
            csum = 0.0
            for i in range(n):  # n x O(1)
                csum += p[i]
                cdf[i] = csum
            samples = np.empty(k, dtype=np.int64)  # O(k)
            for i in range(k):  # k x O(log(n))
                r = np.random.random()
                idx = np.searchsorted(cdf, r)
                samples[i] = idx
            return samples
        else:
            # NON-UNIFORM sampling without replacement using Efraimidis-Spirakis + Exponential keys
            # algorithm description:
            #   Efraimidis:       select k elements corresponding to k largest values of  u_i^{1/p_i} (u_i ~ U(0,1))
            #   Gumbel-Max Trick: select k smallest values of  -log(u_i)/p_i  (u_i ~ U(0,1))
            #   Ziggurat:         (TODO) generate log(u_i) more efficiently, applying the Ziggurat algorithm
            #                            to the exponential distribution, which avoids usage of transcendental
            #                            functions for the majority of the samples.
            keys = np.empty(n, dtype=np.float64)  # O(n)
            u = np.random.random(n)  # O(n)
            for i in range(n):  # n x O(1)
                if p[i] == 0.0:
                    keys[i] = np.inf
                else:
                    keys[i] = -np.log(u[i]) / p[i]

            # Get indices of k smallest keys
            if k > 1:
                return np.argpartition(keys, k)[:k]  # O(n) average case
            else:
                return np.array([np.argmin(keys)])  # O(n)
