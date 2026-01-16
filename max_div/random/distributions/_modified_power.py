"""
This module implements a distribution inspired by the 'power distribution', but modified to be symmetric around 0.5.

Properties:
    - Support: [0, 1]
    - Parameter: m in [0, 1] representing the median of the distribution.

Sampling Method:
    - sample uniform value u in [0, 1]
    - transform to x = f(u)   where f(0)=0, f(1)=1, f(0.5)=m
             --> we will use f(u) = m*u + (1-m)*(u^p) with p such that f(0.5)=m
             --> p = -log2(m/(1-m)) + 1
    - depending on the value of m, we can define f(u) as follows:
        - if m=0.0          --> always sample 0, i.e. f(u)=0
        - if m in (0,0.5)   --> apply f(u) with p as computed above, i.e m*u + (1-m)*(u^p) with p = -log2(m/(1-m)) + 1
        - if m=0.5          --> uniform distribution, i.e. f(u)=u
        - if m in (0.5,1)   --> apply 1-f(u) for m'=1-m
        - if m=1.0          --> always sample 1, i.e. f(u)=1
    - by using the above definitions, we ensure that e.g. m=0.2 and m=0.8 result in mirrored distributions

    NOTE: by adding the term m*u and computing p in an adjusted way (instead of using f(u) = u^p with p = -log2(m)),
          we ensure that the f'(0) = m > 0 and hence that pdf(0) is finite, which makes this a bit better behaved.
"""

import numpy as np
from numba import njit
from numpy.typing import NDArray

from max_div.random.rng import new_rng_state, rand_float32


@njit(fastmath=True, inline="always")
def sample_modified_power_distribution(m: np.float32, rng_state: NDArray[np.uint64]) -> np.float32:
    if m == 0.0:
        return np.float32(0.0)
    elif m == 1.0:
        return np.float32(1.0)
    else:
        # we need to sample a uniform value u in [0, 1]
        u = rand_float32(rng_state)

        # now transform u to the desired distribution
        if m == 0.5:
            # uniform distribution, no transformation
            return u
        elif m < 0.5:
            return _modified_power_transform(u, m)
        else:
            # NOTE: in principle we need to also use 1-u instead of u, but both are random uniform in [0,1],
            #       so it's equivalent.
            return np.float32(1.0) - _modified_power_transform(u, np.float32(1.0) - m)


@njit(fastmath=True, inline="always")
def _modified_power_transform(u: np.float32, m: np.float32) -> np.float32:
    """
    Transform a uniform value u in [0,1] to the modified power distribution with median m in (0,0.5).

    We construct f(u) such that...
      f(0.0) = 0
      f(0.5) = m
      f(1.0) = 1

    Derivatives over [0,1] are strictly positive and finite.
    """
    if u == np.float32(0.0):
        return np.float32(0.0)
    elif u == np.float32(1.0):
        return np.float32(1.0)
    else:
        one_minus_m = np.float32(1.0) - m
        p = -np.log2(m / one_minus_m) + np.float32(1.0)
        return (m * u) + one_minus_m * (u**p)
