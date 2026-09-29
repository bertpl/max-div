import numpy as np
from numpy.typing import NDArray

from max_div._core.jit import lazy_njit


@lazy_njit("float32(float32[::1])", fastmath=True, inline="always", cache=True)
def _p_max(p: NDArray[np.float32]) -> np.float32:
    """Return the maximum value in p array."""
    n = p.size
    max_value = np.float32(0.0)
    for i in range(n):
        max_value = max(max_value, p[i])
    return max_value
