import numpy as np
from numba import njit
from numpy.typing import NDArray


@njit("float32(float32[::1])", fastmath=True, inline="always")
def _p_max(p: NDArray[np.float32]) -> np.float32:
    """Return the maximum value in p array."""
    n = p.size
    max_value = np.float32(0.0)
    for i in range(n):
        max_value = max(max_value, p[i])
    return max_value
