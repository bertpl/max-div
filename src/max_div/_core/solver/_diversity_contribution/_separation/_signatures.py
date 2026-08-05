"""The interface every separation backend implements.

Shared so the three backend modules are interchangeable by construction: a function that does
not match cannot be registered, and a signature change lands in one place rather than nine.
"""

import numba

from max_div._core.metrics._distance import DISTANCE_STORE_TYPE

ELEMENTS_SIGNATURE = numba.void(numba.float32[::1], DISTANCE_STORE_TYPE, numba.int32[::1])
ADD_SIGNATURE = numba.void(numba.float32[::1], DISTANCE_STORE_TYPE, numba.int32)
REMOVE_SIGNATURE = numba.void(numba.float32[::1], DISTANCE_STORE_TYPE, numba.int32, numba.int32[::1])
