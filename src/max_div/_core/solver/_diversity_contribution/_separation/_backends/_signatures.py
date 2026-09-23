"""The interface every separation backend implements.

The signatures live in this one module so the backend modules are interchangeable by
construction: a function that does not match cannot be registered, and a signature change is
made in one place.
"""

import numba

from max_div._core.metrics._distance import DISTANCE_STORE_TYPE

ADD_SIGNATURE = numba.void(numba.float32[::1], DISTANCE_STORE_TYPE, numba.int32)
ADD_MANY_SIGNATURE = numba.void(numba.float32[::1], DISTANCE_STORE_TYPE, numba.int32[::1], numba.boolean)
REMOVE_SIGNATURE = numba.void(numba.float32[::1], DISTANCE_STORE_TYPE, numba.int32, numba.int32[::1])
