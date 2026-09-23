"""The interface every mean-distance backend implements.

Shared so the backend modules are interchangeable by construction: a function that does
not match cannot be registered, and a signature change lands in one place.

Sums accumulate in float64: entries undergo long add/subtract chains over solver iterations, and
float32 drift there would change scores with iteration count.
"""

import numba

from max_div._core.metrics._distance import DISTANCE_STORE_TYPE

UPDATE_SIGNATURE = numba.void(numba.float64[::1], DISTANCE_STORE_TYPE, numba.int32)
