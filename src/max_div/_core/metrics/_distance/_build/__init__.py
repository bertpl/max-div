"""Construction of distance data from vectors: condensed vectors and full matrices.

The two layouts are built the same way — an entry point that owns validation, cosine normalization
and the parallel switch, over a sequential and a parallel fill — so each has its own module of that
shape, with `_common` holding what both depend on.

Every fill writes into a buffer the caller supplies, and each entry point allocates one only when
none is given.  That lets a store be built straight into shared memory rather than built and then
copied.
"""

from ._common import BUILD_BLOCK_WIDTH, parallel_build_enabled
from ._condensed import compute_pdist
from ._full_matrix import compute_full_matrix, expand_condensed

__all__ = [
    "BUILD_BLOCK_WIDTH",
    "compute_full_matrix",
    "compute_pdist",
    "expand_condensed",
    "parallel_build_enabled",
]
