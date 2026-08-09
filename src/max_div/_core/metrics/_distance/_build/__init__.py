"""The build layer turns vectors into distance data, in either the condensed layout or a full matrix.

The two layouts are built the same way — an entry point that owns validation, cosine normalization
and the parallel switch, over a sequential and a parallel fill — so each has its own module of that
shape, with `_common` holding what both depend on.

Every fill writes into a buffer the caller supplies, and each entry point allocates one only when
none is given.  That lets a store be built straight into shared memory rather than built and then
copied.

Only the entry points are re-exported.  The block width and the parallel switch stay internal to the
package, since nothing outside it builds distances itself — anything that reaches for them, tests
included, imports the module that owns them.
"""

from ._condensed import compute_pdist
from ._full_matrix import compute_full_matrix, expand_condensed

__all__ = [
    "compute_full_matrix",
    "compute_pdist",
    "expand_condensed",
]
