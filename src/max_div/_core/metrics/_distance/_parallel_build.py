"""Shared switch and tile constant for the parallel distance-store builds.

The parallel builds are governed by the ``MAXDIV_PARALLEL_BUILD`` environment variable
(enabled unless set to ``"0"``).  Parallel and sequential builds are bit-identical — the
switch controls resource usage (thread count during store construction) only, never results.
"""

import os

# Column-block width for the parallel builds.  The block converts the triangular pair loop into
# uniform slabs (every row above a block computes exactly one block-width of pairs), which is what
# lets prange split the work evenly across threads — parallelizing the raw triangle leaves one
# thread with roughly twice the work of the average.
BUILD_TILE = 64


def parallel_build_enabled() -> bool:
    """Return whether distance-store builds may use multiple threads (default: enabled)."""
    return os.environ.get("MAXDIV_PARALLEL_BUILD", "1") != "0"
