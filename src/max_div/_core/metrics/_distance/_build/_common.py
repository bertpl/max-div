"""What the condensed and full-matrix builds share: the parallel switch, its blocking, and types.

Every build comes in a sequential and a parallel variant with bit-identical output — the parallel
fills run the same pair arithmetic under the same fastmath flags and write each element exactly
once, so thread count cannot affect results.  `MAXDIV_PARALLEL_BUILD` picks the variant (parallel
unless set to ``"0"``).

The parallel fills cut the columns into fixed-width blocks and parallelize the row loop within
each block: every row above a block computes exactly one block-width of pairs, so prange can
split the work evenly across threads — parallelizing the outer row loop over the whole i<j pair
triangle instead leaves one thread with roughly twice the work of the average.
"""

import os

import numba

# Width in columns of the blocks the parallel fills cut the pair space into.
BUILD_BLOCK_WIDTH = 64

# The fills read their input through a read-only array type and write through a writable one.  Numba
# treats read-only as the wider type — a writable argument converts to it, never the reverse — so an
# input typed this way also accepts the read-only views a DistanceStore hands out, while an output
# stays writable because it is written to.  Signature strings cannot spell a read-only array type,
# so every fill takes a signature object.
READONLY_F32_1D = numba.types.Array(numba.float32, 1, "C", readonly=True)
READONLY_F32_2D = numba.types.Array(numba.float32, 2, "C", readonly=True)
WRITABLE_F32_1D = numba.float32[::1]
WRITABLE_F32_2D = numba.float32[:, ::1]


def parallel_build_enabled() -> bool:
    """Return whether distance builds may use multiple threads (default: enabled)."""
    return os.environ.get("MAXDIV_PARALLEL_BUILD", "1") != "0"
