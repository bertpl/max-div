"""Insertion into and deletion from an ascending index list held in a fixed-capacity buffer.

The list occupies the first `n_live` entries of its buffer; the entries beyond that are scratch.
Both operations shift a run of entries by one position, and the source and destination of that
shift overlap, so the move has to be one that defines its behaviour on overlap.

Compiled, that move is `llvm.memmove`, whose whole purpose is to be correct on overlapping
ranges — reaching it needs an intrinsic, because neither an element loop nor a slice assignment
inside numba produces one.  Interpreted, it is numpy slice assignment, which buffers the source
for the same reason.  `move_within` is the single name for both: the plain function below runs
when numba is disabled, the `overload` when it is not.
"""

import numba
import numpy as np
from llvmlite import ir
from numba.core import types
from numba.extending import intrinsic, overload
from numpy.typing import NDArray


# =================================================================================================
#  Overlap-safe move
# =================================================================================================
@intrinsic
def _llvm_memmove(typingctx, dest, dest_offset, src, src_offset, count):  # noqa: ANN001, ANN202
    """Emit a call to `llvm.memmove` over the data pointers of `dest` and `src`."""
    signature = types.void(dest, dest_offset, src, src_offset, count)

    def codegen(context, builder, sig, args):  # noqa: ANN001, ANN202
        dest_array, dest_off, src_array, src_off, n = args
        item_size = context.get_abi_sizeof(context.get_data_type(sig.args[0].dtype))
        dest_struct = context.make_array(sig.args[0])(context, builder, dest_array)
        src_struct = context.make_array(sig.args[2])(context, builder, src_array)
        byte_ptr = ir.IntType(8).as_pointer()
        dest_ptr = builder.bitcast(builder.gep(dest_struct.data, [dest_off]), byte_ptr)
        src_ptr = builder.bitcast(builder.gep(src_struct.data, [src_off]), byte_ptr)
        n_bytes = builder.mul(builder.sext(n, ir.IntType(64)), ir.Constant(ir.IntType(64), item_size))
        memmove = builder.module.declare_intrinsic("llvm.memmove", [byte_ptr, byte_ptr, ir.IntType(64)])
        builder.call(memmove, [dest_ptr, src_ptr, n_bytes, ir.Constant(ir.IntType(1), 0)])
        return context.get_dummy_value()

    return signature, codegen


def move_within(buffer: NDArray[np.int32], dest_offset: int, src_offset: int, count: int) -> None:
    """Move `count` entries of `buffer` from `src_offset` to `dest_offset`, correct on overlap."""
    buffer[dest_offset : dest_offset + count] = buffer[src_offset : src_offset + count]


@overload(move_within)
def _move_within_compiled(buffer, dest_offset, src_offset, count):  # noqa: ANN001, ANN202
    """Compiled form of `move_within`, reaching llvm.memmove rather than a copy loop."""

    def implementation(buffer, dest_offset, src_offset, count):  # noqa: ANN001, ANN202
        _llvm_memmove(buffer, dest_offset, buffer, src_offset, count)

    return implementation


# =================================================================================================
#  Sorted index list
# =================================================================================================
@numba.njit("void(int32[::1], int32, int32)", cache=True)
def insert_sorted(index_list: NDArray[np.int32], n_live: np.int32, value: np.int32) -> None:
    """Insert `value` into the ascending `index_list` of length `n_live`, growing it by one.

    The buffer must hold at least `n_live + 1` entries.  `value` must not already be present;
    inserting a duplicate places it next to its twin and leaves the list ascending, but the
    callers of this function treat their lists as sets.
    """
    position = np.searchsorted(index_list[:n_live], value)
    move_within(index_list, position + 1, position, n_live - position)
    index_list[position] = value


@numba.njit("void(int32[::1], int32, int32)", cache=True)
def delete_sorted(index_list: NDArray[np.int32], n_live: np.int32, value: np.int32) -> None:
    """Remove `value` from the ascending `index_list` of length `n_live`, shrinking it by one.

    `value` must be present.  The entry vacated at the end of the list is left as it was; the
    list is defined by its first `n_live - 1` entries once the caller has shrunk its own count.
    """
    position = np.searchsorted(index_list[:n_live], value)
    move_within(index_list, position, position + 1, n_live - position - 1)
