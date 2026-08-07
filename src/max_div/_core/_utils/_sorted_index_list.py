"""Insertion into and deletion from an ascending index list held in a fixed-capacity buffer.

The list occupies the first `n_live` entries; the rest is scratch.  Both operations shift a run
of entries by one, source and destination overlapping, so the move must be overlap-safe.
`move_within` has two implementations of it: an intrinsic emitting `llvm.memmove` when compiled,
numpy slice assignment when numba is disabled.
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


def move_within(
    buffer: NDArray[np.int32],
    dest_offset: int | np.signedinteger,
    src_offset: int | np.signedinteger,
    count: int | np.signedinteger,
) -> None:
    """Move `count` entries of `buffer` from `src_offset` to `dest_offset`, correct on overlap."""
    buffer[dest_offset : dest_offset + count] = buffer[src_offset : src_offset + count]


@overload(move_within)
def _move_within_compiled(buffer, dest_offset, src_offset, count):  # noqa: ANN001, ANN202
    """Compiled form of `move_within`, reaching llvm.memmove rather than a copy loop."""

    def implementation(buffer, dest_offset, src_offset, count):  # noqa: ANN001, ANN202
        # ty: ignore[missing-argument] -- the stub counts the typingctx parameter, which callers do not pass
        _llvm_memmove(buffer, dest_offset, buffer, src_offset, count)

    return implementation


# =================================================================================================
#  Sorted index list
# =================================================================================================
@numba.njit("void(int32[::1], int32, int32)", cache=True)
def insert_sorted(index_list: NDArray[np.int32], n_live: np.int32, value: np.int32) -> None:
    """Insert `value` into the ascending `index_list` of length `n_live`, growing it by one.

    `value` must not already be present, and the buffer must hold at least `n_live + 1` entries.
    """
    position = np.searchsorted(index_list[:n_live], value)
    move_within(index_list, position + 1, position, n_live - position)
    index_list[position] = value


@numba.njit("void(int32[::1], int32, int32)", cache=True)
def delete_sorted(index_list: NDArray[np.int32], n_live: np.int32, value: np.int32) -> None:
    """Remove `value` from the ascending `index_list` of length `n_live`, shrinking it by one.

    `value` must be present.  The vacated last entry keeps its old contents, so the list is the
    first `n_live - 1` entries once the caller has shrunk its count.
    """
    position = np.searchsorted(index_list[:n_live], value)
    move_within(index_list, position, position + 1, n_live - position - 1)
