"""Insertion into and deletion from an ascending index list held in a fixed-capacity buffer.

The list occupies the first `n_live` entries of its buffer; the rest is scratch.  Both operations
shift a run of entries by one position, and since that run moves within a single buffer, its source
and destination overlap.  The shift therefore needs a move whose behavior on overlapping ranges is
defined — `memcpy` is the one that is not, `memmove` is the one that is.

That move is written twice under the single name `move_within`, because this code runs two ways:

  - **Compiled**, it emits `llvm.memmove` directly.  Reaching that instruction takes an intrinsic,
    since neither an element loop nor a slice assignment written inside numba compiles down to one.
  - **Interpreted** — how the coverage run and the JIT-disabled test legs execute — it is numpy
    slice assignment, which buffers its source and is safe on overlap for the same reason.

Which one a caller gets is decided by numba while it compiles that caller; `move_within` and the
`overload` beneath it document the mechanics.
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
    """Emit a call to `llvm.memmove` over the data pointers of `dest` and `src`.

    `@intrinsic` is numba's hook for generating machine code directly, for cases where no Python
    source compiles to the instruction wanted.  It does not behave like `@njit`: this body never
    runs at call time.  numba runs it *while compiling a caller*, and expects two things back —
    the type signature of the call, and `codegen`, which writes the instructions.

    Inside `codegen`, `builder` is an LLVM instruction writer and `args` holds the caller's
    already-compiled arguments.  The block therefore emits a memmove call into the caller; it does
    not perform one.  The final argument to that call is LLVM's `isvolatile` flag, kept false.
    """
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
    """Supply the implementation numba uses for `move_within` in compiled code.

    `@overload` registers a substitute: when numba compiles a function that calls `move_within`, it
    compiles this instead of the Python body above.  As with `@intrinsic`, this outer body runs at
    compile time — the inner `implementation` is what ends up in the caller.

    Nothing registered here reaches interpreted execution, which is what gives the two paths.  Under
    `NUMBA_DISABLE_JIT=1` the `njit` decorators below become no-ops, `insert_sorted` runs as ordinary
    Python, and its call resolves to `move_within` itself — the numpy version.  Both are therefore
    live, and the tests cover the module in each mode.
    """

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
