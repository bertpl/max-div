"""Insertion into and deletion from an ascending index list held in a fixed-capacity buffer.

The list occupies the first `n_live` entries of its buffer; the rest is scratch.  Both operations
shift a run of entries by one position, and since that run moves within a single buffer, its source
and destination overlap.  The shift therefore needs a move whose behavior on overlapping ranges is
defined — `memcpy` is the one that is not, `memmove` is the one that is.

## Why the shift is written twice

This module runs two ways, and no single implementation serves both:

  - **Compiled** — production, and every test run but coverage.  Nothing written in Python compiles
    to a `memmove`: numba lowers an element loop and a slice assignment alike to a copy loop, which
    is several times slower at large `n_live`.  Reaching the instruction takes an intrinsic.
  - **Interpreted** — the coverage run and the JIT-disabled jobs, where compilation is switched off
    so coverage can see inside these functions.  An intrinsic does not exist in this mode at all: it
    produces instructions for a compiler, and no compiler is running.  numpy slice assignment stands
    in, and buffers its source, so it is equally safe on overlap.

## How the right one gets picked

Two decorators, each doing one half of the job, because neither does both:

  - `@intrinsic` (`_llvm_memmove`) supplies **the instruction**.  What it defines is callable only
    from compiled code and has no Python body to fall back on, so it cannot be called directly by
    the two functions below without breaking the interpreted mode.
  - `@overload(move_within)` supplies **the dispatch**.  It tells numba that a compiled caller of
    `move_within` should get the registered implementation — the one calling the intrinsic — instead
    of a compilation of the plain function.

The plain `move_within` is therefore the interpreted implementation, and the overload redirects away
from it whenever the caller is being compiled:

  - **JIT on:** `insert_sorted` is compiled, numba resolves its `move_within` call through the
    overload registry, and the memmove is emitted inline into `insert_sorted`.
  - **JIT off:** `njit` hands back the undecorated function, so `insert_sorted` is ordinary Python
    and its `move_within` call is an ordinary Python call, landing on the numpy body.  The
    registration still happened — `@overload` runs at import in either mode — but all it does is
    file `_move_within_compiled` under `move_within` in numba's *typing* registry, leaving
    `move_within` itself untouched: not wrapped, not rebound, the same function object either way.
    That registry is read during type inference, and type inference runs only when numba compiles
    something, so with compilation off the entry is simply never looked up.

Both paths are live, and the tests exercise this module in each mode.
"""

import numpy as np
from llvmlite import ir
from numba import types
from numba.extending import intrinsic, overload
from numpy.typing import NDArray

from max_div._core.jit import lazy_njit


# ==================================================================================================
#  Overlap-safe move
# ==================================================================================================
@intrinsic
def _llvm_memmove(typingctx, dest_address, src_address, n_bytes):  # noqa: ANN001, ANN202
    """Emit a call to `llvm.memmove` that moves `n_bytes` bytes from `src_address` to `dest_address`.

    The body runs while numba compiles a caller, not when the function is called, and returns the
    call's type signature plus the `codegen` that writes the instructions.  Inside `codegen`,
    `builder` is an LLVM instruction writer and `args` holds the caller's compiled arguments.

    The addresses and the byte count arrive as plain integers, and `codegen` turns the addresses
    into pointers itself, so the intrinsic does not depend on how numba lays out an array.

    `_llvm_memmove` returns `dest_address`, as C's `memmove` returns its destination, because an
    intrinsic that returns void would need `codegen` to build a dummy return value; callers ignore
    the returned address.

    The `False` ending the emitted call is LLVM's `isvolatile` flag.  Marking an access volatile
    forbids the optimizer from reordering, merging or discarding it, which is what memory needs when
    the accesses themselves are observable — a memory-mapped hardware register being the standard
    case.  A shift within an ordinary array is not that, so the flag stays false and the optimizer
    keeps its freedom.
    """
    signature = types.intp(types.intp, types.intp, types.intp)

    def codegen(context, builder, sig, args):  # noqa: ANN001, ANN202
        dest_address, src_address, n_bytes = args
        # Use a typed byte pointer: every llvmlite version that this package supports accepts it,
        # while the untyped (opaque) `ir.PointerType()` needs llvmlite 0.44 or newer.
        byte_ptr = ir.IntType(8).as_pointer()
        memmove = builder.module.declare_intrinsic("llvm.memmove", [byte_ptr, byte_ptr, n_bytes.type])
        dest_ptr = builder.inttoptr(dest_address, byte_ptr)
        src_ptr = builder.inttoptr(src_address, byte_ptr)
        builder.call(memmove, [dest_ptr, src_ptr, n_bytes, ir.Constant(ir.IntType(1), 0)])
        return dest_address

    return signature, codegen


def move_within(
    buffer: NDArray[np.int32],
    dest_offset: int | np.signedinteger,
    src_offset: int | np.signedinteger,
    count: int | np.signedinteger,
) -> None:
    """Move `count` entries of `buffer` from `src_offset` to `dest_offset`, correct on overlap.

    In compiled code `buffer` must be contiguous (`int32[::1]`), because the compiled move computes
    byte addresses from the buffer's start and item size.
    """
    buffer[dest_offset : dest_offset + count] = buffer[src_offset : src_offset + count]


@overload(move_within)
def _move_within_compiled(buffer, dest_offset, src_offset, count):  # noqa: ANN001, ANN202
    """Register what numba compiles in place of `move_within`; the module docstring has the mechanism.

    This outer body runs at compile time — the inner `implementation` is what lands in the caller.
    """

    def implementation(buffer, dest_offset, src_offset, count):  # noqa: ANN001, ANN202
        # the address arithmetic assumes a contiguous buffer, which the callers' `int32[::1]` signatures guarantee
        base_address = buffer.ctypes.data
        item_size = buffer.itemsize
        # ty: ignore[missing-argument] -- the stub counts the typingctx parameter, which callers do not pass
        _llvm_memmove(base_address + dest_offset * item_size, base_address + src_offset * item_size, count * item_size)

    return implementation


# ==================================================================================================
#  Sorted index list
# ==================================================================================================
@lazy_njit("void(int32[::1], int32, int32)", cache=True)
def insert_sorted(index_list: NDArray[np.int32], n_live: np.int32, value: np.int32) -> None:
    """Insert `value` into the ascending `index_list` of length `n_live`, growing it by one.

    `value` must not already be present, and the buffer must hold at least `n_live + 1` entries.
    """
    position = np.searchsorted(index_list[:n_live], value)
    move_within(index_list, position + 1, position, n_live - position)
    index_list[position] = value


@lazy_njit("void(int32[::1], int32, int32)", cache=True)
def delete_sorted(index_list: NDArray[np.int32], n_live: np.int32, value: np.int32) -> None:
    """Remove `value` from the ascending `index_list` of length `n_live`, shrinking it by one.

    `value` must be present.  The vacated last entry keeps its old contents, so the list is the
    first `n_live - 1` entries once the caller has shrunk its count.

    An absent `value` sorting into the middle silently deletes the wrong entry; the precondition
    is enforced by the callers' tests, not here.
    """
    position = np.searchsorted(index_list[:n_live], value)
    count = n_live - position - 1
    if count < 0:
        # `value` is absent and sorts past every live entry; the negative count would reach
        # memmove as a huge unsigned length -- return instead of writing
        return
    move_within(index_list, position, position + 1, count)
