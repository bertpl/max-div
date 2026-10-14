"""A custom distance metric computes each distance with the user's numba function.

## Why the function is called by its address

Every distance store has the same numba type in the compiled solver code, so each solver function
declares its signature once and is cached.  A numba function passed as an object has a type of its own,
so a distance store that held one would get a different numba type for each function, and every solver
function would compile again for each.

Instead, `address_of_custom_distance_function` compiles the user's function into a wrapper with the C
calling convention, whose address is a plain integer, and `call_custom_distance` calls the wrapper at that
address.  `CustomDistanceMetric` passes the address to the compiled pairwise distance function.

## How it runs with JIT compilation on and off

This module follows the pattern of `max_div._core._utils._sorted_index_list`, whose module docstring
explains how one function runs both compiled and interpreted:

  - **Compiled** — the `@overload` of `call_custom_distance` replaces it in compiled callers with the
    `_call_address` intrinsic, which emits the call to the address.
  - **Interpreted** — with JIT compilation off, no wrapper and no intrinsic exist.  The "address" is
    then a key into a table of the user's Python functions, and `call_custom_distance` is an ordinary
    Python call.
"""

import functools
from collections.abc import Callable

import numba
import numpy as np
from llvmlite import ir
from numba import types
from numba.core.ccallback import CFunc
from numba.extending import intrinsic, overload
from numpy.typing import NDArray

# A custom distance function takes 2 rows of the vectors, each a 1-D float32 array, and returns their distance.
# It is a numba dispatcher, or, with JIT compilation off, the plain Python function that `numba.njit` returns.
CustomDistanceFunction = Callable[[NDArray[np.float32], NDArray[np.float32]], float]

# The wrapper's C signature takes pointers to the first values of 2 rows and the number of values in a row.
_WRAPPER_SIGNATURE = types.float64(types.CPointer(types.float32), types.CPointer(types.float32), types.int64)

# With JIT compilation off, each custom distance function is stored here under its "address".
_PYTHON_FUNCTIONS_BY_ADDRESS: dict[int, CustomDistanceFunction] = {}


# ==================================================================================================
#  Address of a custom distance function
# ==================================================================================================
def address_of_custom_distance_function(distance_function: CustomDistanceFunction) -> int:
    """Return the address, a positive integer, that `call_custom_distance` calls to compute `distance_function`.

    With JIT compilation on, the address is that of the function's C-callable wrapper, compiled once per
    process.  With JIT compilation off, `distance_function` is a plain Python function, and the address is
    its key in a table of Python functions.
    """
    if numba.config.DISABLE_JIT:  # ty: ignore[unresolved-attribute] -- set at runtime from NUMBA_DISABLE_JIT
        _PYTHON_FUNCTIONS_BY_ADDRESS[id(distance_function)] = distance_function
        return id(distance_function)
    else:
        return _compile_wrapper(distance_function).address  # ty: ignore[invalid-return-type] -- set once compiled


@functools.cache
def _compile_wrapper(distance_function: CustomDistanceFunction) -> CFunc:
    """Return the C-callable wrapper of the numba function `distance_function`, compiled on the first call.

    The cache also keeps every wrapper alive for the life of the process.  A distance store holds only the
    wrapper's address, so the wrapper must stay in memory as long as the process runs.

    The wrapper compiles the function's Python source inline, so a distance costs 1 call, not 2.

    The wrapper and the inlined copy of the function both take the function's own njit options: an inlined function
    compiles with its caller's fastmath flags, so the wrapper needs the user's flags to compute what the
    user's function computes.
    """
    # ty does not know a dispatcher's attributes, `targetoptions` and `py_func`
    target_options = distance_function.targetoptions  # ty: ignore[unresolved-attribute]
    # `numba.njit` warns when it is given `nopython`, and the inlined copy sets its own `inline`, so neither of
    # the user's values is passed on.
    options = {name: value for name, value in target_options.items() if name not in ("nopython", "inline")}
    python_function = distance_function.py_func  # ty: ignore[unresolved-attribute]
    inlined_function = numba.njit(inline="always", **options)(python_function)

    @numba.cfunc(_WRAPPER_SIGNATURE, **options)
    def wrapper(row_a, row_b, n_dims):  # noqa: ANN001, ANN202 -- numba types the arguments from the signature
        return inlined_function(numba.carray(row_a, (n_dims,)), numba.carray(row_b, (n_dims,)))

    return wrapper


# ==================================================================================================
#  Calling a custom distance function
# ==================================================================================================
def call_custom_distance(
    address: int | np.integer, vectors: NDArray[np.float32], i: int | np.integer, j: int | np.integer
) -> float:
    """Return the distance between rows i and j of `vectors`, computed by the custom distance at `address`.

    `address` must come from `address_of_custom_distance_function`: in compiled code, any other value crashes the
    process.  This body runs only with JIT compilation off; compiled callers get `_call_address` instead.
    """
    return _PYTHON_FUNCTIONS_BY_ADDRESS[int(address)](vectors[i], vectors[j])


@overload(call_custom_distance, inline="always")
def _call_custom_distance_compiled(address, vectors, i, j):  # noqa: ANN001, ANN202 -- numba's overload protocol
    """Register what numba compiles in place of `call_custom_distance`: a call to the wrapper at `address`.

    The inlining is required, not an optimization: the `_pairwise_distance` docstring explains why everything
    that its branches call must be inlined.
    """

    def implementation(address, vectors, i, j):  # noqa: ANN001, ANN202
        # ty: ignore[missing-argument] -- the stub counts the typingctx parameter, which callers do not pass
        return _call_address(address, vectors, i, j)

    return implementation


@intrinsic
def _call_address(typingctx, address, vectors, i, j):  # noqa: ANN001, ANN202 -- numba intrinsic
    """Emit a call to the C function at `address`, with pointers to rows i and j of `vectors` and the row length.

    `vectors` must be a C-contiguous 2D float32 array: the row pointers are computed from its data pointer and
    its second dimension.
    """
    signature = types.float64(types.int64, vectors, types.int64, types.int64)

    def codegen(context, builder, sig, args):  # noqa: ANN001, ANN202
        address_value, vectors_value, i_value, j_value = args
        array = context.make_array(sig.args[1])(context, builder, value=vectors_value)
        n_dims = builder.extract_value(array.shape, 1)
        row_i = builder.gep(array.data, [builder.mul(i_value, n_dims)])
        row_j = builder.gep(array.data, [builder.mul(j_value, n_dims)])
        function_type = ir.FunctionType(ir.DoubleType(), [array.data.type, array.data.type, n_dims.type])
        function_pointer = builder.inttoptr(address_value, function_type.as_pointer())
        return builder.call(function_pointer, [row_i, row_j, n_dims])

    return signature, codegen
