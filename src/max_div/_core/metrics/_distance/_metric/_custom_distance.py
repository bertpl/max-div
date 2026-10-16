"""A custom distance metric computes each distance with the user's numba function.

## Why the function is called by its address

Every distance store has the same numba type in the compiled solver code, so each solver function
declares its signature once and is cached.  A numba function passed as an object has a type of its own,
so a distance store that held one would get a different numba type for each function, and every solver
function would compile again for each.

Instead, `BoundDistanceFunction.address` compiles the user's function into a wrapper with the C calling
convention, whose address is a plain integer, and `call_custom_distance` calls the wrapper at that address.
`CustomDistanceMetric` passes the address to the compiled pairwise distance function.

## Parameter values

The user's function may take parameter values after its 2 vectors, such as the power of a Minkowski
distance.  The wrapper holds each value as a constant, so its C signature is the same for any values, and
a solve passes nothing more to compiled code.

## How it runs with JIT compilation on and off

This module follows the pattern of `max_div._core._utils._sorted_index_list`, whose module docstring
explains how one function runs both compiled and interpreted:

  - **Compiled** — the `@overload` of `call_custom_distance` replaces it in compiled callers with the
    `_call_address` intrinsic, which emits the call to the address.
  - **Interpreted** — with JIT compilation off, no wrapper and no intrinsic exist.  The "address" is
    then a key into a table of Python functions of 2 rows, each of which calls the user's function with
    its values, and `call_custom_distance` is an ordinary Python call.
"""

import functools
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numba
import numpy as np
from llvmlite import ir
from numba import types
from numba.core.ccallback import CFunc
from numba.core.dispatcher import Dispatcher
from numba.extending import intrinsic, overload
from numpy.typing import NDArray

if TYPE_CHECKING:
    from collections.abc import Hashable

# A custom distance function takes 2 rows of the vectors, each a 1-D float32 array, followed by its parameter values,
# and returns their distance.  It is a numba dispatcher, or, with JIT compilation off, the plain Python function that
# `numba.njit` returns.
CustomDistanceFunction = Callable[..., float]

# A parameter value of a custom distance function.
CustomDistanceParam = bool | int | float

# The wrapper's C signature takes pointers to the first values of 2 rows and the number of values in a row.
_WRAPPER_SIGNATURE = types.float64(types.CPointer(types.float32), types.CPointer(types.float32), types.int64)

# With JIT compilation off, each bound distance function is stored here, as a function of 2 rows, under its "address".
_PYTHON_FUNCTIONS_BY_ADDRESS: dict[int, Callable[[NDArray[np.float32], NDArray[np.float32]], float]] = {}

_INT64_INFO = np.iinfo(np.int64)


# ==================================================================================================
#  BoundDistanceFunction
# ==================================================================================================
@dataclass(frozen=True, eq=False)
class BoundDistanceFunction:
    """A bound distance function is the user's numba function with the parameter values that follow its 2 vectors.

    Creating one converts each numpy bool, integer or floating scalar to its Python type, so the function
    always receives a numba boolean, int64 or float64.

    Values compare by type as well as by value.  Python counts `3`, `3.0` and `True` as equal, but numba
    types them as int64, float64 and boolean and compiles different code for each, so 2 bound functions are
    equal only when they compile the same wrapper.  The wrapper cache and the table of Python functions key
    on this equality.
    """

    function: CustomDistanceFunction
    params: tuple[CustomDistanceParam, ...]

    def __post_init__(self) -> None:
        """Check the function, convert the values to Python types, and check that they fit the function's arguments.

        Raises:
            TypeError: If `function` is not a `numba.njit` function, a value is not a bool, an integer or a
                floating-point number, or `function` does not take the values after its 2 vectors.
            ValueError: If an integer value does not fit in int64.
        """
        # With JIT compilation off, `numba.njit` returns the function itself, so any callable is accepted.
        is_jit_disabled = numba.config.DISABLE_JIT  # ty: ignore[unresolved-attribute] -- set at runtime
        is_numba_function = isinstance(self.function, Dispatcher) or (is_jit_disabled and callable(self.function))
        if not is_numba_function:
            raise TypeError(f"DistanceMetric.custom requires a numba.njit function; got {self.function!r}.")
        # A frozen dataclass rejects `self.params = ...`, so the converted values are set with `object.__setattr__`.
        object.__setattr__(self, "params", tuple(self._as_python_scalar(value) for value in self.params))
        # `bind` raises TypeError when the values do not fit the function's arguments after its 2 vectors.
        inspect.signature(self.function).bind(None, None, *self.params)

    def __eq__(self, other: object) -> bool:
        """Return True for the same function with values of the same types and values."""
        if isinstance(other, BoundDistanceFunction):
            return self._comparison_key == other._comparison_key
        else:
            return NotImplemented

    def __hash__(self) -> int:
        """Return a hash that is consistent with the type-aware equality."""
        return hash(self._comparison_key)

    @property
    def name(self) -> str:
        """Return the function's name, followed by any values in parentheses, such as `f(3.0, True)`."""
        function_name = self.function.__name__  # ty: ignore[unresolved-attribute] -- a function has a name
        if self.params:
            return f"{function_name}({', '.join(repr(value) for value in self.params)})"
        else:
            return function_name

    def address(self) -> int:
        """Return the address, a positive integer, that `call_custom_distance` calls to compute this distance.

        With JIT compilation on, the address is that of the C-callable wrapper, compiled once per process for
        each distinct bound function.  With JIT compilation off, the address is the key of a Python function in
        a table of Python functions.
        """
        if numba.config.DISABLE_JIT:  # ty: ignore[unresolved-attribute] -- set at runtime from NUMBA_DISABLE_JIT
            return _register_python_function(self)
        else:
            return _compile_wrapper(self).address  # ty: ignore[invalid-return-type] -- set once compiled

    # --------------------------------------------------------------------------
    #  Helpers
    # --------------------------------------------------------------------------
    @property
    def _comparison_key(self) -> "Hashable":
        """Return the function and each value paired with its type, which equality and hashing compare."""
        return (self.function, tuple((type(value), value) for value in self.params))

    @staticmethod
    def _as_python_scalar(value: object) -> CustomDistanceParam:
        """Return `value` as a Python bool, int or float, converting a numpy scalar of one of these kinds.

        Raises:
            TypeError: If `value` is not a bool, an integer or a floating-point number.
            ValueError: If `value` is an integer that does not fit in int64.
        """
        # bool comes first, because Python counts a bool as an int
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        elif isinstance(value, (int, np.integer)):
            if _INT64_INFO.min <= int(value) <= _INT64_INFO.max:
                return int(value)
            else:
                raise ValueError(f"A custom distance's int value must fit in int64; got {value}.")
        elif isinstance(value, (float, np.floating)):
            return float(value)
        else:
            raise TypeError(f"A custom distance's value must be a bool, an int or a float; got {value!r}.")


# ==================================================================================================
#  Helpers
# ==================================================================================================
@functools.cache
def _compile_wrapper(bound_function: BoundDistanceFunction) -> CFunc:
    """Return the C-callable wrapper of the numba function of `bound_function`, compiled on the first call.

    The cache also keeps every wrapper alive for the life of the process.  A distance store holds only the
    wrapper's address, so the wrapper must stay in memory as long as the process runs.

    The wrapper compiles the function's Python source inline, so a distance costs 1 call, not 2.  numba
    cannot inline a call that passes `*args`, so the wrapper's source is generated, with a name for each value:
    numba compiles each global of the wrapper's namespace, and so each value, as a constant.

    The wrapper and the inlined copy of the function both take the function's own njit options: an inlined function
    compiles with its caller's fastmath flags, so the wrapper needs the user's flags to compute what the
    user's function computes.
    """
    # only called with JIT compilation on, where `numba.njit` returns a dispatcher
    dispatcher = cast("Dispatcher", bound_function.function)
    target_options = dispatcher.targetoptions
    # `numba.njit` warns when it is given `nopython`, and the inlined copy sets its own `inline`, so neither of
    # the user's values is passed on.
    options = {name: value for name, value in target_options.items() if name not in ("nopython", "inline")}
    inlined_function = numba.njit(inline="always", **options)(dispatcher.py_func)

    value_names = [f"value_{index}" for index in range(len(bound_function.params))]
    arguments = ", ".join(["numba.carray(row_a, (n_dims,))", "numba.carray(row_b, (n_dims,))", *value_names])
    namespace: dict[str, Any] = {
        "numba": numba,
        "inlined_function": inlined_function,
        **dict(zip(value_names, bound_function.params)),
    }
    source = f"def wrapper(row_a, row_b, n_dims):\n    return inlined_function({arguments})\n"
    exec(source, namespace)  # noqa: S102 -- the source holds only names generated here, never a user's string
    return numba.cfunc(_WRAPPER_SIGNATURE, **options)(namespace["wrapper"])


@functools.cache
def _register_python_function(bound_function: BoundDistanceFunction) -> int:
    """Store a Python function of 2 rows that calls `bound_function` with its values, and return its key.

    The cache gives equal bound functions the same key, as `_compile_wrapper` gives them the same wrapper.
    """
    function, params = bound_function.function, bound_function.params

    def distance_of_rows(row_a: NDArray[np.float32], row_b: NDArray[np.float32]) -> float:
        return function(row_a, row_b, *params)

    _PYTHON_FUNCTIONS_BY_ADDRESS[id(distance_of_rows)] = distance_of_rows
    return id(distance_of_rows)


# ==================================================================================================
#  Calling a custom distance function
# ==================================================================================================
def call_custom_distance(
    address: int | np.integer, vectors: NDArray[np.float32], i: int | np.integer, j: int | np.integer
) -> float:
    """Return the distance between rows i and j of `vectors`, computed by the custom distance at `address`.

    `address` must come from `BoundDistanceFunction.address`: in compiled code, any other value crashes the
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
