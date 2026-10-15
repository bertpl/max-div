"""Numba functions that the tests of `numba_precompile` compile, in this process or in spawned ones.

A spawned process finds each function by importing its module. Importing a module also records its functions, so
these live in a module of their own, which a test imports only after emptying the list of recorded functions.
"""

from max_div._core.jit import LazilyCompiledFunction


def double(x: float) -> float:
    """Return twice `x`."""
    return 2.0 * x


def square(x: float) -> float:
    """Return `x` squared."""
    return x * x


# Each `LazilyCompiledFunction` replaces its function under the same name, as `lazy_njit` would, so that pickling by
# reference finds the `LazilyCompiledFunction`. The functions are wrapped directly, not through `lazy_njit`, because
# the coverage run disables the JIT, and `lazy_njit` then records no function.
double = LazilyCompiledFunction(double, "float64(float64)", {"cache": True}, False)
square = LazilyCompiledFunction(square, "float64(float64)", {"cache": True}, False)
