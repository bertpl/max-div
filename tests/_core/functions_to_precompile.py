"""This module holds the numba functions for the tests of `numba_precompile`, which compile them in this process
or in spawned ones.

A spawned process finds each function by importing its module. Importing a module also adds its functions to
`LazilyCompiledFunction.instances()`, so the functions to compile live in this module of their own, which a test
imports only after emptying that list.
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
# the coverage run disables the JIT, and `lazy_njit` then adds no function to `LazilyCompiledFunction.instances()`.
double = LazilyCompiledFunction(double, "float64(float64)", {"cache": True}, False)
square = LazilyCompiledFunction(square, "float64(float64)", {"cache": True}, False)
