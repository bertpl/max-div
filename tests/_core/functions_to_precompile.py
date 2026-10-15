"""Numba functions that the tests of `numba_precompile` compile, in this process or in spawned ones.

A spawned process finds a function by importing its module, and the import records the function, so the functions
live in a module of their own: a test imports it only after emptying the list of recorded functions.
"""

from max_div._core.jit import LazilyCompiledFunction


def _double(x: float) -> float:
    """Return twice `x`."""
    return 2.0 * x


def _square(x: float) -> float:
    """Return `x` squared."""
    return x * x


# Each function is wrapped as `lazy_njit` wraps it with the JIT enabled, because `lazy_njit` records no function
# on the coverage run, which disables the JIT.
double = LazilyCompiledFunction(_double, "float64(float64)", {"cache": True}, False)
square = LazilyCompiledFunction(_square, "float64(float64)", {"cache": True}, False)
