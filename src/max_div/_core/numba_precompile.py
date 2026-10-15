"""Compile all of max-div's numba functions ahead of the solves that call them.

This module sits at the top of `_core`, not in `jit`, because it imports every module of the package,
and nothing inside `_core` may depend on that.
"""

import importlib
import pkgutil

import max_div
from max_div._core.extras import MissingExtraError
from max_div._core.jit import LazilyCompiledFunction


def numba_precompile() -> None:
    """Compile every numba function of max-div, or load it from numba's disk cache, so that later solves compile nothing.

    max-div compiles each numba function the first time that a solve calls it, and that solve's duration includes
    the compile time. Calling this function first moves that cost to a point of your choice, such as the start of an
    application or of a test session.

    After the call, no solve in this process compiles, except that a `DistanceMetric.custom()` metric compiles a
    wrapper around your distance function, in well under a second, once per metric. The compiled code is also written
    to numba's disk cache, from which the worker processes of a parallel solve and later processes load it.

    On an empty disk cache the call takes up to about a minute, because it compiles every function, including those
    that a given solve does not use; once the cache holds them, it takes about a second. With numba's JIT compilation
    disabled, it compiles nothing.
    """
    # importing every module creates every lazily compiled function that the package declares
    for module_info in pkgutil.walk_packages(max_div.__path__, prefix="max_div."):
        try:
            importlib.import_module(module_info.name)
        except MissingExtraError:
            continue  # the module needs an optional extra that is not installed

    for lazily_compiled_function in LazilyCompiledFunction.instances():
        lazily_compiled_function.compile()
