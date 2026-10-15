"""Compile all of max-div's numba functions ahead of the solves that call them.

This module sits directly in `_core`, not in `jit`, because it imports every module of `_core`, and a module
inside `jit` may not import the modules beside `jit`.
"""

import importlib
import pkgutil
from pathlib import Path

from max_div._core.extras import MissingExtraError
from max_div._core.jit import LazilyCompiledFunction


def numba_precompile() -> None:
    """Compile every numba function of max-div, or load it from numba's disk cache, ahead of the solves that call it.

    max-div compiles each numba function the first time that a solve calls it, and that solve's duration includes
    the compile time. Calling this function first moves that cost to a point of your choice, such as the start of an
    application or of a test session.

    After the call, no solve in this process compiles a numba function, with one exception: each
    `DistanceMetric.custom()` metric compiles a wrapper around your distance function once, which takes well under
    a second.

    The compiled code is also written to numba's disk cache, from which the worker processes of a parallel solve and
    later processes load it.

    On an empty disk cache the call takes up to about a minute, because it compiles every function, including those
    that a given solve does not use; once the cache holds them, the call takes about a second. With numba's JIT
    compilation disabled, the call compiles nothing.
    """
    # importing every module runs every `lazy_njit` decorator, which records its function in
    # `LazilyCompiledFunction.instances()`
    for module_info in pkgutil.walk_packages([str(Path(__file__).parent)], prefix=f"{__package__}."):
        try:
            importlib.import_module(module_info.name)
        except MissingExtraError:
            continue

    # an inline-only function never compiles on its own: each numba caller compiles a copy of its source
    for lazily_compiled_function in LazilyCompiledFunction.instances():
        if not lazily_compiled_function.is_inline_only:
            lazily_compiled_function.compile()
