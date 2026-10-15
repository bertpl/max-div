"""Compile all of max-div's numba functions ahead of the solves that call them.

This module sits directly in `_core`, not in `jit`, because it imports every module of `_core`, and a module
inside `jit` may not import the modules beside `jit`.
"""

import importlib
import multiprocessing
import os
import pkgutil
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from contextlib import suppress
from pathlib import Path

from max_div._core.extras import MissingExtraError
from max_div._core.jit import LazilyCompiledFunction

# Beyond 8 processes the call gets little faster, because each process compiles the callees that its functions
# share with the functions of other processes.
_MAX_DEFAULT_N_PROCESSES = 8


def numba_precompile(n_processes: int | None = None) -> None:
    """Compile every numba function of max-div, or load it from numba's disk cache, ahead of the solves that call it.

    max-div compiles each numba function the first time that a solve calls it, and that solve's duration includes
    the compile time. Calling this function first moves that cost to a point of your choice, such as the start of an
    application or of a test session.

    After the call, no solve in this process compiles a numba function, with one exception: each
    `DistanceMetric.custom()` metric compiles a wrapper around your distance function once, which takes well under
    a second.

    The compiled code is also written to numba's disk cache, from which the worker processes of a parallel solve and
    later processes load it.

    The call compiles in `n_processes` spawned processes, which write the compiled code to numba's disk cache, and
    then loads it from there into this process. Spawned processes import your main module again, so a script that
    calls this function must call it under `if __name__ == "__main__":`; otherwise each process fails with an
    error, and the call compiles everything in this process.

    On an empty disk cache the call compiles every function, including those that a given solve does not use; this
    takes about 15 seconds on a machine with 8 or more cores, and up to about a minute with 2 cores. Once the cache
    holds them, the call takes about a second with 1 process, and about 2 seconds with more, which spend that time
    starting. With numba's JIT compilation disabled, the call compiles nothing and starts no process.

    Args:
        n_processes: The number of processes that compile in parallel; by default the number of CPUs, at most 8.
            With 1, the call compiles in this process and starts no other.

    Raises:
        ValueError: If `n_processes` is less than 1.
    """
    if n_processes is None:
        n_processes = min(os.cpu_count() or 1, _MAX_DEFAULT_N_PROCESSES)
    elif n_processes < 1:
        raise ValueError(f"n_processes must be at least 1; got {n_processes}.")

    # importing every module runs every `lazy_njit` decorator, which records its function in
    # `LazilyCompiledFunction.instances()`
    for module_info in pkgutil.walk_packages([str(Path(__file__).parent)], prefix=f"{__package__}."):
        try:
            importlib.import_module(module_info.name)
        except MissingExtraError:
            continue

    # an inline-only function is not compiled here: each numba caller compiles a copy of its source
    functions = [function for function in LazilyCompiledFunction.instances() if not function.is_inline_only]

    if n_processes > 1:
        # Each task is 1 function, in declaration order, and a process takes the next task when it finishes one.
        # A process that dies breaks the pool, and the loop below then compiles whatever no process cached.
        # With the JIT disabled, `lazy_njit` records no function, so the pool gets no task and starts no process.
        context = multiprocessing.get_context("spawn")
        with suppress(BrokenProcessPool), ProcessPoolExecutor(n_processes, mp_context=context) as pool:
            module_names = [function.py_func.__module__ for function in functions]
            qualnames = [function.py_func.__qualname__ for function in functions]
            # consuming the results raises here any error that a process raised
            list(pool.map(_compile_function_by_name, module_names, qualnames))

    # after the processes, this loads each function from numba's disk cache, and compiles the functions that no
    # process cached
    for function in functions:
        function.compile()


def _compile_function_by_name(module_name: str, qualname: str) -> None:
    """Compile the recorded function with this module and qualified name, in a process that `numba_precompile` spawns.

    The function stays at module level, because a spawned process finds it by its module and name. A function
    that importing its module does not record, such as one created inside another function, is skipped here and
    compiled by the caller of `numba_precompile`.
    """
    importlib.import_module(module_name)
    for function in LazilyCompiledFunction.instances():
        if function.py_func.__module__ == module_name and function.py_func.__qualname__ == qualname:
            function.compile()
