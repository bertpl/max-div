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

# Beyond this many processes the call barely gets faster, because each process compiles its own copy of every
# callee that its functions share with the functions in other processes.
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

    The call starts `n_processes` spawned processes, which compile the functions; the call then loads the compiled
    code from numba's disk cache into this process.

    Spawned processes import your main module again, so with more than 1 process, a script that calls this
    function must call it under `if __name__ == "__main__":`; otherwise each spawned process runs the script's
    top-level code again, fails, and the call then compiles every function in this process.

    On an empty disk cache the call compiles every function, including those that a given solve does not use.
    Compiling them takes about 15 seconds on a machine with 8 or more cores, and up to about a minute with 2 cores.

    Once the cache holds the compiled functions, the call takes about a second with 1 process, and about 2 seconds
    with more processes, because starting the processes takes the extra second. With numba's JIT compilation
    disabled, the call compiles nothing and starts no process.

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
    lazily_compiled_functions = [
        lazily_compiled_function
        for lazily_compiled_function in LazilyCompiledFunction.instances()
        if not lazily_compiled_function.is_inline_only
    ]

    if n_processes > 1:
        # Each task is 1 function, in declaration order, and a process takes the next task when it finishes one.
        # A process that dies breaks the pool, which `suppress` ignores.
        # With the JIT disabled, `lazy_njit` records no function, so the pool gets no task and starts no process.
        with (
            suppress(BrokenProcessPool),
            ProcessPoolExecutor(n_processes, mp_context=multiprocessing.get_context("spawn")) as pool,
        ):
            # reading the results re-raises in this process any error that a spawned process raised
            list(pool.map(_compile_into_disk_cache, lazily_compiled_functions))

    # this loop loads each function that a spawned process cached from numba's disk cache, and compiles the others
    # in this process
    for lazily_compiled_function in lazily_compiled_functions:
        lazily_compiled_function.compile()


def _compile_into_disk_cache(lazily_compiled_function: LazilyCompiledFunction) -> None:
    """Compile the function in a process that `numba_precompile` spawns, and send nothing back to the caller.

    The spawned process receives the `LazilyCompiledFunction` pickled by reference, and finds it by importing the
    module that defines it; `_compile_into_disk_cache` stays at module level, so that the process finds it the
    same way. Returning the compiled function would pickle it back to the caller; the caller loads it from numba's
    disk cache instead.
    """
    lazily_compiled_function.compile()
