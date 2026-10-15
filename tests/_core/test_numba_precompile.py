import subprocess
import sys
from types import SimpleNamespace

import numba
import pytest

from max_div._core.extras import MissingExtraError
from max_div._core.jit import LazilyCompiledFunction
from max_div._core.numba_precompile import numba_precompile

_needs_jit = pytest.mark.skipif(numba.config.DISABLE_JIT, reason="with the JIT disabled nothing compiles")

# This script runs in a fresh interpreter, so that the solves below still have to compile or load any function
# that `numba_precompile` misses, even when an earlier test already compiled it.
#
# numba calls `add_overload` both when it compiles a function and when it loads one from its disk cache, so
# counting those calls detects a missed function whatever the state of the cache.
#
# The script's solves cover:
# - every preset
# - every distance storage type other than AUTO
# - both problem constructors
# - a parallel solve
_SOLVES_AFTER_PRECOMPILE_SCRIPT = """
import numpy as np
from numba.core.dispatcher import _DispatcherBase

from max_div import Constraint, MaxDivProblem, numba_precompile
from max_div.solver import (
    DistanceStorageType, MaxDivSolverBuilder, ParallelMaxDivSolverBuilder, SolverPreset, Verbosity, iterations
)

numba_precompile()

n_added_overloads = 0
original_add_overload = _DispatcherBase.add_overload


def counting_add_overload(self, cres):
    global n_added_overloads
    n_added_overloads += 1
    return original_add_overload(self, cres)


_DispatcherBase.add_overload = counting_add_overload

vectors = np.random.default_rng(0).random((200, 3)).astype(np.float32)
constraints = [Constraint(int_set=set(range(40)), min_count=3, max_count=6)]
for problem_constraints in (None, constraints):
    for storage_type in [t for t in DistanceStorageType if t is not DistanceStorageType.AUTO]:
        for preset in SolverPreset:
            problem = MaxDivProblem.new(vectors, k=10, constraints=problem_constraints)
            builder = MaxDivSolverBuilder(problem).with_preset(iterations(50), preset)
            builder.with_distance_storage(storage_type).build().solve(verbosity=Verbosity.SILENT)

distances = np.linalg.norm(vectors[:, None, :] - vectors[None, :, :], axis=-1)
problem = MaxDivProblem.from_distances(distances, k=10, constraints=constraints)
MaxDivSolverBuilder(problem).with_preset(iterations(50)).build().solve(verbosity=Verbosity.SILENT)

problem = MaxDivProblem.new(vectors, k=10, constraints=constraints)
ParallelMaxDivSolverBuilder(problem).with_workers(iterations(50), 2).build().solve(verbosity=Verbosity.SILENT)

print(n_added_overloads)
"""


@_needs_jit
def test_numba_precompile_compiles_every_function_of_the_package_that_is_not_inline_only() -> None:
    """The call compiles every `lazy_njit` function of the package that Python may call."""
    # --- act --------------------------
    numba_precompile()

    # --- assert -----------------------
    non_inline_only_functions = [
        lazily_compiled_function
        for lazily_compiled_function in LazilyCompiledFunction.instances()
        if not lazily_compiled_function.is_inline_only
    ]
    assert non_inline_only_functions
    assert all(lazily_compiled_function.is_compiled for lazily_compiled_function in non_inline_only_functions)


@_needs_jit
def test_every_inline_only_function_of_the_package_compiles_with_its_declared_signature() -> None:
    """Each inline-only function compiles on its own, so a broken signature string fails this test.

    `numba_precompile` leaves inline-only functions out, and each numba caller compiles a copy of the inline-only
    function's source, so nothing else compiles the declared signatures of inline-only functions.
    """
    # --- arrange ----------------------
    # numba_precompile imports every module of the package, which records every inline-only function
    numba_precompile()
    inline_only_functions = [
        lazily_compiled_function
        for lazily_compiled_function in LazilyCompiledFunction.instances()
        if lazily_compiled_function.is_inline_only
    ]

    # --- act --------------------------
    for lazily_compiled_function in inline_only_functions:
        lazily_compiled_function.compile()

    # --- assert -----------------------
    assert inline_only_functions
    assert all(lazily_compiled_function.is_compiled for lazily_compiled_function in inline_only_functions)


@pytest.mark.parametrize(
    "options, is_inline_only, expected_is_compiled",
    [({"cache": False}, False, True), ({"cache": False, "inline": "always"}, True, False)],
)
@pytest.mark.parametrize("is_every_extra_missing", [False, True])
def test_numba_precompile_compiles_each_recorded_function_that_is_not_inline_only(
    monkeypatch: pytest.MonkeyPatch,
    is_every_extra_missing: bool,
    options: dict[str, object],
    is_inline_only: bool,
    expected_is_compiled: bool,
) -> None:
    """The call compiles each recorded function that is not inline-only, also when no module can be imported."""
    # --- arrange ----------------------
    if is_every_extra_missing:

        def raise_missing_extra_error(name: str) -> None:
            """Fail as a module does on import when its optional extra is missing."""
            raise MissingExtraError(name)

        # only numba_precompile's imports fail: numba itself imports modules when it loads a cached function
        monkeypatch.setattr(
            "max_div._core.numba_precompile.importlib", SimpleNamespace(import_module=raise_missing_extra_error)
        )
    monkeypatch.setattr(LazilyCompiledFunction, "_instances", [])
    lazily_compiled_function = LazilyCompiledFunction(lambda x: 2.0 * x, "float64(float64)", options, is_inline_only)

    # --- act --------------------------
    numba_precompile()

    # --- assert -----------------------
    assert lazily_compiled_function.is_compiled == expected_is_compiled


@_needs_jit
def test_no_solve_after_numba_precompile_compiles_or_loads_a_numba_function() -> None:
    """After the call, no solve compiles or loads a numba function."""
    # --- act --------------------------
    output = subprocess.run(  # noqa: S603 -- fixed args
        [sys.executable, "-c", _SOLVES_AFTER_PRECOMPILE_SCRIPT], capture_output=True, text=True, check=True
    ).stdout

    # --- assert -----------------------
    assert int(output.split()[-1]) == 0
