import importlib
import subprocess
import sys

import numba
import pytest

from max_div._core.extras import MissingExtraError
from max_div._core.jit import LazilyCompiledFunction
from max_div._core.numba_precompile import numba_precompile

_needs_jit = pytest.mark.skipif(numba.config.DISABLE_JIT, reason="with the JIT disabled nothing compiles")

# Run in a fresh interpreter, so that no earlier test has compiled or loaded a function that `numba_precompile`
# misses. numba calls `add_overload` both when it compiles a function and when it loads one from its disk cache, so
# counting those calls detects a missed function whatever the state of the cache.
_SOLVES_AFTER_PRECOMPILE = """
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
    for storage_type in (DistanceStorageType.FULL_MATRIX, DistanceStorageType.LAZY):
        for preset in SolverPreset:
            problem = MaxDivProblem.new(vectors, k=10, constraints=problem_constraints)
            builder = MaxDivSolverBuilder(problem).with_preset(iterations(50), preset).with_distance_storage(storage_type)
            builder.build().solve(verbosity=Verbosity.SILENT)

distances = np.linalg.norm(vectors[:, None, :] - vectors[None, :, :], axis=-1)
problem = MaxDivProblem.from_distances(distances, k=10, constraints=constraints)
MaxDivSolverBuilder(problem).with_preset(iterations(50)).build().solve(verbosity=Verbosity.SILENT)

problem = MaxDivProblem.new(vectors, k=10, constraints=constraints)
ParallelMaxDivSolverBuilder(problem).with_workers(iterations(50), 2).build().solve(verbosity=Verbosity.SILENT)

print(n_added_overloads)
"""


@_needs_jit
def test_numba_precompile_compiles_every_function_that_the_package_declares() -> None:
    """After the call, every `lazy_njit` function in the package is compiled, so a broken signature string fails."""
    # --- act --------------------------
    numba_precompile()

    # --- assert -----------------------
    assert LazilyCompiledFunction.instances()
    assert all(lazily_compiled_function.is_compiled for lazily_compiled_function in LazilyCompiledFunction.instances())


@pytest.mark.parametrize("is_every_extra_missing", [False, True])
def test_numba_precompile_compiles_each_recorded_function(
    monkeypatch: pytest.MonkeyPatch, is_every_extra_missing: bool
) -> None:
    """The call compiles a function that is recorded but not yet compiled, also when no module can be imported."""
    # --- arrange ----------------------
    if is_every_extra_missing:

        def import_module_without_extras(name: str) -> None:
            """Fail as a module does on import when its optional extra is missing."""
            raise MissingExtraError(name)

        monkeypatch.setattr(importlib, "import_module", import_module_without_extras)
    monkeypatch.setattr(LazilyCompiledFunction, "_instances", [])
    lazily_compiled_function = LazilyCompiledFunction(lambda x: 2.0 * x, "float64(float64)", {"cache": False})

    # --- act --------------------------
    numba_precompile()

    # --- assert -----------------------
    assert lazily_compiled_function.is_compiled


@_needs_jit
def test_no_solve_after_numba_precompile_compiles_or_loads_a_numba_function() -> None:
    """After the call, solves of every preset, storage type and problem kind add no compiled function in the process."""
    # --- act --------------------------
    output = subprocess.run(  # noqa: S603 -- fixed args
        [sys.executable, "-c", _SOLVES_AFTER_PRECOMPILE], capture_output=True, text=True, check=True
    ).stdout

    # --- assert -----------------------
    assert int(output.split()[-1]) == 0
