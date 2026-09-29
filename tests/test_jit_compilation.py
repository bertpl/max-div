"""Check, across the whole package, when max-div's numba functions compile.

Functions declared with `lazy_njit` compile on first use, not at import, so importing a module does
not reveal a broken signature string. These tests compile every such function, so that a broken
signature string fails a test, and check that importing the package compiles nothing.
"""

import importlib
import pkgutil
import subprocess
import sys

import numba
import pytest

import max_div
from max_div._core.extras import MissingExtraError
from max_div._core.jit import LazilyCompiledFunction

pytestmark = pytest.mark.skipif(numba.config.DISABLE_JIT, reason="with the JIT disabled nothing compiles")


def test_every_lazily_compiled_function_in_the_package_compiles() -> None:
    """Every `lazy_njit` function in the package compiles its declared signatures."""
    # --- arrange ----------------------
    for module_info in pkgutil.walk_packages(max_div.__path__, prefix="max_div."):
        try:
            importlib.import_module(module_info.name)
        except MissingExtraError:
            continue  # the module needs an optional extra that is not installed here

    # --- act --------------------------
    for lazily_compiled_function in LazilyCompiledFunction.instances():
        lazily_compiled_function.compile()

    # --- assert -----------------------
    assert LazilyCompiledFunction.instances()
    assert all(lazily_compiled_function.is_compiled for lazily_compiled_function in LazilyCompiledFunction.instances())


def test_importing_the_package_compiles_nothing() -> None:
    """A fresh interpreter can import max-div without compiling any `lazy_njit` function."""
    # --- arrange ----------------------
    code = (
        "import max_div\n"
        "from max_div._core.jit import LazilyCompiledFunction\n"
        "instances = LazilyCompiledFunction.instances()\n"
        "print(len(instances), sum(f.is_compiled for f in instances))\n"
    )

    # --- act --------------------------
    output = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout  # noqa: S603 -- fixed args

    # --- assert -----------------------
    n_functions, n_compiled = (int(count) for count in output.split())
    assert n_functions > 0
    assert n_compiled == 0
