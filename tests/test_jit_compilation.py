"""These tests check, across the whole package, when max-div's numba functions compile.

Functions declared with `lazy_njit` compile on first use, not at import, so importing a module does
not reveal a broken signature string. These tests build every such function to catch one, and check
that importing the package compiles nothing.
"""

import importlib
import pkgutil
import subprocess
import sys

import numba
import pytest

import max_div
from max_div._core.extras import MissingExtraError
from max_div._core.jit import LazyDispatcher

pytestmark = pytest.mark.skipif(numba.config.DISABLE_JIT, reason="with the JIT disabled nothing compiles")


def test_every_lazy_dispatcher_in_the_package_builds() -> None:
    """Every `lazy_njit` function in the package compiles its declared signatures."""
    # --- arrange ----------------------
    for module_info in pkgutil.walk_packages(max_div.__path__, prefix="max_div."):
        try:
            importlib.import_module(module_info.name)
        except MissingExtraError:
            continue  # the module needs an optional extra that is not installed here

    # --- act --------------------------
    for lazy_dispatcher in LazyDispatcher.instances():
        lazy_dispatcher.build()

    # --- assert -----------------------
    assert LazyDispatcher.instances()
    assert all(lazy_dispatcher.is_built for lazy_dispatcher in LazyDispatcher.instances())


def test_importing_the_package_compiles_nothing() -> None:
    """A fresh interpreter can import max-div without building any `lazy_njit` function."""
    # --- arrange ----------------------
    code = (
        "import max_div\n"
        "from max_div._core.jit import LazyDispatcher\n"
        "print(len(LazyDispatcher.instances()), sum(d.is_built for d in LazyDispatcher.instances()))\n"
    )

    # --- act --------------------------
    output = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout  # noqa: S603 -- fixed args

    # --- assert -----------------------
    n_declared, n_built = (int(count) for count in output.split())
    assert n_declared > 0
    assert n_built == 0
