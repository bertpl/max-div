"""Package-wide checks on when max-div's numba functions compile.

Functions declared with `lazy_njit` compile on first use, not at import, so a broken signature
string no longer fails when its module is imported. These tests restore that check, and guard
against code that would compile at import again.
"""

import importlib
import pkgutil
import subprocess
import sys

import numba
import pytest

import max_div
from max_div._core.extras import MissingExtraError
from max_div._core.jit import lazy_dispatchers

pytestmark = pytest.mark.skipif(numba.config.DISABLE_JIT, reason="with the JIT disabled nothing compiles")


def test_every_lazy_dispatcher_in_the_package_builds() -> None:
    """Every `lazy_njit` function in the package compiles its declared signatures."""
    # --- arrange ----------------------
    for module_info in pkgutil.walk_packages(max_div.__path__, prefix="max_div."):
        try:
            importlib.import_module(module_info.name)
        except MissingExtraError:
            continue  # a module behind an optional extra that is not installed here

    # --- act --------------------------
    for lazy_dispatcher in lazy_dispatchers():
        lazy_dispatcher.build()

    # --- assert -----------------------
    assert lazy_dispatchers()
    assert all(lazy_dispatcher.is_built for lazy_dispatcher in lazy_dispatchers())


def test_importing_the_package_compiles_nothing() -> None:
    """A fresh interpreter can import max-div without building any `lazy_njit` function."""
    # --- arrange ----------------------
    code = (
        "import max_div\n"
        "from max_div._core.jit import lazy_dispatchers\n"
        "print(len(lazy_dispatchers()), sum(d.is_built for d in lazy_dispatchers()))\n"
    )

    # --- act --------------------------
    output = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout  # noqa: S603 -- fixed args

    # --- assert -----------------------
    n_declared, n_built = (int(count) for count in output.split())
    assert n_declared > 0
    assert n_built == 0
