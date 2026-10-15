"""Check that importing max-div compiles none of its numba functions.

Functions declared with `lazy_njit` compile on first use, not at import, so importing a module does not compile
them.
"""

import subprocess
import sys

import numba
import pytest

pytestmark = pytest.mark.skipif(numba.config.DISABLE_JIT, reason="with the JIT disabled nothing compiles")


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
