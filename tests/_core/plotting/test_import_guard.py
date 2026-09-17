"""The plotting package is the only place matplotlib is imported, and it names the `plot` extra when absent."""

import pkgutil
import subprocess
import sys
import textwrap

import pytest

import max_div
from max_div._core.extras import MissingExtraError
from tests._extras import needs_extra, without_extra


@without_extra("plot")
def test_importing_the_plotting_package_without_the_extra_names_it() -> None:
    with pytest.raises(MissingExtraError, match=r"max-div\[plot\]"):
        import max_div._core.plotting  # noqa: F401


@needs_extra("plot")
def test_importing_every_module_outside_plotting_never_loads_matplotlib() -> None:
    # --- arrange ----------------------
    module_names = [
        info.name
        for info in pkgutil.walk_packages(max_div.__path__, prefix="max_div.")
        if not info.name.startswith("max_div._core.plotting")
    ]
    assert module_names  # the walk found the package
    script = textwrap.dedent(
        f"""
        import importlib, sys
        for name in {module_names!r}:
            importlib.import_module(name)
        assert "matplotlib" not in sys.modules, "matplotlib was loaded"
        """
    )

    # --- act --------------------------
    # This runs in a fresh interpreter, so the suite's own imports cannot mask what gets loaded. The
    # command is this interpreter and a literal script, not untrusted input.
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)  # noqa: S603

    # --- assert -----------------------
    assert result.returncode == 0, result.stderr
