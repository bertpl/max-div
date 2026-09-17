"""The plotting package is the only place matplotlib is imported, and it names the `plot` extra when it is missing."""

import importlib
import pkgutil
import subprocess
import sys
import textwrap

import pytest

import max_div
import max_div._core.plotting

_EXTRA = "max-div[plot]"


def _run_in_fresh_interpreter(code: str) -> subprocess.CompletedProcess[str]:
    """Run `code` in a new Python process, so the suite's own imports cannot mask what it loads."""
    # the command is this interpreter and a literal script, not untrusted input
    return subprocess.run(  # noqa: S603
        [sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True, check=False
    )


def test_guard_names_the_extra_when_matplotlib_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    # --- arrange ----------------------
    monkeypatch.setitem(sys.modules, "matplotlib", None)  # This makes `import matplotlib` fail

    # --- act / assert -----------------
    with pytest.raises(ImportError, match=r"max-div\[plot\]"):
        importlib.reload(max_div._core.plotting)


def test_importing_every_module_outside_plotting_never_loads_matplotlib() -> None:
    # --- arrange ----------------------
    module_names = [
        info.name
        for info in pkgutil.walk_packages(max_div.__path__, prefix="max_div.")
        if not info.name.startswith("max_div._core.plotting")
    ]
    assert module_names  # the walk found the package

    # --- act --------------------------
    result = _run_in_fresh_interpreter(
        f"""
        import importlib, sys
        for name in {module_names!r}:
            importlib.import_module(name)
        assert "matplotlib" not in sys.modules, "matplotlib was loaded"
        """
    )

    # --- assert -----------------------
    assert result.returncode == 0, result.stderr


def test_plotting_import_without_matplotlib_fails_naming_the_extra() -> None:
    # --- act --------------------------
    result = _run_in_fresh_interpreter(
        """
        import sys
        sys.modules["matplotlib"] = None
        try:
            import max_div._core.plotting
        except ImportError as e:
            print(e)
        else:
            raise SystemExit("imported without matplotlib")
        """
    )

    # --- assert -----------------------
    assert result.returncode == 0, result.stderr
    assert _EXTRA in result.stdout
