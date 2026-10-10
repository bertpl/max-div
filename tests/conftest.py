import pytest
from numba import config as numba_config

# Import max_div once, up front, in every process pytest starts -- including the xdist
# controller. A warning raised by a test is shipped to the controller, which imports the
# warning's module to deserialize it; that is otherwise the controller's first max_div import,
# arriving concurrently on execnet's per-worker receiver threads and racing over the module
# locks of the package graph until CPython's import deadlock detector aborts the run.
import max_div  # noqa: F401


# ==================================================================================================
#  JIT golden master opt-in
# ==================================================================================================
# The JIT golden master (tests/_core/solver/test_golden_master.py with JIT compilation on) checks
# expected data that holds only for the Python and numba versions that generated it. So whoever runs
# pytest passes `--jit-golden-master` only on a run that has those versions; for `make test`,
# JIT_GOLDEN_MASTER_ARG in the Makefile decides whether to pass it.
#
# With JIT compilation off, the cases check the expected data of interpreted solves, which holds on
# every runtime, so they always run.
def pytest_addoption(parser: pytest.Parser) -> None:
    """Register `--jit-golden-master`, which runs the JIT golden master cases."""
    parser.addoption(
        "--jit-golden-master",
        action="store_true",
        help="run the JIT golden master cases; pass it only on a run with the Python and numba versions "
        "that generated the JIT expected data",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register the `jit_golden_master` marker."""
    config.addinivalue_line("markers", "jit_golden_master: with JIT compilation on, runs only with --jit-golden-master")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip the `jit_golden_master` cases under JIT compilation, unless `--jit-golden-master` is given."""
    if not numba_config.DISABLE_JIT and not config.getoption("--jit-golden-master"):
        skip_marker = pytest.mark.skip(reason="the JIT golden master runs only with --jit-golden-master")
        for item in items:
            if item.get_closest_marker("jit_golden_master"):
                item.add_marker(skip_marker)
