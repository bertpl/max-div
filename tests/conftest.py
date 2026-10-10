import pytest

# Import max_div once, up front, in every process pytest starts -- including the xdist
# controller. A warning raised by a test is shipped to the controller, which imports the
# warning's module to deserialize it; that is otherwise the controller's first max_div import,
# arriving concurrently on execnet's per-worker receiver threads and racing over the module
# locks of the package graph until CPython's import deadlock detector aborts the run.
import max_div  # noqa: F401


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register `--require-jit-golden-master`, which `make test` passes on a run from uv.lock on the default Python."""
    parser.addoption(
        "--require-jit-golden-master",
        action="store_true",
        help="fail the JIT golden master, instead of skipping it, when this run's Python or numba version "
        "differs from the versions that it was generated with",
    )
