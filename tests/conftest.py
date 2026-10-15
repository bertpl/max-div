import pytest

# Import max_div once, up front, in every process pytest starts -- including the xdist
# controller. A warning raised by a test is shipped to the controller, which imports the
# warning's module to deserialize it; that is otherwise the controller's first max_div import,
# arriving concurrently on execnet's per-worker receiver threads and racing over the module
# locks of the package graph until CPython's import deadlock detector aborts the run.
import max_div


def pytest_configure(config: pytest.Config) -> None:
    """Compile max-div's numba functions once, in the process that starts the test run, before any test runs.

    The compiled code is written to numba's disk cache, from which every xdist worker loads it; without
    this step, each worker compiles the functions that its own tests call. A collect-only run skips the
    step, because it runs no test, and CI collects the suite in every matrix job to count its tests.
    """
    is_xdist_worker = hasattr(config, "workerinput")
    if (not is_xdist_worker) and (not config.option.collectonly):
        max_div.numba_precompile()
