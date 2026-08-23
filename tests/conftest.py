# Import max_div once, up front, in every process pytest starts -- including the xdist
# controller. A warning raised by a test is shipped to the controller, which imports the
# warning's module to deserialize it; that is otherwise the controller's first max_div import,
# arriving concurrently on execnet's per-worker receiver threads and racing over the module
# locks of the package graph until CPython's import deadlock detector aborts the run.
import max_div  # noqa: F401
