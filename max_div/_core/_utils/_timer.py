import time


class Timer:
    """
    Class acting as a context manager to measure time elapsed.  Elapsed time can be retrieved using t_elapsed().
    """

    # --- constructor -------------------------------------
    def __init__(self):
        self._start: float | None = None
        self._end: float | None = None

    # --- context manager ---------------------------------
    def __enter__(self):
        self._start = time.perf_counter_ns()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._end = time.perf_counter_ns()

    # --- extract results ---------------------------------
    def t_elapsed_nsec(self) -> float:
        if self._start is None:
            raise RuntimeError("Timer has not been started.")

        if self._end is None:
            # timer still running
            return time.perf_counter_ns() - self._start
        else:
            # timer finished
            return self._end - self._start

    def t_elapsed_sec(self) -> float:
        return self.t_elapsed_nsec() / 1e9
