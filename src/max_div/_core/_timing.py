import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass


# =================================================================================================
#  End-to-end solve timing
# =================================================================================================
@dataclass
class EndToEndTiming:
    """An EndToEndTiming records the measured wall-clock span of one end-to-end solve."""

    t_elapsed_sec: float = 0.0


@contextmanager
def measure_end_to_end() -> Iterator[EndToEndTiming]:
    """Measure an end-to-end solve, so every benchmark shares one timing definition.

    End-to-end means everything a user pays for after the problem data exists: solver
    configuration, `build()` (distance-store construction included), and the solve itself.
    Problem construction stays outside — enter this context after the vectors exist and
    before `build()`.

    Yields:
        An `EndToEndTiming` whose `t_elapsed_sec` is set (via `perf_counter`) when the
        context exits.
    """
    timing = EndToEndTiming()
    t_start = time.perf_counter()
    try:
        yield timing
    finally:
        timing.t_elapsed_sec = time.perf_counter() - t_start
