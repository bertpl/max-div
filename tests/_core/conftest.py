import time

import pytest


class FakeClock:
    """Deterministic stand-in for the wall-clock functions, installed by the `fake_clock` fixture.

    Time is held in seconds and only moves when the test moves it: `advance` steps it explicitly,
    and `auto_advance_sec` (0 by default) adds a fixed step on every read, for code that only needs
    time to progress monotonically. The patched `sleep` advances the clock instead of blocking, so a
    test can keep a `sleep(...)` call and have it take zero wall-clock time.
    """

    def __init__(self, start_sec: float = 1000.0, auto_advance_sec: float = 0.0) -> None:
        self._t = start_sec
        self.auto_advance_sec = auto_advance_sec

    def _read(self) -> float:
        t = self._t
        self._t += self.auto_advance_sec
        return t

    def perf_counter(self) -> float:
        return self._read()

    def perf_counter_ns(self) -> float:
        return self._read() * 1e9

    def sleep(self, seconds: float) -> None:
        self._t += seconds

    def advance(self, dt_sec: float) -> None:
        self._t += dt_sec


@pytest.fixture
def fake_clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    """Replace the process wall-clock with a `FakeClock` for the duration of one test.

    Patches `time.perf_counter`, `time.perf_counter_ns` and `time.sleep`. Production code reads the
    clock through the `time` module (never `from time import ...`), so this single patch reaches
    every reader, turning any "did enough time pass?" assertion from a wall-clock race into an exact
    check.
    """
    clock = FakeClock()
    monkeypatch.setattr(time, "perf_counter", clock.perf_counter)
    monkeypatch.setattr(time, "perf_counter_ns", clock.perf_counter_ns)
    monkeypatch.setattr(time, "sleep", clock.sleep)
    return clock
