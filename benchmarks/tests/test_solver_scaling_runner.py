from benchmarks.solver_scaling import grid, runner
from benchmarks.solver_scaling.outcome import REASON_MEMORY, REASON_TIMEOUT


class _FakeChild:
    """Minimal subprocess.Popen stand-in for supervisor tests: scripted poll, recorded kill."""

    def __init__(self, running: bool):
        self._running = running
        self.pid = 4321
        self.killed = False

    def poll(self):
        return None if self._running else 0

    def kill(self):
        self.killed = True
        self._running = False

    def wait(self):
        return 0


def test_supervise_returns_no_reason_when_the_child_finishes_on_its_own(monkeypatch):
    # --- arrange ----------------------
    monkeypatch.setattr(runner, "_available_bytes", lambda: 10_000)
    monkeypatch.setattr(runner, "_observe_child", lambda _pid: (1024, False))

    # --- act --------------------------
    sup = runner._supervise(_FakeChild(running=False), budget_sec=60.0, baseline_bytes=10_000)

    # --- assert -----------------------
    assert sup.reason is None


def test_supervise_kills_on_crossing_the_memory_cap_machine_level(monkeypatch):
    # --- arrange ----------------------
    child = _FakeChild(running=True)
    monkeypatch.setattr(runner, "_available_bytes", lambda: 0)
    monkeypatch.setattr(runner, "_observe_child", lambda _pid: (1024, True))

    # --- act --------------------------
    sup = runner._supervise(child, budget_sec=60.0, baseline_bytes=grid.MEMORY_CAP_BYTES + 1)

    # --- assert -----------------------
    assert sup.reason == REASON_MEMORY
    assert child.killed
    assert sup.spawned  # child processes were observed before the kill


def test_supervise_kills_on_passing_the_deadline(monkeypatch):
    # --- arrange ----------------------
    child = _FakeChild(running=True)
    monkeypatch.setattr(runner, "_available_bytes", lambda: 10_000)
    monkeypatch.setattr(runner, "_observe_child", lambda _pid: (1024, False))
    monkeypatch.setattr(grid, "SETUP_GRACE_SEC", 0.0)

    # --- act --------------------------
    sup = runner._supervise(child, budget_sec=0.0, baseline_bytes=10_000)

    # --- assert -----------------------
    assert sup.reason == REASON_TIMEOUT
    assert child.killed


def test_run_measurement_kills_a_child_that_outlives_its_budget(monkeypatch):
    # Exercises the real subprocess path end to end via the sleep fixture.
    # --- arrange ----------------------
    monkeypatch.setattr(grid, "SETUP_GRACE_SEC", 1.0)

    # --- act --------------------------
    record = runner.run_measurement("_test_sleep", "sleep", n=20, k=2, seed=0, budget_sec=0.3)

    # --- assert -----------------------
    assert not record.completed
    assert record.reason == REASON_TIMEOUT


def test_is_settled_judges_the_final_window():
    """A footprint still growing in the final window is not settled; a flat one is."""
    # --- arrange ----------------------
    flat = [(t, 100) for t in range(0, 60)]
    growing = [(t, 100 + t) for t in range(0, 60)]

    # --- act / assert -----------------
    assert runner._is_settled(flat)
    assert not runner._is_settled(growing)
    assert not runner._is_settled([(0.0, 100)])  # too short to judge


def test_supervise_kills_on_the_child_rss_alone(monkeypatch):
    """The per-process check backstops the machine-level one, which OS memory compression can mask."""
    # --- arrange ----------------------
    child = _FakeChild(running=True)
    monkeypatch.setattr(runner, "_available_bytes", lambda: 10_000)
    monkeypatch.setattr(runner, "_observe_child", lambda _pid: (grid.MEMORY_CAP_BYTES + 1, False))

    # --- act --------------------------
    sup = runner._supervise(child, budget_sec=60.0, baseline_bytes=10_000)

    # --- assert -----------------------
    assert sup.reason == REASON_MEMORY
    assert child.killed
