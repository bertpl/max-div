"""Parent-side campaign runner: one subprocess per run, with budget and memory kills.

The parent polls the child's resident set size on a fixed interval and kills it the moment
it crosses the memory cap; the time kill fires at the run's budget plus a setup grace,
since the child's untimed setup (imports, problem construction) happens inside the same
process.

A completed child reports its own timed measurement, so the grace never inflates a
measured value — it only decides when a stuck child is declared dead.
"""

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from benchmarks.tool_scaling.configs import Mode
from benchmarks.tool_scaling.grid import MEMORY_CAP_BYTES, SETUP_GRACE_SEC
from benchmarks.tool_scaling.records import ScalingRunRecord

_POLL_SEC = 0.5


def run_measurement(tool: str, mode: Mode, n: int, k: int, seed: int, budget_sec: float) -> ScalingRunRecord:
    """Execute one campaign run in a subprocess and return its record.

    A run ends one of three ways: the child reports a result (completed or failed), the
    memory poll catches it crossing the cap, or the deadline fires — ``timeout`` and
    ``memory`` reasons come from the parent, every other reason from the child itself.
    """
    spec = {"tool": tool, "mode": mode.value, "n": n, "seed": seed, "budget_sec": budget_sec}
    with tempfile.TemporaryDirectory() as tmp:
        result_path = Path(tmp) / "result.json"
        child = subprocess.Popen(  # noqa: S603 -- fixed module invocation, repo-local
            [sys.executable, "-m", "benchmarks.tool_scaling.run_one", json.dumps(spec), str(result_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        reason, peak_seen = _supervise(child, budget_sec)
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}

    completed = bool(result.get("completed", False)) and reason is None
    return ScalingRunRecord(
        tool=tool,
        mode=mode.value,
        n=n,
        k=k,
        seed=seed,
        budget_sec=budget_sec,
        completed=completed,
        reason=reason or (None if completed else result.get("reason", "child exited without a result")),
        measured_sec=result.get("measured_sec"),
        peak_rss_bytes=max(result.get("peak_rss_bytes") or 0, peak_seen) or None,
        min_separation=result.get("min_separation"),
    )


def _supervise(child: subprocess.Popen, budget_sec: float) -> tuple[str | None, int]:
    """Wait for the child while enforcing the deadline and the memory cap.

    Returns:
        ``(reason, peak_rss_seen)`` — reason is None when the child ended on its own,
        ``timeout`` or ``memory`` when the parent killed it. The polled peak complements
        the child's own report, which is lost when the child is killed.
    """
    deadline = time.monotonic() + budget_sec + SETUP_GRACE_SEC
    peak_seen = 0
    while child.poll() is None:
        rss = _rss_bytes(child.pid)
        peak_seen = max(peak_seen, rss)
        if rss > MEMORY_CAP_BYTES:
            child.kill()
            child.wait()
            return "memory", peak_seen
        if time.monotonic() > deadline:
            child.kill()
            child.wait()
            return "timeout", peak_seen
        time.sleep(_POLL_SEC)
    return None, peak_seen


def _rss_bytes(pid: int) -> int:
    """Return the process's current resident set size, via ps (kilobytes on macOS and Linux alike)."""
    try:
        out = subprocess.run(  # noqa: S603 -- fixed command, numeric argument
            ["/bin/ps", "-o", "rss=", "-p", str(pid)], capture_output=True, encoding="utf-8", check=False
        ).stdout.strip()
        return int(out) * 1024 if out else 0
    except (ValueError, OSError):
        # A failed poll must never kill the supervisor: without ps the memory cap simply
        # goes unenforced for that tick, and the child's own peak report still arrives.
        return 0
