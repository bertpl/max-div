"""Parent-side runner: one subprocess per run, with time and memory kills.

Two memory measurements serve two purposes:

* **Enforcement and bracketing are machine-level**: the parent samples the machine's available
  memory just before launching the child and kills the run once the drop below that level
  crosses the memory cap. This counts every process the solver spawns and counts shared memory
  once, with no knowledge of any solver's internals. Its noise (a few hundred MB of unrelated
  OS activity) is negligible at the cap's scale.
* **The recorded footprint is the solver process's peak RSS**: precise to ~1 MB, which the
  extrapolating memory fit needs and machine-level readings are far too noisy to provide. The
  child reports its kernel-maintained high-water mark (catching transients shorter than the
  poll interval) and the parent's poll complements it for killed children. RSS sees one process
  fully, threads included — so the parent also records whether the child was ever observed with
  live child processes, and the memory sweep does not measure such configurations.

The time kill fires at the run's budget plus a setup grace, since the child's untimed setup
(imports, problem construction) happens inside the same process. A completed child reports its
own timed measurement, so the grace never inflates a measured value — it only decides when a
stuck child is declared dead.
"""

import json
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import psutil

from . import grid
from .outcome import REASON_MEMORY, REASON_TIMEOUT
from .records import ScalingRunRecord

_POLL_SEC = 0.5

# A killed run's footprint counts as settled when its peak gained less than this fraction over
# the final window — the solver had stopped allocating, so the recorded peak is trustworthy.
_SETTLE_WINDOW_SEC = 20.0
_SETTLE_TOLERANCE = 0.02


@dataclass(frozen=True)
class _Supervision:
    """What the supervisor observed: how the run ended, and the child's memory behavior."""

    reason: str | None
    peak_rss: int
    spawned: bool
    settled: bool


def run_measurement(tool: str, config: str, n: int, k: int, seed: int, budget_sec: float) -> ScalingRunRecord:
    """Execute one run in a subprocess and return its record.

    A run ends one of three ways: the child reports a result (completed or failed), the
    machine-level memory poll catches it crossing the cap, or the deadline fires — `timeout`
    and `memory` reasons come from the parent, every other reason from the child itself.
    """
    with tempfile.TemporaryDirectory() as tmp:
        vectors_path = Path(tmp) / "vectors.npy"
        _save_problem_vectors(n, vectors_path)
        result_path = Path(tmp) / "result.json"
        spec = {
            "tool": tool,
            "config": config,
            "n": n,
            "k": k,
            "seed": seed,
            "budget_sec": budget_sec,
            "vectors_path": str(vectors_path),
        }
        baseline = _available_bytes()
        child = subprocess.Popen(  # noqa: S603 -- fixed module invocation, repo-local
            [sys.executable, "-m", "benchmarks.solver_scaling.run_one", json.dumps(spec), str(result_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        sup = _supervise(child, budget_sec, baseline)
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}

    completed = bool(result.get("completed", False)) and sup.reason is None
    return ScalingRunRecord(
        tool=tool,
        config=config,
        n=n,
        k=k,
        seed=seed,
        budget_sec=budget_sec,
        completed=completed,
        reason=sup.reason or (None if completed else result.get("reason", "child exited without a result")),
        measured_sec=result.get("measured_sec"),
        peak_memory_bytes=max(result.get("peak_memory_bytes") or 0, sup.peak_rss) or None,
        min_separation=result.get("min_separation"),
        spawned_processes=sup.spawned,
        # a run that ended on its own finished allocating by definition; only a deadline kill
        # leaves the question open, answered by the poll series
        memory_settled=sup.settled if sup.reason == REASON_TIMEOUT else True,
    )


def _supervise(child: subprocess.Popen, budget_sec: float, baseline_bytes: int) -> _Supervision:
    """Wait for the child while enforcing the deadline and the machine-level memory cap.

    Args:
        baseline_bytes: the machine's available memory just before the child was launched; the
            cap is enforced on the drop below this level.

    Returns:
        The run's `_Supervision`: reason is None when the child ended on its own, `timeout` or
        `memory` when the parent killed it.
    """
    deadline = time.monotonic() + budget_sec + grid.SETUP_GRACE_SEC
    samples: list[tuple[float, int]] = []
    spawned = False
    while child.poll() is None:
        rss, has_children = _observe_child(child.pid)
        samples.append((time.monotonic(), rss))
        spawned = spawned or has_children
        peak_rss = max(rss for _, rss in samples)
        # two kill conditions: the child's own RSS (exact, but blind to worker processes) and the
        # machine-level drop (sees any process tree, but macOS memory compression can mask it —
        # observed absorbing a 40 GB single-process matrix without the available-memory drop ever
        # reaching the cap)
        if rss > grid.MEMORY_CAP_BYTES or baseline_bytes - _available_bytes() > grid.MEMORY_CAP_BYTES:
            child.kill()
            child.wait()
            return _Supervision(REASON_MEMORY, peak_rss, spawned, _is_settled(samples))
        if time.monotonic() > deadline:
            child.kill()
            child.wait()
            return _Supervision(REASON_TIMEOUT, peak_rss, spawned, _is_settled(samples))
        time.sleep(_POLL_SEC)
    peak_rss = max((rss for _, rss in samples), default=0)
    return _Supervision(None, peak_rss, spawned, settled=True)


def _is_settled(samples: list[tuple[float, int]]) -> bool:
    """Return whether the sampled footprint had stopped growing over the final window."""
    if not samples:
        return False
    t_end = samples[-1][0]
    before = [rss for t, rss in samples if t <= t_end - _SETTLE_WINDOW_SEC]
    if not before:
        return False  # the run was too short to judge
    peak_total = max(rss for _, rss in samples)
    return peak_total <= max(before) * (1 + _SETTLE_TOLERANCE)


def _observe_child(pid: int) -> tuple[int, bool]:
    """Return the child's current RSS and whether it has live child processes of its own.

    A failed observation returns zeros rather than failing the supervisor: the child may exit
    between the liveness poll and this read.
    """
    try:
        process = psutil.Process(pid)
        return int(process.memory_info().rss), bool(process.children())
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0, False


def _available_bytes() -> int:
    """Return the machine's currently available memory."""
    return int(psutil.virtual_memory().available)


def _save_problem_vectors(n: int, path: Path) -> None:
    """Build the reference problem in the parent and save its float32 vectors for the child.

    Building here rather than in the child keeps the problem-generation transients out of the
    measured memory — generation happens before the baseline is sampled, and the child loads
    only the persistent input array.
    """
    import numpy as np

    from benchmarks.common.problems import build_problem
    from max_div.metrics import DiversityMetric

    problem = build_problem("U1", n=n, diversity_metric=DiversityMetric.MIN_SEPARATION)
    np.save(path, np.ascontiguousarray(problem.vectors))
