"""Largest-n-within-time stage: every tool ascends the size grid at the reference budget.

For each tool: run its fastest-valid configuration once per seed at each candidate size,
smallest first, and stop at the first size where the median seed fails. The largest
passing size is the tool's largest n within the time budget. Peak memory is recorded on every run — the
memory-model fits consume those records, so this stage doubles as memory calibration.

Records are appended after every run to the tracked data file at ``DATA_PATH``, so an
interrupted stage resumes by rerunning: already-recorded runs are skipped.

Usage: python -m benchmarks.tool_scaling.time_stage [tool ...]   # default: all registry tools
"""

import statistics
import sys
from pathlib import Path

from benchmarks.tool_scaling.configs import TOOLS, Mode, seeds_for
from benchmarks.tool_scaling.grid import REFERENCE_BUDGET_SEC, operational_bound, size_grid
from benchmarks.tool_scaling.records import append_scaling_record, load_scaling_records
from benchmarks.tool_scaling.runner import run_measurement

DATA_PATH = Path(__file__).resolve().parent / "data" / "time_stage.jsonl"


def run_time_stage(tools: list[str], data_path: Path = DATA_PATH) -> dict[str, int | None]:
    """Run the stage for the given tools and return each one's largest n within the time budget.

    Returns:
        Tool key -> largest passing size, or None when even the smallest grid size fails.
    """
    done = {
        (r.tool, r.n, r.seed): r
        for r in (load_scaling_records(data_path) if data_path.exists() else [])
        if r.mode == Mode.FASTEST_VALID.value
    }
    max_n_by_tool: dict[str, int | None] = {}
    for tool in tools:
        max_n_by_tool[tool] = _ascend(tool, done, data_path)
        print(f"{tool}: largest n within the time budget = {max_n_by_tool[tool]}")
    return max_n_by_tool


def _ascend(tool: str, done: dict, data_path: Path) -> int | None:
    """Run one tool up the grid; return its largest passing size."""
    limit: int | None = None
    for n in size_grid(operational_bound()):
        outcomes = []
        for seed in seeds_for(tool):
            record = done.get((tool, n, seed))
            if record is None:
                record = run_measurement(tool, Mode.FASTEST_VALID, n, n // 10, seed, REFERENCE_BUDGET_SEC)
                append_scaling_record(record, data_path)
            outcomes.append(record.completed)
            print(f"  {tool} n={n} seed={seed}: {'ok' if record.completed else record.reason}")
        if statistics.median(outcomes) < 1:  # the median seed failed
            return limit
        limit = n
    return limit


if __name__ == "__main__":
    run_time_stage(sys.argv[1:] or list(TOOLS))
