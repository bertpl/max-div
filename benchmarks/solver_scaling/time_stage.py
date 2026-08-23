"""Largest-n-within-time stage: every configuration ascends the size grid at the reference budget.

For each configuration: run it once at each candidate size, smallest first, under the protocol's
fixed seed, and stop at the first size that fails. A run passes when it returns a valid selection
within the time budget, measured end-to-end. The largest passing size is the configuration's
largest n within the time budget; the tool's value is the best over its configurations. The
memory sweep (`memory_stage`) is independent and runs first.

One run decides each size's verdict (see the measurement protocol, IV.C.1).

Records are appended after every run to the tracked data file, so an interrupted stage resumes by
rerunning: already-recorded runs are skipped.

Usage: python -m benchmarks.solver_scaling.time_stage [tool | tool/config ...]   # default: all
"""

import sys
from pathlib import Path

from .configs import CONFIGS, ScalingConfig
from .grid import DEFAULT_SEED, GRID_MIN, REFERENCE_BUDGET_SEC, WARMUP_BUDGET_SEC, operational_bound, size_grid
from .records import ScalingRunRecord, append_scaling_record, load_scaling_records
from .runner import run_measurement

DATA_PATH = Path(__file__).resolve().parent / "data" / "time_stage.jsonl"


def passes_time(record: ScalingRunRecord, budget_sec: float = REFERENCE_BUDGET_SEC) -> bool:
    """Return whether a run counts toward the time limit: completed within the time budget.

    A completed run whose measured end-to-end time exceeds the budget does not pass — completing
    within the runner's setup grace is not the same as answering within the time budget.
    """
    return record.completed and record.measured_sec is not None and record.measured_sec <= budget_sec


def run_time_stage(
    configs: list[ScalingConfig] | None = None, data_path: Path = DATA_PATH, budget_sec: float = REFERENCE_BUDGET_SEC
) -> dict[str, int | None]:
    """Run the stage for the given configurations and return each one's largest n within the time budget.

    Returns:
        `tool/config` -> largest passing size, or None when even the smallest grid size fails.
    """
    configs = list(CONFIGS) if configs is None else configs
    done = {(r.tool, r.config, r.n, r.seed): r for r in (load_scaling_records(data_path) if data_path.exists() else [])}
    limits: dict[str, int | None] = {}
    for config in configs:
        limits[f"{config.tool}/{config.name}"] = _ascend(config, done, data_path, budget_sec)
        print(
            f"{config.tool}/{config.name}: largest n within the time budget = {limits[f'{config.tool}/{config.name}']}"
        )
    return limits


def _ascend(config: ScalingConfig, done: dict, data_path: Path, budget_sec: float) -> int | None:
    """Run one configuration up the grid; return its largest passing size.

    A configuration with no recorded runs first gets one discarded warm-up run at the smallest
    grid size (see `WARMUP_BUDGET_SEC`).
    """
    if not any(key[0] == config.tool and key[1] == config.name for key in done):
        run_measurement(config.tool, config.name, GRID_MIN, GRID_MIN // 10, DEFAULT_SEED, WARMUP_BUDGET_SEC)
    limit: int | None = None
    for n in size_grid(operational_bound()):
        record = done.get((config.tool, config.name, n, DEFAULT_SEED))
        if record is None:
            record = run_measurement(config.tool, config.name, n, n // 10, DEFAULT_SEED, budget_sec)
            append_scaling_record(record, data_path)
        passed = passes_time(record, budget_sec)
        print(f"  {config.tool}/{config.name} n={n}: {'ok' if passed else record.reason}")
        if not passed:
            return limit
        limit = n
    return limit


if __name__ == "__main__":
    selected = sys.argv[1:]
    chosen = [c for c in CONFIGS if c.tool in selected or f"{c.tool}/{c.name}" in selected] if selected else list(CONFIGS)
    run_time_stage(chosen)
