"""Best-known-solution stage: every configuration ascends the size grid at the extended budget.

The runs feed the best-known reference pool and publish no size limit, so the stopping rule is
more lenient than the time stage's: a completed run never ends a configuration's series, however
long it took. Only a kill (the extended budget plus the setup grace, or the memory cap) or a
crash ends it. The measurement protocol's `Q_extended` section carries the rationale.

Sizes run up to the largest n any configuration reached within the time budget: beyond that size
no quality verdict exists that would need a reference.

Records are appended after every run to the tracked data file, so an interrupted stage resumes by
rerunning: already-recorded runs are skipped.

Usage: python -m benchmarks.solver_scaling.best_known_stage [tool | tool/config ...]  # default: all
"""

import sys
from pathlib import Path

from .configs import CONFIGS, ScalingConfig
from .grid import DEFAULT_SEED, EXTENDED_BUDGET_SEC, GRID_MIN, WARMUP_BUDGET_SEC, size_grid
from .outcome import Outcome, classify
from .records import ScalingRunRecord, append_scaling_record, load_scaling_records
from .runner import run_measurement
from .time_stage import DATA_PATH as TIME_DATA_PATH
from .time_stage import passes_time

DATA_PATH = Path(__file__).resolve().parent / "data" / "best_known_stage.jsonl"


def size_bound_from_time_stage(time_data_path: Path = TIME_DATA_PATH) -> int:
    """Return the largest n any configuration passed within the time budget.

    Raises:
        FileNotFoundError: If the time stage has not run yet — the extended stage needs its
            result to bound the grid.
    """
    records = load_scaling_records(time_data_path)
    return max(r.n for r in records if passes_time(r))


def best_known_by_size(records: list[ScalingRunRecord]) -> dict[int, ScalingRunRecord]:
    """Return, per size, the completed run with the best quality (highest minimum separation)."""
    best: dict[int, ScalingRunRecord] = {}
    for record in records:
        if not record.completed or record.min_separation is None:
            continue
        incumbent = best.get(record.n)
        if incumbent is None or record.min_separation > incumbent.min_separation:
            best[record.n] = record
    return dict(sorted(best.items()))


def run_best_known_stage(
    configs: list[ScalingConfig] | None = None,
    data_path: Path = DATA_PATH,
    budget_sec: float = EXTENDED_BUDGET_SEC,
    n_bound: int | None = None,
) -> dict[str, int | None]:
    """Run the stage for the given configurations and return each one's largest completed size.

    Returns:
        `tool/config` -> largest size with a completed run, or None when no size completed.
    """
    configs = list(CONFIGS) if configs is None else configs
    n_bound = size_bound_from_time_stage() if n_bound is None else n_bound
    done = {(r.tool, r.config, r.n, r.seed): r for r in (load_scaling_records(data_path) if data_path.exists() else [])}
    reached: dict[str, int | None] = {}
    for config in configs:
        reached[f"{config.tool}/{config.name}"] = _ascend(config, done, data_path, n_bound, budget_sec)
        print(f"{config.tool}/{config.name}: largest completed size = {reached[f'{config.tool}/{config.name}']}")
    return reached


def _ascend(config: ScalingConfig, done: dict, data_path: Path, n_bound: int, budget_sec: float) -> int | None:
    """Run one configuration up the grid; return its largest completed size.

    A configuration with no recorded runs first gets one discarded warm-up run at the smallest
    grid size (see `WARMUP_BUDGET_SEC`).
    """
    if not any(key[0] == config.tool and key[1] == config.name for key in done):
        run_measurement(config.tool, config.name, GRID_MIN, GRID_MIN // 10, DEFAULT_SEED, WARMUP_BUDGET_SEC)
    largest: int | None = None
    for n in size_grid(n_bound):
        record = done.get((config.tool, config.name, n, DEFAULT_SEED))
        if record is None:
            record = run_measurement(config.tool, config.name, n, n // 10, DEFAULT_SEED, budget_sec)
            append_scaling_record(record, data_path)
        print(f"  {config.tool}/{config.name} n={n}: {'ok' if record.completed else record.reason}")
        if record.completed:
            largest = n
            continue
        # A non-resource failure before any completed size is a small-size degeneracy, not a
        # limit — skip it and try the next size (as in the time stage). Any kill, or any failure
        # once a size has completed, ends the series.
        if largest is None and classify(record.completed, record.reason) is Outcome.SCALING_FAILURE:
            continue
        return largest
    return largest


if __name__ == "__main__":
    selected = sys.argv[1:]
    chosen = [c for c in CONFIGS if c.tool in selected or f"{c.tool}/{c.name}" in selected] if selected else list(CONFIGS)
    run_best_known_stage(chosen)
