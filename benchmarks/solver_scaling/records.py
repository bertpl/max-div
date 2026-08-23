"""Scaling run records: one row per campaign run, with JSONL persistence."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class ScalingRunRecord:
    """One campaign run: a solver in one configuration at one size under one seed.

    `measured_sec` is the child-reported wall-clock of the solver call alone — problem
    construction, imports and scoring are excluded. `peak_memory_bytes` is the solver process's
    peak RSS; `memory_settled` is whether that footprint had stopped growing when the run ended
    (always true for a run that ended on its own); `spawned_processes` records whether the
    solver was ever observed with live child processes, which excludes it from the memory sweep
    (see the runner). A killed or failed run carries `completed=False` and names its `reason`
    (`timeout`, `memory`, or the error text); `outcome.classify` maps those two fields to the
    run's `Outcome`. A run's other measured fields hold whatever was known when it ended (None
    when nothing was).
    """

    tool: str
    config: str
    n: int
    k: int
    seed: int
    budget_sec: float
    completed: bool
    reason: str | None
    measured_sec: float | None
    peak_memory_bytes: int | None
    min_separation: float | None
    spawned_processes: bool = False
    memory_settled: bool = True


def save_scaling_records(records: list[ScalingRunRecord], path: Path) -> None:
    """Write records as JSONL, one per line, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(asdict(record)) + "\n")


def append_scaling_record(record: ScalingRunRecord, path: Path) -> None:
    """Append one record — the stage drivers persist after every run, so a kill loses nothing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(asdict(record)) + "\n")


def load_scaling_records(path: Path) -> list[ScalingRunRecord]:
    """Read records from a JSONL file written by `save_scaling_records` or `append_scaling_record`."""
    with path.open() as f:
        return [ScalingRunRecord(**json.loads(line)) for line in f if line.strip()]
