"""Run records: the flat result rows every runner emits, with JSONL persistence."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class RunRecord:
    """One benchmark measurement: a tool solving one problem once under one budget.

    ``measured_sec`` is always the observed wall-clock (as reported by the tool),
    never the nominal budget; figures plot this value.
    """

    tool: str
    problem: str
    size: int
    n: int
    k: int
    diversity_metric: str
    seed: int
    budget: str  # e.g. "time:0.004s", "iterations:1280", or "single-shot"
    measured_sec: float
    n_iterations: int | None
    quality: dict[str, float] = field(default_factory=dict)
    n_constraints: int = 0
    n_constraints_satisfied: int = 0
    proven_optimal: bool | None = None  # exact solvers only: optimality certified within the timeout


def save_records(records: list[RunRecord], path: Path) -> None:
    """Write records as JSONL (one record per line), creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for rec in records:
            f.write(json.dumps(asdict(rec)) + "\n")


def load_records(path: Path) -> list[RunRecord]:
    """Read records from a JSONL file written by save_records."""
    with path.open() as f:
        return [RunRecord(**json.loads(line)) for line in f if line.strip()]
