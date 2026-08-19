"""Ceiling run records: one row per campaign run, with JSONL persistence."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class CeilingRunRecord:
    """One campaign run: a tool at one size in one mode under one seed.

    ``measured_sec`` is the child-reported wall-clock of the tool call alone — problem
    construction, imports and scoring are not the tool's cost. A killed or failed run
    carries ``completed=False`` and names its reason (``timeout``, ``memory``, or the
    error message); its other measured fields hold whatever was known at the kill.
    """

    tool: str
    mode: str
    n: int
    k: int
    seed: int
    budget_sec: float
    completed: bool
    reason: str | None
    measured_sec: float | None
    peak_rss_bytes: int | None
    min_separation: float | None


def save_ceiling_records(records: list[CeilingRunRecord], path: Path) -> None:
    """Write records as JSONL, one per line, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(asdict(record)) + "\n")


def append_ceiling_record(record: CeilingRunRecord, path: Path) -> None:
    """Append one record — the stage drivers persist after every run, so a kill loses nothing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(asdict(record)) + "\n")


def load_ceiling_records(path: Path) -> list[CeilingRunRecord]:
    """Read records from a JSONL file written by the savers above."""
    with path.open() as f:
        return [CeilingRunRecord(**json.loads(line)) for line in f if line.strip()]
