"""Load the best-known MMDP objective values, vendored as tracked data with their provenance.

Each row carries the columns the data file's header defines, sources included.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

_DATA_CSV = Path(__file__).parent / "data" / "mmdp_best_known.csv"


@dataclass(frozen=True)
class BestKnown:
    """Record one published best-known value: an (instance, k) pairing, its objective value, and where it comes from."""

    family: str
    instance: str
    n: int
    k: int
    best_known: float
    best_known_2010: float
    source: str
    proven_optimal: bool


def load_best_known() -> list[BestKnown]:
    """Read all vendored best-known rows (one per published (instance, k) pairing)."""
    rows = []
    with _DATA_CSV.open() as f:
        for record in csv.DictReader(line for line in f if not line.startswith("#")):
            rows.append(
                BestKnown(
                    family=record["family"],
                    instance=record["instance"],
                    n=int(record["n"]),
                    k=int(record["k"]),
                    best_known=float(record["best_known"]),
                    best_known_2010=float(record["best_known_2010"]),
                    source=record["source"],
                    proven_optimal=record["proven_optimal"] == "true",
                )
            )
    return rows
