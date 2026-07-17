"""Best-known MMDP objective values, vendored as tracked data.

The values are the per-instance best over the six algorithms in the companion results
workbook of Resende, Marti, Gallego & Duarte (2010) — the same source that publishes the
instances (see ``loader``). Vendoring pins exactly what the tier-3 results were compared
against. Later work has improved some of these bounds; the results page states this.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

_DATA_CSV = Path(__file__).parent / "data" / "mmdp_best_known.csv"


@dataclass(frozen=True)
class BestKnown:
    """One published best-known value: an (instance, k) pairing and its objective value."""

    family: str
    instance: str
    n: int
    k: int
    best_known: float


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
                )
            )
    return rows
