"""The published tier-3 run: max-div vs. the MDPLIB MMDP best-known values.

Run with: ``uv run --group benchmarks python -m benchmarks.tier3.full``.
Runs the budget ladder on every published (instance, k) pairing of the Glover/Geo/Ran MMDP
sets (fetched at run time, never redistributed) and writes JSONL records into
``reports/benchmarks/tier3/``; docs artifacts come from ``benchmarks.tier3.report``.
Expect roughly half an hour of sequential compute.
"""

from pathlib import Path

from benchmarks.common import save_records, time_ladder
from benchmarks.mdplib import load_instance
from benchmarks.mdplib.best_known import load_best_known
from benchmarks.runners import run_maxdiv_ladder
from max_div.metrics import DiversityMetric

OUTPUT_DIR = Path("reports/benchmarks/tier3")

SEEDS = (0, 1, 2)
METRIC = DiversityMetric.MIN_SEPARATION  # published MMDP values are max-min

# Every pairing gets the ladder to ~2 s — calibration showed max-div plateaus well within
# that on these instances. Only the largest instances (n = 500) get the extended rungs, so
# the deep-budget story is measured where a residual gap exists without ladder-solving 195
# pairings to 16 s each (~5 h of mostly-flat curves).
TIME_BUDGETS_SEC = time_ladder(0.001, 2.0)
EXTENDED_BUDGETS_SEC = [4.096, 8.192, 16.384]
EXTENDED_MIN_N = 500


def main() -> None:
    """Ladder every published (instance, k) pairing and persist the records."""
    rows = load_best_known()
    records = []
    for i, row in enumerate(rows, start=1):
        problem = load_instance(row.family, row.instance, k=row.k, diversity_metric=METRIC)
        budgets = TIME_BUDGETS_SEC + (EXTENDED_BUDGETS_SEC if row.n >= EXTENDED_MIN_N else [])
        # The instance file name + k pairing identifies the published row; ``size`` carries
        # k so records stay unique per pairing (Glover instances pair with 5 k values).
        records += run_maxdiv_ladder(
            problem,
            problem_name=row.instance,
            size=row.k,
            time_budgets_sec=budgets,
            seeds=SEEDS,
        )
        if i % 10 == 0:
            print(f"{i}/{len(rows)} pairings done", flush=True)
            save_records(records, OUTPUT_DIR / "records.jsonl")
    save_records(records, OUTPUT_DIR / "records.jsonl")
    print(f"tier-3 complete: {len(records)} records for {len(rows)} pairings", flush=True)


if __name__ == "__main__":
    main()
