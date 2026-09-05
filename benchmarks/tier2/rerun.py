"""Re-measure max-div for the tier-2 comparison, keeping the competitor records fixed.

Run with: ``uv run --group benchmarks python -m benchmarks.tier2.rerun``.
Runs only max-div's budget series — same sizes, budgets and seeds as ``benchmarks.tier2.full`` —
and writes them into ``reports/benchmarks/tier2/``. The competitor side of the report comes from
the tracked reference records in ``benchmarks/tier2/data/``.
"""

from benchmarks.tier2.full import OUTPUT_DIR, run_maxdiv


def main() -> None:
    """Re-run max-div's tier-2 budget series under the published protocol."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("tier-2 re-measurement: max-div ...", flush=True)
    run_maxdiv()
    print("tier-2 re-measurement complete", flush=True)


if __name__ == "__main__":
    main()
