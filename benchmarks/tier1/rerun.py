"""Re-measure max-div for the tier-1 comparison, keeping exact-solver references fixed.

Run with: ``uv run --group benchmarks python -m benchmarks.tier1.rerun``.
Runs only max-div's budget series — same problems, budgets, and seeds as
``benchmarks.tier1.full`` — and writes them into ``reports/benchmarks/tier1/``. The
exact-solver side of the report comes from the tracked reference records in
``benchmarks/tier1/data/``: proven optima and long-cap incumbents are properties of the
problems, so re-solving them would re-spend hours to reproduce known values. Expect
roughly 15 minutes of sequential compute on a quiet machine.
"""

from benchmarks.tier1.full import run_incumbent_maxdiv, run_maxmin_maxdiv


def main() -> None:
    """Re-run max-div's tier-1 budget series under the published protocol."""
    print("tier-1 re-measurement: max-div on the proven-optimum problems ...", flush=True)
    run_maxmin_maxdiv()
    print("tier-1 re-measurement: max-div on the incumbent-panel problems ...", flush=True)
    run_incumbent_maxdiv()
    print("tier-1 re-measurement complete", flush=True)


if __name__ == "__main__":
    main()
