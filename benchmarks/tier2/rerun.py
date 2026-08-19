"""Re-measure max-div for the tier-2 comparison, keeping competitor values fixed.

Run with: ``uv run --group benchmarks python -m benchmarks.tier2.rerun``.
Runs only max-div's budget seriess — same problems, budgets, and seeds as
``benchmarks.tier2.full`` — and writes them into ``reports/benchmarks/tier2/``. The
competitor side of the report comes from the tracked reference records in
``benchmarks/tier2/data/``: those are deterministic single-shot values, so re-running
them would add noise and no information. Expect roughly two hours of sequential compute
on a quiet machine.
"""

from benchmarks.tier2.full import run_maxdiv_constrained, run_maxdiv_unconstrained


def main() -> None:
    """Re-run max-div's tier-2 ladders under the published protocol."""
    print("tier-2 re-measurement: max-div unconstrained ...", flush=True)
    run_maxdiv_unconstrained()
    print("tier-2 re-measurement: max-div constrained ...", flush=True)
    run_maxdiv_constrained()
    print("tier-2 re-measurement complete", flush=True)


if __name__ == "__main__":
    main()
