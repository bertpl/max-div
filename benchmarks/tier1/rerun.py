"""Re-measure max-div for the tier-1 comparison, keeping the exact-solver references fixed.

Run with: ``uv run --group benchmarks python -m benchmarks.tier1.rerun``.
Runs only max-div's budget series — same cells, budgets and seeds as ``benchmarks.tier1.full`` —
and writes them into `OUTPUT_DIR`. The certified optima come from the tracked reference files
under `DATA_DIR`: they are properties of the problems, so
re-solving them would re-spend hours to reproduce known values.
"""

import json

from .full import DATA_DIR, EXACT_MAXMIN_FILE, EXACT_NN_FILE, OUTPUT_DIR, run_maxdiv


def main() -> None:
    """Re-run max-div's tier-1 budget series on the cells the tracked references certify."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    exact_rows = json.loads((DATA_DIR / EXACT_MAXMIN_FILE).read_text()) + json.loads((DATA_DIR / EXACT_NN_FILE).read_text())
    print("tier-1 re-measurement: max-div on the certified cells ...", flush=True)
    run_maxdiv(exact_rows)
    print("tier-1 re-measurement complete", flush=True)


if __name__ == "__main__":
    main()
