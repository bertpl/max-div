"""Child-process entry point: execute one campaign run and report its result as JSON.

The run executes in its own process so the parent can kill it — a tool stuck inside compiled code
cannot be interrupted in-process — and so its peak memory is its own, not the campaign
driver's. Problem construction happens before the clock starts and scoring after it
stops: only the tool call itself is timed.

Usage: python -m benchmarks.ceilings.run_one '<spec json>' <result path>
"""

import json
import resource
import sys
import time
from pathlib import Path

import numpy as np


def execute(spec: dict) -> dict:
    """Run the spec's tool once and return the result fields the parent records."""
    from benchmarks.ceilings.configs import Mode, resolve
    from benchmarks.common.problems import build_problem
    from benchmarks.common.quality import evaluate_selection
    from max_div.metrics import DiversityMetric

    problem = build_problem("U1", n=spec["n"], diversity_metric=DiversityMetric.MIN_SEPARATION)
    config = resolve(spec["tool"], Mode(spec["mode"]))

    # Warm-up on a throwaway tiny problem, before the clock: it forces the tool's library
    # imports (and any compilation caches), which are process setup rather than solving
    # cost. max-div's own import happens untimed above, so without this the competitors
    # would pay their imports inside the timed section while max-div never does.
    if spec["tool"] != "_test_sleep":
        try:
            warmup = build_problem("U1", n=20, diversity_metric=DiversityMetric.MIN_SEPARATION)
            config.select(warmup, 0, 5.0)
        except Exception:  # noqa: BLE001, S110 -- a failed warm-up must not fail the run it serves
            pass

    t0 = time.perf_counter()
    selection = config.select(problem, spec["seed"], spec["budget_sec"])
    measured_sec = time.perf_counter() - t0

    selection = np.asarray(selection, dtype=np.int64)
    if len(selection) != problem.k or len(np.unique(selection)) != problem.k:
        raise ValueError(f"invalid selection: {len(selection)} indices for k={problem.k}")
    quality = evaluate_selection(problem, selection)
    return {
        "completed": True,
        "measured_sec": measured_sec,
        "min_separation": quality[DiversityMetric.MIN_SEPARATION.name],
    }


def main() -> int:
    """Parse the spec, execute the run, and write the result JSON for the parent."""
    spec = json.loads(sys.argv[1])
    result_path = Path(sys.argv[2])
    try:
        result = execute(spec)
    except Exception as error:  # noqa: BLE001 -- every failure must reach the parent as data
        result = {"completed": False, "reason": f"{type(error).__name__}: {error}"}
    # ru_maxrss is bytes on macOS and kilobytes on Linux; the campaign machine is a Mac,
    # and the parent's live polling is the enforcement path either way.
    result["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result_path.write_text(json.dumps(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
