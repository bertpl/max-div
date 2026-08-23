"""Child-process entry point: execute one run and report its result as JSON.

The run executes in its own process so the parent can kill it — a solver stuck inside compiled
code cannot be interrupted in-process — and so its peak memory is its own, not the driver's. The
parent builds the problem and hands the child only the final float32 vectors (`vectors_path`), so
the child's peak RSS carries the persistent input and the solver's own allocations but not the
problem-generation transients. The clock runs over the solver call alone: loading the vectors and
scoring the result are outside it.

Usage: python -m benchmarks.solver_scaling.run_one '<spec json>' <result path>
"""

import json
import resource
import sys
import time
from pathlib import Path

import numpy as np


def execute(spec: dict) -> dict:
    """Run the spec's configuration once and return the result fields the parent records."""
    from benchmarks.common.quality import min_separation_nn
    from max_div.metrics import DiversityMetric
    from max_div.problem import MaxDivProblem

    from .configs import resolve

    vectors = np.load(spec["vectors_path"])
    problem = MaxDivProblem.new(vectors=vectors, k=spec["k"], diversity_metric=DiversityMetric.MIN_SEPARATION)
    config = resolve(spec["tool"], spec["config"])

    # Warm-up on a tiny slice of the same vectors, before the clock: it forces the solver's library
    # imports (and any compilation caches), which are process setup rather than solving cost. A
    # slice reuses the loaded array, so the warm-up adds no generation cost. max-div's own import
    # happens untimed inside its select; without this warm-up the other solvers would pay their
    # imports inside the timed section while max-div never does.
    if spec["tool"] != "_test_sleep":
        try:
            warmup = MaxDivProblem.new(vectors=vectors[:20], k=2, diversity_metric=DiversityMetric.MIN_SEPARATION)
            config.select(warmup, 0, 5.0)
        except Exception:  # noqa: BLE001, S110 -- a failed warm-up must not fail the run it serves
            pass

    t0 = time.perf_counter()
    selection = config.select(problem, spec["seed"], spec["budget_sec"])
    measured_sec = time.perf_counter() - t0

    selection = np.asarray(selection, dtype=np.int64)
    if len(selection) != problem.k or len(np.unique(selection)) != problem.k:
        raise ValueError(f"invalid selection: {len(selection)} indices for k={problem.k}")
    # Scored via nearest neighbors rather than `evaluate_selection`: the k x k matrix the latter
    # builds would dominate the campaign's memory at the largest sizes.
    return {
        "completed": True,
        "measured_sec": measured_sec,
        "min_separation": min_separation_nn(problem.vectors, selection),
    }


def main() -> int:
    """Parse the spec, execute the run, and write the result JSON for the parent."""
    spec = json.loads(sys.argv[1])
    result_path = Path(sys.argv[2])
    try:
        result = execute(spec)
    except Exception as error:  # noqa: BLE001 -- every failure must reach the parent as data
        result = {"completed": False, "reason": f"{type(error).__name__}: {error}"}
    # ru_maxrss is bytes on macOS and kilobytes on Linux; the campaign machine is a Mac, and the
    # parent's live polling is the enforcement path either way.
    result["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result_path.write_text(json.dumps(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
