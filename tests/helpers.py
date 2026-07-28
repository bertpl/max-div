"""Helpers shared across the test suite.

The suite runs in two modes, and one of them is far more expensive per test. With
`NUMBA_DISABLE_JIT=1` every `@njit` kernel runs as plain Python — the only way `coverage.py`
can see inside them, and the mode CI measures coverage in. The same work then costs roughly
twenty times what the compiled path does.

That cost is not spread evenly. It concentrates in the suites that sweep a real solver run
across every benchmark problem, which is the axis narrowed below.
"""

import numba

from max_div._core.benchmark_problems import BenchmarkProblemFactory


def swept_benchmark_problems() -> list[str]:
    """The benchmark problems a solver-level test should sweep in the current mode.

    Compiled, that is all of them: the sweep is cheap and the breadth is the point. Interpreted,
    it narrows to one constrained and one unconstrained problem — the distinction the solver
    actually branches on, since the rest of a problem's identity is its data rather than a code
    path through the solver.

    Deliberately not a skip. Skipping these suites outright leaves the swap strategies' inner
    branches and the CLI's solve path uncovered, which is exactly what the interpreted run
    exists to measure; narrowing the sweep keeps every one of those lines while dropping the
    repetitions that only re-walk them.

    Keys on `numba.config.DISABLE_JIT` — numba's own parse of the environment variable — so the
    decision always matches what numba is actually doing.
    """
    names = list(BenchmarkProblemFactory.get_all_benchmark_names())
    if not numba.config.DISABLE_JIT:
        return names
    return [_first_with_prefix(names, "C"), _first_with_prefix(names, "U")]


def _first_with_prefix(names: list[str], prefix: str) -> str:
    """Pick a representative by family, so renaming or reordering the problems cannot silently
    turn the narrowed sweep into two problems of the same kind."""
    return next(name for name in names if name.startswith(prefix))
