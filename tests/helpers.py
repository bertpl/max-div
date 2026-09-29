"""Helpers shared across the test suite."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numba

from max_div._core.benchmark_problems import BenchmarkProblemFactory
from max_div._core.metrics import (
    DiversityObjectiveHybrid,
    DiversityObjectiveSimple,
    HybridAggregationBase,
    HybridAggregationGeometricMean,
)


def swept_benchmark_problems() -> list[str]:
    """The benchmark problems a solver-level test should sweep in the current mode.

    All of them when compiled. With the JIT disabled — where every kernel runs as plain Python
    and a solver sweep costs roughly twenty times more — one constrained and one unconstrained,
    which is the distinction the solver branches on; the rest of a problem's identity is data.

    Narrowed rather than skipped: skipping these suites leaves the swap strategies' inner
    branches and the CLI solve path uncovered, which is what the interpreted run measures.
    """
    names = list(BenchmarkProblemFactory.get_all_benchmark_names())
    if not numba.config.DISABLE_JIT:
        return names
    return [_first_with_prefix(names, "C"), _first_with_prefix(names, "U")]


def _first_with_prefix(names: list[str], prefix: str) -> str:
    """Pick a representative by family, so reordering the problems cannot collapse the narrowed
    sweep into two of the same kind."""
    return next(name for name in names if name.startswith(prefix))


def hybrid_objective(
    *terms: DiversityObjectiveSimple, aggregation_type: type[HybridAggregationBase] = HybridAggregationGeometricMean
) -> DiversityObjectiveHybrid:
    """Build a `DiversityObjectiveHybrid` from simple objectives at equal weights, geometric-mean by default."""
    return DiversityObjectiveHybrid(terms, aggregation_type.with_unit_weights(len(terms)))


def load_script(name: str, path: Path) -> ModuleType:
    """Import a script from `scripts/` by path, because `scripts/` is maintainer tooling, not an importable package.

    The module is registered in `sys.modules` under `name` before it runs, so a script that imports
    another script by that name finds it.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
