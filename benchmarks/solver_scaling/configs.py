"""The per-solver run configurations of the solver-scaling benchmarks — the smoke subset.

Each solver enters the measurements as one or more named configurations, mirroring the
published solver-configurations page; every configuration is measured independently and the
best result per (axis, n) is taken. This module carries only the smoke set:

* max-div's three configurations — `lean`, `optimal-eager`, `optimal-lazy` — spanning init-only
  linear memory, a forced full matrix (quadratic), and forced lazy storage (linear anytime);
* RDKit MaxMinPicker `default` — a one-shot picker;
* DPPy `default` — a sampler whose limit is expressiveness rather than resources.

Together they exercise linear vs quadratic memory growth, anytime vs one-shot solving, and a
non-resource scaling failure — enough to validate the harness and design the results pages.

`select` runs a configuration for a problem, seed and time budget, returning the selected
indices; the runner enforces the budget with a hard kill, and a configuration that accepts a
budget honors it itself. max-div's imports happen inside `select`, not at module load, so a run
of another solver never pays them.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from benchmarks.adapters.base import SelectionAdapter
from max_div.problem import VectorMaxDivProblem

from .grid import SELF_LIMIT_MARGIN_SEC

SelectFn = Callable[[VectorMaxDivProblem, int, float], NDArray[np.int64]]

_QUALITY_WORKERS = 12  # matches the reference machine's performance-core count


@dataclass(frozen=True)
class ScalingConfig:
    """One named configuration of one solver: how to run it, and whether seeds vary its result.

    `tool` is the registry key (`data/solver_registry.yaml`); `name` is the configuration name as
    the solver-configurations page lists it; `description` is a terse echo of that page's wording.
    `stochastic` records whether repeated runs under different seeds can differ; the
    measurement protocol decides when seeds are enumerated.
    """

    tool: str
    name: str
    description: str
    select: SelectFn
    stochastic: bool


# ==================================================================================================
#  select() builders
# ==================================================================================================
def _maxdiv_lean() -> SelectFn:
    """Build max-div's `lean` selector: uniform random one-shot init, no optimization, lazy storage."""

    def select(problem: VectorMaxDivProblem, seed: int, budget_sec: float) -> NDArray[np.int64]:
        from max_div.solver import DistanceStorage, InitializationStrategy, MaxDivSolverBuilder, Verbosity

        # No with_preset: the pipeline is a single init step, so the solve returns that
        # initialization and nothing more. Uniform sampling replaces the default
        # contribution-weighted variant, whose dataset-wide distance sweep would dominate this
        # configuration's runtime.
        builder = (
            MaxDivSolverBuilder(problem)
            .with_seed(seed)
            .with_distance_storage(DistanceStorage.LAZY)
            .set_initialization_strategy(InitializationStrategy.random_one_shot(uniform=True))
        )
        return np.asarray(builder.build().solve(verbosity=Verbosity.SILENT).i_selected, dtype=np.int64)

    return select


def _maxdiv_optimal(lazy: bool) -> SelectFn:
    """Build max-div's `optimal-*` selector: SMART, one cooperative worker group, forced storage, e2e budget.

    The budget handed to the solver is `SELF_LIMIT_MARGIN_SEC` under the run budget, so the real
    end-to-end time — which overshoots by up to one optimization batch — still lands within the
    time budget the run is judged against.
    """

    def select(problem: VectorMaxDivProblem, seed: int, budget_sec: float) -> NDArray[np.int64]:
        from max_div.solver import (
            DistanceStorage,
            ParallelMaxDivSolverBuilder,
            SolverPreset,
            Verbosity,
            WorkerConfig,
            seconds,
        )

        storage = DistanceStorage.LAZY if lazy else DistanceStorage.FULL_MATRIX
        budget = seconds(max(budget_sec - SELF_LIMIT_MARGIN_SEC, 0.1))
        group = [WorkerConfig(SolverPreset.SMART) for _ in range(_QUALITY_WORKERS)]
        builder = (
            ParallelMaxDivSolverBuilder(problem)
            .with_seed(seed)
            .with_distance_storage(storage)
            .with_workers(budget, workers=[group], end_to_end_budget=True)  # one cooperative group
        )
        return np.asarray(builder.build().solve(verbosity=Verbosity.SILENT).i_selected, dtype=np.int64)

    return select


def _adapter_select(make_adapter: Callable) -> SelectFn:
    """Build a selector around a single-shot adapter; the runner's kill enforces its budget."""

    def select(problem: VectorMaxDivProblem, seed: int, budget_sec: float) -> NDArray[np.int64]:
        return np.asarray(make_adapter().select(problem, seed), dtype=np.int64)

    return select


def _rdkit() -> SelectionAdapter:
    """Instantiate the RDKit MaxMinPicker adapter."""
    from benchmarks.adapters.rdkit_maxmin import RdkitMaxMin

    return RdkitMaxMin()


def _dppy() -> SelectionAdapter:
    """Instantiate the DPPy k-DPP adapter."""
    from benchmarks.adapters.dppy_kdpp import DppyKDpp

    return DppyKDpp()


def _test_sleep_select() -> SelectFn:
    """Build a selector that sleeps far past any budget, so the kill path is testable without a slow solver."""

    def select(problem: VectorMaxDivProblem, seed: int, budget_sec: float) -> NDArray[np.int64]:
        import time

        time.sleep(budget_sec + 3600)
        raise AssertionError("unreachable: the runner must have killed this")

    return select


# ==================================================================================================
#  The registry
# ==================================================================================================
CONFIGS: tuple[ScalingConfig, ...] = (
    ScalingConfig(
        "max-div",
        "lean",
        "uniform random one-shot initialization only, no optimization step, lazy distance storage, 1 worker",
        _maxdiv_lean(),
        stochastic=True,
    ),
    ScalingConfig(
        "max-div",
        "optimal-eager",
        "SMART preset, full end-to-end time budget, full-matrix distance storage forced, 12 cooperative workers",
        _maxdiv_optimal(lazy=False),
        stochastic=True,
    ),
    ScalingConfig(
        "max-div",
        "optimal-lazy",
        "SMART preset, full end-to-end time budget, lazy distance storage forced, 12 cooperative workers",
        _maxdiv_optimal(lazy=True),
        stochastic=True,
    ),
    ScalingConfig(
        "rdkit",
        "default",
        "MaxMinPicker with a Euclidean distance callable (its only mode)",
        _adapter_select(_rdkit),
        stochastic=False,
    ),
    ScalingConfig(
        "dppy",
        "default",
        "one exact k-DPP sample over an RBF likelihood kernel, median-pairwise-distance bandwidth",
        _adapter_select(_dppy),
        stochastic=True,
    ),
)

# The kill-path test fixture sits outside CONFIGS: it must never look like a registry
# configuration, but the runner's kill test needs a child that outlives any budget.
TEST_SLEEP_CONFIG = ScalingConfig(
    "_test_sleep", "sleep", "test fixture that sleeps past any budget", _test_sleep_select(), stochastic=False
)


def resolve(tool: str, name: str) -> ScalingConfig:
    """Return the configuration for one solver by name; `_test_sleep` resolves to the kill-path fixture."""
    if tool == TEST_SLEEP_CONFIG.tool:
        return TEST_SLEEP_CONFIG
    for config in CONFIGS:
        if config.tool == tool and config.name == name:
            return config
    raise KeyError(f"no configuration {name!r} for tool {tool!r}")


def configs_for(tool: str) -> list[ScalingConfig]:
    """Return every configuration of one solver, in declaration order."""
    return [config for config in CONFIGS if config.tool == tool]
