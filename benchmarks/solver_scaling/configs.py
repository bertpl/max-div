"""The per-solver run configurations of the solver-scaling benchmarks.

Each solver enters the measurements as one or more named configurations, mirroring the published
solver-configurations page; every configuration is measured independently and the best result per
(axis, n) is taken. `tool` is the registry key (`data/solver_registry.yaml`).

`select` runs a configuration for a problem, seed and time budget, returning the selected indices;
the runner enforces the budget with a hard kill, and a configuration that accepts a budget honors
it itself. Every solver's imports happen inside `select`, not at module load, so a run of one
solver never pays another's.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from benchmarks.adapters.base import SelectionAdapter
from max_div.problem import VectorMaxDivProblem

from .grid import self_limit_margin_sec

SelectFn = Callable[[VectorMaxDivProblem, int, float], NDArray[np.int64]]

_QUALITY_WORKERS = 12  # matches the reference machine's performance-core count


@dataclass(frozen=True)
class ScalingConfig:
    """One named configuration of one solver: how to run it, and whether seeds vary its result.

    `tool` is the registry key (`data/solver_registry.yaml`); `name` is the configuration name as
    the solver-configurations page lists it; `description` is a terse echo of that page's wording.
    `seed_varies_result` records whether the seed can influence the returned selection — through
    a literal seed parameter or an adapter-chosen start item; the measurement protocol decides
    when seeds are enumerated. The flag deliberately does not cover a tool with unseedable
    internal randomness (runs differ, no seed reaches them) — no registry tool behaves that way.
    """

    tool: str
    name: str
    description: str
    select: SelectFn
    seed_varies_result: bool


def _exact_deadline(budget_sec: float) -> float:
    """Return the `time.monotonic()` deadline an exact solver must finish by: now + budget − margin.

    The margin under the run budget covers what a solver's internal time limit cannot: result
    extraction after the solver's clock stops, and the limit's own imprecision.
    """
    return time.monotonic() + max(budget_sec - self_limit_margin_sec(budget_sec), 0.1)


# ==================================================================================================
#  select() builders — max-div
# ==================================================================================================
def _maxdiv_lean() -> SelectFn:
    """Build max-div's `lean` selector: uniform random one-shot init, no optimization, lazy storage."""

    def select(problem: VectorMaxDivProblem, seed: int, budget_sec: float) -> NDArray[np.int64]:
        from max_div.solver import DistanceStorage, InitializationStrategy, MaxDivSolverBuilder, Verbosity

        # No with_preset: the pipeline is a single init step, so the solve returns that
        # initialization and nothing more. Uniform sampling replaces the default
        # contribution-weighted variant, whose dataset-wide distance sweep would dominate this
        # configuration's runtime. The parallel batched tracker update is safe here: this
        # configuration runs a single process, so no worker competes for the CPU cores.
        builder = (
            MaxDivSolverBuilder(problem)
            .with_seed(seed)
            .with_distance_storage(DistanceStorage.LAZY)
            .set_initialization_strategy(InitializationStrategy.random_one_shot(uniform=True, parallel=True))
        )
        return np.asarray(builder.build().solve(verbosity=Verbosity.SILENT).i_selected, dtype=np.int64)

    return select


def _maxdiv_optimal(lazy: bool) -> SelectFn:
    """Build max-div's `optimal-*` selector: SMART workers, dynamic grouping, forced storage, e2e budget.

    The budget handed to the solver is the self-limit margin under the run budget, so the real
    end-to-end time — which overshoots by up to one optimization batch — still lands within the
    time budget the run is judged against.
    """

    def select(problem: VectorMaxDivProblem, seed: int, budget_sec: float) -> NDArray[np.int64]:
        from max_div.solver import (
            DistanceStorage,
            ParallelMaxDivSolverBuilder,
            Verbosity,
            seconds,
        )

        storage = DistanceStorage.LAZY if lazy else DistanceStorage.FULL_MATRIX
        budget = seconds(max(budget_sec - self_limit_margin_sec(budget_sec), 0.1))
        builder = (
            ParallelMaxDivSolverBuilder(problem)
            .with_seed(seed)
            .with_distance_storage(storage)
            .with_workers(budget, _QUALITY_WORKERS)  # dynamic worker groups
            .with_end_to_end_budget()
        )
        return np.asarray(builder.build().solve(verbosity=Verbosity.SILENT).i_selected, dtype=np.int64)

    return select


# ==================================================================================================
#  select() builders — exact solvers
# ==================================================================================================
def _cpsat_select(first_feasible: bool, num_workers: int) -> SelectFn:
    """Build a CP-SAT max-min selector; `first_feasible` stops at the first solution."""

    def select(problem: VectorMaxDivProblem, seed: int, budget_sec: float) -> NDArray[np.int64]:
        from benchmarks.exact.mip_maxmin import solve_maxmin_cpsat_selection

        return solve_maxmin_cpsat_selection(
            problem, _exact_deadline(budget_sec), first_feasible=first_feasible, seed=seed, num_workers=num_workers
        )

    return select


def _scip_select(first_feasible: bool) -> SelectFn:
    """Build a SCIP big-M max-min MIP selector; `first_feasible` stops at the first solution."""

    def select(problem: VectorMaxDivProblem, seed: int, budget_sec: float) -> NDArray[np.int64]:
        from benchmarks.exact.mip_maxmin import solve_maxmin_scip_selection

        return solve_maxmin_scip_selection(
            problem, _exact_deadline(budget_sec), first_feasible=first_feasible, seed=seed
        )

    return select


def _highs_select(first_feasible: bool, num_workers: int) -> SelectFn:
    """Build a HiGHS big-M max-min MIP selector; `first_feasible` stops at the first improving solution."""

    def select(problem: VectorMaxDivProblem, seed: int, budget_sec: float) -> NDArray[np.int64]:
        from benchmarks.exact.mip_maxmin import solve_maxmin_highs_selection

        return solve_maxmin_highs_selection(
            problem, _exact_deadline(budget_sec), first_feasible=first_feasible, seed=seed, num_workers=num_workers
        )

    return select


# ==================================================================================================
#  select() builders — single-shot adapters
# ==================================================================================================
def _adapter_select(make_adapter: Callable[[], SelectionAdapter]) -> SelectFn:
    """Build a selector around a single-shot adapter; the runner's kill enforces its budget."""

    def select(problem: VectorMaxDivProblem, seed: int, budget_sec: float) -> NDArray[np.int64]:
        return np.asarray(make_adapter().select(problem, seed), dtype=np.int64)

    return select


def _rdkit() -> SelectionAdapter:
    """Instantiate the RDKit MaxMinPicker adapter."""
    from benchmarks.adapters.rdkit_maxmin import RdkitMaxMin

    return RdkitMaxMin()


def _fpsample(variant: str) -> Callable[[], SelectionAdapter]:
    """Return a factory for the fpsample adapter in the given variant (`vanilla` or `kdline`)."""
    from benchmarks.adapters.fps import FpsampleFPS

    return lambda: FpsampleFPS(variant=variant)


def _skmatter() -> SelectionAdapter:
    """Instantiate the skmatter FPS adapter."""
    from benchmarks.adapters.fps import SkmatterFPS

    return SkmatterFPS()


def _apricot() -> SelectionAdapter:
    """Instantiate the apricot facility-location adapter."""
    from benchmarks.adapters.apricot_fl import ApricotFacilityLocation

    return ApricotFacilityLocation()


def _qc_selector() -> SelectionAdapter:
    """Instantiate the qc-selector max-min adapter."""
    from benchmarks.adapters.qc_selector import QcSelectorMaxMin

    return QcSelectorMaxMin()


def _kmedoids() -> SelectionAdapter:
    """Instantiate the kmedoids FasterPAM adapter."""
    from benchmarks.adapters.kmedoids_fasterpam import KMedoidsFasterPAM

    return KMedoidsFasterPAM()


def _code_fdm() -> SelectionAdapter:
    """Instantiate the code-FDM single-color adapter (its unconstrained reduction)."""
    from benchmarks.adapters.code_fdm import CodeFdmSingleColor

    return CodeFdmSingleColor()


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
        seed_varies_result=True,
    ),
    ScalingConfig(
        "max-div",
        "optimal-eager",
        "SMART preset, full end-to-end time budget, full-matrix distance storage forced, 12 cooperative workers",
        _maxdiv_optimal(lazy=False),
        seed_varies_result=True,
    ),
    ScalingConfig(
        "max-div",
        "optimal-lazy",
        "SMART preset, full end-to-end time budget, lazy distance storage forced, 12 cooperative workers",
        _maxdiv_optimal(lazy=True),
        seed_varies_result=True,
    ),
    ScalingConfig(
        "ortools-cpsat",
        "feasible",
        "max-min CP-SAT model, stop at the first feasible solution, 1 worker",
        _cpsat_select(first_feasible=True, num_workers=1),
        seed_varies_result=True,
    ),
    ScalingConfig(
        "ortools-cpsat",
        "optimal",
        "max-min CP-SAT model, full time budget, 12 portfolio workers",
        _cpsat_select(first_feasible=False, num_workers=_QUALITY_WORKERS),
        seed_varies_result=True,
    ),
    ScalingConfig(
        "scip",
        "feasible",
        "big-M max-min MIP, stop at the first feasible solution",
        _scip_select(first_feasible=True),
        seed_varies_result=True,
    ),
    ScalingConfig(
        "scip",
        "optimal",
        "big-M max-min MIP, full time budget",
        _scip_select(first_feasible=False),
        seed_varies_result=True,
    ),
    ScalingConfig(
        "highs",
        "feasible",
        "big-M max-min MIP, stop at the first improving solution",
        _highs_select(first_feasible=True, num_workers=1),
        seed_varies_result=True,
    ),
    ScalingConfig(
        "highs",
        "optimal",
        "big-M max-min MIP, full time budget, parallel branch-and-bound",
        _highs_select(first_feasible=False, num_workers=_QUALITY_WORKERS),
        seed_varies_result=True,
    ),
    ScalingConfig(
        "rdkit",
        "default",
        "MaxMinPicker with a Euclidean distance callable (its only mode)",
        _adapter_select(_rdkit),
        seed_varies_result=True,
    ),
    ScalingConfig(
        "fpsample",
        "vanilla",
        "plain farthest-point sampling",
        _adapter_select(_fpsample("vanilla")),
        seed_varies_result=True,
    ),
    ScalingConfig(
        "fpsample",
        "kdline",
        "bucket KD-line farthest-point sampling — the tree-accelerated variant, well suited to d=2",
        _adapter_select(_fpsample("kdline")),
        seed_varies_result=True,
    ),
    ScalingConfig(
        "skmatter",
        "default",
        "FPS selector (its only mode)",
        _adapter_select(_skmatter),
        seed_varies_result=True,
    ),
    ScalingConfig(
        "apricot-select",
        "default",
        "facility-location selection, lazy greedy, RBF similarity matrix",
        _adapter_select(_apricot),
        seed_varies_result=False,
    ),
    ScalingConfig(
        "qc-selector",
        "maxmin",
        "max-min selection on a precomputed distance matrix",
        _adapter_select(_qc_selector),
        seed_varies_result=True,
    ),
    ScalingConfig(
        "dppy",
        "default",
        "one exact k-DPP sample over an RBF likelihood kernel, median-pairwise-distance bandwidth",
        _adapter_select(_dppy),
        seed_varies_result=True,
    ),
    ScalingConfig(
        "code-fdm",
        "default",
        "FairFlow with a single color spanning all items (its unconstrained reduction)",
        _adapter_select(_code_fdm),
        seed_varies_result=False,
    ),
    ScalingConfig(
        "kmedoids",
        "default",
        "FasterPAM k-medoids on a precomputed distance matrix, random initialization",
        _adapter_select(_kmedoids),
        seed_varies_result=True,
    ),
)

# The kill-path test fixture sits outside CONFIGS: it must never look like a registry
# configuration, but the runner's kill test needs a child that outlives any budget.
TEST_SLEEP_CONFIG = ScalingConfig(
    "_test_sleep", "sleep", "test fixture that sleeps past any budget", _test_sleep_select(), seed_varies_result=False
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
