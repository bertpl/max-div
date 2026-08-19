"""The per-tool run configurations of the ceilings campaign.

Every tool in ``data/solver_registry.yaml`` has one entry here, with a configuration per
campaign mode. The three modes mirror the three published definitions:

* ``FASTEST_VALID`` — the fastest standard, user-configurable setting that still produces
  a valid selection; what the time ceiling is measured in.
* ``MEMORY_OPTIMAL`` — the most memory-efficient setting; what the memory ceiling is
  defined against. For most tools this is the same callable as ``FASTEST_VALID``.
* ``QUALITY`` — the tool's standard configuration; what the quality ceiling and the
  best-known-solution runs are measured in.

Each entry also declares the exponent of the tool's documented asymptotic memory growth,
which constrains the memory-model fit — a fit over a handful of measured sizes is only
defensible because the shape of the curve is known, not inferred.

A configuration's ``description`` is the pinned public wording; the definitions page's
per-tool configuration listing renders from here so prose and code cannot drift.
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from max_div.problem import VectorMaxDivProblem

SelectFn = Callable[[VectorMaxDivProblem, int, float], NDArray[np.int64]]


class Mode(Enum):
    """The three campaign run modes, one per published ceiling definition."""

    FASTEST_VALID = "fastest_valid"
    MEMORY_OPTIMAL = "memory_optimal"
    QUALITY = "quality"


@dataclass(frozen=True)
class ToolConfig:
    """One tool in one mode: the pinned public description and the callable that runs it."""

    description: str
    select: SelectFn


@dataclass(frozen=True)
class ToolEntry:
    """Everything the campaign knows about one registry tool.

    ``memory_exponent`` is the power of n in the tool's documented dominant memory term
    (1 for linear structures, 2 for anything materializing an n x n object);
    ``memory_note`` names that structure, so the fit's constraint is a stated fact.
    ``stochastic`` decides the seed count (see ``seeds_for``).
    """

    memory_exponent: int
    memory_note: str
    stochastic: bool
    configs: dict[Mode, ToolConfig]


# ==================================================================================================
#  select() builders
# ==================================================================================================
def _maxdiv_select(preset_name: str, budget_kind: str) -> SelectFn:
    """Build a select function running the named max-div preset under the given budget kind."""

    def select(problem: VectorMaxDivProblem, seed: int, budget_sec: float) -> NDArray[np.int64]:
        from max_div.solver import MaxDivSolverBuilder, SolverPreset, Verbosity, iterations, seconds

        target = iterations(1) if budget_kind == "single-iteration" else seconds(budget_sec)
        preset = SolverPreset[preset_name]
        solver = MaxDivSolverBuilder(problem).with_preset(target, preset).with_seed(seed).build()
        return np.asarray(solver.solve(verbosity=Verbosity.SILENT).i_selected, dtype=np.int64)

    return select


def _adapter_select(make_adapter: Callable) -> SelectFn:
    """Build a select function around a single-shot adapter; the runner's kill enforces its budget."""

    def select(problem: VectorMaxDivProblem, seed: int, budget_sec: float) -> NDArray[np.int64]:
        return np.asarray(make_adapter().select(problem, seed), dtype=np.int64)

    return select


def _cpsat_select(first_feasible: bool) -> SelectFn:
    """Build a select function for CP-SAT on the max-min model; first_feasible stops at the first solution."""

    def select(problem: VectorMaxDivProblem, seed: int, budget_sec: float) -> NDArray[np.int64]:
        from benchmarks.exact.mip_maxmin import solve_maxmin_cpsat_selection

        return solve_maxmin_cpsat_selection(problem, budget_sec, first_feasible=first_feasible, seed=seed)

    return select


def _scip_select(first_feasible: bool) -> SelectFn:
    """Build a select function for SCIP on the big-M max-min MIP; first_feasible stops at the first solution."""

    def select(problem: VectorMaxDivProblem, seed: int, budget_sec: float) -> NDArray[np.int64]:
        from benchmarks.exact.mip_maxmin import solve_maxmin_scip

        return solve_maxmin_scip(problem, budget_sec, first_feasible=first_feasible, seed=seed)

    return select


def _highs_select(first_feasible: bool) -> SelectFn:
    """Build a select function for HiGHS on the big-M max-min MIP; first_feasible stops at the first improving one."""

    def select(problem: VectorMaxDivProblem, seed: int, budget_sec: float) -> NDArray[np.int64]:
        from benchmarks.exact.mip_maxmin import solve_maxmin_highs

        return solve_maxmin_highs(problem, budget_sec, first_feasible=first_feasible, seed=seed)

    return select


def _dppy_select() -> SelectFn:
    """Build a select function drawing one DPPy k-DPP sample over an RBF likelihood kernel."""

    def select(problem: VectorMaxDivProblem, seed: int, budget_sec: float) -> NDArray[np.int64]:
        from benchmarks.adapters.dppy_kdpp import DppyKDpp

        return np.asarray(DppyKDpp().select(problem, seed), dtype=np.int64)

    return select


def _test_sleep_select() -> SelectFn:
    """Build a select function that sleeps past any budget, so the kill path is testable without a slow tool."""

    def select(problem: VectorMaxDivProblem, seed: int, budget_sec: float) -> NDArray[np.int64]:
        import time

        time.sleep(budget_sec + 3600)
        raise AssertionError("unreachable: the runner must have killed this")

    return select


def _same_for_all_modes(description: str, select: SelectFn) -> dict[Mode, ToolConfig]:
    """Return one configuration for all three modes — the single-shot common case."""
    config = ToolConfig(description, select)
    return {mode: config for mode in Mode}


# ==================================================================================================
#  The registry
# ==================================================================================================
def _rdkit():
    """Instantiate the RDKit MaxMinPicker adapter."""
    from benchmarks.adapters.rdkit_maxmin import RdkitMaxMin

    return RdkitMaxMin()


def _fpsample():
    """Instantiate the fpsample FPS adapter."""
    from benchmarks.adapters.fps import FpsampleFPS

    return FpsampleFPS()


def _skmatter():
    """Instantiate the skmatter FPS adapter."""
    from benchmarks.adapters.fps import SkmatterFPS

    return SkmatterFPS()


def _apricot():
    """Instantiate the apricot facility-location adapter."""
    from benchmarks.adapters.apricot_fl import ApricotFacilityLocation

    return ApricotFacilityLocation()


def _qc_selector():
    """Instantiate the qc-selector max-min adapter."""
    from benchmarks.adapters.qc_selector_maxmin import QcSelectorMaxMin

    return QcSelectorMaxMin()


def _code_fdm():
    """Instantiate the single-color code-FDM adapter."""
    from benchmarks.adapters.code_fdm import CodeFdmSingleColor

    return CodeFdmSingleColor()


TOOLS: dict[str, ToolEntry] = {
    "max-div": ToolEntry(
        memory_exponent=1,
        memory_note="lazy distance storage holds the vectors and O(n) working arrays, never the n^2 matrix",
        stochastic=True,
        configs={
            Mode.FASTEST_VALID: ToolConfig(
                "RANDOM preset with a one-iteration budget: random initialization, effectively no optimization",
                _maxdiv_select("RANDOM", "single-iteration"),
            ),
            Mode.MEMORY_OPTIMAL: ToolConfig(
                "RANDOM preset with a one-iteration budget (the lazy distance storage is the default at large n)",
                _maxdiv_select("RANDOM", "single-iteration"),
            ),
            Mode.QUALITY: ToolConfig(
                "SMART preset under the wall-clock budget",
                _maxdiv_select("SMART", "time"),
            ),
        },
    ),
    "ortools-cpsat": ToolEntry(
        memory_exponent=2,
        memory_note="the max-min model carries a variable or constraint per candidate pair",
        stochastic=False,
        configs={
            Mode.FASTEST_VALID: ToolConfig(
                "max-min CP-SAT model, stopping at the first feasible solution",
                _cpsat_select(first_feasible=True),
            ),
            Mode.MEMORY_OPTIMAL: ToolConfig(
                "max-min CP-SAT model, stopping at the first feasible solution",
                _cpsat_select(first_feasible=True),
            ),
            Mode.QUALITY: ToolConfig(
                "max-min CP-SAT model under the wall-clock budget",
                _cpsat_select(first_feasible=False),
            ),
        },
    ),
    "scip": ToolEntry(
        memory_exponent=2,
        memory_note="the big-M max-min MIP carries a constraint per candidate pair",
        stochastic=False,
        configs={
            Mode.FASTEST_VALID: ToolConfig(
                "big-M max-min MIP, stopping at the first feasible solution",
                _scip_select(first_feasible=True),
            ),
            Mode.MEMORY_OPTIMAL: ToolConfig(
                "big-M max-min MIP, stopping at the first feasible solution",
                _scip_select(first_feasible=True),
            ),
            Mode.QUALITY: ToolConfig(
                "big-M max-min MIP under the wall-clock budget",
                _scip_select(first_feasible=False),
            ),
        },
    ),
    "highs": ToolEntry(
        memory_exponent=2,
        memory_note="the big-M max-min MIP carries a constraint per candidate pair",
        stochastic=False,
        configs={
            Mode.FASTEST_VALID: ToolConfig(
                "big-M max-min MIP, stopping at the first improving solution",
                _highs_select(first_feasible=True),
            ),
            Mode.MEMORY_OPTIMAL: ToolConfig(
                "big-M max-min MIP, stopping at the first improving solution",
                _highs_select(first_feasible=True),
            ),
            Mode.QUALITY: ToolConfig(
                "big-M max-min MIP under the wall-clock budget",
                _highs_select(first_feasible=False),
            ),
        },
    ),
    "rdkit": ToolEntry(
        memory_exponent=1,
        memory_note="lazy distance evaluation over a caller-supplied function; no pairwise matrix",
        stochastic=False,
        configs=_same_for_all_modes(
            "MaxMinPicker with a Euclidean distance function (its only mode)", _adapter_select(_rdkit)
        ),
    ),
    "fpsample": ToolEntry(
        memory_exponent=1,
        memory_note="KD-tree-backed farthest-point sampling holds the tree and the vectors",
        stochastic=False,
        configs=_same_for_all_modes("vanilla FPS sampling (its only mode)", _adapter_select(_fpsample)),
    ),
    "skmatter": ToolEntry(
        memory_exponent=1,
        memory_note="incremental FPS holds per-candidate distances-to-selection, not a pairwise matrix",
        stochastic=False,
        configs=_same_for_all_modes("FPS selector (its only mode)", _adapter_select(_skmatter)),
    ),
    "apricot-select": ToolEntry(
        memory_exponent=2,
        memory_note="facility-location runs on a materialized n x n similarity matrix",
        stochastic=False,
        configs=_same_for_all_modes(
            "facility-location selection on an RBF similarity matrix (single greedy pass)",
            _adapter_select(_apricot),
        ),
    ),
    "qc-selector": ToolEntry(
        memory_exponent=2,
        memory_note="operates on a materialized pairwise distance matrix",
        stochastic=True,
        configs=_same_for_all_modes(
            "max-min selection on a precomputed distance matrix (single pass)", _adapter_select(_qc_selector)
        ),
    ),
    "dppy": ToolEntry(
        memory_exponent=2,
        memory_note="the k-DPP likelihood kernel is a materialized n x n matrix",
        stochastic=True,
        configs=_same_for_all_modes(
            "exact k-DPP sample over an RBF likelihood kernel (its standard usage)", _dppy_select()
        ),
    ),
    "code-fdm": ToolEntry(
        memory_exponent=2,
        memory_note="builds pairwise structures over the candidate set",
        stochastic=False,
        configs=_same_for_all_modes("FairFlow with a single color spanning all items (its unconstrained reduction)", _adapter_select(_code_fdm)),
    ),
}

# Runner-test fixture, deliberately outside TOOLS: it must never look like a registry tool,
# but the kill path needs a subprocess that outlives any budget.
TEST_SLEEP_ENTRY = ToolEntry(
    memory_exponent=1,
    memory_note="test fixture",
    stochastic=False,
    configs=_same_for_all_modes("test fixture that sleeps past any budget", _test_sleep_select()),
)


def resolve(tool: str, mode: Mode) -> ToolConfig:
    """Return the configuration for one tool in one mode; the test fixture by its own name."""
    if tool == "_test_sleep":
        return TEST_SLEEP_ENTRY.configs[mode]
    return TOOLS[tool].configs[mode]


def seeds_for(tool: str) -> tuple[int, ...]:
    """Return the campaign seeds for a tool: five for stochastic tools, three otherwise."""
    if tool == "_test_sleep":
        return (0,)
    return (0, 1, 2, 3, 4) if TOOLS[tool].stochastic else (0, 1, 2)
