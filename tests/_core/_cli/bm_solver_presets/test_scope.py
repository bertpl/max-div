import pytest

from max_div._core._cli.bm_solver_presets.scope import (
    K_TARGET,
    LADDER_N_POINTS,
    LADDER_T_MAX_SEC,
    LADDER_T_MIN_SEC,
    PARALLEL_ARM_T_MIN_SEC,
    determine_benchmark_scope,
    determine_benchmark_scope_for_max_duration,
    determine_problem_size_for_k,
)
from max_div._core.benchmark_problems import BenchmarkProblemFactory
from max_div._core.solver import SolverPreset

ALL_PROBLEMS = BenchmarkProblemFactory.get_all_benchmark_names()


# =================================================================================================
#  determine_problem_size_for_k
# =================================================================================================
@pytest.mark.parametrize("problem_name", ALL_PROBLEMS)
def test_determine_problem_size_for_k(problem_name: str):
    """Every problem resolves to the largest n at which it selects exactly K_TARGET items."""
    # --- act ---------------------------------------------
    n = determine_problem_size_for_k(problem_name)

    # --- assert ------------------------------------------
    _d, _n, k, _m, _n_con = BenchmarkProblemFactory.get_problem_dimensions(problem_name, n=n)
    _d, _n, k_next, _m, _n_con = BenchmarkProblemFactory.get_problem_dimensions(problem_name, n=n + 1)
    assert k == K_TARGET
    assert k_next > K_TARGET


def test_determine_problem_size_for_k_unreachable(monkeypatch: pytest.MonkeyPatch):
    """A k(n) mapping that skips the target raises instead of silently returning a nearby size."""

    # --- arrange -----------------------------------------
    def _even_k_only(problem_name: str, n: int) -> tuple[int, int, int, int, int]:
        return 2, n, 2 * ((n + 9) // 10), 0, 0  # k only takes even values

    monkeypatch.setattr(BenchmarkProblemFactory, "get_problem_dimensions", staticmethod(_even_k_only))

    # --- act & assert ------------------------------------
    with pytest.raises(ValueError, match="no size n"):
        determine_problem_size_for_k("U1", k_target=101)


# =================================================================================================
#  determine_benchmark_scope
# =================================================================================================
def test_determine_benchmark_scope_full_ladder():
    """speed=0 produces the full docs-page configuration: 50-point ladder + parallel arm."""
    # --- arrange -----------------------------------------
    presets = [SolverPreset.RANDOM, SolverPreset.SMART]

    # --- act ---------------------------------------------
    scope = determine_benchmark_scope(presets, ["U1", "C3"], None, speed=0.0)

    # --- assert ------------------------------------------
    singles = [s for s in scope if not s.is_parallel]
    parallels = [s for s in scope if s.is_parallel]

    # ladder: LADDER_N_POINTS log-spaced budgets spanning [T_MIN, T_MAX]
    durations = sorted({s.duration.value() for s in singles})
    assert len(durations) == LADDER_N_POINTS
    assert durations[0] == pytest.approx(LADDER_T_MIN_SEC)
    assert durations[-1] == pytest.approx(LADDER_T_MAX_SEC)

    # per-problem sizes follow the k=K_TARGET rule
    assert {s.problem_size for s in scope if s.problem_name == "U1"} == {1000}
    assert {s.problem_size for s in scope if s.problem_name == "C3"} == {1500}

    # each (problem, preset)-curve has a fresh seed per budget point
    u1_random = [s for s in singles if (s.problem_name, s.preset) == ("U1", SolverPreset.RANDOM)]
    assert len({s.seed for s in u1_random}) == len(u1_random) == LADDER_N_POINTS

    # parallel arm: SMART only, ladder points >= its minimum budget, machine default workers
    assert {s.preset for s in parallels} == {SolverPreset.SMART}
    assert all(s.duration.value() >= PARALLEL_ARM_T_MIN_SEC for s in parallels)
    assert len({s.n_workers for s in parallels}) == 1
    assert all(s.n_workers > 1 for s in parallels)


def test_determine_benchmark_scope_no_parallel_arm_without_smart():
    """The parallel arm only accompanies SMART."""
    # --- act ---------------------------------------------
    scope = determine_benchmark_scope([SolverPreset.RANDOM], ["U1"], None, speed=0.0)

    # --- assert ------------------------------------------
    assert all(not s.is_parallel for s in scope)


def test_determine_benchmark_scope_turbo():
    """speed=1 shrinks to a scope of sub-second runs with no parallel arm."""
    # --- act ---------------------------------------------
    scope = determine_benchmark_scope([SolverPreset.SMART], ["U1"], 100, speed=1.0)

    # --- assert ------------------------------------------
    assert 0 < len(scope) <= 2
    assert all(not s.is_parallel for s in scope)
    assert all(s.duration.value() < 1.0 for s in scope)
    assert all(s.problem_size == 100 for s in scope)  # explicit n overrides the k-rule


def test_determine_benchmark_scope_max_run_duration_override():
    """An explicit max run duration caps the ladder's longest budget."""
    # --- act ---------------------------------------------
    scope = determine_benchmark_scope([SolverPreset.SMART], ["U1"], 100, speed=0.0, max_run_duration_sec=10.0)

    # --- assert ------------------------------------------
    assert max(s.duration.value() for s in scope) == pytest.approx(10.0)


def test_determine_benchmark_scope_degenerate_ladder():
    """A max run duration at the ladder's short end collapses the ladder to a single budget."""
    # --- act ---------------------------------------------
    scope = determine_benchmark_scope(
        [SolverPreset.RANDOM], ["U1"], 100, speed=0.0, max_run_duration_sec=LADDER_T_MIN_SEC
    )

    # --- assert ------------------------------------------
    assert {s.duration.value() for s in scope} == {LADDER_T_MIN_SEC}


# =================================================================================================
#  determine_benchmark_scope_for_max_duration
# =================================================================================================
@pytest.mark.parametrize(
    "max_duration_sec, expected_speed",
    [
        (1e12, 0.0),  # slowest setting already fits
        (1e-6, 1.0),  # fastest setting still too slow
        (600.0, None),  # in between: bisected speed in (0, 1)
    ],
)
def test_determine_benchmark_scope_for_max_duration(max_duration_sec: float, expected_speed: float | None):
    """The auto-tuner returns the boundary speeds directly and bisects in between."""
    # --- act ---------------------------------------------
    speed, scope = determine_benchmark_scope_for_max_duration(
        [SolverPreset.SMART], ["U1"], 100, max_duration_sec=max_duration_sec
    )

    # --- assert ------------------------------------------
    assert len(scope) > 0
    if expected_speed is not None:
        assert speed == expected_speed
    else:
        assert 0.0 < speed < 1.0
