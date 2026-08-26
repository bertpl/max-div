import pytest

from max_div._core._cli.benchmarks.solver_presets.scope import (
    PARALLEL_SERIES,
    SINGLE_SERIES,
    determine_benchmark_scope,
)
from max_div._core.solver import SolverPreset


# =================================================================================================
#  determine_benchmark_scope
# =================================================================================================
def test_determine_benchmark_scope_full_series():
    """speed=0 produces the full docs-page configuration: single and parallel runs each on their own budget series."""
    # --- arrange ----------------------
    presets = [SolverPreset.RANDOM, SolverPreset.SMART]

    # --- act --------------------------
    scope = determine_benchmark_scope(presets, ["U1", "C3"], None, speed=0.0)

    # --- assert -----------------------
    singles = [s for s in scope if not s.is_parallel]
    parallels = [s for s in scope if s.is_parallel]

    # the single-worker series has n_points log-spaced budgets spanning [t_min, t_max]
    single_durations = sorted({s.duration.value() for s in singles})
    assert len(single_durations) == SINGLE_SERIES.n_points.at(0.0)
    assert single_durations[0] == pytest.approx(SINGLE_SERIES.t_min_sec.at(0.0))
    assert single_durations[-1] == pytest.approx(SINGLE_SERIES.t_max_sec.at(0.0))

    # the parallel series keeps its own point count and range, independent of the single series
    parallel_durations = sorted({s.duration.value() for s in parallels})
    assert len(parallel_durations) == PARALLEL_SERIES.n_points.at(0.0)
    assert parallel_durations[0] == pytest.approx(PARALLEL_SERIES.t_min_sec.at(0.0))
    assert parallel_durations[-1] == pytest.approx(PARALLEL_SERIES.t_max_sec.at(0.0))

    # per-problem sizes follow the k=K_TARGET rule
    assert {s.problem_size for s in scope if s.problem_name == "U1"} == {1000}
    assert {s.problem_size for s in scope if s.problem_name == "C3"} == {1500}

    # each curve has a fresh seed per budget point, for single and parallel runs separately
    u1_random = [s for s in singles if (s.problem_name, s.preset) == ("U1", SolverPreset.RANDOM)]
    assert len({s.seed for s in u1_random}) == len(u1_random) == SINGLE_SERIES.n_points.at(0.0)
    u1_parallel = [s for s in parallels if s.problem_name == "U1"]
    assert len({s.seed for s in u1_parallel}) == len(u1_parallel) == PARALLEL_SERIES.n_points.at(0.0)

    # parallel runs use SMART only, on the machine's default worker count
    assert {s.preset for s in parallels} == {SolverPreset.SMART}
    assert len({s.n_workers for s in parallels}) == 1
    assert all(s.n_workers > 1 for s in parallels)


def test_determine_benchmark_scope_no_parallel_runs_without_smart():
    """Parallel runs only accompany SMART."""
    # --- act --------------------------
    scope = determine_benchmark_scope([SolverPreset.RANDOM], ["U1"], 100, speed=0.0)

    # --- assert -----------------------
    assert all(not s.is_parallel for s in scope)


def test_determine_benchmark_scope_turbo():
    """speed=1 shrinks to sub-second single runs plus one short parallel run."""
    # --- act --------------------------
    scope = determine_benchmark_scope([SolverPreset.SMART], ["U1"], 100, speed=1.0)

    # --- assert -----------------------
    singles = [s for s in scope if not s.is_parallel]
    parallels = [s for s in scope if s.is_parallel]
    assert 0 < len(singles) <= 2
    assert all(s.duration.value() < 1.0 for s in singles)
    assert len(parallels) == 1
    assert parallels[0].duration.value() == pytest.approx(PARALLEL_SERIES.t_max_sec.at(1.0))
    assert all(s.problem_size == 100 for s in scope)  # explicit n overrides the k-rule


def test_determine_benchmark_scope_max_run_duration_override():
    """An explicit max run duration replaces both series' longest budget."""
    # --- act --------------------------
    scope = determine_benchmark_scope([SolverPreset.SMART], ["U1"], 100, speed=0.0, max_run_duration_sec=10.0)

    # --- assert -----------------------
    singles = [s for s in scope if not s.is_parallel]
    parallels = [s for s in scope if s.is_parallel]
    assert max(s.duration.value() for s in singles) == pytest.approx(10.0)
    assert max(s.duration.value() for s in parallels) == pytest.approx(10.0)


def test_determine_benchmark_scope_degenerate_series():
    """A max run duration at the series' short end collapses it to a single budget."""
    # --- arrange ----------------------
    t_min_sec = SINGLE_SERIES.t_min_sec.at(0.0)

    # --- act --------------------------
    scope = determine_benchmark_scope([SolverPreset.RANDOM], ["U1"], 100, speed=0.0, max_run_duration_sec=t_min_sec)

    # --- assert -----------------------
    assert {s.duration.value() for s in scope} == {t_min_sec}
