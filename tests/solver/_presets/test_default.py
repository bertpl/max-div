import pytest

from max_div.benchmarks._factory import BenchmarkProblemFactory
from max_div.solver import DiversityMetric
from max_div.solver._duration import TargetDuration, iterations, seconds
from max_div.solver._presets.preset_guided import get_preset_strategies_guided
from max_div.solver._strategies._initialization._init_fast import InitFast


@pytest.mark.parametrize(
    "target_duration",
    [
        seconds(1e-6),
        seconds(1e3),
        iterations(1_000_000),
        iterations(1),
    ],
)
def test_preset_default_get_strategies(target_duration: TargetDuration):
    """Perform some rudimentary checks for expected outcome."""

    # --- arrange -----------------------------------------
    problem = BenchmarkProblemFactory.construct_problem(
        name="U1",
        size=10,
        diversity_metric=DiversityMetric.min_separation(),
    )

    # --- act ---------------------------------------------
    init_strat, optim_steps = get_preset_strategies_guided(problem, target_duration)

    # --- assert ------------------------------------------
    assert isinstance(init_strat, InitFast)  # by default we choose fast initialization
    assert len(optim_steps) > 0  # at least 1 optimization step
    assert optim_steps[0]._duration == target_duration  # should be as requested
