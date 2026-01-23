import pytest

from max_div.benchmarks._factory import BenchmarkProblemFactory
from max_div.solver import DiversityMetric
from max_div.solver._duration import TargetDuration, iterations, seconds
from max_div.solver._presets.preset_guided import get_preset_strategies_guided
from max_div.solver._strategies._initialization._init_eager import InitEager
from max_div.solver._strategies._initialization._init_fast import InitFast


@pytest.mark.parametrize("target_duration", [seconds(1e-6), iterations(1)])
@pytest.mark.parametrize("initialization_included", [True, False])
def test_preset_default_get_strategies_short(target_duration: TargetDuration, initialization_included: bool):
    """Perform some rudimentary checks for expected outcome of a very short run."""

    # --- arrange -----------------------------------------
    problem = BenchmarkProblemFactory.construct_problem(
        name="U1",
        size=10,
        diversity_metric=DiversityMetric.min_separation(),
    )

    # --- act ---------------------------------------------
    init_strat, optim_steps = get_preset_strategies_guided(problem, target_duration, initialization_included)

    # --- assert ------------------------------------------
    assert isinstance(init_strat, InitFast)  # we should choose the fastest in this case
    if initialization_included:
        assert len(optim_steps) == 0
    else:
        assert len(optim_steps) > 0
        assert optim_steps[0]._duration == target_duration  # should be unchanged


@pytest.mark.parametrize("target_duration", [seconds(1e3), iterations(1_000_000)])
@pytest.mark.parametrize("initialization_included", [True, False])
def test_preset_default_get_strategies_long(target_duration: TargetDuration, initialization_included: bool):
    """Perform some rudimentary checks for expected outcome of a very long run."""

    # --- arrange -----------------------------------------
    problem = BenchmarkProblemFactory.construct_problem(
        name="U1",
        size=10,
        diversity_metric=DiversityMetric.min_separation(),
    )

    # --- act ---------------------------------------------
    init_strat, optim_steps = get_preset_strategies_guided(problem, target_duration, initialization_included)

    # --- assert ------------------------------------------
    assert isinstance(init_strat, InitEager)  # we should have time to initialize with InitEager
    assert len(optim_steps) > 0  # we should have time left to optimize
    if initialization_included:
        assert optim_steps[0]._duration != target_duration  # should be changed
        assert optim_steps[0]._duration.value() < target_duration.value()  # should be reduced
    else:
        assert len(optim_steps) > 0
        assert optim_steps[0]._duration == target_duration  # should be unchanged


def test_preset_default_get_strategies_invalid_duration():
    """Test that an invalid TargetDuration type raises a TypeError."""
    with pytest.raises(TypeError):
        _ = get_preset_strategies_guided("dummy_problem", "invalid_duration_type")
