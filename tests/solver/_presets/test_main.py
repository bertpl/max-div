from unittest.mock import Mock

import pytest

from max_div.benchmarks._factory import BenchmarkProblemFactory
from max_div.solver import DiversityMetric, SolverPreset
from max_div.solver._duration import TargetDuration, iterations, seconds
from max_div.solver._presets import get_preset_strategies
from max_div.solver._strategies._initialization._init_random_one_shot import InitRandomOneShot


@pytest.mark.parametrize(
    "target_duration",
    [
        seconds(1e-6),
        seconds(1e3),
        iterations(1_000_000),
        iterations(1),
    ],
)
@pytest.mark.parametrize("preset", list(SolverPreset))
def test_get_preset_strategies(preset: SolverPreset, target_duration: TargetDuration):
    """Perform some rudimentary checks for expected outcome."""

    # --- act ---------------------------------------------
    init_strat, optim_steps = get_preset_strategies(SolverPreset.DEFAULT, target_duration)

    # --- assert ------------------------------------------
    assert isinstance(init_strat, InitRandomOneShot)  # by default we choose fast initialization
    assert len(optim_steps) > 0  # at least 1 optimization step
    assert optim_steps[0]._duration == target_duration  # should be as requested


def test_get_preset_strategies_invalid_preset():
    """Test that an invalid preset raises a ValueError."""

    # --- arrange -----------------------------------------
    invalid_preset = Mock(resolve_alias=lambda: "INVALID")

    # --- act & assert ------------------------------------
    with pytest.raises(ValueError):
        get_preset_strategies(invalid_preset, seconds(1))
