from unittest.mock import Mock

import pytest

from max_div._core.solver import SolverPreset
from max_div._core.solver._duration import TargetDuration, iterations, seconds
from max_div._core.solver._presets import get_preset_strategies
from max_div._core.solver._strategies._initialization import InitializationStrategy
from max_div._core.solver._strategies._initialization._init_farthest_point import InitFarthestPoint
from max_div._core.solver._strategies._initialization._init_random_one_shot import InitRandomOneShot

# expected init strategy per resolved preset (DEFAULT resolves to its alias's entry)
_EXPECTED_INIT: dict[SolverPreset, type[InitializationStrategy]] = {
    SolverPreset.RANDOM: InitRandomOneShot,
    SolverPreset.GUIDED: InitRandomOneShot,
    SolverPreset.SMART: InitFarthestPoint,
    SolverPreset.THOROUGH: InitFarthestPoint,
}


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
    """Each preset yields its expected init strategy, at least one optim step, and the requested duration."""

    # --- act ---------------------------------------------
    init_strat, optim_steps = get_preset_strategies(preset, target_duration)

    # --- assert ------------------------------------------
    assert isinstance(init_strat, _EXPECTED_INIT[preset.resolve_alias()])
    assert len(optim_steps) > 0  # at least 1 optimization step
    assert optim_steps[0]._duration == target_duration  # should be as requested


def test_get_preset_strategies_invalid_preset():
    """Test that an invalid preset raises a ValueError."""

    # --- arrange -----------------------------------------
    invalid_preset = Mock(resolve_alias=lambda: "INVALID")

    # --- act & assert ------------------------------------
    with pytest.raises(ValueError):
        get_preset_strategies(invalid_preset, seconds(1))


def test_solver_preset_all_sorted():
    assert SolverPreset.all_sorted() == [
        SolverPreset.RANDOM,
        SolverPreset.GUIDED,
        SolverPreset.SMART,
        SolverPreset.THOROUGH,
    ]
