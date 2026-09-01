from unittest.mock import Mock

import pytest

from max_div._core.metrics import DiversityMetric
from max_div._core.solver import SolverPreset
from max_div._core.solver._duration import TargetDuration, iterations, seconds
from max_div._core.solver._presets import get_preset_strategies
from max_div._core.solver._strategies._initialization import InitializationStrategy
from max_div._core.solver._strategies._initialization._init_farthest_point import InitFarthestPoint
from max_div._core.solver._strategies._initialization._init_farthest_point_batched import InitFarthestPointBatched
from max_div._core.solver._strategies._initialization._init_most_feasible import InitMostFeasible
from max_div._core.solver._strategies._initialization._init_random_one_shot import InitRandomOneShot

# Each preset (by resolved alias, so DEFAULT follows SMART) yields this init for an unconstrained
# problem under a separation-family diversity metric.
_EXPECTED_INIT_UNCONSTRAINED: dict[SolverPreset, type[InitializationStrategy]] = {
    SolverPreset.RANDOM: InitRandomOneShot,
    SolverPreset.GUIDED: InitRandomOneShot,
    SolverPreset.SMART: InitFarthestPointBatched,
    SolverPreset.THOROUGH: InitFarthestPointBatched,
}

# Under `MEAN_PAIRWISE_DISTANCE`, SMART/THOROUGH use the per-pick farthest-point construction.
_EXPECTED_INIT_MEAN_DISTANCE: dict[SolverPreset, type[InitializationStrategy]] = {
    SolverPreset.RANDOM: InitRandomOneShot,
    SolverPreset.GUIDED: InitRandomOneShot,
    SolverPreset.SMART: InitFarthestPoint,
    SolverPreset.THOROUGH: InitFarthestPoint,
}

# With constraints present, SMART/THOROUGH start from `most_feasible()`.
_EXPECTED_INIT_CONSTRAINED: dict[SolverPreset, type[InitializationStrategy]] = {
    SolverPreset.RANDOM: InitRandomOneShot,
    SolverPreset.GUIDED: InitRandomOneShot,
    SolverPreset.SMART: InitMostFeasible,
    SolverPreset.THOROUGH: InitMostFeasible,
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

    # --- act --------------------------
    init_strat, optim_steps = get_preset_strategies(preset, target_duration, DiversityMetric.GEOMEAN_SEPARATION)

    # --- assert -----------------------
    assert isinstance(init_strat, _EXPECTED_INIT_UNCONSTRAINED[preset.resolve_alias()])
    assert len(optim_steps) > 0  # at least 1 optimization step
    assert optim_steps[0]._duration == target_duration  # should be as requested


@pytest.mark.parametrize(
    "diversity_metric, has_constraints, expected_init",
    [
        (DiversityMetric.GEOMEAN_SEPARATION, True, _EXPECTED_INIT_CONSTRAINED),
        (DiversityMetric.MEAN_PAIRWISE_DISTANCE, False, _EXPECTED_INIT_MEAN_DISTANCE),
    ],
    ids=["constrained", "mean_distance"],
)
@pytest.mark.parametrize("preset", list(SolverPreset))
def test_get_preset_strategies_init_follows_the_problem(
    preset: SolverPreset,
    diversity_metric: DiversityMetric,
    has_constraints: bool,
    expected_init: dict[SolverPreset, type[InitializationStrategy]],
):
    """SMART/THOROUGH start from most_feasible() when constrained, and from the per-pick farthest-point
    construction under MEAN_PAIRWISE_DISTANCE.
    """

    # --- act --------------------------
    init_strat, _ = get_preset_strategies(preset, iterations(30), diversity_metric, has_constraints=has_constraints)

    # --- assert -----------------------
    assert isinstance(init_strat, expected_init[preset.resolve_alias()])


def test_get_preset_strategies_invalid_preset():
    """Test that an invalid preset raises a ValueError."""

    # --- arrange ----------------------
    invalid_preset = Mock(resolve_alias=lambda: "INVALID")

    # --- act & assert -----------------
    with pytest.raises(ValueError):
        get_preset_strategies(invalid_preset, seconds(1), DiversityMetric.GEOMEAN_SEPARATION)


def test_solver_preset_all_sorted():
    assert SolverPreset.all_sorted() == [
        SolverPreset.RANDOM,
        SolverPreset.GUIDED,
        SolverPreset.SMART,
        SolverPreset.THOROUGH,
    ]
