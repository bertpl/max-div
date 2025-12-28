import pytest

from max_div.solver._strategies import OptimizationStrategy
from max_div.solver._strategies._optimization._presets import OptimPreset
from max_div.solver._strategies._optimization._time_model import OptimizationTimeModel


def test_optim_preset_count():
    """Forces focus on these unit tests when changing OptimPreset."""
    assert len(list(OptimPreset)) == 10
    assert OptimPreset.all() == list(OptimPreset)


@pytest.mark.parametrize("optim_preset", list(OptimPreset))
def test_optim_preset_properties(optim_preset: OptimPreset):
    """Test properties of OptimPreset enum members."""

    # --- act ---------------------------------------------
    optim_strat = optim_preset.create()

    # --- assert ------------------------------------------
    assert isinstance(optim_strat, OptimizationStrategy)
    assert optim_strat.__class__.__name__ == optim_preset.class_name()
    assert isinstance(optim_preset.class_kwargs(), dict)
    assert all([isinstance(k, str) for k in optim_preset.class_kwargs().keys()])

    assert isinstance(optim_preset.is_constraint_aware(), bool)
    assert isinstance(optim_preset.is_relevant_for_problem(problem_has_constraints=True), bool)
    assert isinstance(optim_preset.is_relevant_for_problem(problem_has_constraints=False), bool)
    assert isinstance(optim_preset.time_model(), OptimizationTimeModel)
