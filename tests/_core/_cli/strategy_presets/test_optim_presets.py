import pytest

from max_div._core._cli.bm_solver_strategies.presets import OptimPreset
from max_div._core.solver._strategies import OptimizationStrategy


def test_optim_preset_count():
    """Forces focus on these unit tests when changing OptimPreset."""
    assert len(list(OptimPreset)) == 9
    assert OptimPreset.all() == list(OptimPreset)


def test_optim_preset_notes():
    """The strategies matching a shipped preset's optimizer exactly carry a note."""
    noted = {preset for preset in OptimPreset if preset.preset_note()}
    assert noted == {OptimPreset.GS_GUIDED, OptimPreset.SM_8, OptimPreset.SM_THOROUGH}


@pytest.mark.parametrize("optim_preset", list(OptimPreset))
def test_optim_preset_properties(optim_preset: OptimPreset):
    """Test properties of OptimPreset enum members."""

    # --- act ---------------------------------------------
    optim_strat = optim_preset.create()

    # --- assert ------------------------------------------
    assert isinstance(optim_strat, OptimizationStrategy)
    assert optim_strat.__class__.__name__ == optim_preset.class_name()
    assert isinstance(optim_preset.class_kwargs(), dict)
    assert all(isinstance(k, str) for k in optim_preset.class_kwargs())

    assert isinstance(optim_preset.is_constraint_aware(), bool)
    assert isinstance(optim_preset.is_relevant_for_problem(problem_has_constraints=True), bool)
    assert isinstance(optim_preset.is_relevant_for_problem(problem_has_constraints=False), bool)
