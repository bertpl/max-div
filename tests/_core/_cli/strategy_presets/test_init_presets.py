import pytest

from max_div._core._cli.bm_solver_strategies.presets import InitPreset
from max_div._core.solver._strategies import InitializationStrategy


def test_init_preset_count():
    """Forces focus on these unit tests when changing InitPreset."""
    assert len(list(InitPreset)) == 13
    assert InitPreset.all() == list(InitPreset)


@pytest.mark.parametrize("init_preset", list(InitPreset))
def test_init_preset_properties(init_preset: InitPreset):
    """Test properties of InitPreset enum members."""

    # --- act ---------------------------------------------
    init_strat = init_preset.create()

    # --- assert ------------------------------------------
    assert isinstance(init_strat, InitializationStrategy)
    assert init_strat.__class__.__name__ == init_preset.class_name()
    assert isinstance(init_preset.class_kwargs(), dict)
    assert all(isinstance(k, str) for k in init_preset.class_kwargs())

    assert isinstance(init_preset.is_constraint_aware(), bool)
    assert isinstance(init_preset.is_relevant_for_problem(problem_has_constraints=True), bool)
    assert isinstance(init_preset.is_relevant_for_problem(problem_has_constraints=False), bool)
