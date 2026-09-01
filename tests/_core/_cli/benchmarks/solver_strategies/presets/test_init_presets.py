import pytest

from max_div._core._cli.benchmarks.solver_strategies.presets import InitPreset
from max_div._core.solver._strategies import InitializationStrategy


def test_init_preset_count():
    """Forces focus on these unit tests when changing InitPreset."""
    assert len(list(InitPreset)) == 13
    assert InitPreset.all() == list(InitPreset)


def test_init_preset_notes():
    """The strategies matching a shipped preset's initialization exactly carry a note."""
    noted = {preset for preset in InitPreset if preset.preset_note()}
    assert noted == {InitPreset.ROS_U_UNCON, InitPreset.FPSB_8, InitPreset.MF}


def test_most_feasible_relevant_only_on_constrained_problems():
    """most_feasible raises without constraints, so it is dropped from the unconstrained pages."""
    for preset in (InitPreset.MF,):
        assert preset.is_relevant_for_problem(problem_has_constraints=True)
        assert not preset.is_relevant_for_problem(problem_has_constraints=False)


@pytest.mark.parametrize("init_preset", list(InitPreset))
def test_init_preset_properties(init_preset: InitPreset):
    """Test properties of InitPreset enum members."""

    # --- act --------------------------
    init_strat = init_preset.create()

    # --- assert -----------------------
    assert isinstance(init_strat, InitializationStrategy)
    assert init_strat.__class__.__name__ == init_preset.class_name()
    assert isinstance(init_preset.class_kwargs(), dict)
    assert all(isinstance(k, str) for k in init_preset.class_kwargs())

    assert isinstance(init_preset.is_constraint_aware(), bool)
    assert isinstance(init_preset.is_relevant_for_problem(problem_has_constraints=True), bool)
    assert isinstance(init_preset.is_relevant_for_problem(problem_has_constraints=False), bool)
