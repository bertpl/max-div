import pytest

from max_div._core._timing import EndToEndTiming, measure_end_to_end


def test_measure_end_to_end():
    """The context yields a timing whose elapsed span is set on exit."""
    # --- act ---------------------------------------------
    with measure_end_to_end() as timing:
        inside_value = timing.t_elapsed_sec

    # --- assert ------------------------------------------
    assert isinstance(timing, EndToEndTiming)
    assert inside_value == 0.0
    assert timing.t_elapsed_sec > 0.0


def test_measure_end_to_end_sets_elapsed_on_exception():
    """The elapsed span is set even when the timed block raises."""
    # --- act ---------------------------------------------
    with pytest.raises(RuntimeError), measure_end_to_end() as timing:
        raise RuntimeError("boom")

    # --- assert ------------------------------------------
    assert timing.t_elapsed_sec > 0.0
