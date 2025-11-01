import pytest

from max_div.internal.utils import clip


@pytest.mark.parametrize(
    "value, min_value, max_value, expected",
    [
        (5, 0, 10, 5),
        (-5, 0, 10, 0),
        (15, 0, 10, 10),
        (5, 5, 5, 5),
        (2, 10, 5, 7.5),
        (5, 10, 5, 7.5),
        (7, 10, 5, 7.5),
        (10, 10, 5, 7.5),
        (18, 10, 5, 7.5),
    ],
)
def test_clip(value: float, min_value: float, max_value: float, expected: float):
    # --- act ---------------------------------------------
    result = clip(value, min_value, max_value)

    # --- assert ------------------------------------------
    assert result == expected
