import numpy as np
import pytest

from max_div._core._utils import is_non_bool_int


@pytest.mark.parametrize(
    "value, expected",
    [
        (3, True),
        (-3, True),
        (np.int32(3), True),
        (np.int64(3), True),
        (True, False),
        (np.bool_(True), False),
        (3.0, False),
        (np.float32(3.0), False),
        ("3", False),
        (None, False),
    ],
)
def test_is_non_bool_int(value: object, expected: bool):
    # --- act & assert -----------------
    assert is_non_bool_int(value) is expected
