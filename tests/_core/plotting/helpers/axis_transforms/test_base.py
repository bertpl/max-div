import numpy as np
import pytest

from tests._extras import skip_module_unless_extra

skip_module_unless_extra("plot")

from max_div._core.plotting.helpers import AxisTransform, NullTransform  # noqa: E402


def test_axis_transform_is_abstract() -> None:
    with pytest.raises(TypeError):
        AxisTransform()  # type: ignore[abstract]


def test_null_transform_returns_a_float_for_a_scalar() -> None:
    # --- act --------------------------
    to_axis = NullTransform().to_axis(3)
    from_axis = NullTransform().from_axis(3)

    # --- assert -----------------------
    assert to_axis == 3.0
    assert isinstance(to_axis, float)
    assert from_axis == 3.0
    assert isinstance(from_axis, float)


def test_null_transform_returns_an_array_for_a_list() -> None:
    # --- act --------------------------
    to_axis = NullTransform().to_axis([1, 2])
    from_axis = NullTransform().from_axis([1, 2])

    # --- assert -----------------------
    np.testing.assert_array_equal(to_axis, np.array([1.0, 2.0]))
    np.testing.assert_array_equal(from_axis, np.array([1.0, 2.0]))
