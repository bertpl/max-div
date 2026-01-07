import numpy as np
import pytest

from max_div.solver._parameters._parameter_schedule import (
    EaseInOutSchedule,
    EaseInSchedule,
    EaseOutSchedule,
    LinearSchedule,
    ParameterSchedule,
    _evaluate_schedules,
    _schedules_to_2d_numpy_array,
    ease_in,
    ease_in_out,
    ease_out,
    linear,
)


# =================================================================================================
#  TEST - Base Class
# =================================================================================================
@pytest.mark.parametrize(
    "v0, v1, expected_min_value, expected_max_value",
    [
        (0.0, 1.0, 0.0, 1.0),
        (1.0, 0.0, 0.0, 1.0),
        (-1.0, 1.0, -1.0, 1.0),
        (2.7, -3.2, -3.2, 2.7),
    ],
)
def test_parameter_schedule_properties(v0: float, v1: float, expected_min_value: float, expected_max_value: float):
    # --- arrange -----------------------------------------
    c_poly = [-2.0, 10.0, 0.0, -20.0]  # set strange values; the result should only depend on v0, v1
    schedule = ParameterSchedule(v0, v1, c_poly)

    # --- act ---------------------------------------------
    min_value, max_value = schedule.min_value, schedule.max_value

    # --- assert ------------------------------------------
    assert min_value == expected_min_value
    assert max_value == expected_max_value


@pytest.mark.parametrize(
    "c_poly, f, v_expected",
    [
        ([0.0, 1.0, 0.0, 0.0], -0.1, 1.23),  # f clipped to 0.0
        ([0.0, 1.0, 0.0, 0.0], 0.0, 1.23),
        ([0.0, 1.0, 0.0, 0.0], 0.4, 1.23 + 0.4 * 3.33),
        ([0.0, 1.0, 0.0, 0.0], 1.0, 4.56),
        ([0.0, 1.0, 0.0, 0.0], 1.1, 4.56),  # f clipped to 1.0
        ([0.1, -1.0, 0.05, 3.0], 0.0, 1.23 + 0.1 * 3.33),
        ([0.1, -1.0, 0.05, 3.0], 0.2, 1.23),  # clipped to min_value
        ([0.1, -1.0, 0.05, 3.0], 0.6, 1.23 + 3.33 * (0.1 - 0.6 + (0.05 * 0.6 * 0.6) + (3 * 0.6 * 0.6 * 0.6))),
        ([0.1, -1.0, 0.05, 3.0], 1.0, 4.56),  # clipped to max_value
    ],
    ids=[
        "linear_below_0_clipped",
        "linear_at_0",
        "linear_at_0.4",
        "linear_at_1",
        "linear_above_1_clipped",
        "complex_at_0",
        "complex_at_0_2_clipped",
        "complex_at_0_6",
        "complex_at_1_clipped",
    ],
)
def test_parameter_schedule_get_value(c_poly: list[float], f: float, v_expected: float):
    # --- arrange -----------------------------------------
    schedule = ParameterSchedule(1.23, 4.56, c_poly)
    v_min = schedule.min_value
    v_max = schedule.max_value

    # --- act ---------------------------------------------
    v = schedule.get_value(f)

    # --- assert ------------------------------------------
    assert v_min <= v <= v_max
    assert v == pytest.approx(v_expected)


@pytest.mark.parametrize(
    "sched",
    [
        linear(1.0, 2.0),
        ease_in(1.1, 2.1),
        ease_out(3.0, 4.0),
        ease_in_out(5.1, 6.1),
    ],
)
def test_parameter_schedule_get_initial_value(sched: ParameterSchedule):
    assert sched.min_value <= sched.get_initial_value() <= sched.max_value


# =================================================================================================
#  TEST - Child Classes
# =================================================================================================
@pytest.mark.parametrize("cls", [LinearSchedule, EaseInSchedule, EaseOutSchedule, EaseInOutSchedule])
def test_parameter_schedule_child_classes_well_behaved(cls):
    """Test if child classes are well-behaved: v(0) = v0, v(1) = v1, min_value <= v(f) <= max_value."""

    # --- arrange -----------------------------------------
    v0_exact = 1.23
    v1_exact = 4.56
    schedule = cls(v0=v0_exact, v1=v1_exact)

    # --- act ---------------------------------------------
    v0 = schedule.get_value(0.0)
    v1 = schedule.get_value(1.0)
    v_internal = [schedule.get_value(float(f)) for f in np.linspace(0.001, 0.999, num=1000)]

    # --- assert ------------------------------------------
    assert v0 == pytest.approx(v0_exact)
    assert v1 == pytest.approx(v1_exact)
    assert all([1.23 < v < 4.56 for v in v_internal])


@pytest.mark.parametrize("cls", [EaseInSchedule, EaseInOutSchedule])
def test_parameter_schedule_ease_in(cls):
    # --- arrange -----------------------------------------
    schedule = cls(v0=0.0, v1=1.0)
    delta = 0.001

    # --- act ---------------------------------------------
    v0001 = schedule.get_value(delta)

    # --- assert ------------------------------------------
    assert 0.0 < v0001 < 10 * (delta * delta)  # v(f) should ease in quadratically


@pytest.mark.parametrize("cls", [EaseOutSchedule, EaseInOutSchedule])
def test_parameter_schedule_ease_out(cls):
    # --- arrange -----------------------------------------
    schedule = cls(v0=0.0, v1=1.0)
    delta = 0.001

    # --- act ---------------------------------------------
    v0999 = schedule.get_value(1.0 - delta)

    # --- assert ------------------------------------------
    assert 1.0 - (10 * (delta * delta)) < v0999 < 1.0  # v(f) should ease out quadratically


# =================================================================================================
#  TEST - Aliases
# =================================================================================================
@pytest.mark.parametrize(
    "alias_fun, expected_cls",
    [
        (linear, LinearSchedule),
        (ease_in, EaseInSchedule),
        (ease_out, EaseOutSchedule),
        (ease_in_out, EaseInOutSchedule),
    ],
)
def test_parameter_schedule_aliases(alias_fun, expected_cls):
    # --- act ---------------------------------------------
    schedule = alias_fun(v0=2.46, v1=3.69)

    # --- assert ------------------------------------------
    assert isinstance(schedule, expected_cls)
    assert schedule.v0 == 2.46
    assert schedule.v1 == 3.69


@pytest.mark.parametrize(
    "schedule,expected_str",
    [
        (linear(0.1, 0.2), "linear(0.10,0.20)"),
        (ease_in(1.5, -2.5), "ease_in(1.50,-2.50)"),
        (ease_out(-3.0, 4.0), "ease_out(-3.00,4.00)"),
        (ease_in_out(0.0, 1.0), "ease_in_out(0.00,1.00)"),
        (
            ParameterSchedule(v0=0.0, v1=1.0, c_poly=[0.2, 0.4, 0.6, 0.8]),
            "ParameterSchedule(v0=0.0, v1=1.0, c_poly=[0.2, 0.4, 0.6, 0.8])",
        ),
    ],
    ids=["linear", "ease_in", "ease_out", "ease_in_out", "ParameterSchedule"],
)
def test_parameter_schedule_str(schedule: ParameterSchedule, expected_str: str):
    assert str(schedule) == expected_str


# =================================================================================================
#  TEST - numba acceleration
# =================================================================================================
def test_schedules_to_2d_numpy_array():
    # --- arrange -----------------------------------------
    schedules = [
        ParameterSchedule(v0=1.0, v1=2.0, c_poly=[0.1, 0.2, 0.4, 0.8]),
        ParameterSchedule(v0=1.6, v1=-3.1, c_poly=[0.8, 0.4, 0.2, 0.1]),
    ]

    # --- act ---------------------------------------------
    arr = _schedules_to_2d_numpy_array(schedules)

    # --- assert ------------------------------------------
    assert arr.shape == (2, 6)
    assert arr.dtype == np.float64
    assert np.allclose(
        arr,
        [
            [1.0, 2.0, 1.1, 0.2, 0.4, 0.8],
            [-3.1, 1.6, 1.6 - 4.7 * 0.8, -4.7 * 0.4, -4.7 * 0.2, -4.7 * 0.1],
        ],
    )


def test_schedules_to_2d_numpy_array_empty():
    # --- act ---------------------------------------------
    arr = _schedules_to_2d_numpy_array([])

    # --- assert ------------------------------------------
    assert arr.shape == (0, 6)
    assert arr.dtype == np.float64


@pytest.mark.parametrize("f", [-0.1, 0.0, 0.1, 0.5, 0.9, 1.0, 1.1])
def test_evaluate_schedules(f: float):
    # --- arrange -----------------------------------------
    schedules = [
        linear(1.1, 2.3),
        ease_in(2.1, -0.6),
        ease_out(-1.5, 3.4),
        ease_in_out(0.1, 5.0),
    ]
    v_expected = [schedule.get_value(f) for schedule in schedules]

    arr = _schedules_to_2d_numpy_array(schedules)

    # --- act ---------------------------------------------
    v_actual = _evaluate_schedules(arr, f)

    # --- assert ------------------------------------------
    assert np.allclose(v_expected, v_actual)
