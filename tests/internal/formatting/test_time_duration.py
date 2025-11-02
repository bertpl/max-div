import pytest

from max_div.internal.formatting._time_duration import (
    _format_long_time_duration_to_spec,
    _format_short_time_duration_to_spec,
    format_long_time_duration,
    format_short_time_duration,
    format_time_duration,
)


# =================================================================================================
#  GENERIC duration
# =================================================================================================
@pytest.mark.parametrize(
    "value",
    [
        1e-9,
        1e-8,
        1e-7,
        1e-6,
        1e-5,
        1e-4,
        1e-3,
        1e-2,
        1e-1,
        1.0,
        1e2,
        1e3,
        1e4,
        1e5,
        1e6,
    ],
)
@pytest.mark.parametrize("n_chars", [5, 6, 7, 8, 9, 10, 11, 12, 20])
def test_format_time_duration_length(value: float, n_chars: int):
    """Check if we can always exactly meet n_chars if value<100d and n_chars>=5."""

    # --- act ---------------------------------------------
    result = format_time_duration(value, n_chars)

    # --- assert ------------------------------------------
    assert len(result) == n_chars


# =================================================================================================
#  LONG duration
# =================================================================================================
@pytest.mark.parametrize(
    "value",
    [
        1e-9,
        1e-8,
        1e-7,
        1e-6,
        1e-5,
        1e-4,
        1e-3,
        1e-2,
        1e-1,
        1.0,
        1e2,
        1e3,
        1e4,
        1e5,
        1e6,
    ],
)
@pytest.mark.parametrize("n_chars", [5, 6, 7, 8, 9, 10, 11, 12, 20])
def test_format_long_time_duration_length(value: float, n_chars: int):
    """Check if we can always exactly meet n_chars if value<100d and n_chars>=5."""

    # --- act ---------------------------------------------
    result = format_long_time_duration(value, n_chars)

    # --- assert ------------------------------------------
    assert len(result) == n_chars


# =================================================================================================
#  SHORT duration
# =================================================================================================
@pytest.mark.parametrize(
    "value",
    [
        999.4,
        1.234e0,
        1.234e-1,
        1.234e-2,
        1.234e-3,
        1.234e-4,
        1.234e-5,
        1.234e-6,
        1.234e-7,
        1.234e-8,
        1.234e-9,
        1.234e-10,
        1.234e-11,
    ],
)
@pytest.mark.parametrize("n_chars", [5, 6, 7, 8, 9, 10, 11, 12, 20])
def test_format_short_time_duration_length(value: float, n_chars: int):
    """Check if we can always exactly meet n_chars if value<999.5 and n_chars>=5."""

    # --- act ---------------------------------------------
    result = format_short_time_duration(value, n_chars)

    # --- assert ------------------------------------------
    assert len(result) == n_chars


# =================================================================================================
#  HELPERS
# =================================================================================================
@pytest.mark.parametrize(
    "value, precision, expected",
    [
        # --- "s.ss" precision ------------------
        (0.0, "s.ss", "0.00s"),
        (0.001, "s.ss", "0.00s"),
        (0.006, "s.ss", "0.01s"),
        (0.011, "s.ss", "0.01s"),
        (0.04, "s.ss", "0.04s"),
        (0.9, "s.ss", "0.90s"),
        (10.0, "s.ss", "10.00s"),
        (60.0, "s.ss", "1m0.00s"),
        (61.234, "s.ss", "1m1.23s"),
        (661.234, "s.ss", "11m1.23s"),
        (3661.234, "s.ss", "1h1m1.23s"),
        (86400.00, "s.ss", "1d0h0m0.00s"),
        (90000.00, "s.ss", "1d1h0m0.00s"),
        (90360.00, "s.ss", "1d1h6m0.00s"),
        (90361.23, "s.ss", "1d1h6m1.23s"),
        (90361.236, "s.ss", "1d1h6m1.24s"),
        # --- "s.s" precision -------------------
        (0.0, "s.s", "0.0s"),
        (0.01, "s.s", "0.0s"),
        (0.04, "s.s", "0.0s"),
        (0.06, "s.s", "0.1s"),
        (0.9, "s.s", "0.9s"),
        (10.0, "s.s", "10.0s"),
        (60.0, "s.s", "1m0.0s"),
        (61.234, "s.s", "1m1.2s"),
        (661.234, "s.s", "11m1.2s"),
        (3661.234, "s.s", "1h1m1.2s"),
        (86400.00, "s.s", "1d0h0m0.0s"),
        (90000.00, "s.s", "1d1h0m0.0s"),
        (90360.00, "s.s", "1d1h6m0.0s"),
        (90361.23, "s.s", "1d1h6m1.2s"),
        (90361.26, "s.s", "1d1h6m1.3s"),
        # --- "s" precision ---------------------
        (0.0, "s", "0s"),
        (0.4, "s", "0s"),
        (0.6, "s", "1s"),
        (10.0, "s", "10s"),
        (60.0, "s", "1m0s"),
        (61.234, "s", "1m1s"),
        (661.234, "s", "11m1s"),
        (3661.234, "s", "1h1m1s"),
        (86400.00, "s", "1d0h0m0s"),
        (90000.00, "s", "1d1h0m0s"),
        (90360.00, "s", "1d1h6m0s"),
        (90361.23, "s", "1d1h6m1s"),
        (90361.6, "s", "1d1h6m2s"),
        # --- "m" precision ---------------------
        (0.0, "m", "0m"),
        (10.0, "m", "0m"),
        (31.0, "m", "1m"),
        (60.0, "m", "1m"),
        (61.234, "m", "1m"),
        (661.234, "m", "11m"),
        (3661.234, "m", "1h1m"),
        (86400.00, "m", "1d0h0m"),
        (90000.00, "m", "1d1h0m"),
        (90360.00, "m", "1d1h6m"),
        (90361.23, "m", "1d1h6m"),
        (90390.1, "m", "1d1h7m"),
        # --- "h" precision ---------------------
        (0.0, "h", "0h"),
        (661.234, "h", "0h"),
        (1800.01, "h", "1h"),
        (3661.234, "h", "1h"),
        (86400.00, "h", "1d0h"),
        (90361.23, "h", "1d1h"),
        (91800.01, "h", "1d2h"),
        # --- "d" precision ---------------------
        (0.0, "d", "0d"),
        (43000.00, "d", "0d"),
        (43200.01, "d", "1d"),
        (90361.23, "d", "1d"),
        (86400 * 1.49999, "d", "1d"),
        (86400 * 1.50001, "d", "2d"),
    ],
)
def test_format_long_time_duration_to_spec(value: float, precision: str, expected: str):
    # --- act ---------------------------------------------
    result = _format_long_time_duration_to_spec(value, precision)

    # --- assert ------------------------------------------
    assert result == expected


@pytest.mark.parametrize(
    "dt_sec,n_digits,right_aligned,spaced,long_units,expected",
    [
        (123.45, 2, True, True, True, "123.45 sec "),
        (0.12345, 2, True, True, True, "123.45 msec"),
        (0.12345 * 1e-3, 2, True, True, True, "123.45 μsec"),
        (0.12345 * 1e-6, 2, True, True, True, "123.45 nsec"),
        (0.12345 * 1e-9, 2, True, True, True, "0.12 nsec"),
        (0.12345, 2, False, False, False, "123.45ms"),
        (0.12345, 1, False, False, False, "123.5ms"),
        (0.12345, 0, False, False, False, "123ms"),
        (0.999999, 2, False, False, False, "1.00s"),
        (0.999999, 2, True, False, False, "1.00s "),
        (0.999999, 3, True, False, False, "999.999ms"),
    ],
)
def test_format_short_time_duration_to_spec(
    dt_sec: float,
    n_digits: int,
    right_aligned: bool,
    spaced: bool,
    long_units: bool,
    expected: str,
):
    # --- act ---------------------------------------------
    result = _format_short_time_duration_to_spec(dt_sec, n_digits, right_aligned, spaced, long_units)

    # --- assert ------------------------------------------
    assert result == expected
