from typing import Literal

from max_div.internal.utils import HALF_EPS


# =================================================================================================
#  Main entrypoints
# =================================================================================================
def format_time_duration(dt_sec: float, n_chars: int = 10) -> str:
    """Format a time duration in seconds."""
    if dt_sec < 1.0:
        return format_short_time_duration(dt_sec, n_chars=n_chars, spaced=False, long_units=False, right_aligned=False)
    else:
        return format_long_time_duration(dt_sec, n_chars=n_chars)


def format_long_time_duration(dt_sec: float, n_chars: int = 10) -> str:
    """
    Format a time duration that is expected to be ~1sec or larger.

    The max lengths of the resulting string (n_chars) will always be exactly matched if n_chars>=5 and dt_sec<100d.
    """
    # --- main loop ---------------------------------------
    result = ""
    for precision in ["d", "h", "m", "s", "s.s", "s.ss"]:
        # choose representation with the highest precision with length <= n_chars
        cand = _format_long_time_duration_to_spec(dt_sec, precision)
        if (result == "") or (len(cand) <= n_chars):
            result = cand

    # --- final result ------------------------------------
    result = result.rjust(n_chars)  # pad with spaces on the left, if too short
    return result


def format_short_time_duration(
    dt_sec: float,
    n_chars: int = 10,
    right_aligned: bool | None = None,
    spaced: bool | None = None,
    long_units: bool | None = None,
) -> str:
    """
    Format a time duration that is expected to be <<1sec.  Different styling options are chosen to optimally meet
    the given character count.

    The options 'right_aligned', 'spaced', and 'long_units' are optional and, when omitted, automatically chosen
     based on heuristics.

    The requested n_chars is guaranteed to be met exactly if n_chars>=5, dt_sec<999.5 and with default styling options.

    :param dt_sec: (float) Time duration in seconds.
    :param n_chars: (int >= 5) Desired number of characters in the output
    :param right_aligned: (bool) If True, the result is right-aligned such that values & units align vertically.
    :param spaced: (bool) If True, a space is added between the number and the unit.
    :param long_units: (bool) If True, long unit names are used (e.g. "nsec" instead of "ns").
    """

    # --- styling heuristics ------------------------------
    if right_aligned is None:
        right_aligned = n_chars > 6
    if spaced is None:
        spaced = n_chars > 8
    if long_units is None:
        long_units = n_chars > 10

    # --- main loop ---------------------------------------
    result = ""
    for n_digits in range(n_chars - 2):
        # choose representation with largest n_digits with length <= n_chars
        cand = _format_short_time_duration_to_spec(dt_sec, n_digits, right_aligned, spaced, long_units)
        if (result == "") or (len(cand) <= n_chars):
            result = cand

    # --- final result ------------------------------------
    result = result.rjust(n_chars)  # pad with spaces on the left, if too short
    return result


# =================================================================================================
#  Helpers
# =================================================================================================
def _format_long_time_duration_to_spec(
    dt_sec: float, precision: str = Literal["d", "h", "m", "s", "s.s", "s.ss", "s.sss"]
) -> str:
    # --- rounding ----------------------------------------
    match precision:
        case "d":
            round_to = 60 * 60 * 24
        case "h":
            round_to = 60 * 60
        case "m":
            round_to = 60
        case "s":
            round_to = 1
        case "s.s":
            round_to = 0.1
        case "s.ss":
            round_to = 0.01
    dt_sec = round_to * round(dt_sec / round_to) * (1 + HALF_EPS)  # avoid rounding issues

    # --- split in components -----------------------------
    rem_m, s = divmod(dt_sec, 60)
    rem_h, m = divmod(rem_m, 60)
    d, h = divmod(rem_h, 24)

    # --- construct string-valued components --------------
    d, h, m, s = int(d), int(h), int(m), s  # convert to int where appropriate
    match precision:
        case "d":
            components = [f"{d}d"]
        case "h":
            components = [f"{d}d", f"{h}h"]
        case "m":
            components = [f"{d}d", f"{h}h", f"{m}m"]
        case "s":
            components = [f"{d}d", f"{h}h", f"{m}m", f"{int(s)}s"]
        case "s.s":
            components = [f"{d}d", f"{h}h", f"{m}m", f"{s:.1f}s"]
        case "s.ss":
            components = [f"{d}d", f"{h}h", f"{m}m", f"{s:.2f}s"]

    # prune leading zero components
    while (len(components) > 1) and components[0].startswith("0"):
        components.pop(0)

    # --- build final result ------------------------------
    return "".join(components)


def _format_short_time_duration_to_spec(
    dt_sec: float, n_digits: int, right_aligned: bool, spaced: bool, long_units: bool
) -> str:
    """
    Format a time duration that is expected to be <<1sec to the given specification.
    :param dt_sec: (float) Time duration in seconds.
    :param n_digits: (int >= 0) Number of digits to show after the decimal point.
    :param right_aligned: (bool) If True, the result is right-aligned such that values & units align vertically.
    :param spaced: (bool) If True, a space is added between the number and the unit.
    :param long_units: (bool) If True, long unit names are used (e.g. "nsec" instead of "ns").
    :return: string representation of the time duration.
    """

    # --- init --------------------------------------------
    if long_units:
        c_and_unit = [
            (1.0, "sec " if right_aligned else "s"),
            (1e3, "msec"),
            (1e6, "μsec"),
            (1e9, "nsec"),
        ]
    else:
        c_and_unit = [
            (1.0, "s " if right_aligned else "s"),
            (1e3, "ms"),
            (1e6, "μs"),
            (1e9, "ns"),
        ]

    # --- main loop ---------------------------------------
    value_str, unit_str = "", ""
    for c, unit in c_and_unit:
        value = round(dt_sec * c, ndigits=n_digits)
        if (value < 1000) or (value_str == ""):
            unit_str = unit
            if n_digits > 0:
                value_str = f"{value:.{n_digits}f}".lstrip()
            else:
                value_str = str(int(value))

    # --- final result ------------------------------------
    if spaced:
        return f"{value_str} {unit_str}"
    else:
        return f"{value_str}{unit_str}"
