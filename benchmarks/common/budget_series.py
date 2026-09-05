"""Build the sequences of solve budgets used for anytime measurements."""


def time_budget_series(t_min_sec: float = 0.001, t_max_sec: float = 1.0, factor: float = 2.0) -> list[float]:
    """Build a geometric series of wall-clock budgets, in seconds.

    Args:
        t_max_sec: Largest budget to cover; the series ends at the first value >= this.

    Returns:
        Increasing list of budgets in seconds.

    Raises:
        ValueError: If t_min_sec <= 0, t_max_sec < t_min_sec, or factor <= 1.
    """
    if t_min_sec <= 0 or t_max_sec < t_min_sec or factor <= 1.0:
        raise ValueError("A budget series requires 0 < t_min_sec <= t_max_sec and factor > 1.")
    budgets = [t_min_sec]
    while budgets[-1] < t_max_sec:
        budgets.append(budgets[-1] * factor)
    return budgets


def grid_budget_series(t_min_sec: float, t_max_sec: float) -> list[float]:
    """Build a 1-2-5 series of wall-clock budgets from `t_min_sec`, ending exactly at `t_max_sec`.

    The series walks the 1-2-5 sequence from `t_min_sec`, which must itself lie on it, and ends at
    `t_max_sec` itself, so the largest budget is the one the results are judged at: every grid
    value within a factor 2 below `t_max_sec` is dropped, and `t_max_sec` replaces them.

    Raises:
        ValueError: If `t_min_sec` is not a 1-2-5 value, or `t_max_sec` <= `t_min_sec`.
    """
    if t_max_sec <= t_min_sec:
        raise ValueError("A 1-2-5 budget series requires t_max_sec > t_min_sec.")
    mantissa, exponent = _split_grid_value(t_min_sec)
    budgets: list[float] = []
    while (value := round(mantissa * 10.0**exponent, 12)) * 2 <= t_max_sec:
        budgets.append(value)
        mantissa, exponent = (2, exponent) if mantissa == 1 else (5, exponent) if mantissa == 2 else (1, exponent + 1)
    budgets.append(t_max_sec)
    return budgets


def _split_grid_value(value: float) -> tuple[int, int]:
    """Return the (mantissa, exponent) of a 1-2-5 grid value, mantissa in {1, 2, 5}."""
    for exponent in range(-6, 7):
        for mantissa in (1, 2, 5):
            if abs(value - mantissa * 10.0**exponent) <= 1e-9 * max(1.0, value):
                return mantissa, exponent
    raise ValueError(f"{value} is not a 1-2-5 grid value.")


def iteration_budget_series(i_min: int = 10, i_max: int = 100_000, factor: float = 2.0) -> list[int]:
    """Build a geometric series of iteration-count budgets.

    Args:
        i_max: Largest budget to cover; the series ends at the first value >= this.

    Returns:
        Increasing list of iteration counts (rounded; each value is forced to exceed the previous one).

    Raises:
        ValueError: If i_min <= 0, i_max < i_min, or factor <= 1.
    """
    if i_min <= 0 or i_max < i_min or factor <= 1.0:
        raise ValueError("A budget series requires 0 < i_min <= i_max and factor > 1.")
    budgets = [i_min]
    while budgets[-1] < i_max:
        next_budget = max(budgets[-1] + 1, round(budgets[-1] * factor))
        budgets.append(next_budget)
    return budgets
