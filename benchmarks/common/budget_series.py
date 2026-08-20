"""Budget series: the geometric sequences of solve budgets used for anytime measurements."""


def time_budget_series(t_min_sec: float = 0.001, t_max_sec: float = 1.0, factor: float = 2.0) -> list[float]:
    """Build a geometric wall-clock budget series, in seconds.

    Args:
        t_min_sec: First budget of the series.
        t_max_sec: Limit; the last value is the first one >= this, so the
            limit itself is always covered.
        factor: Multiplicative step between consecutive budgets.

    Returns:
        Increasing list of budgets in seconds.
    """
    if t_min_sec <= 0 or t_max_sec < t_min_sec or factor <= 1.0:
        raise ValueError("A budget series requires 0 < t_min_sec <= t_max_sec and factor > 1.")
    budgets = [t_min_sec]
    while budgets[-1] < t_max_sec:
        budgets.append(budgets[-1] * factor)
    return budgets


def iteration_budget_series(i_min: int = 10, i_max: int = 100_000, factor: float = 2.0) -> list[int]:
    """Build a geometric iteration-count budget series.

    Args:
        i_min: First budget of the series.
        i_max: Limit; the last value is the first one >= this.
        factor: Multiplicative step between consecutive budgets.

    Returns:
        Increasing list of iteration counts (deduplicated after rounding).
    """
    if i_min <= 0 or i_max < i_min or factor <= 1.0:
        raise ValueError("A budget series requires 0 < i_min <= i_max and factor > 1.")
    budgets = [i_min]
    while budgets[-1] < i_max:
        nxt = max(budgets[-1] + 1, round(budgets[-1] * factor))
        budgets.append(nxt)
    return budgets
