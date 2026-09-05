"""Build the geometric sequences of solve budgets used for anytime measurements."""


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
