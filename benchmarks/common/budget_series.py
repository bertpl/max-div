"""Budget series: the geometric sequences of solve budgets used for anytime measurements."""


def time_budget_series(t_min_sec: float = 0.001, t_max_sec: float = 1.0, factor: float = 2.0) -> list[float]:
    """Build a geometric wall-clock budget series, in seconds.

    Args:
        t_min_sec: Bottom rung (first budget).
        t_max_sec: Ceiling; the last rung is the first value >= this, so the
            ceiling itself is always covered.
        factor: Multiplicative step between rungs.

    Returns:
        Increasing list of budgets in seconds.
    """
    if t_min_sec <= 0 or t_max_sec < t_min_sec or factor <= 1.0:
        raise ValueError("A budget series requires 0 < t_min_sec <= t_max_sec and factor > 1.")
    rungs = [t_min_sec]
    while rungs[-1] < t_max_sec:
        rungs.append(rungs[-1] * factor)
    return rungs


def iteration_budget_series(i_min: int = 10, i_max: int = 100_000, factor: float = 2.0) -> list[int]:
    """Build a geometric iteration-count budget series.

    Args:
        i_min: Bottom rung (first budget).
        i_max: Ceiling; the last rung is the first value >= this.
        factor: Multiplicative step between rungs.

    Returns:
        Increasing list of iteration counts (deduplicated after rounding).
    """
    if i_min <= 0 or i_max < i_min or factor <= 1.0:
        raise ValueError("A budget series requires 0 < i_min <= i_max and factor > 1.")
    rungs = [i_min]
    while rungs[-1] < i_max:
        nxt = max(rungs[-1] + 1, round(rungs[-1] * factor))
        rungs.append(nxt)
    return rungs
