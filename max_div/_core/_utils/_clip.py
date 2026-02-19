def clip(value: float | int, min_value: float | int, max_value: float | int) -> float | int:
    if min_value >= max_value:
        return (min_value + max_value) / 2
    else:
        return max(min_value, min(max_value, value))
