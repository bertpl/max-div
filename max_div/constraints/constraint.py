from dataclasses import dataclass


@dataclass
class Constraint:
    """Constraint indicating we want to sample at least `min_count` and at most `max_count` integers from `int_set`."""

    int_set: set[int]
    min_count: int
    max_count: int
