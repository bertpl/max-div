import numpy as np


def is_non_bool_int(value: object) -> bool:
    """Return True for a Python or numpy integer, and False for a bool, which Python counts as an integer."""
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)
