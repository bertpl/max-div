import numpy as np


def normalize_p(p: np.ndarray) -> np.ndarray:
    """Normalize the probability array p (making a copy) in-place so that its maximum value is 1.0."""
    p_max = p.max()
    p_out = p.copy()
    if p_max > 0:
        for i in range(p.size):
            p_out[i] /= p_max
        return p_out
    else:
        return p_out
