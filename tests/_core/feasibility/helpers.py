import numpy as np

from max_div._core.constraints import Constraint, to_numpy_constraints


def constraint_arrays(cons: list[Constraint], n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert constraints to (con_values, con_indices, weights) as the pipeline ingests them."""
    con_values, con_indices = to_numpy_constraints(cons, n)
    weights = np.array([con.weight for con in cons], dtype=np.float64)
    return con_values, con_indices, weights
