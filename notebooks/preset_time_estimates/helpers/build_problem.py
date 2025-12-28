import numpy as np

from max_div.solver import Constraint, MaxDivProblem
from max_div.solver._distance import DistanceMetric
from max_div.solver._diversity import DiversityMetric


def construct_problem(n: int, k: int, m: int, n_con_indices: int) -> MaxDivProblem:
    # --- validation ------------------
    if k > n:
        raise ValueError(f"k={k} cannot be larger than n={n}.")
    if n_con_indices <= m:
        raise ValueError(f"n_con_indices={n_con_indices} must be larger than or equal to m={m}.")
    if n_con_indices >= (n * m):
        raise ValueError(f"n_con_indices={n_con_indices} must be smaller than n*m={n}*{m}={n * m}.")

    # --- init ------------------------
    d = 1
    np.random.seed(42)

    # --- vectors ---------------------
    vectors = np.random.random((n, d)).astype(np.float32)

    # --- constraints -----------------
    n_con_ind_remaining = n_con_indices
    m_remaining = m
    constraints: list[Constraint] = []
    for _ in range(m):
        # determine size of index set
        con_size = n_con_ind_remaining // m_remaining

        # add constraint
        min_count = max(1, min(n, con_size) // 4)
        max_count = max(min_count, min(n, con_size) // 2)
        constraints.append(
            Constraint(
                int_set=set(np.random.choice(n, size=con_size, replace=False).tolist()),
                min_count=min_count,
                max_count=max_count,
            )
        )

        # update tracking variables
        n_con_ind_remaining -= con_size
        m_remaining -= 1

    # --- build problem ---------------
    problem = MaxDivProblem(
        vectors=vectors,
        k=k,
        distance_metric=DistanceMetric.L1_MANHATTAN,
        diversity_metric=DiversityMetric.approx_geomean_separation(),
        constraints=constraints,
    )
    return problem
