from dataclasses import dataclass

import numpy as np

from max_div.solver import MaxDivProblem

from .build_problem import construct_problem


@dataclass(frozen=True)
class ProblemDimensions:
    n: int  # number of items
    k: int  # number of selections
    m: int  # number of constraints
    n_con_indices: int  # number of constraint indices

    def build_problem(self) -> MaxDivProblem:
        return construct_problem(n=self.n, k=self.k, m=self.m, n_con_indices=self.n_con_indices)


def get_problem_dimensions(n_grid: int = 8) -> list[ProblemDimensions]:
    dimension_ranges: list[ProblemDimensions] = []
    for n in int_log_range(5, 2_000, n_grid):
        for k in int_log_range(3, 1_000, n_grid):
            for m in [0] + int_log_range(1, 1_000, n_grid - 1):
                for n_con_indices in [0] + int_log_range(1, 100_000, n_grid - 1):
                    if k >= n:
                        continue  # skip invalid case
                    if not (m < n_con_indices < n * m):
                        continue  # skip invalid case
                    dimension_ranges.append(
                        ProblemDimensions(
                            n=n,
                            k=k,
                            m=m,
                            n_con_indices=n_con_indices,
                        )
                    )
    return dimension_ranges


def int_log_range(lb: int, ub: int, n: int) -> list[int]:
    return sorted(set(int(x) for x in np.logspace(np.log10(lb), np.log10(ub), num=n)))
