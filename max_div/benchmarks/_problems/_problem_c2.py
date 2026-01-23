from typing import Any

import numpy as np

from max_div.benchmarks._registry import BenchmarkProblem
from max_div.solver import Constraint, DistanceMetric, DiversityMetric, MaxDivProblem

from ._helpers import sort_vectors


# =================================================================================================
#  C2 - Non-Uniform - Medium-Easy constraints
# =================================================================================================
class BenchmarkProblem_C2(BenchmarkProblem):
    @classmethod
    def name(cls) -> str:
        return "C2"

    @classmethod
    def description(cls) -> str:
        return "Problem with semi-non-uniform vector density and simple constraints"

    @classmethod
    def supported_params(cls) -> dict[str, str]:
        return dict(
            size="(int) value in [1, ...].  Problem size, with d=2, n=100*size, k=10*size, m=2*size",
            diversity_metric="(DiversityMetric) diversity metric to be maximized",
        )

    @classmethod
    def get_example_parameters(cls) -> dict[str, Any]:
        return dict(
            size=1,
            diversity_metric=DiversityMetric.approx_geomean_separation(),
        )

    @classmethod
    def get_problem_dimensions(cls, **kwargs) -> tuple[int, int, int, int, int]:
        size = kwargs.get("size")
        d = 2
        n = 100 * size
        k = 10 * size
        m = 2 * size
        n_con_indices = n
        return d, n, k, m, n_con_indices

    @classmethod
    def _create_problem_instance(cls, size: int, diversity_metric: DiversityMetric, **kwargs) -> MaxDivProblem:
        d, n, k, m, _ = cls.get_problem_dimensions(size=size)

        # Generate semi-non-uniform random vectors (uniform + gaussian)
        np.random.seed(42)
        uniform_col = np.random.rand(n, 1)
        gaussian_col = np.random.randn(n, 1)
        vectors = np.concatenate((uniform_col, gaussian_col), axis=1).astype(np.float32)
        vectors = sort_vectors(vectors)  # sort by increasing L2 norm of rows

        # Generate constraints
        constraints: list[Constraint] = []
        for i in range(m):
            # generate m bands [v_min, v_max] spanning dimension 0   (total range [0,1])
            # add specify constraint that EXACTLY 5 samples should be taken from each band
            # (k=5*m and n=50*m, so this should always be feasible)
            v_min, v_max = i / m, (i + 1) / m  # range of values in dimension 0
            indices_in_range = [idx for idx in range(n) if v_min <= vectors[idx, 0] < v_max]
            constraints.append(Constraint(int_set=set(indices_in_range), min_count=5, max_count=5))

        return MaxDivProblem(
            vectors=vectors,
            k=k,
            distance_metric=DistanceMetric.L2_EUCLIDEAN,
            diversity_metric=diversity_metric,
            constraints=constraints,
        )
