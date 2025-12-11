from typing import Any

import numpy as np

from max_div.benchmarks._registry import BenchmarkProblem
from max_div.solver import Constraint, DistanceMetric, DiversityMetric, MaxDivProblem


# =================================================================================================
#  A3 - Gaussian - Simple constraints
# =================================================================================================
class BenchmarkProblem_A3(BenchmarkProblem):
    @classmethod
    def name(cls) -> str:
        return "A3"

    @classmethod
    def description(cls) -> str:
        return "Problem with non-uniform vector density (gaussian distribution) and simple constrains"

    @classmethod
    def supported_params(cls) -> dict[str, str]:
        return dict(
            size="(int) value in [1, ...].  Problem size, with d=size, n=100*size, k=10*size, m=2*size",
            diversity_metric="(DiversityMetric) diversity metric to be maximized",
        )

    @classmethod
    def get_example_parameters(cls) -> dict[str, Any]:
        return dict(
            size=1,
            diversity_metric=DiversityMetric.approx_geomean_separation(),
        )

    @classmethod
    def _create_problem_instance(cls, size: int, diversity_metric: DiversityMetric, **kwargs) -> MaxDivProblem:
        d = size
        n = 100 * size
        k = 10 * size

        # Generate gaussian random vectors
        np.random.seed(42)
        vectors = np.random.randn(n, d).astype(np.float32) + 1.0  # shift by 1.0 (distribution of signs ~84%-16%)

        # Generate constraints
        constraints: list[Constraint] = []
        for i in range(d):
            # half of k samples should have positive or 0 value in dimension i
            indices_positive = [idx for idx in range(n) if vectors[idx, i] >= 0.0]
            constraints.append(Constraint(int_set=set(indices_positive), min_count=k // 2, max_count=k))

            # half of k samples should have negative or 0 value in dimension i
            indices_negative = [idx for idx in range(n) if vectors[idx, i] <= 0.0]
            constraints.append(Constraint(int_set=set(indices_negative), min_count=k // 2, max_count=k))

        return MaxDivProblem(
            vectors=vectors,
            k=k,
            distance_metric=DistanceMetric.L2_EUCLIDEAN,
            diversity_metric=diversity_metric,
            constraints=constraints,
        )
