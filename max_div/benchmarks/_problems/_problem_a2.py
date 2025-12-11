from typing import Any

import numpy as np

from max_div.benchmarks._registry import BenchmarkProblem
from max_div.solver import DistanceMetric, DiversityMetric, MaxDivProblem


# =================================================================================================
#  A2 - Gaussian - Unconstrained
# =================================================================================================
class BenchmarkProblem_A2(BenchmarkProblem):
    @classmethod
    def name(cls) -> str:
        return "A2"

    @classmethod
    def description(cls) -> str:
        return "Unconstrained problem with non-uniform vector density (gaussian distribution)"

    @classmethod
    def supported_params(cls) -> dict[str, str]:
        return dict(
            size="(int) value in [1, ...].  Problem size, with d=size, n=100*size, k=10*size",
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
        vectors = np.random.randn(n, d).astype(np.float32)

        return MaxDivProblem(
            vectors=vectors,
            k=k,
            distance_metric=DistanceMetric.L2_EUCLIDEAN,
            diversity_metric=diversity_metric,
            constraints=[],
        )
