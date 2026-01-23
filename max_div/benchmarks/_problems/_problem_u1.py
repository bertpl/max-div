from typing import Any

import numpy as np

from max_div.benchmarks._registry import BenchmarkProblem
from max_div.solver import DistanceMetric, DiversityMetric, MaxDivProblem

from ._helpers import sort_vectors


# =================================================================================================
#  U1 - Uniform - Unconstrained
# =================================================================================================
class BenchmarkProblem_U1(BenchmarkProblem):
    @classmethod
    def name(cls) -> str:
        return "U1"

    @classmethod
    def description(cls) -> str:
        return "Unconstrained problem with uniform vector density"

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
    def get_problem_dimensions(cls, **kwargs) -> tuple[int, int, int, int, int]:
        size = kwargs.get("size")
        d = size
        n = 100 * size
        k = 10 * size
        m = 0
        n_con_indices = 0
        return d, n, k, m, n_con_indices

    @classmethod
    def _create_problem_instance(cls, size: int, diversity_metric: DiversityMetric, **kwargs) -> MaxDivProblem:
        d, n, k, _, _ = cls.get_problem_dimensions(size=size)

        # Generate uniform random vectors
        np.random.seed(42)
        vectors = np.random.random_sample(size=(n, d)).astype(np.float32)
        vectors = sort_vectors(vectors)  # sort by increasing L2 norm of rows

        return MaxDivProblem(
            vectors=vectors,
            k=k,
            distance_metric=DistanceMetric.L2_EUCLIDEAN,
            diversity_metric=diversity_metric,
            constraints=[],
        )
