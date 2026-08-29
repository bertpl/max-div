import math

import numpy as np

from max_div._core.benchmark_problems._registry import BenchmarkProblem
from max_div._core.metrics import DistanceMetric, DiversityMetric
from max_div._core.problem import VectorMaxDivProblem

from ._helpers import sort_vectors


# =================================================================================================
#  U2 - Uniform - Unconstrained
# =================================================================================================
class BenchmarkProblem_U2(BenchmarkProblem):
    """U2 draws vectors uniformly over the unit hypercube."""

    @classmethod
    def name(cls) -> str:
        return "U2"

    @classmethod
    def description(cls) -> str:
        return "Unconstrained problem with uniform vector density"

    @classmethod
    def _get_problem_dimensions(cls, n: int) -> tuple[int, int, int, int, int]:
        d = math.ceil(n / 100)
        k = math.ceil(n / 10)
        m = 0
        n_con_indices = 0
        return d, n, k, m, n_con_indices

    @classmethod
    def _create_problem_instance(cls, n: int, diversity_metric: DiversityMetric) -> VectorMaxDivProblem:
        d, _, k, _, _ = cls._get_problem_dimensions(n)

        # Generate uniform random vectors
        np.random.seed(42)
        vectors = np.random.random_sample(size=(n, d)).astype(np.float32)
        vectors = sort_vectors(vectors)  # sort by increasing L2 norm of rows

        return VectorMaxDivProblem(
            vectors=vectors,
            k=k,
            distance_metric=DistanceMetric.l2_euclidean(),
            diversity_metric=diversity_metric,
            constraints=[],
        )
