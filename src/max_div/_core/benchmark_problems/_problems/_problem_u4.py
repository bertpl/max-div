import math

import numpy as np

from max_div._core.benchmark_problems._registry import BenchmarkProblem
from max_div._core.metrics import DistanceMetric, DiversityMetric
from max_div._core.problem import VectorMaxDivProblem

from ._helpers import sort_vectors


# =================================================================================================
#  U4 - Conic - Unconstrained
# =================================================================================================
class BenchmarkProblem_U4(BenchmarkProblem):
    """U4 draws vectors in a cone around the all-ones direction.

    Volumetric density concentrates toward the cone's tip, ever more sharply as d grows.
    """

    @classmethod
    def name(cls) -> str:
        return "U4"

    @classmethod
    def description(cls) -> str:
        return "Unconstrained problem with non-uniform vector density (conic)"

    @classmethod
    def _get_problem_dimensions(cls, n: int) -> tuple[int, int, int, int, int]:
        d = math.ceil(n / 100)
        k = math.ceil(n / 10)
        m = 0
        n_con_indices = 0
        return d, n, k, m, n_con_indices

    @classmethod
    def _create_problem_instance(cls, n: int, diversity_metric: DiversityMetric) -> VectorMaxDivProblem:
        """We will generate vectors in d-dim. space as a * ([1, 1, 1, ..., 1] + (r * [x1, x2, ..., xd])).

        - a is sampled uniformly in [0,1]
        - xi-values are sampled uniformly in the hyper-box [-1, +1]^d and then rescaled to have L2 norm = 1
        - r = 0.1 * sqrt(d), such that the perceived angle of the cone from the origin remains constant as d increases
        """
        d, _, k, _, _ = cls._get_problem_dimensions(n)
        r = 0.1 * np.sqrt(d)

        # step 1 - generate n x (d+1) matrix of random values in [0,1]
        np.random.seed(42)
        random_data = np.random.random_sample(size=(n, d + 1)).astype(np.float32)

        # step 2 - generate vectors as described above
        vectors = np.empty(shape=(n, d), dtype=np.float32)
        for i in range(n):
            a = random_data[i, 0]
            x_vals = (random_data[i, 1:] * 2.0) - 1.0  # map [0,1] to [-1,+1]
            x_norm = np.linalg.norm(x_vals, ord=2)
            if x_norm > 0:
                x_vals = (x_vals / x_norm) * r  # rescale to have L2 norm = a*r
            vectors[i, :] = a * (1 + x_vals)

        # step 3 - sort vectors by increasing L2 norm of rows & return problem instance
        vectors = sort_vectors(vectors)  # sort by increasing L2 norm of rows
        return VectorMaxDivProblem(
            vectors=vectors,
            k=k,
            distance_metric=DistanceMetric.L2_EUCLIDEAN,
            diversity_metric=diversity_metric,
            constraints=[],
        )
